"""Debug script for CVI calibration on a single expiry."""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from pathlib import Path

from sdg.market_data.barchart.pipeline import load_from_csv
from sdg.volatility.cvi import CVICalibrator, estimate_anchor_atm_vol, evaluate_vol
from sdg.volatility.types import CVIConfig

# Load data
fixtures = Path(__file__).parent.parent / "tests" / "fixtures"

with open(fixtures / "ko_quotes_20260203.json") as f:
    data = json.load(f)
    forwards = data["forwards"]
    valuation_date = date.fromisoformat(data["valuation_date"])

expiries = load_from_csv(
    fixtures / "ko_options_20260203.csv",
    forwards,
    valuation_date=valuation_date,
    rate=0.03,
    min_strikes=3,
)

# Select expiry to debug (change index as needed)
EXPIRY_INDEX = 0
exp = expiries[EXPIRY_INDEX]

print(f"=== Debugging Expiry {EXPIRY_INDEX + 1} ===")
print(f"Valuation date: {valuation_date}")
print(f"Time to expiry: {exp.time_to_expiry:.4f} ({int(exp.time_to_expiry * 365)} days)")
print(f"Forward: {exp.forward}")
print(f"Discount factor: {exp.discount_factor:.6f}")
print()

# Market data
print("=== Market Data ===")
print(f"Strikes: {len(exp.strikes)}")
print(f"  Range: [{exp.strikes.min():.0f}, {exp.strikes.max():.0f}]")
print()

# Estimate ATM vol
atm_vol = estimate_anchor_atm_vol(exp)
print(f"Estimated ATM vol (anchor): {atm_vol:.2%}")
exp.anchor_atm_vol = atm_vol
print()

# Show bid/ask vols
print("Strike     Bid Vol    Ask Vol    Mid Vol    Log-M")
print("-" * 55)
for i, k in enumerate(exp.strikes):
    bid = exp.bid_vols[i]
    ask = exp.ask_vols[i]
    mid = (bid + ask) / 2 if np.isfinite(bid) and np.isfinite(ask) else np.nan
    log_m = np.log(k / exp.forward)
    print(f"{k:>6.0f}    {bid:>7.2%}    {ask:>7.2%}    {mid:>7.2%}    {log_m:>6.3f}")
print()

# Run calibration
print("=== Calibration ===")
config = CVIConfig()  # Use defaults
print(f"Config: n_knots={config.n_knots}, z_range={config.z_range}, reg={config.regularization}")
print(f"        calendar_penalty={config.calendar_penalty}, max_iter={config.max_iterations}")
print()

calibrator = CVICalibrator(config)
try:
    result = calibrator.calibrate([exp])
    print("Calibration successful!")
    print()

    # Show calibrated vols
    print("=== Calibrated Surface ===")
    print(f"B-spline weights: {result.bspline_weights.shape}")
    print(f"Breakpoints: {result.breakpoints}")
    print()

    print("Strike     Mkt Mid    Cal Vol    Diff")
    print("-" * 45)
    cal_vols = evaluate_vol(result, exp.strikes, 0)
    mid_vols = np.array([
        (exp.bid_vols[i] + exp.ask_vols[i]) / 2
        if np.isfinite(exp.bid_vols[i]) and np.isfinite(exp.ask_vols[i])
        else np.nan
        for i in range(len(exp.strikes))
    ])
    for i, k in enumerate(exp.strikes):
        mid = mid_vols[i]
        cal = cal_vols[i]
        diff = cal - mid if np.isfinite(mid) else np.nan
        print(f"{k:>6.0f}    {mid:>7.2%}    {cal:>7.2%}    {diff:>+6.2%}" if np.isfinite(diff) else f"{k:>6.0f}    {mid:>7.2%}    {cal:>7.2%}       N/A")

    print()
    atm_cal = evaluate_vol(result, np.array([exp.forward]), 0)[0]
    print(f"ATM calibrated vol: {atm_cal:.2%}")

    # === Plot ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

    # Top plot: Market vs Calibrated vols
    ax1.plot(exp.strikes, mid_vols * 100, 'o-', label='Market Mid', color='blue', markersize=6)
    ax1.fill_between(exp.strikes, exp.bid_vols * 100, exp.ask_vols * 100,
                     alpha=0.2, color='blue', label='Bid-Ask')

    # Smooth calibrated curve
    strikes_smooth = np.linspace(exp.strikes.min(), exp.strikes.max(), 100)
    cal_vols_smooth = evaluate_vol(result, strikes_smooth, 0)
    ax1.plot(strikes_smooth, cal_vols_smooth * 100, '-', label='Calibrated', color='red', linewidth=2)

    ax1.axvline(exp.forward, color='gray', linestyle='--', alpha=0.5, label=f'Forward={exp.forward:.0f}')
    ax1.set_xlabel('Strike')
    ax1.set_ylabel('Implied Volatility (%)')
    ax1.set_title(f'Expiry {EXPIRY_INDEX + 1}: T={exp.time_to_expiry:.4f} ({int(exp.time_to_expiry * 365)} days)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Calibration error
    errors = (cal_vols - mid_vols) * 100
    ax2.bar(exp.strikes, errors, width=40, color=['green' if e >= 0 else 'red' for e in errors], alpha=0.7)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Strike')
    ax2.set_ylabel('Error (%)')
    ax2.set_title('Calibration Error (Calibrated - Market)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'debug_expiry_{EXPIRY_INDEX + 1}.png', dpi=150)
    print(f"\nChart saved to: debug_expiry_{EXPIRY_INDEX + 1}.png")
    plt.show()

except Exception as e:
    print(f"Calibration failed: {e}")
    import traceback
    traceback.print_exc()
