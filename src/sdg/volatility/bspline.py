"""B-spline basis functions and dual parameterization for CVI.

This module provides utilities for:
- Building clamped B-spline knot vectors
- Evaluating B-spline basis functions and their derivatives
- Linear extrapolation outside the knot range
- Dual transformation between B-spline and cubic spline parameter spaces

References:
    Deschatres (2025), "Convex Volatility Interpolation", Sections 2.1-2.3
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


def build_knot_vector(breakpoints: np.ndarray, degree: int = 3) -> np.ndarray:
    """Build a clamped B-spline knot vector from breakpoints.

    For n breakpoints and degree 3, the clamped knot vector repeats the
    first and last breakpoints to produce n + 2 basis functions.

    Args:
        breakpoints: Spline breakpoints (knots) z_0, ..., z_{n-1},
            sorted ascending.
        degree: B-spline degree (3 = cubic).

    Returns:
        Augmented knot vector of length n + 2 * degree.
    """
    return np.concatenate([
        np.full(degree, breakpoints[0]),
        breakpoints,
        np.full(degree, breakpoints[-1]),
    ])


def eval_basis(
    z: np.ndarray,
    knot_vector: np.ndarray,
    degree: int = 3,
    deriv: int = 0,
) -> np.ndarray:
    """Evaluate B-spline basis functions or their derivatives.

    Args:
        z: Evaluation points, shape (p,).
        knot_vector: Augmented B-spline knot vector.
        degree: B-spline degree.
        deriv: Derivative order (0 = value, 1 = first, 2 = second).

    Returns:
        Design matrix of shape (p, n_basis) where
        n_basis = len(knot_vector) - degree - 1.
    """
    n_basis = len(knot_vector) - degree - 1
    z = np.atleast_1d(np.asarray(z, dtype=float))
    result = np.zeros((len(z), n_basis))

    for j in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[j] = 1.0
        spline = BSpline(knot_vector, coeffs, degree, extrapolate=False)
        if deriv > 0:
            spline = spline.derivative(deriv)
        vals = spline(z)
        result[:, j] = np.nan_to_num(vals, nan=0.0)

    return result


def eval_basis_extrap(
    z: np.ndarray,
    knot_vector: np.ndarray,
    breakpoints: np.ndarray,
    degree: int = 3,
) -> np.ndarray:
    """Evaluate B-spline basis with linear extrapolation outside breakpoints.

    Inside [z_0, z_{n-1}]: standard B-spline evaluation.
    Outside: linear extrapolation v(z) = v(boundary) + v'(boundary) * (z - boundary).

    This matches the CVI assumption that variance is linear in log-strike
    beyond the edge knots (Section 2.1, boundary conditions).

    Args:
        z: Evaluation points, shape (p,).
        knot_vector: Augmented B-spline knot vector.
        breakpoints: Original breakpoints z_0, ..., z_{n-1}.
        degree: B-spline degree.

    Returns:
        Design matrix of shape (p, n_basis).
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    z_left, z_right = breakpoints[0], breakpoints[-1]

    # Precompute basis values and first derivatives at boundaries
    B_left = eval_basis(np.array([z_left]), knot_vector, degree, 0)[0]
    B_left_d = eval_basis(np.array([z_left]), knot_vector, degree, 1)[0]
    B_right = eval_basis(np.array([z_right]), knot_vector, degree, 0)[0]
    B_right_d = eval_basis(np.array([z_right]), knot_vector, degree, 1)[0]

    # Evaluate at clipped z for interior points
    z_clipped = np.clip(z, z_left, z_right)
    result = eval_basis(z_clipped, knot_vector, degree, 0)

    # Left extrapolation: v(z) = v(z_left) + v'(z_left) * (z - z_left)
    left_mask = z < z_left
    if np.any(left_mask):
        dz = (z[left_mask] - z_left)[:, np.newaxis]
        result[left_mask] = B_left[np.newaxis, :] + dz * B_left_d[np.newaxis, :]

    # Right extrapolation: v(z) = v(z_right) + v'(z_right) * (z - z_right)
    right_mask = z > z_right
    if np.any(right_mask):
        dz = (z[right_mask] - z_right)[:, np.newaxis]
        result[right_mask] = B_right[np.newaxis, :] + dz * B_right_d[np.newaxis, :]

    return result


def build_dual_transform(
    breakpoints: np.ndarray,
    knot_vector: np.ndarray,
    degree: int = 3,
) -> np.ndarray:
    """Build the linear transformation between B-spline and cubic spline spaces.

    The cubic spline parameters (physical space) are:
        [v(z=0), dv/dz(z=0), d2v/dz2(z_0), ..., d2v/dz2(z_{n-1})]

    The B-spline parameters (mathematical space) are:
        [alpha_0, alpha_1, ..., alpha_{n+1}]

    Returns M such that: cubic_params = M @ bspline_weights

    The matrix M is constructed by evaluating the B-spline basis functions
    and their derivatives at the relevant points:
    - Row 0: basis values at z=0 (gives ATM variance)
    - Row 1: first derivatives at z=0 (gives ATM skew dv/dz)
    - Rows 2..n+1: second derivatives at each knot (gives convexities)

    References:
        Deschatres (2025), Section 2.3 (Dual Parameterization, Figure 5)

    Args:
        breakpoints: Spline breakpoints z_0, ..., z_{n-1}.
        knot_vector: Augmented B-spline knot vector.
        degree: B-spline degree.

    Returns:
        Transformation matrix M of shape (n+2, n+2).
    """
    # Row 0: v(z=0) — ATM variance
    row_v_atm = eval_basis(np.array([0.0]), knot_vector, degree, deriv=0)[0]

    # Row 1: dv/dz(z=0) — ATM skew
    row_dv_atm = eval_basis(np.array([0.0]), knot_vector, degree, deriv=1)[0]

    # Rows 2..n+1: d2v/dz2(z_i) — convexities at each knot
    rows_d2v = eval_basis(breakpoints, knot_vector, degree, deriv=2)

    return np.vstack([row_v_atm[np.newaxis, :], row_dv_atm[np.newaxis, :], rows_d2v])
