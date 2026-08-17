"""Reusable PES fitting utilities extracted from the project notebooks.

Data arrays are expected to have columns R, theta1, theta2, phi, energy. The
module intentionally does not provide text-file input handling for new model
classes; callers should read/normalize each dataset before constructing a class.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import numpy
import scipy
import scipy.interpolate
import scipy.linalg
import scipy.special
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional for headless use.
    plt = None

# Standalone dependencies for the second-pass hysteresis acquisition cell below.
# Run this cell, then the next cell, without rerunning the entire notebook.

# Shared fitting utilities copied from CO2_H2_fitting.ipynb.
# This cell intentionally defines helpers only; it does not load data or run a fit.





DEFAULT_MAX_L1 = 24
DEFAULT_MAX_L2 = 6
DEFAULT_TERMS = 158
DEFAULT_TRAIN_FRACTION = 0.80
DEFAULT_RANDOM_SEED = 202605702
DEFAULT_RELATIVE_RMSE_CUTOFF = 0.01
DEFAULT_SVD_THRESHOLD = 1e-8
DEFAULT_THRESHOLD_MODE = "relative"


def _phase(power):
    return -1.0 if power % 2 else 1.0


@lru_cache(maxsize=None)
def wigner_3j(j1, j2, j3, m1, m2, m3):
    """Compute Wigner 3j symbols for integer quantum numbers."""
    if m1 + m2 + m3 != 0:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0

    delta = (
        math.factorial(j1 + j2 - j3)
        * math.factorial(j1 - j2 + j3)
        * math.factorial(-j1 + j2 + j3)
        / math.factorial(j1 + j2 + j3 + 1)
    )
    norm = math.sqrt(
        delta
        * math.factorial(j1 + m1)
        * math.factorial(j1 - m1)
        * math.factorial(j2 + m2)
        * math.factorial(j2 - m2)
        * math.factorial(j3 + m3)
        * math.factorial(j3 - m3)
    )

    z_min = max(0, j2 - j3 - m1, j1 + m2 - j3)
    z_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)

    total = 0.0
    for z in range(z_min, z_max + 1):
        denom = (
            math.factorial(z)
            * math.factorial(j1 + j2 - j3 - z)
            * math.factorial(j1 - m1 - z)
            * math.factorial(j2 + m2 - z)
            * math.factorial(j3 - j2 + m1 + z)
            * math.factorial(j3 - j1 - m2 + z)
        )
        total += _phase(z) / denom

    return _phase(j1 - j2 - m3) * norm * total


def normalized_associated_legendre(l, m, x):
    # scipy.special.lpmv includes the Condon-Shortley phase used by the paper's normalization.
    normalization = math.sqrt((2 * l + 1) / 2 * math.factorial(l - m) / math.factorial(l + m))
    return normalization * scipy.special.lpmv(m, l, x)


def angular_basis(L, l1, l2, theta1, theta2, phi):
    x1 = numpy.cos(theta1)
    x2 = numpy.cos(theta2)

    value = (
        _phase(l1 - l2)
        * wigner_3j(l1, l2, L, 0, 0, 0)
        * normalized_associated_legendre(l1, 0, x1)
        * normalized_associated_legendre(l2, 0, x2)
    )

    for m in range(1, min(l1, l2) + 1):
        value += (
            2
            * _phase(m + l1 - l2)
            * wigner_3j(l1, l2, L, m, -m, 0)
            * normalized_associated_legendre(l1, m, x1)
            * normalized_associated_legendre(l2, m, x2)
            * numpy.cos(m * phi)
        )

    return math.sqrt((2 * L + 1) / (4 * math.pi)) * value


def generate_all_terms(max_l1, max_l2):
    valid_terms = []
    for l1 in range(0, max_l1 + 1, 2):
        for l2 in range(0, max_l2 + 1, 2):
            for L in range(abs(l1 - l2), l1 + l2 + 1):
                if (l1 + l2 + L) % 2 != 0:
                    continue
                valid_terms.append((L, l1, l2))

    valid_terms.sort(key=lambda term: (sum(term), term[0], term[1], term[2]))
    return valid_terms


def generate_terms(max_l1, max_l2, terms=None):
    valid_terms = generate_all_terms(max_l1, max_l2)
    if terms is None:
        return valid_terms
    if terms > len(valid_terms):
        raise ValueError(f"Requested {terms} terms, but only {len(valid_terms)} valid terms exist.")
    return valid_terms[:terms]


def design_matrix_for(data_subset, basis_terms):
    subset_theta1 = numpy.deg2rad(data_subset[:, 1])
    subset_theta2 = numpy.deg2rad(data_subset[:, 2])
    subset_phi = numpy.deg2rad(data_subset[:, 3])

    return numpy.column_stack([
        angular_basis(L, l1, l2, subset_theta1, subset_theta2, subset_phi)
        for L, l1, l2 in basis_terms
    ])


def relative_rmse(residuals, targets):
    residuals = numpy.asarray(residuals, dtype=float)
    targets = numpy.asarray(targets, dtype=float)
    target_rms = float(numpy.sqrt(numpy.mean(targets**2)))
    if target_rms == 0:
        return numpy.nan
    return float(numpy.sqrt(numpy.mean(residuals**2)) / target_rms)


def average_relative_absolute_error(residuals, targets):
    residuals = numpy.asarray(residuals, dtype=float)
    targets = numpy.asarray(targets, dtype=float)
    nonzero_targets = numpy.abs(targets) > 0
    if not numpy.any(nonzero_targets):
        return numpy.nan
    return float(numpy.mean(numpy.abs(residuals[nonzero_targets]) / numpy.abs(targets[nonzero_targets])))


def fit_basis(data_subset, basis_terms):
    X_fit = design_matrix_for(data_subset, basis_terms)
    y_fit = data_subset[:, 4]
    coefficients_fit, residual_sum_fit, rank_fit, singular_values_fit = scipy.linalg.lstsq(X_fit, y_fit)
    fitted_values = X_fit @ coefficients_fit
    residuals = y_fit - fitted_values

    return {
        "basis_terms": basis_terms,
        "X": X_fit,
        "V": y_fit,
        "coefficients": coefficients_fit,
        "residual_sum": residual_sum_fit,
        "rank": rank_fit,
        "singular_values": singular_values_fit,
        "fit_values": fitted_values,
        "fit_residuals": residuals,
        "sse": float(numpy.sum(residuals**2)),
        "rmse": float(numpy.sqrt(numpy.mean(residuals**2))),
        "relative_rmse": relative_rmse(residuals, y_fit),
        "average_relative_absolute_error": average_relative_absolute_error(residuals, y_fit),
        "max_abs_error": float(numpy.max(numpy.abs(residuals))),
    }


def fit_naive(data_subset, max_l1, max_l2, terms):
    basis_terms = generate_terms(max_l1, max_l2, terms)
    return fit_basis(data_subset, basis_terms)


def print_fit_summary(fit_result, radius=None):
    if radius is not None:
        print(f"Radius: {radius}")
    print(f"Geometries: {len(fit_result['V'])}")
    print(f"Terms used: {len(fit_result['basis_terms'])}")
    print(f"Design matrix shape: {fit_result['X'].shape}")
    print(f"Rank: {fit_result['rank']}")
    print(f"SSE: {fit_result['sse']:.8g}")
    print(f"RMSE: {fit_result['rmse']:.8g}")
    print(f"Relative RMSE: {fit_result['relative_rmse']:.8g}")
    print(f"Average relative absolute error: {fit_result['average_relative_absolute_error']:.8g}")
    print(f"Max abs error: {fit_result['max_abs_error']:.8g}")


def create_train_test_split(data_subset, train_fraction=DEFAULT_TRAIN_FRACTION, random_seed=DEFAULT_RANDOM_SEED):
    rng = numpy.random.default_rng(random_seed)
    shuffled_indices = rng.permutation(len(data_subset))
    train_count = int(train_fraction * len(data_subset))
    training_indices = shuffled_indices[:train_count]
    testing_indices = shuffled_indices[train_count:]

    return data_subset[training_indices], data_subset[testing_indices], training_indices, testing_indices


def fit_and_score(train_subset, test_subset, basis_terms):
    train_fit = fit_basis(train_subset, basis_terms)
    X_test = design_matrix_for(test_subset, basis_terms)
    y_test = test_subset[:, 4]
    test_residuals = y_test - X_test @ train_fit["coefficients"]

    return {
        "test_rmse": float(numpy.sqrt(numpy.mean(test_residuals**2))),
        "test_relative_rmse": relative_rmse(test_residuals, y_test),
        "test_average_relative_absolute_error": average_relative_absolute_error(test_residuals, y_test),
        "train_rmse": train_fit["rmse"],
        "train_relative_rmse": train_fit["relative_rmse"],
        "train_average_relative_absolute_error": train_fit["average_relative_absolute_error"],
        "term_count": len(basis_terms),
        "rank": int(train_fit["rank"]),
        "basis_terms": basis_terms,
        "coefficients": train_fit["coefficients"],
        "test_residuals": test_residuals,
    }


def run_basis_cutoff_sweep(train_subset, test_subset, max_l1_grid, max_l2_grid, terms_cap=None):
    test_rmse_grid = numpy.full((len(max_l2_grid), len(max_l1_grid)), numpy.nan)
    train_rmse_grid = numpy.full_like(test_rmse_grid, numpy.nan)
    test_relative_rmse_grid = numpy.full_like(test_rmse_grid, numpy.nan)
    train_relative_rmse_grid = numpy.full_like(test_rmse_grid, numpy.nan)
    test_average_relative_absolute_error_grid = numpy.full_like(test_rmse_grid, numpy.nan)
    train_average_relative_absolute_error_grid = numpy.full_like(test_rmse_grid, numpy.nan)
    term_count_grid = numpy.zeros_like(test_rmse_grid, dtype=int)
    rank_grid = numpy.zeros_like(test_rmse_grid, dtype=int)
    split_fit_results = {}

    for l2_index, sweep_max_l2 in enumerate(max_l2_grid):
        for l1_index, sweep_max_l1 in enumerate(max_l1_grid):
            basis_terms = generate_all_terms(sweep_max_l1, sweep_max_l2)
            if terms_cap is not None:
                basis_terms = basis_terms[:terms_cap]

            result = fit_and_score(train_subset, test_subset, basis_terms)
            test_rmse_grid[l2_index, l1_index] = result["test_rmse"]
            train_rmse_grid[l2_index, l1_index] = result["train_rmse"]
            test_relative_rmse_grid[l2_index, l1_index] = result["test_relative_rmse"]
            train_relative_rmse_grid[l2_index, l1_index] = result["train_relative_rmse"]
            test_average_relative_absolute_error_grid[l2_index, l1_index] = result[
                "test_average_relative_absolute_error"
            ]
            train_average_relative_absolute_error_grid[l2_index, l1_index] = result[
                "train_average_relative_absolute_error"
            ]
            term_count_grid[l2_index, l1_index] = result["term_count"]
            rank_grid[l2_index, l1_index] = result["rank"]
            split_fit_results[(sweep_max_l1, sweep_max_l2)] = result

    best_l2_index, best_l1_index = numpy.unravel_index(numpy.nanargmin(test_rmse_grid), test_rmse_grid.shape)
    best_max_l1 = max_l1_grid[best_l1_index]
    best_max_l2 = max_l2_grid[best_l2_index]
    best_result = split_fit_results[(best_max_l1, best_max_l2)]

    return {
        "max_l1_grid": max_l1_grid,
        "max_l2_grid": max_l2_grid,
        "terms_cap": terms_cap,
        "test_rmse_grid": test_rmse_grid,
        "train_rmse_grid": train_rmse_grid,
        "test_relative_rmse_grid": test_relative_rmse_grid,
        "train_relative_rmse_grid": train_relative_rmse_grid,
        "test_average_relative_absolute_error_grid": test_average_relative_absolute_error_grid,
        "train_average_relative_absolute_error_grid": train_average_relative_absolute_error_grid,
        "term_count_grid": term_count_grid,
        "rank_grid": rank_grid,
        "split_fit_results": split_fit_results,
        "color_min": float(numpy.nanmin(test_rmse_grid)),
        "color_max": 10 * float(numpy.nanmin(test_rmse_grid)),
        "best_l2_index": best_l2_index,
        "best_l1_index": best_l1_index,
        "best_max_l1": best_max_l1,
        "best_max_l2": best_max_l2,
        "best_result": best_result,
    }


def plot_sweep_rmse_heatmap(sweep_result, title=None):
    max_l1_grid = sweep_result["max_l1_grid"]
    max_l2_grid = sweep_result["max_l2_grid"]
    test_rmse_grid = sweep_result["test_rmse_grid"]

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    heatmap = ax.imshow(
        test_rmse_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
        vmin=sweep_result["color_min"],
        vmax=sweep_result["color_max"],
        extent=[min(max_l1_grid) - 1, max(max_l1_grid) + 1, min(max_l2_grid) - 1, max(max_l2_grid) + 1],
    )

    if title is None:
        if sweep_result["terms_cap"] is None:
            title = "80/20 held-out RMSE by basis cutoff"
        else:
            title = f"80/20 held-out RMSE by basis cutoff, capped at {sweep_result['terms_cap']} terms"

    ax.set_title(title)
    ax.set_xlabel("max l1")
    ax.set_ylabel("max l2")
    ax.set_xticks(max_l1_grid)
    ax.set_yticks(max_l2_grid)
    fig.colorbar(heatmap, ax=ax, label="Testing RMSE", extend="max")

    for l2_index, sweep_max_l2 in enumerate(max_l2_grid):
        for l1_index, sweep_max_l1 in enumerate(max_l1_grid):
            ax.text(
                sweep_max_l1,
                sweep_max_l2,
                f"{test_rmse_grid[l2_index, l1_index]:.3g}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    return fig, ax


def print_sweep_best(sweep_result, label=""):
    prefix = f"{label} " if label else ""
    best_result = sweep_result["best_result"]
    print(f"Best {prefix}testing RMSE: {best_result['test_rmse']:.8g}")
    print(f"Best {prefix}testing relative RMSE: {best_result['test_relative_rmse']:.8g}")
    print(
        f"Best {prefix}testing average relative absolute error: "
        f"{best_result['test_average_relative_absolute_error']:.8g}"
    )
    print(f"Best {prefix}training relative RMSE: {best_result['train_relative_rmse']:.8g}")
    print(f"Best {prefix}max_l1: {sweep_result['best_max_l1']}")
    print(f"Best {prefix}max_l2: {sweep_result['best_max_l2']}")
    print(f"Terms in best {prefix}fit: {best_result['term_count']}")
    print(f"Training rank in best {prefix}fit: {best_result['rank']}")


def truncated_svd_lstsq(X, y, svd_threshold=DEFAULT_SVD_THRESHOLD, threshold_mode=DEFAULT_THRESHOLD_MODE):
    """Solve least squares by discarding singular values below the threshold."""
    U, singular_values, Vh = scipy.linalg.svd(X, full_matrices=False, check_finite=False)

    if singular_values.size == 0:
        return numpy.zeros(X.shape[1]), 0, singular_values, 0.0

    if threshold_mode == "relative":
        cutoff = float(svd_threshold) * float(singular_values[0])
    elif threshold_mode == "absolute":
        cutoff = float(svd_threshold)
    else:
        raise ValueError("threshold_mode must be 'relative' or 'absolute'.")

    retained = singular_values > cutoff
    rank = int(numpy.count_nonzero(retained))
    if rank == 0:
        return numpy.zeros(X.shape[1]), rank, singular_values, cutoff

    coefficients = Vh[retained, :].T @ ((U[:, retained].T @ y) / singular_values[retained])
    return coefficients, rank, singular_values, cutoff


def fit_basis_truncated_svd(data_subset, basis_terms, svd_threshold=DEFAULT_SVD_THRESHOLD, threshold_mode=DEFAULT_THRESHOLD_MODE):
    X_fit = design_matrix_for(data_subset, basis_terms)
    y_fit = data_subset[:, 4]
    coefficients_fit, rank_fit, singular_values_fit, svd_cutoff = truncated_svd_lstsq(
        X_fit,
        y_fit,
        svd_threshold=svd_threshold,
        threshold_mode=threshold_mode,
    )
    fitted_values = X_fit @ coefficients_fit
    residuals = y_fit - fitted_values

    if rank_fit > 0:
        retained_singular_values = singular_values_fit[singular_values_fit > svd_cutoff]
        retained_condition_number = float(retained_singular_values[0] / retained_singular_values[-1])
    else:
        retained_condition_number = numpy.inf

    full_condition_number = numpy.inf
    if singular_values_fit.size > 0 and singular_values_fit[-1] > 0:
        full_condition_number = float(singular_values_fit[0] / singular_values_fit[-1])

    return {
        "basis_terms": basis_terms,
        "X": X_fit,
        "V": y_fit,
        "coefficients": coefficients_fit,
        "rank": rank_fit,
        "singular_values": singular_values_fit,
        "svd_threshold": svd_threshold,
        "threshold_mode": threshold_mode,
        "svd_cutoff": svd_cutoff,
        "discarded_singular_count": int(singular_values_fit.size - rank_fit),
        "retained_condition_number": retained_condition_number,
        "full_condition_number": full_condition_number,
        "fit_values": fitted_values,
        "fit_residuals": residuals,
        "sse": float(numpy.sum(residuals**2)),
        "rmse": float(numpy.sqrt(numpy.mean(residuals**2))),
        "relative_rmse": relative_rmse(residuals, y_fit),
        "average_relative_absolute_error": average_relative_absolute_error(residuals, y_fit),
        "max_abs_error": float(numpy.max(numpy.abs(residuals))),
        "max_abs_coefficient": float(numpy.max(numpy.abs(coefficients_fit))) if coefficients_fit.size else 0.0,
        "coefficient_l2_norm": float(numpy.linalg.norm(coefficients_fit)),
    }


def fit_and_score_truncated_svd(
    train_subset,
    test_subset,
    basis_terms,
    svd_threshold=DEFAULT_SVD_THRESHOLD,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
):
    train_fit = fit_basis_truncated_svd(
        train_subset,
        basis_terms,
        svd_threshold=svd_threshold,
        threshold_mode=threshold_mode,
    )
    X_test = design_matrix_for(test_subset, basis_terms)
    y_test = test_subset[:, 4]
    test_residuals = y_test - X_test @ train_fit["coefficients"]

    return {
        "test_rmse": float(numpy.sqrt(numpy.mean(test_residuals**2))),
        "test_relative_rmse": relative_rmse(test_residuals, y_test),
        "test_average_relative_absolute_error": average_relative_absolute_error(test_residuals, y_test),
        "train_rmse": train_fit["rmse"],
        "train_relative_rmse": train_fit["relative_rmse"],
        "train_average_relative_absolute_error": train_fit["average_relative_absolute_error"],
        "term_count": len(basis_terms),
        "rank": int(train_fit["rank"]),
        "discarded_singular_count": train_fit["discarded_singular_count"],
        "svd_cutoff": train_fit["svd_cutoff"],
        "retained_condition_number": train_fit["retained_condition_number"],
        "full_condition_number": train_fit["full_condition_number"],
        "max_abs_coefficient": train_fit["max_abs_coefficient"],
        "coefficient_l2_norm": train_fit["coefficient_l2_norm"],
        "basis_terms": basis_terms,
        "coefficients": train_fit["coefficients"],
        "test_residuals": test_residuals,
    }


def basis_support_cutoffs(basis_terms):
    if not basis_terms:
        return {"max_l1": 0, "max_l2": 0, "max_L": 0}
    return {
        "max_L": max(term[0] for term in basis_terms),
        "max_l1": max(term[1] for term in basis_terms),
        "max_l2": max(term[2] for term in basis_terms),
    }


def incremental_term_candidates(universe_max_l1, universe_max_l2, min_terms=1, max_terms=None):
    ordered_terms = generate_all_terms(universe_max_l1, universe_max_l2)
    if max_terms is None:
        max_terms = len(ordered_terms)
    max_terms = min(int(max_terms), len(ordered_terms))
    min_terms = max(1, int(min_terms))

    if min_terms > max_terms:
        raise ValueError("min_terms cannot be larger than max_terms.")

    for term_count in range(min_terms, max_terms + 1):
        basis_terms = ordered_terms[:term_count]
        yield {
            "basis_terms": basis_terms,
            "term_count": term_count,
            **basis_support_cutoffs(basis_terms),
        }


def basis_cutoff_candidates(max_l1_grid, max_l2_grid, min_l1=8, min_l2=4, terms_cap=None):
    """Return candidate bases ordered by smallest total number of terms first."""
    candidates = []
    seen_bases = set()

    for candidate_max_l1 in sorted(int(value) for value in max_l1_grid):
        for candidate_max_l2 in sorted(int(value) for value in max_l2_grid):
            if candidate_max_l1 < min_l1 or candidate_max_l2 < min_l2:
                continue

            all_candidate_terms = generate_all_terms(candidate_max_l1, candidate_max_l2)
            if terms_cap is None:
                candidate_terms = all_candidate_terms
            else:
                candidate_terms = all_candidate_terms[:int(terms_cap)]

            basis_key = tuple(candidate_terms)
            if basis_key in seen_bases:
                continue
            seen_bases.add(basis_key)

            candidates.append({
                "max_l1": candidate_max_l1,
                "max_l2": candidate_max_l2,
                "basis_terms": candidate_terms,
                "term_count": len(candidate_terms),
            })

    candidates.sort(key=lambda candidate: (
        candidate["term_count"],
        candidate["max_l1"] + candidate["max_l2"],
        candidate["max_l1"],
        candidate["max_l2"],
    ))
    return candidates


# --- R -> R' point-count helper definitions ---

# R -> R' ab initio point-count prior.
# This implements through-radius N estimates with per-radius incremental term selection.

RADIUS_TO_RADIUS_DATA_PATH = "data/all_avcbs_uniform.dat"
RADIUS_TO_RADIUS_GAMMA = 1.0
RADIUS_TO_RADIUS_ANGULAR_DIMENSION = 3
RADIUS_TO_RADIUS_CURVATURE_GROWTH_EXPONENT = 0.5
RADIUS_TO_RADIUS_LARGE_R_DAMPING_RADIUS = 15.0
RADIUS_TO_RADIUS_GRID_STEP_DEGREES = 15.0
RADIUS_TO_RADIUS_DERIVATIVE_STEP_DEGREES = 1.0
RADIUS_TO_RADIUS_DELTA_RELATIVE_SCALE = 1e-12
RADIUS_TO_RADIUS_DELTA_FLOOR = 1e-12
RADIUS_TO_RADIUS_DEFAULT_TERM_COUNT = DEFAULT_TERMS
RADIUS_TO_RADIUS_MIN_TERM_COUNT = 1
RADIUS_TO_RADIUS_MAX_TERM_COUNT = None
RADIUS_TO_RADIUS_INITIAL_FIT_POINT_COUNT = 40
RADIUS_TO_RADIUS_INITIAL_TEST_POINT_COUNT = 100


def resolve_data_path(data_path=RADIUS_TO_RADIUS_DATA_PATH):
    data_path = Path(data_path)
    candidates = [
        data_path,
        Path.cwd() / data_path,
        Path.cwd() / "Performance Code" / data_path,
        Path.cwd() / "data" / data_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find data file {data_path!s}.")


def load_all_radius_data(data_path=RADIUS_TO_RADIUS_DATA_PATH):
    return numpy.loadtxt(resolve_data_path(data_path))


def radius_subset(all_data, r_value):
    return all_data[all_data[:, 0] == r_value]


def grouped_radius_data(all_data):
    radii = numpy.asarray(sorted(numpy.unique(all_data[:, 0]), reverse=True), dtype=float)
    return radii, {float(r_value): radius_subset(all_data, r_value) for r_value in radii}


def validate_common_angular_grid(radius_data_by_r, atol=1e-12):
    radii = list(radius_data_by_r)
    reference_angles = radius_data_by_r[radii[0]][:, 1:4]
    for r_value in radii[1:]:
        candidate_angles = radius_data_by_r[r_value][:, 1:4]
        if candidate_angles.shape != reference_angles.shape or not numpy.allclose(
            candidate_angles,
            reference_angles,
            atol=atol,
            rtol=0.0,
        ):
            return False
    return True


def make_angular_candidate_grid(
    theta1_step=RADIUS_TO_RADIUS_GRID_STEP_DEGREES,
    theta2_step=RADIUS_TO_RADIUS_GRID_STEP_DEGREES,
    phi_step=RADIUS_TO_RADIUS_GRID_STEP_DEGREES,
):
    theta1_values = numpy.arange(0.0, 180.0 + 0.5 * theta1_step, theta1_step)
    theta2_values = numpy.arange(0.0, 180.0 + 0.5 * theta2_step, theta2_step)
    phi_values = numpy.arange(0.0, 180.0 + 0.5 * phi_step, phi_step)
    theta1_grid, theta2_grid, phi_grid = numpy.meshgrid(
        theta1_values,
        theta2_values,
        phi_values,
        indexing="ij",
    )
    return numpy.column_stack([
        theta1_grid.ravel(),
        theta2_grid.ravel(),
        phi_grid.ravel(),
    ])


def design_matrix_for_angles(theta1, theta2, phi, basis_terms, angles_in_degrees=True):
    theta1_values = numpy.asarray(theta1, dtype=float)
    theta2_values = numpy.asarray(theta2, dtype=float)
    phi_values = numpy.asarray(phi, dtype=float)
    if angles_in_degrees:
        theta1_values = numpy.deg2rad(theta1_values)
        theta2_values = numpy.deg2rad(theta2_values)
        phi_values = numpy.deg2rad(phi_values)

    return numpy.column_stack([
        angular_basis(L, l1, l2, theta1_values, theta2_values, phi_values)
        for L, l1, l2 in basis_terms
    ])


def canonicalize_angular_grid_degrees(angular_grid_degrees):
    """Map angular coordinates to the physical ``[0, 180]`` degree domain.

    ``theta1`` and ``theta2`` are polar angles and ``phi`` is their relative
    azimuth.  A polar-angle perturbation that passes through either pole must
    therefore rotate that monomer's azimuth by 180 degrees.  In relative
    coordinates this shifts ``phi`` by -180 degrees for a ``theta1`` crossing
    and +180 degrees for a ``theta2`` crossing.  Applying both corrections
    jointly is important for mixed finite-difference stencils: simultaneous
    crossings cancel modulo 360 degrees.

    Finally, reflection symmetry folds the relative azimuth from its
    360-degree periodic representation into ``[0, 180]``.  The function is
    vectorized and accepts either one ``(3,)`` coordinate or an ``(n, 3)``
    array.  It does not mutate its input.
    """
    angular_grid = numpy.asarray(angular_grid_degrees, dtype=float)
    if angular_grid.ndim == 0 or angular_grid.shape[-1] != 3:
        raise ValueError("angular_grid_degrees must have a final dimension of length 3.")

    canonical_grid = angular_grid.copy()

    theta1_modulo = numpy.mod(canonical_grid[..., 0], 360.0)
    theta2_modulo = numpy.mod(canonical_grid[..., 1], 360.0)
    theta1_crossed_pole = theta1_modulo > 180.0
    theta2_crossed_pole = theta2_modulo > 180.0

    canonical_grid[..., 0] = numpy.where(
        theta1_crossed_pole,
        360.0 - theta1_modulo,
        theta1_modulo,
    )
    canonical_grid[..., 1] = numpy.where(
        theta2_crossed_pole,
        360.0 - theta2_modulo,
        theta2_modulo,
    )

    relative_azimuth = (
        canonical_grid[..., 2]
        + 180.0 * theta2_crossed_pole.astype(float)
        - 180.0 * theta1_crossed_pole.astype(float)
    )
    relative_azimuth_modulo = numpy.mod(relative_azimuth, 360.0)
    canonical_grid[..., 2] = numpy.where(
        relative_azimuth_modulo > 180.0,
        360.0 - relative_azimuth_modulo,
        relative_azimuth_modulo,
    )
    return canonical_grid


def evaluate_fit_on_angles(fit_result, angular_grid_degrees):
    X_eval = design_matrix_for_angles(
        angular_grid_degrees[:, 0],
        angular_grid_degrees[:, 1],
        angular_grid_degrees[:, 2],
        fit_result["basis_terms"],
        angles_in_degrees=True,
    )
    return X_eval @ fit_result["coefficients"]


def derivative_complexity_for_fit(
    fit_result,
    angular_grid_degrees,
    derivative_step_degrees=RADIUS_TO_RADIUS_DERIVATIVE_STEP_DEGREES,
    delta_relative_scale=RADIUS_TO_RADIUS_DELTA_RELATIVE_SCALE,
    delta_floor=RADIUS_TO_RADIUS_DELTA_FLOOR,
    percentile=95.0,
):
    angular_grid_degrees = canonicalize_angular_grid_degrees(angular_grid_degrees)
    h_degrees = float(derivative_step_degrees)
    h_radians = numpy.deg2rad(h_degrees)

    values = evaluate_fit_on_angles(fit_result, angular_grid_degrees)
    amplitude_rms = float(numpy.sqrt(numpy.mean(values**2)))
    delta = max(float(delta_floor), float(delta_relative_scale) * float(numpy.max(numpy.abs(values))))

    gradients = []
    shifted_plus_values = []
    shifted_minus_values = []
    for axis in range(3):
        plus_grid = angular_grid_degrees.copy()
        minus_grid = angular_grid_degrees.copy()
        plus_grid[:, axis] += h_degrees
        minus_grid[:, axis] -= h_degrees
        plus_grid = canonicalize_angular_grid_degrees(plus_grid)
        minus_grid = canonicalize_angular_grid_degrees(minus_grid)
        plus_values = evaluate_fit_on_angles(fit_result, plus_grid)
        minus_values = evaluate_fit_on_angles(fit_result, minus_grid)
        shifted_plus_values.append(plus_values)
        shifted_minus_values.append(minus_values)
        gradients.append((plus_values - minus_values) / (2.0 * h_radians))

    gradient_norm = numpy.sqrt(sum(component**2 for component in gradients))

    hessian_frobenius_squared = numpy.zeros_like(values, dtype=float)
    for axis in range(3):
        second_derivative = (shifted_plus_values[axis] - 2.0 * values + shifted_minus_values[axis]) / (h_radians**2)
        hessian_frobenius_squared += second_derivative**2

    for axis_a in range(3):
        for axis_b in range(axis_a + 1, 3):
            plus_plus_grid = angular_grid_degrees.copy()
            plus_minus_grid = angular_grid_degrees.copy()
            minus_plus_grid = angular_grid_degrees.copy()
            minus_minus_grid = angular_grid_degrees.copy()

            plus_plus_grid[:, axis_a] += h_degrees
            plus_plus_grid[:, axis_b] += h_degrees
            plus_minus_grid[:, axis_a] += h_degrees
            plus_minus_grid[:, axis_b] -= h_degrees
            minus_plus_grid[:, axis_a] -= h_degrees
            minus_plus_grid[:, axis_b] += h_degrees
            minus_minus_grid[:, axis_a] -= h_degrees
            minus_minus_grid[:, axis_b] -= h_degrees

            plus_plus_grid = canonicalize_angular_grid_degrees(plus_plus_grid)
            plus_minus_grid = canonicalize_angular_grid_degrees(plus_minus_grid)
            minus_plus_grid = canonicalize_angular_grid_degrees(minus_plus_grid)
            minus_minus_grid = canonicalize_angular_grid_degrees(minus_minus_grid)

            mixed_second_derivative = (
                evaluate_fit_on_angles(fit_result, plus_plus_grid)
                - evaluate_fit_on_angles(fit_result, plus_minus_grid)
                - evaluate_fit_on_angles(fit_result, minus_plus_grid)
                + evaluate_fit_on_angles(fit_result, minus_minus_grid)
            ) / (4.0 * h_radians**2)
            hessian_frobenius_squared += 2.0 * mixed_second_derivative**2

    curvature_norm = numpy.sqrt(hessian_frobenius_squared)
    gradient_scale = float(numpy.percentile(gradient_norm, percentile))
    curvature_scale = float(numpy.percentile(curvature_norm, percentile))

    return {
        "amplitude_rms": amplitude_rms,
        "delta": delta,
        "gradient_scale": gradient_scale,
        "curvature_scale": curvature_scale,
        "normalized_gradient_scale": gradient_scale / (amplitude_rms + delta),
        "normalized_curvature_scale": curvature_scale / (amplitude_rms + delta),
        "values": values,
        "gradient_norm": gradient_norm,
        "curvature_norm": curvature_norm,
    }


def basis_sampling_requirement(term_count, gamma=RADIUS_TO_RADIUS_GAMMA):
    term_count = max(1, int(term_count))
    return int(numpy.ceil(float(gamma) * term_count * numpy.log(max(term_count, 2))))


def curvature_growth_diagnostics(
    current_complexity,
    previous_complexity=None,
    current_radius=None,
    base_exponent=RADIUS_TO_RADIUS_CURVATURE_GROWTH_EXPONENT,
    large_r_damping_radius=RADIUS_TO_RADIUS_LARGE_R_DAMPING_RADIUS,
):
    if previous_complexity is None:
        curvature_ratio = 1.0
    else:
        numerator = current_complexity["normalized_curvature_scale"]
        denominator = previous_complexity["normalized_curvature_scale"]
        if denominator <= 0 or not numpy.isfinite(denominator):
            curvature_ratio = 1.0
        else:
            curvature_ratio = max(float(numerator) / float(denominator), 0.0)

    if current_radius is None:
        radial_damping = 1.0
    else:
        radius_scale = max(abs(float(current_radius)), 1e-12)
        radial_damping = min(1.0, float(large_r_damping_radius) / radius_scale)

    effective_exponent = float(base_exponent) * float(radial_damping)
    curvature_growth_factor = max(1.0, float(curvature_ratio) ** effective_exponent)
    return {
        "curvature_ratio": float(curvature_ratio),
        "radial_damping": float(radial_damping),
        "effective_curvature_exponent": float(effective_exponent),
        "curvature_growth_factor": float(curvature_growth_factor),
    }


def predict_next_point_count(
    current_point_count,
    current_complexity,
    previous_complexity=None,
    next_term_count=None,
    gamma=RADIUS_TO_RADIUS_GAMMA,
    angular_dimension=RADIUS_TO_RADIUS_ANGULAR_DIMENSION,
    max_available_points=None,
    current_radius=None,
    curvature_growth_exponent=RADIUS_TO_RADIUS_CURVATURE_GROWTH_EXPONENT,
    large_r_damping_radius=RADIUS_TO_RADIUS_LARGE_R_DAMPING_RADIUS,
    return_diagnostics=False,
):
    del angular_dimension  # Kept in the signature for compatibility with earlier calls.
    growth_diagnostics = curvature_growth_diagnostics(
        current_complexity,
        previous_complexity=previous_complexity,
        current_radius=current_radius,
        base_exponent=curvature_growth_exponent,
        large_r_damping_radius=large_r_damping_radius,
    )
    derivative_estimate = float(current_point_count) * growth_diagnostics["curvature_growth_factor"]

    basis_estimate = 0 if next_term_count is None else basis_sampling_requirement(next_term_count, gamma=gamma)
    estimated_count = int(numpy.ceil(max(float(basis_estimate), derivative_estimate)))
    if max_available_points is not None:
        estimated_count = min(estimated_count, int(max_available_points))
    estimated_count = max(1, estimated_count)

    if return_diagnostics:
        return estimated_count, {
            **growth_diagnostics,
            "basis_estimate": int(basis_estimate),
            "derivative_estimate": float(derivative_estimate),
        }
    return estimated_count


def extend_point_indices(previous_indices, target_count, available_indices, rng):
    previous_indices = numpy.asarray(previous_indices, dtype=int)
    available_indices = numpy.asarray(available_indices, dtype=int)
    target_count = min(int(target_count), available_indices.size)

    previous_set = set(int(index) for index in previous_indices)
    reusable_indices = numpy.asarray([index for index in available_indices if int(index) in previous_set], dtype=int)
    if reusable_indices.size >= target_count:
        return numpy.sort(reusable_indices[:target_count])

    remaining_indices = numpy.asarray([index for index in available_indices if int(index) not in previous_set], dtype=int)
    additional_count = target_count - reusable_indices.size
    if additional_count > 0:
        added_indices = rng.choice(remaining_indices, size=additional_count, replace=False)
        selected_indices = numpy.concatenate([reusable_indices, added_indices])
    else:
        selected_indices = reusable_indices
    return numpy.sort(selected_indices)


def select_incremental_terms_for_radius_indices(
    radius_data,
    point_indices,
    test_indices,
    universe_max_l1=DEFAULT_MAX_L1,
    universe_max_l2=DEFAULT_MAX_L2,
    relative_rmse_cutoff=DEFAULT_RELATIVE_RMSE_CUTOFF,
    svd_threshold=DEFAULT_SVD_THRESHOLD,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
    min_terms=RADIUS_TO_RADIUS_MIN_TERM_COUNT,
    max_terms=RADIUS_TO_RADIUS_MAX_TERM_COUNT,
):
    train_data = radius_data[numpy.asarray(point_indices, dtype=int)]
    test_data = radius_data[numpy.asarray(test_indices, dtype=int)]

    tried_results = []
    best_result = None
    chosen_result = None
    passed_cutoff = False

    for candidate in incremental_term_candidates(
        universe_max_l1,
        universe_max_l2,
        min_terms=min_terms,
        max_terms=max_terms,
    ):
        score = fit_and_score_truncated_svd(
            train_data,
            test_data,
            candidate["basis_terms"],
            svd_threshold=svd_threshold,
            threshold_mode=threshold_mode,
        )
        score.update(candidate)
        tried_results.append(score)

        if best_result is None or score["test_relative_rmse"] < best_result["test_relative_rmse"]:
            best_result = score

        if numpy.isfinite(score["test_relative_rmse"]) and score["test_relative_rmse"] <= relative_rmse_cutoff:
            chosen_result = score
            passed_cutoff = True
            break

    if chosen_result is None:
        chosen_result = best_result
    if chosen_result is None:
        raise ValueError("No incremental term candidates were available for this radius.")

    fit_result = dict(chosen_result)
    fit_result.update({
        "point_count": int(len(point_indices)),
        "point_indices": numpy.asarray(point_indices, dtype=int),
        "test_indices": numpy.asarray(test_indices, dtype=int),
        "passed_basis_cutoff": passed_cutoff,
        "relative_rmse_cutoff": relative_rmse_cutoff,
        "tried_results": tried_results,
    })
    return fit_result


def fit_radius_with_indices(
    radius_data,
    point_indices,
    term_count,
    test_indices=None,
    max_l1=DEFAULT_MAX_L1,
    max_l2=DEFAULT_MAX_L2,
    svd_threshold=DEFAULT_SVD_THRESHOLD,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
):
    basis_terms = generate_terms(max_l1, max_l2, int(term_count))
    fit_result = fit_basis_truncated_svd(
        radius_data[numpy.asarray(point_indices, dtype=int)],
        basis_terms,
        svd_threshold=svd_threshold,
        threshold_mode=threshold_mode,
    )

    score = {}
    if test_indices is not None and len(test_indices) > 0:
        test_data = radius_data[numpy.asarray(test_indices, dtype=int)]
        X_test = design_matrix_for(test_data, basis_terms)
        y_test = test_data[:, 4]
        test_residuals = y_test - X_test @ fit_result["coefficients"]
        score = {
            "test_rmse": float(numpy.sqrt(numpy.mean(test_residuals**2))),
            "test_relative_rmse": relative_rmse(test_residuals, y_test),
            "test_average_relative_absolute_error": average_relative_absolute_error(test_residuals, y_test),
            "test_residuals": test_residuals,
        }

    fit_result.update({
        "term_count": int(term_count),
        "point_count": int(len(point_indices)),
        "point_indices": numpy.asarray(point_indices, dtype=int),
        **score,
    })
    return fit_result


def leverage_scores_for_candidates(candidate_grid_degrees, selected_radius_data, basis_terms, ridge=1e-10):
    X_selected = design_matrix_for(selected_radius_data, basis_terms)
    X_candidates = design_matrix_for_angles(
        candidate_grid_degrees[:, 0],
        candidate_grid_degrees[:, 1],
        candidate_grid_degrees[:, 2],
        basis_terms,
        angles_in_degrees=True,
    )
    gram = X_selected.T @ X_selected
    regularized_inverse = numpy.linalg.pinv(gram + float(ridge) * numpy.eye(gram.shape[0]))
    return numpy.einsum("ij,jk,ik->i", X_candidates, regularized_inverse, X_candidates)


def acquisition_scores(
    curvature_norm,
    leverage=None,
    disagreement=None,
    weights=None,
    alpha=1.0,
    beta=0.0,
    eta=0.0,
):
    curvature_norm = numpy.asarray(curvature_norm, dtype=float)
    score = float(alpha) * curvature_norm / max(float(numpy.max(curvature_norm)), 1e-300)

    if disagreement is not None:
        disagreement = numpy.asarray(disagreement, dtype=float)
        score = score + float(beta) * disagreement / max(float(numpy.max(disagreement)), 1e-300)

    if leverage is not None:
        leverage = numpy.asarray(leverage, dtype=float)
        score = score + float(eta) * leverage / max(float(numpy.max(leverage)), 1e-300)

    if weights is not None:
        score = numpy.asarray(weights, dtype=float) * score

    return score


def run_radius_to_radius_point_count_experiment(
    all_data,
    term_count_by_radius=None,
    default_term_count=RADIUS_TO_RADIUS_DEFAULT_TERM_COUNT,
    gamma=RADIUS_TO_RADIUS_GAMMA,
    train_fraction=DEFAULT_TRAIN_FRACTION,
    random_seed=DEFAULT_RANDOM_SEED,
    max_l1=DEFAULT_MAX_L1,
    max_l2=DEFAULT_MAX_L2,
    svd_threshold=DEFAULT_SVD_THRESHOLD,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
    relative_rmse_cutoff=DEFAULT_RELATIVE_RMSE_CUTOFF,
    min_terms=RADIUS_TO_RADIUS_MIN_TERM_COUNT,
    max_terms=RADIUS_TO_RADIUS_MAX_TERM_COUNT,
    candidate_grid=None,
    initial_fit_point_count=RADIUS_TO_RADIUS_INITIAL_FIT_POINT_COUNT,
    initial_test_point_count=RADIUS_TO_RADIUS_INITIAL_TEST_POINT_COUNT,
):
    radii, radius_data_by_r = grouped_radius_data(all_data)
    if not validate_common_angular_grid(radius_data_by_r):
        raise ValueError("All radii must share the same angular grid to reuse point indices across R.")

    point_count_per_radius = len(radius_data_by_r[float(radii[0])])
    all_indices = numpy.arange(point_count_per_radius)
    rng = numpy.random.default_rng(random_seed)
    shuffled_indices = rng.permutation(all_indices)

    if initial_test_point_count is None:
        test_count = point_count_per_radius - int(train_fraction * point_count_per_radius)
    else:
        test_count = int(initial_test_point_count)
    if test_count <= 0 or test_count >= point_count_per_radius:
        raise ValueError("initial_test_point_count must be between 1 and the number of angular points - 1.")

    test_indices = numpy.sort(shuffled_indices[:test_count])
    train_pool_indices = numpy.sort(shuffled_indices[test_count:])

    if candidate_grid is None:
        candidate_grid = make_angular_candidate_grid()

    if term_count_by_radius is None:
        term_count_by_radius = {}

    results_by_radius = {}
    transitions = []
    previous_complexity = None
    previous_point_indices = None
    previous_point_count = None
    previous_term_count = None

    for radius_index, r_value in enumerate(radii):
        r_value = float(r_value)
        radius_data = radius_data_by_r[r_value]
        forced_term_count = term_count_by_radius.get(r_value)

        if radius_index == 0:
            if initial_fit_point_count is None:
                starting_term_count = int(forced_term_count if forced_term_count is not None else default_term_count)
                target_point_count = basis_sampling_requirement(starting_term_count, gamma=gamma)
            else:
                target_point_count = int(initial_fit_point_count)
            target_point_count = min(target_point_count, train_pool_indices.size)
            point_indices = extend_point_indices([], target_point_count, train_pool_indices, rng)
            predicted_point_count = target_point_count
            derivative_point_count = target_point_count
            prediction_term_count = None
            basis_point_count = None
            point_count_prediction_diagnostics = {
                "curvature_ratio": 1.0,
                "radial_damping": 1.0,
                "effective_curvature_exponent": 0.0,
                "curvature_growth_factor": 1.0,
                "basis_estimate": 0,
                "derivative_estimate": float(target_point_count),
            }
        else:
            prediction_term_count = int(forced_term_count if forced_term_count is not None else previous_term_count)
            predicted_point_count, point_count_prediction_diagnostics = predict_next_point_count(
                previous_point_count,
                current_complexity=results_by_radius[float(radii[radius_index - 1])]["complexity"],
                previous_complexity=previous_complexity,
                next_term_count=prediction_term_count,
                gamma=gamma,
                angular_dimension=RADIUS_TO_RADIUS_ANGULAR_DIMENSION,
                max_available_points=train_pool_indices.size,
                current_radius=float(radii[radius_index - 1]),
                return_diagnostics=True,
            )
            basis_point_count = basis_sampling_requirement(prediction_term_count, gamma=gamma)
            if previous_complexity is None:
                derivative_point_count = previous_point_count
            else:
                derivative_point_count = predicted_point_count
            point_indices = extend_point_indices(previous_point_indices, predicted_point_count, train_pool_indices, rng)

        if forced_term_count is None:
            fit_result = select_incremental_terms_for_radius_indices(
                radius_data,
                point_indices,
                test_indices,
                universe_max_l1=max_l1,
                universe_max_l2=max_l2,
                relative_rmse_cutoff=relative_rmse_cutoff,
                svd_threshold=svd_threshold,
                threshold_mode=threshold_mode,
                min_terms=min_terms,
                max_terms=max_terms,
            )
        else:
            fit_result = fit_radius_with_indices(
                radius_data,
                point_indices,
                int(forced_term_count),
                test_indices=test_indices,
                max_l1=max_l1,
                max_l2=max_l2,
                svd_threshold=svd_threshold,
                threshold_mode=threshold_mode,
            )
            fit_result.update({
                "passed_basis_cutoff": numpy.isfinite(fit_result.get("test_relative_rmse", numpy.nan))
                and fit_result.get("test_relative_rmse", numpy.inf) <= relative_rmse_cutoff,
                "relative_rmse_cutoff": relative_rmse_cutoff,
                "tried_results": [fit_result],
                **basis_support_cutoffs(fit_result["basis_terms"]),
            })

        term_count = int(fit_result["term_count"])
        complexity = derivative_complexity_for_fit(fit_result, candidate_grid)
        capped_basis_point_count = int(min(basis_sampling_requirement(term_count, gamma=gamma), train_pool_indices.size))

        results_by_radius[r_value] = {
            "R": r_value,
            "fit": fit_result,
            "complexity": complexity,
            "term_count": term_count,
            "point_count": int(len(point_indices)),
            "point_indices": point_indices,
            "test_indices": test_indices,
            "basis_point_count": capped_basis_point_count,
            "prediction_term_count": None if prediction_term_count is None else int(prediction_term_count),
            "predicted_point_count": int(predicted_point_count),
            "derivative_point_count": int(min(numpy.ceil(derivative_point_count), train_pool_indices.size)),
            "passed_basis_cutoff": bool(fit_result.get("passed_basis_cutoff", False)),
            "point_count_prediction_diagnostics": point_count_prediction_diagnostics,
        }

        if radius_index > 0:
            previous_r = float(radii[radius_index - 1])
            reused_point_count = int(numpy.intersect1d(previous_point_indices, point_indices).size)
            transitions.append({
                "from_R": previous_r,
                "to_R": r_value,
                "predicted_point_count": int(predicted_point_count),
                "used_point_count": int(len(point_indices)),
                "reused_point_count": reused_point_count,
                "added_point_count": int(len(point_indices) - reused_point_count),
                "prediction_term_count": int(prediction_term_count),
                "term_count": term_count,
                "basis_point_count": capped_basis_point_count,
                "passed_basis_cutoff": bool(fit_result.get("passed_basis_cutoff", False)),
                "test_rmse": fit_result.get("test_rmse", numpy.nan),
                "test_relative_rmse": fit_result.get("test_relative_rmse", numpy.nan),
                "test_average_relative_absolute_error": fit_result.get("test_average_relative_absolute_error", numpy.nan),
                "normalized_curvature_scale": complexity["normalized_curvature_scale"],
                **point_count_prediction_diagnostics,
            })

        previous_complexity = None if radius_index == 0 else results_by_radius[float(radii[radius_index - 1])]["complexity"]
        previous_point_indices = point_indices
        previous_point_count = int(len(point_indices))
        previous_term_count = term_count

    return {
        "radii_descending": radii,
        "train_pool_indices": train_pool_indices,
        "test_indices": test_indices,
        "candidate_grid": candidate_grid,
        "results_by_radius": results_by_radius,
        "transitions": transitions,
        "gamma": gamma,
        "default_term_count": default_term_count,
        "relative_rmse_cutoff": relative_rmse_cutoff,
        "min_terms": min_terms,
        "max_terms": max_terms,
        "train_fraction": train_fraction,
        "initial_fit_point_count": int(initial_fit_point_count) if initial_fit_point_count is not None else None,
        "initial_test_point_count": int(test_count),
        "random_seed": random_seed,
    }


all_radius_data = load_all_radius_data()
radius_to_radius_candidate_grid = make_angular_candidate_grid()


# --- Acquisition helper definitions ---

# Acquisition-based R -> R' point selection using max-l1/max-l2 sweep basis selection.
# This leaves the previous random point-extension function untouched.

ACQUISITION_REQUIRED_ANGLE_POINTS_DEGREES = numpy.asarray([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 90.0],
    [0.0, 90.0, 0.0],
    [90.0, 0.0, 0.0],
    [90.0, 90.0, 90.0],
    [180.0, 0.0, 0.0],
], dtype=float)
ACQUISITION_ALPHA = 0.5
ACQUISITION_BETA = 0.5
ACQUISITION_MIN_PEAK_SEPARATION_DEGREES = 20.0
ACQUISITION_SWEEP_MAX_L1_GRID = numpy.arange(0, DEFAULT_MAX_L1 + 1, 2)
ACQUISITION_SWEEP_MAX_L2_GRID = numpy.arange(0, DEFAULT_MAX_L2 + 1, 2)
ACQUISITION_SWEEP_MIN_L1 = 0
ACQUISITION_SWEEP_MIN_L2 = 0


def nearest_angle_indices(radius_data, target_angles_degrees):
    angles = numpy.asarray(radius_data[:, 1:4], dtype=float)
    targets = numpy.asarray(target_angles_degrees, dtype=float)
    nearest_indices = []
    for target in targets:
        distances_squared = numpy.sum((angles - target) ** 2, axis=1)
        nearest_indices.append(int(numpy.argmin(distances_squared)))
    return numpy.asarray(sorted(set(nearest_indices)), dtype=int)


def split_train_pool_with_required_points(
    radius_data,
    required_point_indices,
    initial_test_point_count=RADIUS_TO_RADIUS_INITIAL_TEST_POINT_COUNT,
    train_fraction=DEFAULT_TRAIN_FRACTION,
    random_seed=DEFAULT_RANDOM_SEED,
):
    point_count = len(radius_data)
    all_indices = numpy.arange(point_count)
    required_point_indices = numpy.asarray(required_point_indices, dtype=int)
    rng = numpy.random.default_rng(random_seed)
    eligible_test_indices = numpy.asarray(
        [index for index in all_indices if index not in set(required_point_indices)],
        dtype=int,
    )

    if initial_test_point_count is None:
        test_count = point_count - int(train_fraction * point_count)
    else:
        test_count = int(initial_test_point_count)
    test_count = min(test_count, eligible_test_indices.size)
    if test_count <= 0:
        raise ValueError("The requested test set leaves no eligible test points.")

    shuffled_eligible_test_indices = rng.permutation(eligible_test_indices)
    test_indices = numpy.sort(shuffled_eligible_test_indices[:test_count])
    train_pool_indices = numpy.asarray(
        [index for index in all_indices if index not in set(test_indices)],
        dtype=int,
    )
    return numpy.sort(train_pool_indices), test_indices


def select_spread_indices(previous_indices, target_count, available_indices, radius_data):
    available_indices = numpy.asarray(available_indices, dtype=int)
    selected = list(dict.fromkeys(int(index) for index in numpy.asarray(previous_indices, dtype=int)))
    selected_set = set(selected)
    target_count = max(int(target_count), len(selected))

    if len(selected) >= target_count:
        return numpy.asarray(sorted(selected), dtype=int)

    available_set = set(int(index) for index in available_indices)
    remaining = [index for index in available_indices if int(index) not in selected_set]
    angles = numpy.asarray(radius_data[:, 1:4], dtype=float)

    if not selected and remaining:
        centroid = numpy.mean(angles[numpy.asarray(remaining, dtype=int)], axis=0)
        first_index = int(remaining[int(numpy.argmax(numpy.sum((angles[remaining] - centroid) ** 2, axis=1)))])
        selected.append(first_index)
        selected_set.add(first_index)
        remaining = [index for index in remaining if int(index) != first_index]

    while len(selected) < target_count and remaining:
        selected_angles = angles[numpy.asarray(selected, dtype=int)]
        remaining_angles = angles[numpy.asarray(remaining, dtype=int)]
        distance_squared = numpy.sum(
            (remaining_angles[:, None, :] - selected_angles[None, :, :]) ** 2,
            axis=2,
        )
        next_index = int(remaining[int(numpy.argmax(numpy.min(distance_squared, axis=1)))])
        selected.append(next_index)
        selected_set.add(next_index)
        remaining = [index for index in remaining if int(index) != next_index]

    return numpy.asarray(sorted(index for index in selected if index in available_set), dtype=int)


def tensor_local_maximum_indices(angular_grid_degrees, scores):
    angular_grid_degrees = numpy.asarray(angular_grid_degrees, dtype=float)
    scores = numpy.asarray(scores, dtype=float)
    axis_values = [numpy.unique(angular_grid_degrees[:, axis]) for axis in range(3)]
    expected_size = numpy.prod([len(values) for values in axis_values])
    if int(expected_size) != len(scores):
        return numpy.arange(len(scores), dtype=int)

    score_cube = scores.reshape(tuple(len(values) for values in axis_values))
    local_maximum_indices = []
    for flat_index, score in enumerate(scores):
        if not numpy.isfinite(score):
            continue
        grid_index = numpy.unravel_index(flat_index, score_cube.shape)
        slices = tuple(
            slice(max(0, coordinate - 1), min(score_cube.shape[axis], coordinate + 2))
            for axis, coordinate in enumerate(grid_index)
        )
        if score >= numpy.nanmax(score_cube[slices]):
            local_maximum_indices.append(flat_index)
    return numpy.asarray(local_maximum_indices, dtype=int)


def normalized_leverage_on_candidate_grid(candidate_grid_degrees, selected_radius_data, basis_terms, ridge=1e-10):
    raw_leverage = leverage_scores_for_candidates(
        candidate_grid_degrees,
        selected_radius_data,
        basis_terms,
        ridge=ridge,
    )
    max_leverage = max(float(numpy.nanmax(raw_leverage)), 1e-300)
    return raw_leverage / max_leverage


def acquisition_values_for_fit(
    fit_result,
    selected_radius_data,
    candidate_grid_degrees,
    alpha=ACQUISITION_ALPHA,
    beta=ACQUISITION_BETA,
):
    complexity = derivative_complexity_for_fit(fit_result, candidate_grid_degrees)
    curvature = numpy.asarray(complexity["curvature_norm"], dtype=float)
    curvature_score = curvature / max(float(numpy.nanmax(curvature)), 1e-300)
    leverage_score = normalized_leverage_on_candidate_grid(
        candidate_grid_degrees,
        selected_radius_data,
        fit_result["basis_terms"],
    )
    acquisition = float(alpha) * curvature_score + float(beta) * leverage_score
    return acquisition, {
        "complexity": complexity,
        "curvature_score": curvature_score,
        "leverage_score": leverage_score,
        "acquisition": acquisition,
    }


def snap_peak_angles_to_available_indices(
    peak_angles_degrees,
    already_selected_indices,
    available_indices,
    radius_data,
    target_new_count,
):
    already_selected = set(int(index) for index in numpy.asarray(already_selected_indices, dtype=int))
    available = [int(index) for index in numpy.asarray(available_indices, dtype=int) if int(index) not in already_selected]
    angles = numpy.asarray(radius_data[:, 1:4], dtype=float)
    snapped = []
    used = set(already_selected)

    for peak_angle in numpy.asarray(peak_angles_degrees, dtype=float):
        if len(snapped) >= int(target_new_count):
            break
        candidates = [index for index in available if index not in used]
        if not candidates:
            break
        candidate_angles = angles[numpy.asarray(candidates, dtype=int)]
        nearest_position = int(numpy.argmin(numpy.sum((candidate_angles - peak_angle) ** 2, axis=1)))
        nearest_index = int(candidates[nearest_position])
        snapped.append(nearest_index)
        used.add(nearest_index)
    return numpy.asarray(snapped, dtype=int)


def acquisition_extend_point_indices(
    previous_indices,
    target_count,
    available_indices,
    radius_data,
    fit_result=None,
    candidate_grid=None,
    alpha=ACQUISITION_ALPHA,
    beta=ACQUISITION_BETA,
    required_point_indices=None,
    min_peak_separation_degrees=ACQUISITION_MIN_PEAK_SEPARATION_DEGREES,
):
    available_indices = numpy.asarray(available_indices, dtype=int)
    available_set = set(int(index) for index in available_indices)
    selected = []
    for index in numpy.asarray(previous_indices, dtype=int):
        if int(index) in available_set and int(index) not in selected:
            selected.append(int(index))
    if required_point_indices is not None:
        for index in numpy.asarray(required_point_indices, dtype=int):
            if int(index) in available_set and int(index) not in selected:
                selected.append(int(index))

    target_count = max(int(target_count), len(selected))
    if len(selected) >= target_count:
        return numpy.asarray(sorted(selected), dtype=int), {
            "added_indices": numpy.asarray([], dtype=int),
            "peak_angles": numpy.empty((0, 3), dtype=float),
            "acquisition": None,
        }

    if fit_result is None or candidate_grid is None:
        spread_indices = select_spread_indices(selected, target_count, available_indices, radius_data)
        added_indices = numpy.asarray([index for index in spread_indices if int(index) not in set(selected)], dtype=int)
        return spread_indices, {
            "added_indices": added_indices,
            "peak_angles": numpy.empty((0, 3), dtype=float),
            "acquisition": None,
        }

    selected_radius_data = radius_data[numpy.asarray(selected, dtype=int)]
    acquisition, acquisition_detail = acquisition_values_for_fit(
        fit_result,
        selected_radius_data,
        candidate_grid,
        alpha=alpha,
        beta=beta,
    )
    local_maximum_indices = tensor_local_maximum_indices(candidate_grid, acquisition)
    local_maximum_indices = local_maximum_indices[numpy.argsort(acquisition[local_maximum_indices])[::-1]]

    accepted_peak_angles = []
    min_distance_squared = float(min_peak_separation_degrees) ** 2
    for candidate_index in local_maximum_indices:
        candidate_angle = candidate_grid[int(candidate_index)]
        if accepted_peak_angles:
            distances_squared = numpy.sum((numpy.asarray(accepted_peak_angles) - candidate_angle) ** 2, axis=1)
            if numpy.min(distances_squared) < min_distance_squared:
                continue
        accepted_peak_angles.append(candidate_angle)
        if len(accepted_peak_angles) >= target_count - len(selected):
            break

    if len(accepted_peak_angles) < target_count - len(selected):
        fallback_indices = numpy.argsort(acquisition)[::-1]
        for candidate_index in fallback_indices:
            candidate_angle = candidate_grid[int(candidate_index)]
            if accepted_peak_angles:
                distances_squared = numpy.sum((numpy.asarray(accepted_peak_angles) - candidate_angle) ** 2, axis=1)
                if numpy.min(distances_squared) < min_distance_squared:
                    continue
            accepted_peak_angles.append(candidate_angle)
            if len(accepted_peak_angles) >= target_count - len(selected):
                break

    added_indices = snap_peak_angles_to_available_indices(
        numpy.asarray(accepted_peak_angles, dtype=float),
        selected,
        available_indices,
        radius_data,
        target_count - len(selected),
    )
    point_indices = numpy.asarray(sorted(set(selected).union(int(index) for index in added_indices)), dtype=int)
    if len(point_indices) < target_count:
        point_indices = select_spread_indices(point_indices, target_count, available_indices, radius_data)
        added_indices = numpy.asarray([index for index in point_indices if int(index) not in set(selected)], dtype=int)

    return point_indices, {
        "added_indices": added_indices,
        "peak_angles": numpy.asarray(accepted_peak_angles, dtype=float),
        **acquisition_detail,
    }


def select_basis_cutoff_for_radius_indices_truncated_svd(
    radius_data,
    point_indices,
    test_indices,
    max_l1_grid=ACQUISITION_SWEEP_MAX_L1_GRID,
    max_l2_grid=ACQUISITION_SWEEP_MAX_L2_GRID,
    relative_rmse_cutoff=DEFAULT_RELATIVE_RMSE_CUTOFF,
    min_l1=ACQUISITION_SWEEP_MIN_L1,
    min_l2=ACQUISITION_SWEEP_MIN_L2,
    terms_cap=None,
    svd_threshold=DEFAULT_SVD_THRESHOLD,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
):
    train_data = radius_data[numpy.asarray(point_indices, dtype=int)]
    test_data = radius_data[numpy.asarray(test_indices, dtype=int)]
    candidates = basis_cutoff_candidates(
        max_l1_grid,
        max_l2_grid,
        min_l1=min_l1,
        min_l2=min_l2,
        terms_cap=terms_cap,
    )
    if not candidates:
        raise ValueError("No max-l1/max-l2 basis candidates are available.")

    tried_results = []
    best_result = None
    chosen_result = None
    passed_cutoff = False
    for candidate in candidates:
        score = fit_and_score_truncated_svd(
            train_data,
            test_data,
            candidate["basis_terms"],
            svd_threshold=svd_threshold,
            threshold_mode=threshold_mode,
        )
        score.update(candidate)
        tried_results.append(score)
        if best_result is None or score["test_relative_rmse"] < best_result["test_relative_rmse"]:
            best_result = score
        if numpy.isfinite(score["test_relative_rmse"]) and score["test_relative_rmse"] <= relative_rmse_cutoff:
            chosen_result = score
            passed_cutoff = True
            break

    if chosen_result is None:
        chosen_result = best_result

    fit_result = dict(chosen_result)
    fit_result.update({
        "point_count": int(len(point_indices)),
        "point_indices": numpy.asarray(point_indices, dtype=int),
        "test_indices": numpy.asarray(test_indices, dtype=int),
        "passed_basis_cutoff": passed_cutoff,
        "relative_rmse_cutoff": relative_rmse_cutoff,
        "tried_results": tried_results,
        **basis_support_cutoffs(chosen_result["basis_terms"]),
    })
    return fit_result


def fit_radius_with_acquisition_selector(
    radius_data,
    point_indices,
    test_indices,
    basis_selection_method="sweep_l1_l2",
    max_l1_grid=ACQUISITION_SWEEP_MAX_L1_GRID,
    max_l2_grid=ACQUISITION_SWEEP_MAX_L2_GRID,
    max_l1=DEFAULT_MAX_L1,
    max_l2=DEFAULT_MAX_L2,
    relative_rmse_cutoff=DEFAULT_RELATIVE_RMSE_CUTOFF,
    svd_threshold=DEFAULT_SVD_THRESHOLD,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
):
    if basis_selection_method == "sweep_l1_l2":
        return select_basis_cutoff_for_radius_indices_truncated_svd(
            radius_data,
            point_indices,
            test_indices,
            max_l1_grid=max_l1_grid,
            max_l2_grid=max_l2_grid,
            relative_rmse_cutoff=relative_rmse_cutoff,
            svd_threshold=svd_threshold,
            threshold_mode=threshold_mode,
        )
    if basis_selection_method == "incremental_terms":
        return select_incremental_terms_for_radius_indices(
            radius_data,
            point_indices,
            test_indices,
            universe_max_l1=max_l1,
            universe_max_l2=max_l2,
            relative_rmse_cutoff=relative_rmse_cutoff,
            svd_threshold=svd_threshold,
            threshold_mode=threshold_mode,
            min_terms=1,
            max_terms=None,
        )
    raise ValueError(f"Unknown basis_selection_method: {basis_selection_method!r}")


def run_radius_to_radius_acquisition_point_count_experiment(
    all_data,
    basis_selection_method="sweep_l1_l2",
    gamma=RADIUS_TO_RADIUS_GAMMA,
    train_fraction=DEFAULT_TRAIN_FRACTION,
    random_seed=DEFAULT_RANDOM_SEED,
    candidate_grid=None,
    initial_fit_point_count=RADIUS_TO_RADIUS_INITIAL_FIT_POINT_COUNT,
    initial_test_point_count=RADIUS_TO_RADIUS_INITIAL_TEST_POINT_COUNT,
    required_angle_points_degrees=ACQUISITION_REQUIRED_ANGLE_POINTS_DEGREES,
    acquisition_alpha=ACQUISITION_ALPHA,
    acquisition_beta=ACQUISITION_BETA,
    min_peak_separation_degrees=ACQUISITION_MIN_PEAK_SEPARATION_DEGREES,
    relative_rmse_cutoff=DEFAULT_RELATIVE_RMSE_CUTOFF,
    svd_threshold=DEFAULT_SVD_THRESHOLD,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
):
    radii, radius_data_by_r = grouped_radius_data(all_data)
    if not validate_common_angular_grid(radius_data_by_r):
        raise ValueError("All radii must share the same angular grid to reuse point indices across R.")
    if candidate_grid is None:
        candidate_grid = make_angular_candidate_grid()

    first_radius_data = radius_data_by_r[float(radii[0])]
    required_point_indices = nearest_angle_indices(first_radius_data, required_angle_points_degrees)
    train_pool_indices, test_indices = split_train_pool_with_required_points(
        first_radius_data,
        required_point_indices,
        initial_test_point_count=initial_test_point_count,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )

    results_by_radius = {}
    transitions = []
    previous_complexity = None
    previous_point_indices = None
    previous_point_count = None
    previous_term_count = None
    previous_fit_result = None

    for radius_index, r_value in enumerate(radii):
        r_value = float(r_value)
        radius_data = radius_data_by_r[r_value]

        if radius_index == 0:
            target_point_count = min(int(initial_fit_point_count), train_pool_indices.size)
            point_indices, acquisition_detail = acquisition_extend_point_indices(
                [],
                target_point_count,
                train_pool_indices,
                radius_data,
                fit_result=None,
                candidate_grid=None,
                required_point_indices=required_point_indices,
            )
            predicted_point_count = int(len(point_indices))
            derivative_point_count = predicted_point_count
            prediction_term_count = None
            basis_point_count = None
        else:
            prediction_term_count = int(previous_term_count)
            predicted_point_count, point_count_prediction_diagnostics = predict_next_point_count(
                previous_point_count,
                current_complexity=results_by_radius[float(radii[radius_index - 1])]["complexity"],
                previous_complexity=previous_complexity,
                next_term_count=prediction_term_count,
                gamma=gamma,
                angular_dimension=RADIUS_TO_RADIUS_ANGULAR_DIMENSION,
                max_available_points=train_pool_indices.size,
                current_radius=float(radii[radius_index - 1]),
                return_diagnostics=True,
            )
            basis_point_count = basis_sampling_requirement(prediction_term_count, gamma=gamma)
            derivative_point_count = predicted_point_count
            point_indices, acquisition_detail = acquisition_extend_point_indices(
                previous_point_indices,
                predicted_point_count,
                train_pool_indices,
                radius_data,
                fit_result=previous_fit_result,
                candidate_grid=candidate_grid,
                alpha=acquisition_alpha,
                beta=acquisition_beta,
                required_point_indices=required_point_indices,
                min_peak_separation_degrees=min_peak_separation_degrees,
            )
            predicted_point_count = max(int(predicted_point_count), int(len(point_indices)))

        if radius_index == 0:
            point_count_prediction_diagnostics = {
                "curvature_ratio": 1.0,
                "radial_damping": 1.0,
                "effective_curvature_exponent": 0.0,
                "curvature_growth_factor": 1.0,
                "basis_estimate": 0,
                "derivative_estimate": float(len(point_indices)),
            }

        fit_result = fit_radius_with_acquisition_selector(
            radius_data,
            point_indices,
            test_indices,
            basis_selection_method=basis_selection_method,
            relative_rmse_cutoff=relative_rmse_cutoff,
            svd_threshold=svd_threshold,
            threshold_mode=threshold_mode,
        )
        term_count = int(fit_result["term_count"])
        complexity = derivative_complexity_for_fit(fit_result, candidate_grid)
        capped_basis_point_count = int(min(basis_sampling_requirement(term_count, gamma=gamma), train_pool_indices.size))

        results_by_radius[r_value] = {
            "R": r_value,
            "fit": fit_result,
            "complexity": complexity,
            "term_count": term_count,
            "point_count": int(len(point_indices)),
            "point_indices": point_indices,
            "test_indices": test_indices,
            "basis_point_count": capped_basis_point_count,
            "prediction_term_count": None if prediction_term_count is None else int(prediction_term_count),
            "predicted_point_count": int(predicted_point_count),
            "derivative_point_count": int(min(numpy.ceil(derivative_point_count), train_pool_indices.size)),
            "passed_basis_cutoff": bool(fit_result.get("passed_basis_cutoff", False)),
            "acquisition_detail": acquisition_detail,
            "point_count_prediction_diagnostics": point_count_prediction_diagnostics,
        }

        if radius_index > 0:
            previous_r = float(radii[radius_index - 1])
            reused_point_count = int(numpy.intersect1d(previous_point_indices, point_indices).size)
            transitions.append({
                "from_R": previous_r,
                "to_R": r_value,
                "predicted_point_count": int(predicted_point_count),
                "used_point_count": int(len(point_indices)),
                "reused_point_count": reused_point_count,
                "added_point_count": int(len(point_indices) - reused_point_count),
                "prediction_term_count": int(prediction_term_count),
                "term_count": term_count,
                "basis_point_count": capped_basis_point_count,
                "passed_basis_cutoff": bool(fit_result.get("passed_basis_cutoff", False)),
                "test_rmse": fit_result.get("test_rmse", numpy.nan),
                "test_relative_rmse": fit_result.get("test_relative_rmse", numpy.nan),
                "test_average_relative_absolute_error": fit_result.get("test_average_relative_absolute_error", numpy.nan),
                "normalized_curvature_scale": complexity["normalized_curvature_scale"],
                "new_point_indices": acquisition_detail.get("added_indices", numpy.asarray([], dtype=int)),
                "peak_angles": acquisition_detail.get("peak_angles", numpy.empty((0, 3), dtype=float)),
                **point_count_prediction_diagnostics,
            })

        previous_complexity = None if radius_index == 0 else results_by_radius[float(radii[radius_index - 1])]["complexity"]
        previous_point_indices = point_indices
        previous_point_count = int(len(point_indices))
        previous_term_count = term_count
        previous_fit_result = fit_result

    return {
        "basis_selection_method": basis_selection_method,
        "radii_descending": radii,
        "train_pool_indices": train_pool_indices,
        "test_indices": test_indices,
        "required_point_indices": required_point_indices,
        "required_angle_points_degrees": numpy.asarray(required_angle_points_degrees, dtype=float),
        "candidate_grid": candidate_grid,
        "results_by_radius": results_by_radius,
        "transitions": transitions,
        "gamma": gamma,
        "acquisition_alpha": acquisition_alpha,
        "acquisition_beta": acquisition_beta,
        "min_peak_separation_degrees": min_peak_separation_degrees,
        "relative_rmse_cutoff": relative_rmse_cutoff,
        "initial_fit_point_count": int(initial_fit_point_count) if initial_fit_point_count is not None else None,
        "initial_test_point_count": int(len(test_indices)),
        "random_seed": random_seed,
    }


acquisition_all_radius_data = load_all_radius_data()
acquisition_candidate_grid = make_angular_candidate_grid()


# --- Hysteresis helper definitions ---

# Acquisition-based R -> R' point selection with max-l1/max-l2 hysteresis.
# This returns to the acquisition procedure and stabilizes basis cutoffs across radii.

HYSTERESIS_MEMORY_RADIUS_COUNT = 3
HYSTERESIS_RELATIVE_RMSE_TARGET = DEFAULT_RELATIVE_RMSE_CUTOFF
HYSTERESIS_ABS_RMSE_TARGET_FRACTION = DEFAULT_RELATIVE_RMSE_CUTOFF
HYSTERESIS_RELATIVE_RMSE_FACTOR = 1.3
HYSTERESIS_LARGER_ABS_RMSE_FACTOR = 1.0
HYSTERESIS_SMALLER_ABS_RMSE_FACTOR = 1.5
HYSTERESIS_CONDITION_LIMIT = 1e10

_REQUIRED_HYSTERESIS_HELPERS = [
    "run_radius_to_radius_acquisition_point_count_experiment",
    "select_basis_cutoff_for_radius_indices_truncated_svd",
    "fit_and_score_truncated_svd",
    "generate_all_terms",
    "predict_next_point_count",
    "acquisition_extend_point_indices",
]
missing_hysteresis_helpers = [helper for helper in _REQUIRED_HYSTERESIS_HELPERS if helper not in globals()]
if missing_hysteresis_helpers:
    raise NameError(
        "Run the acquisition-helper cell before this hysteresis cell. Missing: "
        + ", ".join(missing_hysteresis_helpers)
    )


def basis_terms_for_cutoff_pair(max_l1, max_l2):
    return generate_all_terms(int(max_l1), int(max_l2))


def componentwise_anchor_cutoff(previous_fit_results, memory_count=HYSTERESIS_MEMORY_RADIUS_COUNT):
    recent = list(previous_fit_results)[-int(memory_count):]
    if not recent:
        return None
    max_l1_values = [fit.get("max_l1", basis_support_cutoffs(fit["basis_terms"])["max_l1"]) for fit in recent]
    max_l2_values = [fit.get("max_l2", basis_support_cutoffs(fit["basis_terms"])["max_l2"]) for fit in recent]
    return {
        "max_l1": int(max(max_l1_values)),
        "max_l2": int(max(max_l2_values)),
    }


def hysteresis_anchor_cutoff_for_raw(raw_result, previous_fit_results, memory_count=HYSTERESIS_MEMORY_RADIUS_COUNT):
    previous_anchor = componentwise_anchor_cutoff(previous_fit_results, memory_count=memory_count)
    if previous_anchor is None:
        return None

    raw_pair = (int(raw_result["max_l1"]), int(raw_result["max_l2"]))
    previous_pair = (int(previous_anchor["max_l1"]), int(previous_anchor["max_l2"]))
    previous_is_larger = previous_pair[0] >= raw_pair[0] and previous_pair[1] >= raw_pair[1]
    previous_is_smaller = previous_pair[0] <= raw_pair[0] and previous_pair[1] <= raw_pair[1]

    if previous_is_larger or previous_is_smaller:
        return {
            "max_l1": previous_pair[0],
            "max_l2": previous_pair[1],
            "source": "previous_anchor",
            "previous_anchor_pair": previous_pair,
            "raw_pair": raw_pair,
        }

    union_pair = (max(previous_pair[0], raw_pair[0]), max(previous_pair[1], raw_pair[1]))
    return {
        "max_l1": union_pair[0],
        "max_l2": union_pair[1],
        "source": "componentwise_union",
        "previous_anchor_pair": previous_pair,
        "raw_pair": raw_pair,
    }


def score_cutoff_pair_on_radius_indices(
    radius_data,
    point_indices,
    test_indices,
    max_l1,
    max_l2,
    svd_threshold=DEFAULT_SVD_THRESHOLD,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
):
    basis_terms = basis_terms_for_cutoff_pair(max_l1, max_l2)
    score = fit_and_score_truncated_svd(
        radius_data[numpy.asarray(point_indices, dtype=int)],
        radius_data[numpy.asarray(test_indices, dtype=int)],
        basis_terms,
        svd_threshold=svd_threshold,
        threshold_mode=threshold_mode,
    )
    score.update({
        "max_l1": int(max_l1),
        "max_l2": int(max_l2),
        **basis_support_cutoffs(basis_terms),
    })
    return score


def radius_abs_rmse_target(reference_energy, target_fraction=HYSTERESIS_ABS_RMSE_TARGET_FRACTION):
    reference_energy = numpy.asarray(reference_energy, dtype=float)
    return float(target_fraction) * float(numpy.sqrt(numpy.mean(reference_energy**2)))


def choose_hysteresis_basis_result(
    raw_result,
    anchor_result,
    reference_energy,
    relative_rmse_target=HYSTERESIS_RELATIVE_RMSE_TARGET,
    abs_rmse_target_fraction=HYSTERESIS_ABS_RMSE_TARGET_FRACTION,
    relative_rmse_factor=HYSTERESIS_RELATIVE_RMSE_FACTOR,
    larger_abs_rmse_factor=HYSTERESIS_LARGER_ABS_RMSE_FACTOR,
    smaller_abs_rmse_factor=HYSTERESIS_SMALLER_ABS_RMSE_FACTOR,
    condition_limit=HYSTERESIS_CONDITION_LIMIT,
):
    if anchor_result is None:
        raw_result = dict(raw_result)
        raw_result.update({
            "basis_source": "raw",
            "hysteresis_kept_anchor": False,
            "hysteresis_reason": "no previous anchor basis",
        })
        return raw_result

    raw_pair = (int(raw_result["max_l1"]), int(raw_result["max_l2"]))
    anchor_pair = (int(anchor_result["max_l1"]), int(anchor_result["max_l2"]))
    anchor_is_larger = anchor_pair[0] >= raw_pair[0] and anchor_pair[1] >= raw_pair[1] and anchor_pair != raw_pair
    anchor_is_smaller = anchor_pair[0] <= raw_pair[0] and anchor_pair[1] <= raw_pair[1] and anchor_pair != raw_pair
    anchor_relation = "larger" if anchor_is_larger else "smaller" if anchor_is_smaller else "mixed_or_equal"
    abs_rmse_target = radius_abs_rmse_target(reference_energy, target_fraction=abs_rmse_target_fraction)

    anchor_rel_ok = anchor_result["test_relative_rmse"] <= float(relative_rmse_factor) * float(relative_rmse_target)
    larger_anchor_abs_ok = (
        anchor_result["test_rmse"] <= float(larger_abs_rmse_factor) * raw_result["test_rmse"]
        and anchor_result["test_rmse"] <= abs_rmse_target
    )
    smaller_anchor_abs_ok = (
        anchor_result["test_rmse"] <= float(smaller_abs_rmse_factor) * raw_result["test_rmse"]
        or anchor_result["test_rmse"] <= abs_rmse_target
    )
    anchor_abs_ok = (
        larger_anchor_abs_ok if anchor_is_larger
        else smaller_anchor_abs_ok if anchor_is_smaller
        else False
    )
    anchor_condition = anchor_result.get("retained_condition_number", numpy.inf)
    anchor_condition_ok = numpy.isfinite(anchor_condition) and anchor_condition <= float(condition_limit)

    if (anchor_is_larger or anchor_is_smaller) and anchor_rel_ok and anchor_abs_ok and anchor_condition_ok:
        chosen = dict(anchor_result)
        chosen.update({
            "basis_source": f"hysteresis_{anchor_relation}_anchor",
            "hysteresis_kept_anchor": True,
            "hysteresis_anchor_relation": anchor_relation,
            "hysteresis_anchor_source": anchor_result.get("hysteresis_anchor_source", "previous_anchor"),
            "hysteresis_previous_anchor_pair": anchor_result.get("hysteresis_previous_anchor_pair"),
            "hysteresis_raw_pair": anchor_result.get("hysteresis_raw_pair"),
            "hysteresis_reason": f"{anchor_relation} anchor passed relative, absolute, and conditioning checks",
            "raw_basis_result": raw_result,
            "anchor_basis_result": anchor_result,
            "abs_rmse_target": abs_rmse_target,
            "larger_abs_rmse_factor": larger_abs_rmse_factor,
            "smaller_abs_rmse_factor": smaller_abs_rmse_factor,
        })
        return chosen

    chosen = dict(raw_result)
    drop_reasons = []
    if not (anchor_is_larger or anchor_is_smaller):
        drop_reasons.append("anchor was neither componentwise larger nor componentwise smaller than raw")
    if not anchor_rel_ok:
        drop_reasons.append("anchor relative RMSE too large")
    if (anchor_is_larger or anchor_is_smaller) and not anchor_abs_ok:
        drop_reasons.append(f"{anchor_relation} anchor absolute RMSE too large")
    if not anchor_condition_ok:
        drop_reasons.append("anchor condition number too large")
    chosen.update({
        "basis_source": "raw",
        "hysteresis_kept_anchor": False,
        "hysteresis_anchor_relation": anchor_relation,
        "hysteresis_anchor_source": anchor_result.get("hysteresis_anchor_source", "previous_anchor"),
        "hysteresis_previous_anchor_pair": anchor_result.get("hysteresis_previous_anchor_pair"),
        "hysteresis_raw_pair": anchor_result.get("hysteresis_raw_pair"),
        "hysteresis_reason": "; ".join(drop_reasons),
        "raw_basis_result": raw_result,
        "anchor_basis_result": anchor_result,
        "abs_rmse_target": abs_rmse_target,
        "larger_abs_rmse_factor": larger_abs_rmse_factor,
        "smaller_abs_rmse_factor": smaller_abs_rmse_factor,
    })
    return chosen


def fit_radius_with_hysteresis_selector(
    radius_data,
    point_indices,
    test_indices,
    previous_fit_results,
    max_l1_grid=ACQUISITION_SWEEP_MAX_L1_GRID,
    max_l2_grid=ACQUISITION_SWEEP_MAX_L2_GRID,
    relative_rmse_cutoff=DEFAULT_RELATIVE_RMSE_CUTOFF,
    svd_threshold=DEFAULT_SVD_THRESHOLD,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
):
    raw_result = select_basis_cutoff_for_radius_indices_truncated_svd(
        radius_data,
        point_indices,
        test_indices,
        max_l1_grid=max_l1_grid,
        max_l2_grid=max_l2_grid,
        relative_rmse_cutoff=relative_rmse_cutoff,
        svd_threshold=svd_threshold,
        threshold_mode=threshold_mode,
    )
    anchor_cutoff = hysteresis_anchor_cutoff_for_raw(raw_result, previous_fit_results)
    anchor_result = None
    if anchor_cutoff is not None:
        anchor_result = score_cutoff_pair_on_radius_indices(
            radius_data,
            point_indices,
            test_indices,
            anchor_cutoff["max_l1"],
            anchor_cutoff["max_l2"],
            svd_threshold=svd_threshold,
            threshold_mode=threshold_mode,
        )
        anchor_result.update({
            "hysteresis_anchor_source": anchor_cutoff["source"],
            "hysteresis_previous_anchor_pair": anchor_cutoff["previous_anchor_pair"],
            "hysteresis_raw_pair": anchor_cutoff["raw_pair"],
        })

    chosen = choose_hysteresis_basis_result(
        raw_result,
        anchor_result,
        radius_data[numpy.asarray(test_indices, dtype=int), 4],
        relative_rmse_target=relative_rmse_cutoff,
    )
    chosen.update({
        "point_count": int(len(point_indices)),
        "point_indices": numpy.asarray(point_indices, dtype=int),
        "test_indices": numpy.asarray(test_indices, dtype=int),
        "passed_basis_cutoff": numpy.isfinite(chosen.get("test_relative_rmse", numpy.nan))
        and chosen.get("test_relative_rmse", numpy.inf) <= relative_rmse_cutoff,
    })
    return chosen


def run_radius_to_radius_hysteresis_acquisition_experiment(
    all_data,
    gamma=RADIUS_TO_RADIUS_GAMMA,
    train_fraction=DEFAULT_TRAIN_FRACTION,
    random_seed=DEFAULT_RANDOM_SEED,
    candidate_grid=None,
    initial_fit_point_count=RADIUS_TO_RADIUS_INITIAL_FIT_POINT_COUNT,
    initial_test_point_count=RADIUS_TO_RADIUS_INITIAL_TEST_POINT_COUNT,
    required_angle_points_degrees=ACQUISITION_REQUIRED_ANGLE_POINTS_DEGREES,
    acquisition_alpha=ACQUISITION_ALPHA,
    acquisition_beta=ACQUISITION_BETA,
    min_peak_separation_degrees=ACQUISITION_MIN_PEAK_SEPARATION_DEGREES,
    relative_rmse_cutoff=DEFAULT_RELATIVE_RMSE_CUTOFF,
    svd_threshold=DEFAULT_SVD_THRESHOLD,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
):
    radii, radius_data_by_r = grouped_radius_data(all_data)
    if not validate_common_angular_grid(radius_data_by_r):
        raise ValueError("All radii must share the same angular grid to reuse point indices across R.")
    if candidate_grid is None:
        candidate_grid = make_angular_candidate_grid()

    first_radius_data = radius_data_by_r[float(radii[0])]
    required_point_indices = nearest_angle_indices(first_radius_data, required_angle_points_degrees)
    train_pool_indices, test_indices = split_train_pool_with_required_points(
        first_radius_data,
        required_point_indices,
        initial_test_point_count=initial_test_point_count,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )

    results_by_radius = {}
    transitions = []
    previous_complexity = None
    previous_point_indices = None
    previous_point_count = None
    previous_term_count = None
    previous_fit_result = None
    previous_fit_results = []

    for radius_index, r_value in enumerate(radii):
        r_value = float(r_value)
        radius_data = radius_data_by_r[r_value]

        if radius_index == 0:
            target_point_count = min(int(initial_fit_point_count), train_pool_indices.size)
            point_indices, acquisition_detail = acquisition_extend_point_indices(
                [],
                target_point_count,
                train_pool_indices,
                radius_data,
                fit_result=None,
                candidate_grid=None,
                required_point_indices=required_point_indices,
            )
            predicted_point_count = int(len(point_indices))
            derivative_point_count = predicted_point_count
            prediction_term_count = None
            basis_point_count = None
            point_count_prediction_diagnostics = {
                "curvature_ratio": 1.0,
                "radial_damping": 1.0,
                "effective_curvature_exponent": 0.0,
                "curvature_growth_factor": 1.0,
                "basis_estimate": 0,
                "derivative_estimate": float(len(point_indices)),
            }
        else:
            prediction_term_count = int(previous_term_count)
            predicted_point_count, point_count_prediction_diagnostics = predict_next_point_count(
                previous_point_count,
                current_complexity=results_by_radius[float(radii[radius_index - 1])]["complexity"],
                previous_complexity=previous_complexity,
                next_term_count=prediction_term_count,
                gamma=gamma,
                angular_dimension=RADIUS_TO_RADIUS_ANGULAR_DIMENSION,
                max_available_points=train_pool_indices.size,
                current_radius=float(radii[radius_index - 1]),
                return_diagnostics=True,
            )
            basis_point_count = basis_sampling_requirement(prediction_term_count, gamma=gamma)
            derivative_point_count = predicted_point_count
            point_indices, acquisition_detail = acquisition_extend_point_indices(
                previous_point_indices,
                predicted_point_count,
                train_pool_indices,
                radius_data,
                fit_result=previous_fit_result,
                candidate_grid=candidate_grid,
                alpha=acquisition_alpha,
                beta=acquisition_beta,
                required_point_indices=required_point_indices,
                min_peak_separation_degrees=min_peak_separation_degrees,
            )
            predicted_point_count = max(int(predicted_point_count), int(len(point_indices)))

        fit_result = fit_radius_with_hysteresis_selector(
            radius_data,
            point_indices,
            test_indices,
            previous_fit_results,
            relative_rmse_cutoff=relative_rmse_cutoff,
            svd_threshold=svd_threshold,
            threshold_mode=threshold_mode,
        )
        term_count = int(fit_result["term_count"])
        complexity = derivative_complexity_for_fit(fit_result, candidate_grid)
        capped_basis_point_count = int(min(basis_sampling_requirement(term_count, gamma=gamma), train_pool_indices.size))

        results_by_radius[r_value] = {
            "R": r_value,
            "fit": fit_result,
            "complexity": complexity,
            "term_count": term_count,
            "point_count": int(len(point_indices)),
            "point_indices": point_indices,
            "test_indices": test_indices,
            "basis_point_count": capped_basis_point_count,
            "prediction_term_count": None if prediction_term_count is None else int(prediction_term_count),
            "predicted_point_count": int(predicted_point_count),
            "derivative_point_count": int(min(numpy.ceil(derivative_point_count), train_pool_indices.size)),
            "passed_basis_cutoff": bool(fit_result.get("passed_basis_cutoff", False)),
            "basis_source": fit_result.get("basis_source", "raw"),
            "hysteresis_kept_anchor": bool(fit_result.get("hysteresis_kept_anchor", False)),
            "hysteresis_anchor_relation": fit_result.get("hysteresis_anchor_relation", "none"),
            "hysteresis_anchor_source": fit_result.get("hysteresis_anchor_source", "none"),
            "hysteresis_reason": fit_result.get("hysteresis_reason", ""),
            "acquisition_detail": acquisition_detail,
            "point_count_prediction_diagnostics": point_count_prediction_diagnostics,
        }

        if radius_index > 0:
            previous_r = float(radii[radius_index - 1])
            reused_point_count = int(numpy.intersect1d(previous_point_indices, point_indices).size)
            transitions.append({
                "from_R": previous_r,
                "to_R": r_value,
                "predicted_point_count": int(predicted_point_count),
                "used_point_count": int(len(point_indices)),
                "reused_point_count": reused_point_count,
                "added_point_count": int(len(point_indices) - reused_point_count),
                "prediction_term_count": int(prediction_term_count),
                "term_count": term_count,
                "max_l1": int(fit_result["max_l1"]),
                "max_l2": int(fit_result["max_l2"]),
                "basis_source": fit_result.get("basis_source", "raw"),
                "hysteresis_kept_anchor": bool(fit_result.get("hysteresis_kept_anchor", False)),
                "hysteresis_anchor_relation": fit_result.get("hysteresis_anchor_relation", "none"),
                "hysteresis_anchor_source": fit_result.get("hysteresis_anchor_source", "none"),
                "hysteresis_reason": fit_result.get("hysteresis_reason", ""),
                "basis_point_count": capped_basis_point_count,
                "passed_basis_cutoff": bool(fit_result.get("passed_basis_cutoff", False)),
                "test_rmse": fit_result.get("test_rmse", numpy.nan),
                "test_relative_rmse": fit_result.get("test_relative_rmse", numpy.nan),
                "test_average_relative_absolute_error": fit_result.get("test_average_relative_absolute_error", numpy.nan),
                "retained_condition_number": fit_result.get("retained_condition_number", numpy.nan),
                "normalized_curvature_scale": complexity["normalized_curvature_scale"],
                "new_point_indices": acquisition_detail.get("added_indices", numpy.asarray([], dtype=int)),
                "peak_angles": acquisition_detail.get("peak_angles", numpy.empty((0, 3), dtype=float)),
                **point_count_prediction_diagnostics,
            })

        previous_complexity = None if radius_index == 0 else results_by_radius[float(radii[radius_index - 1])]["complexity"]
        previous_point_indices = point_indices
        previous_point_count = int(len(point_indices))
        previous_term_count = term_count
        previous_fit_result = fit_result
        previous_fit_results.append(fit_result)

    return {
        "basis_selection_method": "sweep_l1_l2_with_hysteresis",
        "radii_descending": radii,
        "train_pool_indices": train_pool_indices,
        "test_indices": test_indices,
        "required_point_indices": required_point_indices,
        "required_angle_points_degrees": numpy.asarray(required_angle_points_degrees, dtype=float),
        "candidate_grid": candidate_grid,
        "results_by_radius": results_by_radius,
        "transitions": transitions,
        "gamma": gamma,
        "hysteresis_memory_radius_count": HYSTERESIS_MEMORY_RADIUS_COUNT,
        "relative_rmse_target": HYSTERESIS_RELATIVE_RMSE_TARGET,
        "abs_rmse_target_fraction": HYSTERESIS_ABS_RMSE_TARGET_FRACTION,
        "larger_abs_rmse_factor": HYSTERESIS_LARGER_ABS_RMSE_FACTOR,
        "smaller_abs_rmse_factor": HYSTERESIS_SMALLER_ABS_RMSE_FACTOR,
        "condition_limit": HYSTERESIS_CONDITION_LIMIT,
        "acquisition_alpha": acquisition_alpha,
        "acquisition_beta": acquisition_beta,
        "min_peak_separation_degrees": min_peak_separation_degrees,
        "initial_fit_point_count": int(initial_fit_point_count) if initial_fit_point_count is not None else None,
        "initial_test_point_count": int(len(test_indices)),
        "random_seed": random_seed,
    }


def radius_data_for_transition(all_data, radius):
    radius = float(radius)
    if isinstance(all_data, dict):
        return all_data[radius]
    _, radius_data_by_r = grouped_radius_data(all_data)
    return radius_data_by_r[radius]


def acquisition_demanded_points_for_transition(transition, all_data, decimals=6):
    added_indices = numpy.asarray(transition.get("new_point_indices", []), dtype=int)
    peak_angles = numpy.asarray(transition.get("peak_angles", numpy.empty((0, 3))), dtype=float)
    if added_indices.size == 0:
        return []

    radius_data = radius_data_for_transition(all_data, transition["to_R"])
    demanded_points = []
    for local_position, point_index in enumerate(added_indices):
        grid_angles = radius_data[int(point_index), 1:4]
        if local_position < len(peak_angles):
            requested_angles = peak_angles[local_position]
        else:
            requested_angles = numpy.full(3, numpy.nan)
        demanded_points.append({
            "index": int(point_index),
            "requested_theta1": round(float(requested_angles[0]), decimals),
            "requested_theta2": round(float(requested_angles[1]), decimals),
            "requested_phi": round(float(requested_angles[2]), decimals),
            "theta1": round(float(grid_angles[0]), decimals),
            "theta2": round(float(grid_angles[1]), decimals),
            "phi": round(float(grid_angles[2]), decimals),
        })
    return demanded_points


# --- Adaptive basis/point selection helpers ---

ADAPTIVE_POINT_COUNT_RULE = "N > p"

ADAPTIVE_CANDIDATE_TERM_STEP = 4

ADAPTIVE_VALIDATION_SPLITS = 5

ADAPTIVE_VALIDATION_FRACTION = 0.25

ADAPTIVE_GEOMETRY_TOLERANCE = 0.5

ADAPTIVE_PREDICTION_STABILITY_TOLERANCE = 0.01

ADAPTIVE_COEFFICIENT_STABILITY_TOLERANCE = 0.5

ADAPTIVE_CONFIDENCE_ALPHA = 0.05

ADAPTIVE_EQUIVALENCE_EPSILON = 0.001

ADAPTIVE_GAIN_EPSILON = 0.001

ADAPTIVE_POINT_BATCH_SIZE = 20


def adaptive_minimum_point_count_for_basis(term_count):
    """Smallest N satisfying the current basis-size rule N > p."""
    return max(2, int(term_count) + 1)

def adaptive_candidate_term_counts(
    point_count,
    universe_max_l1=DEFAULT_MAX_L1,
    universe_max_l2=DEFAULT_MAX_L2,
    min_terms=1,
    max_terms=None,
    candidate_step=ADAPTIVE_CANDIDATE_TERM_STEP,
):
    ordered_terms = generate_all_terms(universe_max_l1, universe_max_l2)
    if max_terms is None:
        max_terms = len(ordered_terms)
    max_terms = min(int(max_terms), len(ordered_terms))
    allowed_counts = [
        term_count
        for term_count in range(max(1, int(min_terms)), max_terms + 1)
        if adaptive_minimum_point_count_for_basis(term_count) <= int(point_count)
    ]
    if not allowed_counts:
        return numpy.asarray([1], dtype=int)

    candidate_step = max(1, int(candidate_step))
    sampled_counts = allowed_counts[::candidate_step]
    if sampled_counts[-1] != allowed_counts[-1]:
        sampled_counts.append(allowed_counts[-1])
    return numpy.asarray(sorted(set(sampled_counts)), dtype=int)

def make_repeated_validation_splits(point_count, n_splits=ADAPTIVE_VALIDATION_SPLITS, validation_fraction=ADAPTIVE_VALIDATION_FRACTION, random_seed=DEFAULT_RANDOM_SEED):
    point_count = int(point_count)
    validation_count = max(1, int(numpy.ceil(float(validation_fraction) * point_count)))
    if validation_count >= point_count:
        validation_count = point_count - 1
    if validation_count <= 0:
        raise ValueError("Need at least two selected points for repeated validation splits.")

    rng = numpy.random.default_rng(random_seed)
    splits = []
    for _ in range(int(n_splits)):
        permutation = rng.permutation(point_count)
        validation_positions = numpy.sort(permutation[:validation_count])
        train_positions = numpy.sort(permutation[validation_count:])
        splits.append((train_positions, validation_positions))
    return splits

def symmetric_inverse_square_root(matrix, ridge=1e-10):
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = numpy.linalg.eigh(matrix)
    clipped = numpy.maximum(eigenvalues, float(ridge))
    return (eigenvectors * (1.0 / numpy.sqrt(clipped))) @ eigenvectors.T

def gram_geometry_delta(reference_data, selected_data, basis_terms, ridge=1e-10):
    X_reference = design_matrix_for(reference_data, basis_terms)
    X_selected = design_matrix_for(selected_data, basis_terms)
    G_reference = (X_reference.T @ X_reference) / max(1, X_reference.shape[0])
    G_selected = (X_selected.T @ X_selected) / max(1, X_selected.shape[0])
    reference_inverse_sqrt = symmetric_inverse_square_root(G_reference, ridge=ridge)
    whitened_selected = reference_inverse_sqrt @ G_selected @ reference_inverse_sqrt
    return float(numpy.linalg.norm(whitened_selected - numpy.eye(len(basis_terms)), ord=2))

def score_candidate_basis_repeated(
    radius_data,
    point_indices,
    test_indices,
    basis_terms,
    validation_splits,
    svd_threshold=DEFAULT_SVD_THRESHOLD,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
    geometry_tolerance=ADAPTIVE_GEOMETRY_TOLERANCE,
    prediction_stability_tolerance=ADAPTIVE_PREDICTION_STABILITY_TOLERANCE,
    coefficient_stability_tolerance=ADAPTIVE_COEFFICIENT_STABILITY_TOLERANCE,
):
    point_indices = numpy.asarray(point_indices, dtype=int)
    selected_data = radius_data[point_indices]
    test_data = radius_data[numpy.asarray(test_indices, dtype=int)]
    validation_errors = []
    ranks = []
    condition_numbers = []
    coefficients = []
    test_predictions = []

    for train_positions, validation_positions in validation_splits:
        train_data = selected_data[numpy.asarray(train_positions, dtype=int)]
        validation_data = selected_data[numpy.asarray(validation_positions, dtype=int)]
        split_fit = fit_basis_truncated_svd(
            train_data,
            basis_terms,
            svd_threshold=svd_threshold,
            threshold_mode=threshold_mode,
        )
        X_validation = design_matrix_for(validation_data, basis_terms)
        validation_targets = validation_data[:, 4]
        validation_residuals = validation_targets - X_validation @ split_fit["coefficients"]
        validation_errors.append(relative_rmse(validation_residuals, validation_targets))
        ranks.append(int(split_fit["rank"]))
        condition_numbers.append(float(split_fit["retained_condition_number"]))
        coefficients.append(split_fit["coefficients"])
        test_predictions.append(design_matrix_for(test_data, basis_terms) @ split_fit["coefficients"])

    validation_errors = numpy.asarray(validation_errors, dtype=float)
    ranks = numpy.asarray(ranks, dtype=int)
    condition_numbers = numpy.asarray(condition_numbers, dtype=float)
    coefficients = numpy.vstack(coefficients)
    test_predictions = numpy.vstack(test_predictions)

    prediction_scale = max(float(numpy.sqrt(numpy.mean(test_data[:, 4] ** 2))), 1e-300)
    prediction_stability = float(numpy.sqrt(numpy.mean(numpy.var(test_predictions, axis=0))) / prediction_scale)
    coefficient_scale = max(float(numpy.linalg.norm(numpy.mean(coefficients, axis=0))), 1e-300)
    coefficient_stability = float(numpy.linalg.norm(numpy.std(coefficients, axis=0)) / coefficient_scale)
    geometry_delta = gram_geometry_delta(radius_data, selected_data, basis_terms)
    term_count = len(basis_terms)

    stable = (
        bool(numpy.all(ranks == term_count))
        and geometry_delta <= float(geometry_tolerance)
        and prediction_stability <= float(prediction_stability_tolerance)
        and coefficient_stability <= float(coefficient_stability_tolerance)
    )

    return {
        "basis_terms": basis_terms,
        "term_count": term_count,
        **basis_support_cutoffs(basis_terms),
        "validation_errors": validation_errors,
        "mean_validation_relative_rmse": float(numpy.mean(validation_errors)),
        "rank_min": int(numpy.min(ranks)),
        "rank_max": int(numpy.max(ranks)),
        "retained_condition_number_max": float(numpy.nanmax(condition_numbers)),
        "geometry_delta": geometry_delta,
        "prediction_stability": prediction_stability,
        "coefficient_stability": coefficient_stability,
        "stable": stable,
    }

def confidence_adjustment(values, confidence_alpha=ADAPTIVE_CONFIDENCE_ALPHA):
    values = numpy.asarray(values, dtype=float)
    if values.size <= 1:
        return float(numpy.mean(values)), numpy.inf
    mean_value = float(numpy.mean(values))
    standard_error = float(numpy.std(values, ddof=1) / numpy.sqrt(values.size))
    t_value = float(scipy.stats.t.ppf(1.0 - float(confidence_alpha), df=values.size - 1))
    return mean_value, t_value * standard_error

def select_adaptive_basis(candidate_scores, equivalence_epsilon=ADAPTIVE_EQUIVALENCE_EPSILON, gain_epsilon=ADAPTIVE_GAIN_EPSILON, confidence_alpha=ADAPTIVE_CONFIDENCE_ALPHA):
    stable_scores = [score for score in candidate_scores if score["stable"]]
    if not stable_scores:
        best_available = min(candidate_scores, key=lambda score: score["mean_validation_relative_rmse"])
        return {
            "selected_score": best_available,
            "best_stable_score": None,
            "richer_score": None,
            "stable_scores": [],
            "action": "add diagnostic points; no candidate was stable",
            "passed_stability": False,
            "noninferiority_ucb": numpy.nan,
            "richer_improvement_lcb": numpy.nan,
        }

    best_stable = min(stable_scores, key=lambda score: score["mean_validation_relative_rmse"])
    selected_score = best_stable
    selected_ucb = numpy.inf
    for score in sorted(stable_scores, key=lambda candidate: candidate["term_count"]):
        paired_excess = score["validation_errors"] - best_stable["validation_errors"]
        mean_excess, excess_adjustment = confidence_adjustment(paired_excess, confidence_alpha=confidence_alpha)
        upper_bound = mean_excess + excess_adjustment
        if upper_bound <= float(equivalence_epsilon):
            selected_score = score
            selected_ucb = upper_bound
            break

    richer_candidates = [score for score in candidate_scores if score["term_count"] > selected_score["term_count"]]
    richer_score = min(richer_candidates, key=lambda score: score["term_count"]) if richer_candidates else None
    richer_improvement_lcb = numpy.nan
    action = "retain smaller basis"
    final_score = selected_score

    if richer_score is not None:
        improvement = selected_score["validation_errors"] - richer_score["validation_errors"]
        mean_improvement, improvement_adjustment = confidence_adjustment(improvement, confidence_alpha=confidence_alpha)
        richer_improvement_lcb = mean_improvement - improvement_adjustment
        meaningful_gain = richer_improvement_lcb > float(gain_epsilon)
        if meaningful_gain and richer_score["stable"]:
            final_score = richer_score
            action = "add terms without adding points"
        elif meaningful_gain and not richer_score["stable"]:
            action = "add points targeted to support richer basis"
        elif not meaningful_gain and selected_score["stable"]:
            action = "retain smaller basis"
        else:
            action = "add diagnostic points or test another basis block"

    return {
        "selected_score": final_score,
        "best_stable_score": best_stable,
        "richer_score": richer_score,
        "stable_scores": stable_scores,
        "action": action,
        "passed_stability": True,
        "noninferiority_ucb": selected_ucb,
        "richer_improvement_lcb": richer_improvement_lcb,
    }

def evaluate_adaptive_basis_candidates(
    radius_data,
    point_indices,
    test_indices,
    universe_max_l1=DEFAULT_MAX_L1,
    universe_max_l2=DEFAULT_MAX_L2,
    candidate_step=ADAPTIVE_CANDIDATE_TERM_STEP,
    validation_splits=ADAPTIVE_VALIDATION_SPLITS,
    validation_fraction=ADAPTIVE_VALIDATION_FRACTION,
    random_seed=DEFAULT_RANDOM_SEED,
    svd_threshold=DEFAULT_SVD_THRESHOLD,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
):
    term_counts = adaptive_candidate_term_counts(
        len(point_indices),
        universe_max_l1=universe_max_l1,
        universe_max_l2=universe_max_l2,
        candidate_step=candidate_step,
    )
    splits = make_repeated_validation_splits(
        len(point_indices),
        n_splits=validation_splits,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
    )
    ordered_terms = generate_all_terms(universe_max_l1, universe_max_l2)
    candidate_scores = []
    for term_count in term_counts:
        candidate_scores.append(
            score_candidate_basis_repeated(
                radius_data,
                point_indices,
                test_indices,
                ordered_terms[:int(term_count)],
                splits,
                svd_threshold=svd_threshold,
                threshold_mode=threshold_mode,
            )
        )
    return candidate_scores

def fit_adaptive_selected_basis(radius_data, point_indices, test_indices, selected_score, svd_threshold=DEFAULT_SVD_THRESHOLD, threshold_mode=DEFAULT_THRESHOLD_MODE):
    fit_result = fit_and_score_truncated_svd(
        radius_data[numpy.asarray(point_indices, dtype=int)],
        radius_data[numpy.asarray(test_indices, dtype=int)],
        selected_score["basis_terms"],
        svd_threshold=svd_threshold,
        threshold_mode=threshold_mode,
    )
    fit_result.update({
        "point_count": int(len(point_indices)),
        "point_indices": numpy.asarray(point_indices, dtype=int),
        "test_indices": numpy.asarray(test_indices, dtype=int),
        "adaptive_score": selected_score,
        "passed_basis_cutoff": bool(selected_score.get("stable", False)),
        **basis_support_cutoffs(selected_score["basis_terms"]),
    })
    return fit_result

def adapt_basis_and_points_at_radius(
    radius_data,
    point_indices,
    test_indices,
    train_pool_indices,
    candidate_grid,
    random_seed=DEFAULT_RANDOM_SEED,
    point_batch_size=ADAPTIVE_POINT_BATCH_SIZE,
    acquisition_alpha=ACQUISITION_ALPHA,
    acquisition_beta=ACQUISITION_BETA,
    rerun_after_adding_points=True,
):
    candidate_scores = evaluate_adaptive_basis_candidates(
        radius_data,
        point_indices,
        test_indices,
        random_seed=random_seed,
    )
    selection = select_adaptive_basis(candidate_scores)
    fit_result = fit_adaptive_selected_basis(radius_data, point_indices, test_indices, selection["selected_score"])
    added_points = numpy.asarray([], dtype=int)

    should_add_points = selection["action"] in {
        "add points targeted to support richer basis",
        "add diagnostic points; no candidate was stable",
        "add diagnostic points or test another basis block",
    }
    if should_add_points:
        target_term_count = fit_result["term_count"]
        if selection.get("richer_score") is not None:
            target_term_count = max(target_term_count, selection["richer_score"]["term_count"])
        target_count = max(
            len(point_indices) + int(point_batch_size),
            adaptive_minimum_point_count_for_basis(target_term_count),
        )
        target_count = min(target_count, len(train_pool_indices))
        expanded_indices, acquisition_detail = acquisition_extend_point_indices(
            point_indices,
            target_count,
            train_pool_indices,
            radius_data,
            fit_result=fit_result,
            candidate_grid=candidate_grid,
            alpha=acquisition_alpha,
            beta=acquisition_beta,
            required_point_indices=None,
            min_peak_separation_degrees=ACQUISITION_MIN_PEAK_SEPARATION_DEGREES,
        )
        added_points = numpy.asarray([index for index in expanded_indices if int(index) not in set(point_indices)], dtype=int)
        point_indices = expanded_indices
        if rerun_after_adding_points and len(added_points) > 0:
            candidate_scores = evaluate_adaptive_basis_candidates(
                radius_data,
                point_indices,
                test_indices,
                random_seed=random_seed + 1009,
            )
            selection = select_adaptive_basis(candidate_scores)
            fit_result = fit_adaptive_selected_basis(radius_data, point_indices, test_indices, selection["selected_score"])
    else:
        acquisition_detail = {"added_indices": added_points, "peak_angles": numpy.empty((0, 3), dtype=float)}

    return {
        "point_indices": numpy.asarray(point_indices, dtype=int),
        "fit": fit_result,
        "candidate_scores": candidate_scores,
        "selection": selection,
        "added_points": added_points,
        "acquisition_detail": acquisition_detail,
    }

def run_adaptive_basis_point_selection_experiment(
    all_data,
    initial_fit_point_count=RADIUS_TO_RADIUS_INITIAL_FIT_POINT_COUNT,
    initial_test_point_count=RADIUS_TO_RADIUS_INITIAL_TEST_POINT_COUNT,
    random_seed=DEFAULT_RANDOM_SEED,
    candidate_grid=None,
    required_angle_points_degrees=ACQUISITION_REQUIRED_ANGLE_POINTS_DEGREES,
):
    radii, radius_data_by_r = grouped_radius_data(all_data)
    if not validate_common_angular_grid(radius_data_by_r):
        raise ValueError("All radii must share the same angular grid to reuse point indices across R.")
    if candidate_grid is None:
        candidate_grid = make_angular_candidate_grid()

    first_radius_data = radius_data_by_r[float(radii[0])]
    required_point_indices = nearest_angle_indices(first_radius_data, required_angle_points_degrees)
    train_pool_indices, test_indices = split_train_pool_with_required_points(
        first_radius_data,
        required_point_indices,
        initial_test_point_count=initial_test_point_count,
        random_seed=random_seed,
    )

    results_by_radius = {}
    transitions = []
    previous_complexity = None
    previous_point_indices = None
    previous_point_count = None
    previous_term_count = None
    previous_fit_result = None

    for radius_index, r_value in enumerate(radii):
        r_value = float(r_value)
        radius_data = radius_data_by_r[r_value]
        if radius_index == 0:
            point_indices, acquisition_detail = acquisition_extend_point_indices(
                [],
                initial_fit_point_count,
                train_pool_indices,
                radius_data,
                fit_result=None,
                candidate_grid=None,
                required_point_indices=required_point_indices,
            )
            predicted_point_count = int(len(point_indices))
            point_count_prediction_diagnostics = {
                "curvature_ratio": 1.0,
                "radial_damping": 1.0,
                "effective_curvature_exponent": 0.0,
                "curvature_growth_factor": 1.0,
                "basis_estimate": 0,
                "derivative_estimate": float(len(point_indices)),
            }
        else:
            predicted_point_count, point_count_prediction_diagnostics = predict_next_point_count(
                previous_point_count,
                current_complexity=results_by_radius[float(radii[radius_index - 1])]["complexity"],
                previous_complexity=previous_complexity,
                next_term_count=previous_term_count,
                gamma=RADIUS_TO_RADIUS_GAMMA,
                angular_dimension=RADIUS_TO_RADIUS_ANGULAR_DIMENSION,
                max_available_points=train_pool_indices.size,
                current_radius=float(radii[radius_index - 1]),
                return_diagnostics=True,
            )
            point_indices, acquisition_detail = acquisition_extend_point_indices(
                previous_point_indices,
                predicted_point_count,
                train_pool_indices,
                radius_data,
                fit_result=previous_fit_result,
                candidate_grid=candidate_grid,
                alpha=ACQUISITION_ALPHA,
                beta=ACQUISITION_BETA,
                required_point_indices=required_point_indices,
                min_peak_separation_degrees=ACQUISITION_MIN_PEAK_SEPARATION_DEGREES,
            )

        adapted = adapt_basis_and_points_at_radius(
            radius_data,
            point_indices,
            test_indices,
            train_pool_indices,
            candidate_grid,
            random_seed=random_seed + radius_index,
        )
        point_indices = adapted["point_indices"]
        fit_result = adapted["fit"]
        complexity = derivative_complexity_for_fit(fit_result, candidate_grid)
        term_count = int(fit_result["term_count"])

        results_by_radius[r_value] = {
            "R": r_value,
            "point_indices": point_indices,
            "point_count": int(len(point_indices)),
            "test_indices": test_indices,
            "fit": fit_result,
            "complexity": complexity,
            "term_count": term_count,
            "selection": adapted["selection"],
            "candidate_scores": adapted["candidate_scores"],
            "added_points_after_adaptation": adapted["added_points"],
            "point_count_prediction_diagnostics": point_count_prediction_diagnostics,
        }

        if radius_index > 0:
            previous_r = float(radii[radius_index - 1])
            reused_point_count = int(numpy.intersect1d(previous_point_indices, point_indices).size)
            transitions.append({
                "from_R": previous_r,
                "to_R": r_value,
                "N_pred": int(predicted_point_count),
                "N_used": int(len(point_indices)),
                "N_reused": reused_point_count,
                "N_added": int(len(point_indices) - reused_point_count),
                "terms": term_count,
                "action": adapted["selection"]["action"],
                "stable": bool(adapted["selection"].get("passed_stability", False)),
                "test_relative_rmse": fit_result.get("test_relative_rmse", numpy.nan),
                "test_avg_relative_abs_error": fit_result.get("test_average_relative_absolute_error", numpy.nan),
                **point_count_prediction_diagnostics,
            })

        previous_complexity = None if radius_index == 0 else results_by_radius[float(radii[radius_index - 1])]["complexity"]
        previous_point_indices = point_indices
        previous_point_count = int(len(point_indices))
        previous_term_count = term_count
        previous_fit_result = fit_result

    return {
        "radii_descending": radii,
        "train_pool_indices": train_pool_indices,
        "test_indices": test_indices,
        "required_point_indices": required_point_indices,
        "candidate_grid": candidate_grid,
        "results_by_radius": results_by_radius,
        "transitions": transitions,
        "point_count_rule": ADAPTIVE_POINT_COUNT_RULE,
        "equivalence_epsilon": ADAPTIVE_EQUIVALENCE_EPSILON,
        "gain_epsilon": ADAPTIVE_GAIN_EPSILON,
        "random_seed": random_seed,
    }


# --- Second-pass hysteresis helpers ---

# Hysteresis acquisition with a second point-count pass after basis/conditioning changes.
# This cell leaves the previous hysteresis algorithm intact for comparison.

SECOND_PASS_TERM_GROWTH_FRACTION = 0.50
SECOND_PASS_TERM_GROWTH_MIN_INCREASE = 8
SECOND_PASS_MAX_CONDITION_NUMBER = 100.0
SECOND_PASS_MIN_EXTRA_POINTS = 1

_REQUIRED_SECOND_PASS_HELPERS = [
    "grouped_radius_data",
    "validate_common_angular_grid",
    "nearest_angle_indices",
    "split_train_pool_with_required_points",
    "acquisition_extend_point_indices",
    "fit_radius_with_hysteresis_selector",
    "fit_and_score_truncated_svd",
    "basis_support_cutoffs",
    "derivative_complexity_for_fit",
    "predict_next_point_count",
    "basis_sampling_requirement",
]
missing_second_pass_helpers = [helper for helper in _REQUIRED_SECOND_PASS_HELPERS if helper not in globals()]
if missing_second_pass_helpers:
    raise NameError(
        "Run the shared, acquisition, and hysteresis-helper cells before this second-pass cell. Missing: "
        + ", ".join(missing_second_pass_helpers)
    )


def fixed_basis_score_for_radius_indices(
    radius_data,
    point_indices,
    test_indices,
    basis_terms,
    svd_threshold=DEFAULT_SVD_THRESHOLD,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
):
    score = fit_and_score_truncated_svd(
        radius_data[numpy.asarray(point_indices, dtype=int)],
        radius_data[numpy.asarray(test_indices, dtype=int)],
        basis_terms,
        svd_threshold=svd_threshold,
        threshold_mode=threshold_mode,
    )
    score.update(basis_support_cutoffs(basis_terms))
    return score


def second_pass_trigger_diagnostics(
    previous_term_count,
    first_pass_term_count,
    first_pass_condition_number,
    growth_fraction_threshold=SECOND_PASS_TERM_GROWTH_FRACTION,
    growth_min_terms=SECOND_PASS_TERM_GROWTH_MIN_INCREASE,
    max_condition_number=SECOND_PASS_MAX_CONDITION_NUMBER,
):
    previous_term_count = int(previous_term_count)
    first_pass_term_count = int(first_pass_term_count)
    term_increase = first_pass_term_count - previous_term_count
    relative_growth = term_increase / max(previous_term_count, 1)
    term_growth_trigger = (
        term_increase > int(growth_min_terms)
        and relative_growth > float(growth_fraction_threshold)
    )
    condition_trigger = (
        numpy.isfinite(first_pass_condition_number)
        and float(first_pass_condition_number) > float(max_condition_number)
    )
    return {
        "term_increase": int(term_increase),
        "term_relative_growth": float(relative_growth),
        "term_growth_trigger": bool(term_growth_trigger),
        "condition_trigger": bool(condition_trigger),
        "triggered": bool(term_growth_trigger or condition_trigger),
    }


def second_pass_target_point_count(
    first_pass_point_count,
    current_complexity,
    previous_radius_complexity_at_fixed_basis,
    first_pass_term_count,
    first_pass_retained_rank,
    current_radius,
    gamma=RADIUS_TO_RADIUS_GAMMA,
    max_available_points=None,
    min_extra_points=SECOND_PASS_MIN_EXTRA_POINTS,
):
    retained_rank = max(1, int(first_pass_retained_rank))
    target_count, diagnostics = predict_next_point_count(
        int(first_pass_point_count),
        current_complexity=current_complexity,
        previous_complexity=previous_radius_complexity_at_fixed_basis,
        next_term_count=retained_rank,
        gamma=gamma,
        angular_dimension=RADIUS_TO_RADIUS_ANGULAR_DIMENSION,
        max_available_points=max_available_points,
        current_radius=current_radius,
        return_diagnostics=True,
    )
    diagnostics.update({
        "basis_estimate_uses": "retained_rank",
        "basis_term_count": int(first_pass_term_count),
        "basis_retained_rank": retained_rank,
    })
    if max_available_points is None:
        target_count = max(int(target_count), int(first_pass_point_count) + int(min_extra_points))
    else:
        target_count = min(
            int(max_available_points),
            max(int(target_count), int(first_pass_point_count) + int(min_extra_points)),
        )
    target_count = max(int(first_pass_point_count), int(target_count))
    return target_count, diagnostics


def apply_second_point_pass_if_needed(
    previous_radius_data,
    current_radius_data,
    point_indices,
    train_pool_indices,
    test_indices,
    first_pass_fit,
    first_pass_complexity,
    previous_fit_results,
    previous_term_count,
    current_radius,
    candidate_grid,
    gamma=RADIUS_TO_RADIUS_GAMMA,
    acquisition_alpha=ACQUISITION_ALPHA,
    acquisition_beta=ACQUISITION_BETA,
    required_point_indices=None,
    min_peak_separation_degrees=ACQUISITION_MIN_PEAK_SEPARATION_DEGREES,
    relative_rmse_cutoff=DEFAULT_RELATIVE_RMSE_CUTOFF,
    svd_threshold=DEFAULT_SVD_THRESHOLD,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
    growth_fraction_threshold=SECOND_PASS_TERM_GROWTH_FRACTION,
    growth_min_terms=SECOND_PASS_TERM_GROWTH_MIN_INCREASE,
    max_condition_number=SECOND_PASS_MAX_CONDITION_NUMBER,
    min_extra_points=SECOND_PASS_MIN_EXTRA_POINTS,
):
    first_pass_term_count = int(first_pass_fit["term_count"])
    first_pass_condition = first_pass_fit.get("retained_condition_number", numpy.inf)
    trigger = second_pass_trigger_diagnostics(
        previous_term_count,
        first_pass_term_count,
        first_pass_condition,
        growth_fraction_threshold=growth_fraction_threshold,
        growth_min_terms=growth_min_terms,
        max_condition_number=max_condition_number,
    )
    if not trigger["triggered"]:
        return {
            "fit": first_pass_fit,
            "complexity": first_pass_complexity,
            "point_indices": numpy.asarray(point_indices, dtype=int),
            "target_point_count": int(len(point_indices)),
            "trigger_diagnostics": trigger,
            "point_count_diagnostics": None,
            "previous_radius_fixed_basis_fit": None,
            "previous_radius_fixed_basis_complexity": None,
            "acquisition_detail": {
                "added_indices": numpy.asarray([], dtype=int),
                "peak_angles": numpy.empty((0, 3), dtype=float),
                "acquisition": None,
            },
            "performed": False,
        }

    basis_terms = first_pass_fit["basis_terms"]
    previous_radius_fixed_basis_fit = fixed_basis_score_for_radius_indices(
        previous_radius_data,
        point_indices,
        test_indices,
        basis_terms,
        svd_threshold=svd_threshold,
        threshold_mode=threshold_mode,
    )
    previous_radius_fixed_basis_complexity = derivative_complexity_for_fit(
        previous_radius_fixed_basis_fit,
        candidate_grid,
    )
    target_point_count, point_count_diagnostics = second_pass_target_point_count(
        len(point_indices),
        first_pass_complexity,
        previous_radius_fixed_basis_complexity,
        first_pass_term_count,
        first_pass_fit["rank"],
        current_radius=current_radius,
        gamma=gamma,
        max_available_points=len(train_pool_indices),
        min_extra_points=min_extra_points,
    )
    second_point_indices, acquisition_detail = acquisition_extend_point_indices(
        point_indices,
        target_point_count,
        train_pool_indices,
        current_radius_data,
        fit_result=first_pass_fit,
        candidate_grid=candidate_grid,
        alpha=acquisition_alpha,
        beta=acquisition_beta,
        required_point_indices=required_point_indices,
        min_peak_separation_degrees=min_peak_separation_degrees,
    )
    second_pass_fixed_basis_fit = fixed_basis_score_for_radius_indices(
        current_radius_data,
        second_point_indices,
        test_indices,
        basis_terms,
        svd_threshold=svd_threshold,
        threshold_mode=threshold_mode,
    )
    second_pass_fit = fit_radius_with_hysteresis_selector(
        current_radius_data,
        second_point_indices,
        test_indices,
        previous_fit_results,
        relative_rmse_cutoff=relative_rmse_cutoff,
        svd_threshold=svd_threshold,
        threshold_mode=threshold_mode,
    )
    recomputed_basis_source = second_pass_fit.get("basis_source", "raw")
    second_pass_fit.update({
        "basis_source": f"second_pass_recomputed_{recomputed_basis_source}",
        "recomputed_basis_source": recomputed_basis_source,
        "first_pass_basis_source": first_pass_fit.get("basis_source", "raw"),
        "first_pass_fit": first_pass_fit,
        "second_pass_fixed_first_pass_basis_fit": second_pass_fixed_basis_fit,
        "second_pass_triggered": True,
        "second_pass_target_point_count": int(target_point_count),
        "second_pass_added_point_count": int(len(second_point_indices) - len(point_indices)),
        "second_pass_recomputed_basis": True,
    })
    second_pass_complexity = derivative_complexity_for_fit(second_pass_fit, candidate_grid)
    return {
        "fit": second_pass_fit,
        "complexity": second_pass_complexity,
        "point_indices": second_point_indices,
        "target_point_count": int(target_point_count),
        "trigger_diagnostics": trigger,
        "point_count_diagnostics": point_count_diagnostics,
        "previous_radius_fixed_basis_fit": previous_radius_fixed_basis_fit,
        "previous_radius_fixed_basis_complexity": previous_radius_fixed_basis_complexity,
        "second_pass_fixed_first_pass_basis_fit": second_pass_fixed_basis_fit,
        "acquisition_detail": acquisition_detail,
        "performed": True,
    }


def run_radius_to_radius_hysteresis_second_pass_experiment(
    all_data,
    gamma=RADIUS_TO_RADIUS_GAMMA,
    train_fraction=DEFAULT_TRAIN_FRACTION,
    random_seed=DEFAULT_RANDOM_SEED,
    candidate_grid=None,
    initial_fit_point_count=RADIUS_TO_RADIUS_INITIAL_FIT_POINT_COUNT,
    initial_test_point_count=RADIUS_TO_RADIUS_INITIAL_TEST_POINT_COUNT,
    required_angle_points_degrees=ACQUISITION_REQUIRED_ANGLE_POINTS_DEGREES,
    acquisition_alpha=ACQUISITION_ALPHA,
    acquisition_beta=ACQUISITION_BETA,
    min_peak_separation_degrees=ACQUISITION_MIN_PEAK_SEPARATION_DEGREES,
    relative_rmse_cutoff=DEFAULT_RELATIVE_RMSE_CUTOFF,
    svd_threshold=DEFAULT_SVD_THRESHOLD,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
    growth_fraction_threshold=SECOND_PASS_TERM_GROWTH_FRACTION,
    growth_min_terms=SECOND_PASS_TERM_GROWTH_MIN_INCREASE,
    max_condition_number=SECOND_PASS_MAX_CONDITION_NUMBER,
    min_extra_points=SECOND_PASS_MIN_EXTRA_POINTS,
):
    radii, radius_data_by_r = grouped_radius_data(all_data)
    if not validate_common_angular_grid(radius_data_by_r):
        raise ValueError("All radii must share the same angular grid to reuse point indices across R.")
    if candidate_grid is None:
        candidate_grid = make_angular_candidate_grid()

    first_radius_data = radius_data_by_r[float(radii[0])]
    required_point_indices = nearest_angle_indices(first_radius_data, required_angle_points_degrees)
    train_pool_indices, test_indices = split_train_pool_with_required_points(
        first_radius_data,
        required_point_indices,
        initial_test_point_count=initial_test_point_count,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )

    results_by_radius = {}
    transitions = []
    previous_complexity = None
    previous_point_indices = None
    previous_point_count = None
    previous_term_count = None
    previous_fit_result = None
    previous_fit_results = []

    for radius_index, r_value in enumerate(radii):
        r_value = float(r_value)
        radius_data = radius_data_by_r[r_value]

        if radius_index == 0:
            target_point_count = min(int(initial_fit_point_count), train_pool_indices.size)
            point_indices, acquisition_detail = acquisition_extend_point_indices(
                [],
                target_point_count,
                train_pool_indices,
                radius_data,
                fit_result=None,
                candidate_grid=None,
                required_point_indices=required_point_indices,
            )
            predicted_point_count = int(len(point_indices))
            derivative_point_count = predicted_point_count
            prediction_term_count = None
            basis_point_count = None
            point_count_prediction_diagnostics = {
                "curvature_ratio": 1.0,
                "radial_damping": 1.0,
                "effective_curvature_exponent": 0.0,
                "curvature_growth_factor": 1.0,
                "basis_estimate": 0,
                "derivative_estimate": float(len(point_indices)),
            }
        else:
            prediction_term_count = int(previous_term_count)
            predicted_point_count, point_count_prediction_diagnostics = predict_next_point_count(
                previous_point_count,
                current_complexity=results_by_radius[float(radii[radius_index - 1])]["complexity"],
                previous_complexity=previous_complexity,
                next_term_count=prediction_term_count,
                gamma=gamma,
                angular_dimension=RADIUS_TO_RADIUS_ANGULAR_DIMENSION,
                max_available_points=train_pool_indices.size,
                current_radius=float(radii[radius_index - 1]),
                return_diagnostics=True,
            )
            basis_point_count = basis_sampling_requirement(prediction_term_count, gamma=gamma)
            derivative_point_count = predicted_point_count
            point_indices, acquisition_detail = acquisition_extend_point_indices(
                previous_point_indices,
                predicted_point_count,
                train_pool_indices,
                radius_data,
                fit_result=previous_fit_result,
                candidate_grid=candidate_grid,
                alpha=acquisition_alpha,
                beta=acquisition_beta,
                required_point_indices=required_point_indices,
                min_peak_separation_degrees=min_peak_separation_degrees,
            )
            predicted_point_count = max(int(predicted_point_count), int(len(point_indices)))

        first_pass_fit = fit_radius_with_hysteresis_selector(
            radius_data,
            point_indices,
            test_indices,
            previous_fit_results,
            relative_rmse_cutoff=relative_rmse_cutoff,
            svd_threshold=svd_threshold,
            threshold_mode=threshold_mode,
        )
        first_pass_term_count = int(first_pass_fit["term_count"])
        first_pass_complexity = derivative_complexity_for_fit(first_pass_fit, candidate_grid)
        first_pass_point_indices = numpy.asarray(point_indices, dtype=int)
        first_pass_acquisition_detail = acquisition_detail

        second_pass = None
        if radius_index > 0:
            previous_radius_data = radius_data_by_r[float(radii[radius_index - 1])]
            second_pass = apply_second_point_pass_if_needed(
                previous_radius_data,
                radius_data,
                first_pass_point_indices,
                train_pool_indices,
                test_indices,
                first_pass_fit,
                first_pass_complexity,
                previous_fit_results,
                previous_term_count,
                current_radius=r_value,
                candidate_grid=candidate_grid,
                gamma=gamma,
                acquisition_alpha=acquisition_alpha,
                acquisition_beta=acquisition_beta,
                required_point_indices=required_point_indices,
                min_peak_separation_degrees=min_peak_separation_degrees,
                relative_rmse_cutoff=relative_rmse_cutoff,
                svd_threshold=svd_threshold,
                threshold_mode=threshold_mode,
                growth_fraction_threshold=growth_fraction_threshold,
                growth_min_terms=growth_min_terms,
                max_condition_number=max_condition_number,
                min_extra_points=min_extra_points,
            )
            fit_result = second_pass["fit"]
            complexity = second_pass["complexity"]
            point_indices = second_pass["point_indices"]
        else:
            fit_result = first_pass_fit
            complexity = first_pass_complexity
            point_indices = first_pass_point_indices
            second_pass = {
                "performed": False,
                "target_point_count": int(len(point_indices)),
                "trigger_diagnostics": {
                    "term_increase": 0,
                    "term_relative_growth": 0.0,
                    "term_growth_trigger": False,
                    "condition_trigger": False,
                    "triggered": False,
                },
                "point_count_diagnostics": None,
                "acquisition_detail": {
                    "added_indices": numpy.asarray([], dtype=int),
                    "peak_angles": numpy.empty((0, 3), dtype=float),
                    "acquisition": None,
                },
            }

        term_count = int(fit_result["term_count"])
        capped_basis_point_count = int(min(basis_sampling_requirement(term_count, gamma=gamma), train_pool_indices.size))

        results_by_radius[r_value] = {
            "R": r_value,
            "fit": fit_result,
            "first_pass_fit": first_pass_fit,
            "complexity": complexity,
            "first_pass_complexity": first_pass_complexity,
            "term_count": term_count,
            "first_pass_term_count": first_pass_term_count,
            "point_count": int(len(point_indices)),
            "first_pass_point_count": int(len(first_pass_point_indices)),
            "point_indices": point_indices,
            "first_pass_point_indices": first_pass_point_indices,
            "test_indices": test_indices,
            "basis_point_count": capped_basis_point_count,
            "prediction_term_count": None if prediction_term_count is None else int(prediction_term_count),
            "predicted_point_count": int(predicted_point_count),
            "derivative_point_count": int(min(numpy.ceil(derivative_point_count), train_pool_indices.size)),
            "passed_basis_cutoff": bool(fit_result.get("passed_basis_cutoff", False)),
            "basis_source": fit_result.get("basis_source", "raw"),
            "hysteresis_kept_anchor": bool(fit_result.get("hysteresis_kept_anchor", False)),
            "hysteresis_anchor_relation": fit_result.get("hysteresis_anchor_relation", "none"),
            "hysteresis_anchor_source": fit_result.get("hysteresis_anchor_source", "none"),
            "hysteresis_reason": fit_result.get("hysteresis_reason", ""),
            "acquisition_detail": acquisition_detail,
            "second_pass": second_pass,
            "point_count_prediction_diagnostics": point_count_prediction_diagnostics,
        }

        if radius_index > 0:
            previous_r = float(radii[radius_index - 1])
            reused_point_count = int(numpy.intersect1d(previous_point_indices, point_indices).size)
            second_added_indices = numpy.asarray(second_pass["acquisition_detail"].get("added_indices", numpy.asarray([], dtype=int)), dtype=int)
            first_added_indices = numpy.asarray(first_pass_acquisition_detail.get("added_indices", numpy.asarray([], dtype=int)), dtype=int)
            combined_added_indices = numpy.asarray(sorted(set(int(i) for i in numpy.concatenate([first_added_indices, second_added_indices]))), dtype=int)
            transitions.append({
                "from_R": previous_r,
                "to_R": r_value,
                "predicted_point_count": int(predicted_point_count),
                "used_point_count": int(len(point_indices)),
                "first_pass_point_count": int(len(first_pass_point_indices)),
                "second_pass_target_point_count": int(second_pass["target_point_count"]),
                "reused_point_count": reused_point_count,
                "added_point_count": int(len(point_indices) - reused_point_count),
                "first_pass_added_point_count": int(len(first_added_indices)),
                "second_pass_added_point_count": int(len(second_added_indices)),
                "prediction_term_count": int(prediction_term_count),
                "term_count": term_count,
                "first_pass_term_count": first_pass_term_count,
                "max_l1": int(fit_result["max_l1"]),
                "max_l2": int(fit_result["max_l2"]),
                "basis_source": fit_result.get("basis_source", "raw"),
                "hysteresis_kept_anchor": bool(fit_result.get("hysteresis_kept_anchor", False)),
                "hysteresis_anchor_relation": fit_result.get("hysteresis_anchor_relation", "none"),
                "hysteresis_anchor_source": fit_result.get("hysteresis_anchor_source", "none"),
                "hysteresis_reason": fit_result.get("hysteresis_reason", ""),
                "basis_point_count": capped_basis_point_count,
                "passed_basis_cutoff": bool(fit_result.get("passed_basis_cutoff", False)),
                "test_rmse": fit_result.get("test_rmse", numpy.nan),
                "test_relative_rmse": fit_result.get("test_relative_rmse", numpy.nan),
                "test_average_relative_absolute_error": fit_result.get("test_average_relative_absolute_error", numpy.nan),
                "retained_condition_number": fit_result.get("retained_condition_number", numpy.nan),
                "first_pass_test_relative_rmse": first_pass_fit.get("test_relative_rmse", numpy.nan),
                "first_pass_retained_condition_number": first_pass_fit.get("retained_condition_number", numpy.nan),
                "first_pass_retained_rank": int(first_pass_fit.get("rank", 0)),
                "recomputed_basis_source": fit_result.get("recomputed_basis_source", "none"),
                "second_pass_recomputed_basis": bool(fit_result.get("second_pass_recomputed_basis", False)),
                "normalized_curvature_scale": complexity["normalized_curvature_scale"],
                "first_pass_normalized_curvature_scale": first_pass_complexity["normalized_curvature_scale"],
                "second_pass_performed": bool(second_pass["performed"]),
                "second_pass_trigger": second_pass["trigger_diagnostics"],
                "second_pass_point_count_diagnostics": second_pass["point_count_diagnostics"],
                "new_point_indices": combined_added_indices,
                "first_pass_new_point_indices": first_added_indices,
                "second_pass_new_point_indices": second_added_indices,
                "peak_angles": first_pass_acquisition_detail.get("peak_angles", numpy.empty((0, 3), dtype=float)),
                "second_pass_peak_angles": second_pass["acquisition_detail"].get("peak_angles", numpy.empty((0, 3), dtype=float)),
                **point_count_prediction_diagnostics,
            })

        previous_complexity = None if radius_index == 0 else results_by_radius[float(radii[radius_index - 1])]["complexity"]
        previous_point_indices = point_indices
        previous_point_count = int(len(point_indices))
        previous_term_count = term_count
        previous_fit_result = fit_result
        previous_fit_results.append(fit_result)

    return {
        "basis_selection_method": "sweep_l1_l2_with_hysteresis_second_point_pass",
        "radii_descending": radii,
        "train_pool_indices": train_pool_indices,
        "test_indices": test_indices,
        "required_point_indices": required_point_indices,
        "required_angle_points_degrees": numpy.asarray(required_angle_points_degrees, dtype=float),
        "candidate_grid": candidate_grid,
        "results_by_radius": results_by_radius,
        "transitions": transitions,
        "gamma": gamma,
        "second_pass_term_growth_fraction": growth_fraction_threshold,
        "second_pass_term_growth_min_increase": growth_min_terms,
        "second_pass_max_condition_number": max_condition_number,
        "second_pass_min_extra_points": min_extra_points,
        "hysteresis_memory_radius_count": HYSTERESIS_MEMORY_RADIUS_COUNT,
        "relative_rmse_target": HYSTERESIS_RELATIVE_RMSE_TARGET,
        "condition_limit": HYSTERESIS_CONDITION_LIMIT,
        "acquisition_alpha": acquisition_alpha,
        "acquisition_beta": acquisition_beta,
        "min_peak_separation_degrees": min_peak_separation_degrees,
        "initial_fit_point_count": int(initial_fit_point_count) if initial_fit_point_count is not None else None,
        "initial_test_point_count": int(len(test_indices)),
        "random_seed": random_seed,
    }


def second_pass_transition_audit(result):
    all_data = result.get("all_data_for_audit")
    audit_rows = []
    for transition in result["transitions"]:
        row = {
            "from_R": transition["from_R"],
            "to_R": transition["to_R"],
            "N_pred": transition["predicted_point_count"],
            "N_first": transition["first_pass_point_count"],
            "N_used": transition["used_point_count"],
            "N_added_first": transition["first_pass_added_point_count"],
            "N_added_second": transition["second_pass_added_point_count"],
            "terms_for_N": transition["prediction_term_count"],
            "terms": transition["term_count"],
            "second_pass": transition["second_pass_performed"],
            "second_pass_trigger": transition["second_pass_trigger"],
            "test_relative_rmse": transition["test_relative_rmse"],
            "condition": transition["retained_condition_number"],
            "first_pass_retained_rank": transition["first_pass_retained_rank"],
            "recomputed_basis_source": transition["recomputed_basis_source"],
            "second_pass_recomputed_basis": transition["second_pass_recomputed_basis"],
        }
        if all_data is not None and "acquisition_demanded_points_for_transition" in globals():
            first_transition = dict(transition)
            first_transition["new_point_indices"] = transition["first_pass_new_point_indices"]
            first_transition["peak_angles"] = transition["peak_angles"]
            row["first_pass_demanded_points"] = acquisition_demanded_points_for_transition(first_transition, all_data)
            second_transition = dict(transition)
            second_transition["new_point_indices"] = transition["second_pass_new_point_indices"]
            second_transition["peak_angles"] = transition["second_pass_peak_angles"]
            row["second_pass_demanded_points"] = acquisition_demanded_points_for_transition(second_transition, all_data)
        else:
            row["first_pass_new_point_indices"] = transition["first_pass_new_point_indices"]
            row["second_pass_new_point_indices"] = transition["second_pass_new_point_indices"]
        audit_rows.append(row)
    return audit_rows


# --- YUMI coefficient export helpers ---

# Generate and save coefficients

YUMI_COEFFICIENT_OUTPUT_PATH = Path("CO2-H2_minimized_coefficients_yumi_format.txt")
YUMI_COEFFICIENT_TITLE = "CO2 H2 PES fitted from minimizing_ab_initio_points.ipynb"
YUMI_MISSING_COEFFICIENT_VALUE = 0.0
YUMI_APPLY_COEFFICIENT_NORMALIZATION = True


def yumi_coefficient_normalization(L):
    return 2.0 * math.pi / math.sqrt(2 * int(L) + 1)


def yumi_term_record_from_basis_term(term):
    L, l1, l2 = (int(value) for value in term)
    return (l1, 0, l2, L)


def yumi_basis_term_from_record(record):
    l1, _, l2, L = (int(value) for value in record)
    return (L, l1, l2)


def is_yumi_allowed_basis_term(term):
    L, l1, l2 = (int(value) for value in term)
    return (l1 + l2 + L) % 2 == 0


def collect_yumi_coefficient_table(
    point_count_result,
    missing_value=YUMI_MISSING_COEFFICIENT_VALUE,
    apply_normalization=YUMI_APPLY_COEFFICIENT_NORMALIZATION,
):
    results_by_radius = point_count_result["results_by_radius"]
    radii = numpy.asarray(sorted(float(radius) for radius in results_by_radius), dtype=float)

    coefficient_by_radius_and_term = {}
    yumi_records = set()
    for radius in radii:
        fit = results_by_radius[float(radius)]["fit"]
        basis_terms = [tuple(int(value) for value in term) for term in fit["basis_terms"]]
        coefficients = numpy.asarray(fit["coefficients"], dtype=float)
        if len(basis_terms) != coefficients.size:
            raise ValueError(
                f"R={radius:g} has {len(basis_terms)} basis terms but {coefficients.size} coefficients."
            )
        term_coefficients = {}
        for term, coefficient in zip(basis_terms, coefficients):
            if not is_yumi_allowed_basis_term(term):
                continue
            record = yumi_term_record_from_basis_term(term)
            yumi_records.add(record)
            export_coefficient = float(coefficient)
            if apply_normalization:
                export_coefficient *= yumi_coefficient_normalization(term[0])
            term_coefficients[record] = export_coefficient
        coefficient_by_radius_and_term[float(radius)] = term_coefficients

    sorted_records = sorted(yumi_records, key=lambda record: (record[0], record[2], record[3]))
    coefficient_rows = []
    for record in sorted_records:
        coefficient_rows.append([
            coefficient_by_radius_and_term[float(radius)].get(record, float(missing_value))
            for radius in radii
        ])

    return {
        "radii": radii,
        "records": sorted_records,
        "coefficients": numpy.asarray(coefficient_rows, dtype=float),
        "normalization": "2*pi/sqrt(2*L+1)" if apply_normalization else None,
    }


def format_yumi_float(value):
    return f"{float(value):.11E}"


def format_yumi_coefficients(table, title=YUMI_COEFFICIENT_TITLE):
    radii = numpy.asarray(table["radii"], dtype=float)
    records = list(table["records"])
    coefficients = numpy.asarray(table["coefficients"], dtype=float)

    if coefficients.shape != (len(records), len(radii)):
        raise ValueError("Coefficient table shape does not match the number of records and radii.")

    lines = [
        str(title),
        f"{len(records) - 1:4d}{len(radii) - 1:4d}{float(radii[0]):8.4f}{0.0:8.4f}",
        " ".join(f"{radius:.4f}" for radius in radii),
    ]
    for record, coefficient_row in zip(records, coefficients):
        l1, m, l2, L = record
        lines.append(f"{l1:4d}{m:4d}{l2:4d}{L:4d}")
        lines.append(" ".join(format_yumi_float(value) for value in coefficient_row))
    return "\n".join(lines) + "\n"


def save_yumi_coefficients(
    point_count_result,
    output_path=YUMI_COEFFICIENT_OUTPUT_PATH,
    title=YUMI_COEFFICIENT_TITLE,
    apply_normalization=YUMI_APPLY_COEFFICIENT_NORMALIZATION,
):
    table = collect_yumi_coefficient_table(point_count_result, apply_normalization=apply_normalization)
    output_path = Path(output_path)
    output_path.write_text(format_yumi_coefficients(table, title=title))
    return {
        "output_path": output_path,
        "radii_count": int(len(table["radii"])),
        "term_count": int(len(table["records"])),
        "radii": table["radii"],
        "records": table["records"],
        "normalization": table["normalization"],
    }


# --- Miscellaneous notebook helpers ---

def isotropic_coefficient_series_from_result(point_count_result, term=(0, 0, 0)):
    results_by_radius = point_count_result["results_by_radius"]
    radii = numpy.asarray(sorted(float(radius) for radius in results_by_radius), dtype=float)
    coefficients = []
    for radius in radii:
        fit = results_by_radius[float(radius)]["fit"]
        basis_terms = [tuple(int(value) for value in basis_term) for basis_term in fit["basis_terms"]]
        if term not in basis_terms:
            coefficients.append(numpy.nan)
            continue
        term_index = basis_terms.index(term)
        coefficients.append(float(numpy.asarray(fit["coefficients"], dtype=float)[term_index]))
    return radii, numpy.asarray(coefficients, dtype=float)


# --- CO2_H2 fitting notebook radial/model helpers ---

def select_minimal_basis_for_shell(
    r_data,
    max_l1_grid,
    max_l2_grid,
    relative_rmse_cutoff=0.01,
    min_l1=8,
    min_l2=4,
    terms_cap=None,
    train_fraction=0.80,
    random_seed=202605702,
    refit_on_all_data=True,
):
    """Choose the first smallest-term basis with testing relative RMSE below cutoff."""
    r_value = float(r_data[0, 0])
    r_training_data, r_testing_data, r_training_indices, r_testing_indices = create_train_test_split(
        r_data,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )

    candidates = basis_cutoff_candidates(
        max_l1_grid,
        max_l2_grid,
        min_l1=min_l1,
        min_l2=min_l2,
        terms_cap=terms_cap,
    )
    if not candidates:
        raise ValueError("No basis candidates satisfy the requested l1/l2 minima.")

    tried_results = []
    best_result = None
    chosen_result = None
    passed_cutoff = False

    for candidate in candidates:
        score = fit_and_score(r_training_data, r_testing_data, candidate["basis_terms"])
        score.update(candidate)
        tried_results.append(score)

        if best_result is None or score["test_relative_rmse"] < best_result["test_relative_rmse"]:
            best_result = score

        if numpy.isfinite(score["test_relative_rmse"]) and score["test_relative_rmse"] <= relative_rmse_cutoff:
            chosen_result = score
            passed_cutoff = True
            break

    if chosen_result is None:
        chosen_result = best_result

    final_fit_data = r_data if refit_on_all_data else r_training_data
    final_fit = fit_basis(final_fit_data, chosen_result["basis_terms"])
    coefficient_by_term = dict(zip(chosen_result["basis_terms"], final_fit["coefficients"]))

    return {
        "R": r_value,
        "basis_terms": chosen_result["basis_terms"],
        "coefficient_by_term": coefficient_by_term,
        "coefficients": final_fit["coefficients"],
        "final_fit": final_fit,
        "selection_result": chosen_result,
        "tried_results": tried_results,
        "passed_cutoff": passed_cutoff,
        "max_l1": chosen_result["max_l1"],
        "max_l2": chosen_result["max_l2"],
        "term_count": chosen_result["term_count"],
        "training_rank": chosen_result["rank"],
        "test_rmse": chosen_result["test_rmse"],
        "test_relative_rmse": chosen_result["test_relative_rmse"],
        "train_rmse": chosen_result["train_rmse"],
        "train_relative_rmse": chosen_result["train_relative_rmse"],
        "training_indices": r_training_indices,
        "testing_indices": r_testing_indices,
        "refit_on_all_data": refit_on_all_data,
    }

def select_minimal_bases_by_radius(
    all_data,
    max_l1_grid,
    max_l2_grid,
    relative_rmse_cutoff=0.01,
    min_l1=8,
    min_l2=4,
    terms_cap=None,
    train_fraction=0.80,
    random_seed=202605702,
    refit_on_all_data=True,
    verbose=True,
):
    shell_fits = {}

    for radius_index, r_value in enumerate(numpy.unique(all_data[:, 0])):
        r_data = all_data[all_data[:, 0] == r_value]
        shell_seed = None if random_seed is None else int(random_seed) + radius_index
        shell_fit = select_minimal_basis_for_shell(
            r_data,
            max_l1_grid,
            max_l2_grid,
            relative_rmse_cutoff=relative_rmse_cutoff,
            min_l1=min_l1,
            min_l2=min_l2,
            terms_cap=terms_cap,
            train_fraction=train_fraction,
            random_seed=shell_seed,
            refit_on_all_data=refit_on_all_data,
        )
        shell_fits[float(r_value)] = shell_fit

        if verbose:
            status = "passed" if shell_fit["passed_cutoff"] else "best available"
            print(
                f"R = {r_value:g}: {status}, "
                f"test relative RMSE = {shell_fit['test_relative_rmse']:.6g}, "
                f"terms = {shell_fit['term_count']}, "
                f"max_l1 = {shell_fit['max_l1']}, max_l2 = {shell_fit['max_l2']}"
            )

    return shell_fits

def _constant_coefficient_model(value):
    def coefficient_model(R):
        R_array = numpy.asarray(R, dtype=float)
        result = numpy.full_like(R_array, float(value), dtype=float)
        return float(result) if result.shape == () else result

    return coefficient_model

def build_radial_coefficient_models(shell_fits, spline_smoothing=0.0, max_spline_degree=3):
    radii = sorted(shell_fits)
    Q_union = sorted(
        {term for shell_fit in shell_fits.values() for term in shell_fit["basis_terms"]},
        key=lambda term: (sum(term), term[0], term[1], term[2]),
    )
    coefficient_models = {}

    for term in Q_union:
        active_radii = []
        active_coefficients = []

        for r_value in radii:
            coefficient = shell_fits[r_value]["coefficient_by_term"].get(term)
            if coefficient is not None:
                active_radii.append(float(r_value))
                active_coefficients.append(float(coefficient))

        active_radii = numpy.asarray(active_radii, dtype=float)
        active_coefficients = numpy.asarray(active_coefficients, dtype=float)

        if active_radii.size == 1:
            spline_degree = 0
            coefficient_model = _constant_coefficient_model(active_coefficients[0])
        else:
            # Default spline parameters are here: interpolating UnivariateSpline, s=0.0, ext=0.
            # The degree is lowered automatically when fewer than 4 active radii are available.
            spline_degree = min(int(max_spline_degree), active_radii.size - 1)
            coefficient_model = scipy.interpolate.UnivariateSpline(
                active_radii,
                active_coefficients,
                k=spline_degree,
                s=spline_smoothing,
                ext=0,
            )

        coefficient_models[term] = {
            "term": term,
            "radii": active_radii,
            "coefficients": active_coefficients,
            "coefficient_model": coefficient_model,
            "spline_degree": spline_degree,
            "R_cutoff": float(active_radii.max()),
        }

    return Q_union, coefficient_models

def radial_switch(R, R_cutoff, decay_a=1.0):
    R_array = numpy.asarray(R, dtype=float)
    switched = numpy.where(
        R_array <= R_cutoff,
        1.0,
        numpy.exp(-float(decay_a) * (R_array - R_cutoff) ** 2),
    )
    return float(switched) if switched.shape == () else switched

def evaluate_minimized_pes(model, R, theta1, theta2, phi, angles_in_degrees=True):
    """Evaluate V(R, Omega) for scalar R and scalar or array-valued angles."""
    if numpy.ndim(R) != 0:
        raise NotImplementedError("evaluate_minimized_pes currently expects scalar R.")

    theta1_values = numpy.asarray(theta1, dtype=float)
    theta2_values = numpy.asarray(theta2, dtype=float)
    phi_values = numpy.asarray(phi, dtype=float)
    if angles_in_degrees:
        theta1_values = numpy.deg2rad(theta1_values)
        theta2_values = numpy.deg2rad(theta2_values)
        phi_values = numpy.deg2rad(phi_values)

    value = numpy.zeros_like(theta1_values, dtype=float)
    for term in model["Q_union"]:
        L, l1, l2 = term
        coefficient_info = model["coefficient_models"][term]
        coefficient = float(coefficient_info["coefficient_model"](R))
        switch = radial_switch(R, coefficient_info["R_cutoff"], decay_a=model["decay_a"])
        value = value + switch * coefficient * angular_basis(L, l1, l2, theta1_values, theta2_values, phi_values)

    return float(value) if value.shape == () else value

def build_minimized_pes_model(
    all_data,
    max_l1_grid,
    max_l2_grid,
    relative_rmse_cutoff=0.01,
    min_l1=8,
    min_l2=4,
    terms_cap=None,
    train_fraction=0.80,
    random_seed=202605702,
    refit_on_all_data=True,
    spline_smoothing=0.0,
    max_spline_degree=3,
    decay_a=1.0,
    verbose=True,
):
    shell_fits = select_minimal_bases_by_radius(
        all_data,
        max_l1_grid,
        max_l2_grid,
        relative_rmse_cutoff=relative_rmse_cutoff,
        min_l1=min_l1,
        min_l2=min_l2,
        terms_cap=terms_cap,
        train_fraction=train_fraction,
        random_seed=random_seed,
        refit_on_all_data=refit_on_all_data,
        verbose=verbose,
    )
    Q_union, coefficient_models = build_radial_coefficient_models(
        shell_fits,
        spline_smoothing=spline_smoothing,
        max_spline_degree=max_spline_degree,
    )

    return {
        "shell_fits": shell_fits,
        "Q_union": Q_union,
        "coefficient_models": coefficient_models,
        "relative_rmse_cutoff": relative_rmse_cutoff,
        "min_l1": min_l1,
        "min_l2": min_l2,
        "terms_cap": terms_cap,
        "train_fraction": train_fraction,
        "random_seed": random_seed,
        "refit_on_all_data": refit_on_all_data,
        "spline_smoothing": spline_smoothing,
        "max_spline_degree": max_spline_degree,
        "decay_a": decay_a,
    }

def select_incremental_terms_for_shell_truncated_svd(
    r_data,
    universe_max_l1,
    universe_max_l2,
    relative_rmse_cutoff=0.01,
    svd_threshold=1e-8,
    threshold_mode="relative",
    min_terms=1,
    max_terms=None,
    train_fraction=0.80,
    random_seed=202605702,
    refit_on_all_data=True,
):
    r_value = float(r_data[0, 0])
    r_training_data, r_testing_data, r_training_indices, r_testing_indices = create_train_test_split(
        r_data,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )

    tried_results = []
    best_result = None
    chosen_result = None
    passed_cutoff = False

    for candidate in incremental_term_candidates(
        universe_max_l1,
        universe_max_l2,
        min_terms=min_terms,
        max_terms=max_terms,
    ):
        score = fit_and_score_truncated_svd(
            r_training_data,
            r_testing_data,
            candidate["basis_terms"],
            svd_threshold=svd_threshold,
            threshold_mode=threshold_mode,
        )
        score.update(candidate)
        tried_results.append(score)

        if best_result is None or score["test_relative_rmse"] < best_result["test_relative_rmse"]:
            best_result = score

        if numpy.isfinite(score["test_relative_rmse"]) and score["test_relative_rmse"] <= relative_rmse_cutoff:
            chosen_result = score
            passed_cutoff = True
            break

    if chosen_result is None:
        chosen_result = best_result

    final_fit_data = r_data if refit_on_all_data else r_training_data
    final_fit = fit_basis_truncated_svd(
        final_fit_data,
        chosen_result["basis_terms"],
        svd_threshold=svd_threshold,
        threshold_mode=threshold_mode,
    )
    coefficient_by_term = dict(zip(chosen_result["basis_terms"], final_fit["coefficients"]))

    return {
        "R": r_value,
        "basis_terms": chosen_result["basis_terms"],
        "coefficient_by_term": coefficient_by_term,
        "coefficients": final_fit["coefficients"],
        "final_fit": final_fit,
        "selection_result": chosen_result,
        "tried_results": tried_results,
        "passed_cutoff": passed_cutoff,
        "term_count": chosen_result["term_count"],
        "max_L": chosen_result["max_L"],
        "max_l1": chosen_result["max_l1"],
        "max_l2": chosen_result["max_l2"],
        "training_rank": chosen_result["rank"],
        "discarded_singular_count": chosen_result["discarded_singular_count"],
        "svd_cutoff": chosen_result["svd_cutoff"],
        "retained_condition_number": chosen_result["retained_condition_number"],
        "full_condition_number": chosen_result["full_condition_number"],
        "max_abs_coefficient": final_fit["max_abs_coefficient"],
        "coefficient_l2_norm": final_fit["coefficient_l2_norm"],
        "test_rmse": chosen_result["test_rmse"],
        "test_relative_rmse": chosen_result["test_relative_rmse"],
        "train_rmse": chosen_result["train_rmse"],
        "train_relative_rmse": chosen_result["train_relative_rmse"],
        "training_indices": r_training_indices,
        "testing_indices": r_testing_indices,
        "refit_on_all_data": refit_on_all_data,
    }

def select_incremental_terms_by_radius_truncated_svd(
    all_data,
    universe_max_l1,
    universe_max_l2,
    relative_rmse_cutoff=0.01,
    svd_threshold=1e-8,
    threshold_mode="relative",
    min_terms=1,
    max_terms=None,
    train_fraction=0.80,
    random_seed=202605702,
    refit_on_all_data=True,
    verbose=True,
):
    shell_fits = {}

    for radius_index, r_value in enumerate(numpy.unique(all_data[:, 0])):
        r_data = all_data[all_data[:, 0] == r_value]
        shell_seed = None if random_seed is None else int(random_seed) + radius_index
        shell_fit = select_incremental_terms_for_shell_truncated_svd(
            r_data,
            universe_max_l1,
            universe_max_l2,
            relative_rmse_cutoff=relative_rmse_cutoff,
            svd_threshold=svd_threshold,
            threshold_mode=threshold_mode,
            min_terms=min_terms,
            max_terms=max_terms,
            train_fraction=train_fraction,
            random_seed=shell_seed,
            refit_on_all_data=refit_on_all_data,
        )
        shell_fits[float(r_value)] = shell_fit

        if verbose:
            status = "passed" if shell_fit["passed_cutoff"] else "best available"
            print(
                f"R = {r_value:g}: {status}, "
                f"test relative RMSE = {shell_fit['test_relative_rmse']:.6g}, "
                f"terms = {shell_fit['term_count']}, "
                f"rank = {shell_fit['training_rank']}, "
                f"discarded singular values = {shell_fit['discarded_singular_count']}, "
                f"max abs coefficient = {shell_fit['max_abs_coefficient']:.6g}"
            )

    return shell_fits

def build_incremental_truncated_svd_pes_model(
    all_data,
    universe_max_l1,
    universe_max_l2,
    relative_rmse_cutoff=0.01,
    svd_threshold=1e-8,
    threshold_mode="relative",
    min_terms=1,
    max_terms=None,
    train_fraction=0.80,
    random_seed=202605702,
    refit_on_all_data=True,
    spline_smoothing=0.0,
    max_spline_degree=3,
    decay_a=1.0,
    verbose=True,
):
    shell_fits = select_incremental_terms_by_radius_truncated_svd(
        all_data,
        universe_max_l1,
        universe_max_l2,
        relative_rmse_cutoff=relative_rmse_cutoff,
        svd_threshold=svd_threshold,
        threshold_mode=threshold_mode,
        min_terms=min_terms,
        max_terms=max_terms,
        train_fraction=train_fraction,
        random_seed=random_seed,
        refit_on_all_data=refit_on_all_data,
        verbose=verbose,
    )
    Q_union, coefficient_models = build_radial_coefficient_models(
        shell_fits,
        spline_smoothing=spline_smoothing,
        max_spline_degree=max_spline_degree,
    )

    return {
        "shell_fits": shell_fits,
        "Q_union": Q_union,
        "coefficient_models": coefficient_models,
        "relative_rmse_cutoff": relative_rmse_cutoff,
        "svd_threshold": svd_threshold,
        "threshold_mode": threshold_mode,
        "universe_max_l1": universe_max_l1,
        "universe_max_l2": universe_max_l2,
        "min_terms": min_terms,
        "max_terms": max_terms,
        "train_fraction": train_fraction,
        "random_seed": random_seed,
        "refit_on_all_data": refit_on_all_data,
        "spline_smoothing": spline_smoothing,
        "max_spline_degree": max_spline_degree,
        "decay_a": decay_a,
    }

def select_minimal_basis_for_shell_truncated_svd(
    r_data,
    max_l1_grid,
    max_l2_grid,
    relative_rmse_cutoff=0.01,
    svd_threshold=1e-8,
    threshold_mode="relative",
    min_l1=8,
    min_l2=4,
    terms_cap=None,
    train_fraction=0.80,
    random_seed=202605702,
    refit_on_all_data=True,
):
    """Choose a max-l1/max-l2 basis using truncated-SVD least squares."""
    r_value = float(r_data[0, 0])
    r_training_data, r_testing_data, r_training_indices, r_testing_indices = create_train_test_split(
        r_data,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )

    candidates = basis_cutoff_candidates(
        max_l1_grid,
        max_l2_grid,
        min_l1=min_l1,
        min_l2=min_l2,
        terms_cap=terms_cap,
    )
    if not candidates:
        raise ValueError("No basis candidates satisfy the requested l1/l2 minima.")

    tried_results = []
    best_result = None
    chosen_result = None
    passed_cutoff = False

    for candidate in candidates:
        score = fit_and_score_truncated_svd(
            r_training_data,
            r_testing_data,
            candidate["basis_terms"],
            svd_threshold=svd_threshold,
            threshold_mode=threshold_mode,
        )
        score.update(candidate)
        tried_results.append(score)

        if best_result is None or score["test_relative_rmse"] < best_result["test_relative_rmse"]:
            best_result = score

        if numpy.isfinite(score["test_relative_rmse"]) and score["test_relative_rmse"] <= relative_rmse_cutoff:
            chosen_result = score
            passed_cutoff = True
            break

    if chosen_result is None:
        chosen_result = best_result

    final_fit_data = r_data if refit_on_all_data else r_training_data
    final_fit = fit_basis_truncated_svd(
        final_fit_data,
        chosen_result["basis_terms"],
        svd_threshold=svd_threshold,
        threshold_mode=threshold_mode,
    )
    coefficient_by_term = dict(zip(chosen_result["basis_terms"], final_fit["coefficients"]))

    return {
        "R": r_value,
        "basis_terms": chosen_result["basis_terms"],
        "coefficient_by_term": coefficient_by_term,
        "coefficients": final_fit["coefficients"],
        "final_fit": final_fit,
        "selection_result": chosen_result,
        "tried_results": tried_results,
        "passed_cutoff": passed_cutoff,
        "max_l1": chosen_result["max_l1"],
        "max_l2": chosen_result["max_l2"],
        "term_count": chosen_result["term_count"],
        "training_rank": chosen_result["rank"],
        "discarded_singular_count": chosen_result["discarded_singular_count"],
        "svd_cutoff": chosen_result["svd_cutoff"],
        "retained_condition_number": chosen_result["retained_condition_number"],
        "full_condition_number": chosen_result["full_condition_number"],
        "max_abs_coefficient": final_fit["max_abs_coefficient"],
        "coefficient_l2_norm": final_fit["coefficient_l2_norm"],
        "test_rmse": chosen_result["test_rmse"],
        "test_relative_rmse": chosen_result["test_relative_rmse"],
        "train_rmse": chosen_result["train_rmse"],
        "train_relative_rmse": chosen_result["train_relative_rmse"],
        "training_indices": r_training_indices,
        "testing_indices": r_testing_indices,
        "refit_on_all_data": refit_on_all_data,
    }

def select_minimal_bases_by_radius_truncated_svd(
    all_data,
    max_l1_grid,
    max_l2_grid,
    relative_rmse_cutoff=0.01,
    svd_threshold=1e-8,
    threshold_mode="relative",
    min_l1=8,
    min_l2=4,
    terms_cap=None,
    train_fraction=0.80,
    random_seed=202605702,
    refit_on_all_data=True,
    verbose=True,
):
    shell_fits = {}

    for radius_index, r_value in enumerate(numpy.unique(all_data[:, 0])):
        r_data = all_data[all_data[:, 0] == r_value]
        shell_seed = None if random_seed is None else int(random_seed) + radius_index
        shell_fit = select_minimal_basis_for_shell_truncated_svd(
            r_data,
            max_l1_grid,
            max_l2_grid,
            relative_rmse_cutoff=relative_rmse_cutoff,
            svd_threshold=svd_threshold,
            threshold_mode=threshold_mode,
            min_l1=min_l1,
            min_l2=min_l2,
            terms_cap=terms_cap,
            train_fraction=train_fraction,
            random_seed=shell_seed,
            refit_on_all_data=refit_on_all_data,
        )
        shell_fits[float(r_value)] = shell_fit

        if verbose:
            status = "passed" if shell_fit["passed_cutoff"] else "best available"
            print(
                f"R = {r_value:g}: {status}, "
                f"test relative RMSE = {shell_fit['test_relative_rmse']:.6g}, "
                f"terms = {shell_fit['term_count']}, "
                f"max_l1 = {shell_fit['max_l1']}, max_l2 = {shell_fit['max_l2']}, "
                f"rank = {shell_fit['training_rank']}, "
                f"discarded singular values = {shell_fit['discarded_singular_count']}, "
                f"max abs coefficient = {shell_fit['max_abs_coefficient']:.6g}"
            )

    return shell_fits

def build_minimized_truncated_svd_pes_model(
    all_data,
    max_l1_grid,
    max_l2_grid,
    relative_rmse_cutoff=0.01,
    svd_threshold=1e-8,
    threshold_mode="relative",
    min_l1=8,
    min_l2=4,
    terms_cap=None,
    train_fraction=0.80,
    random_seed=202605702,
    refit_on_all_data=True,
    spline_smoothing=0.0,
    max_spline_degree=3,
    decay_a=1.0,
    verbose=True,
):
    shell_fits = select_minimal_bases_by_radius_truncated_svd(
        all_data,
        max_l1_grid,
        max_l2_grid,
        relative_rmse_cutoff=relative_rmse_cutoff,
        svd_threshold=svd_threshold,
        threshold_mode=threshold_mode,
        min_l1=min_l1,
        min_l2=min_l2,
        terms_cap=terms_cap,
        train_fraction=train_fraction,
        random_seed=random_seed,
        refit_on_all_data=refit_on_all_data,
        verbose=verbose,
    )
    Q_union, coefficient_models = build_radial_coefficient_models(
        shell_fits,
        spline_smoothing=spline_smoothing,
        max_spline_degree=max_spline_degree,
    )

    return {
        "shell_fits": shell_fits,
        "Q_union": Q_union,
        "coefficient_models": coefficient_models,
        "relative_rmse_cutoff": relative_rmse_cutoff,
        "svd_threshold": svd_threshold,
        "threshold_mode": threshold_mode,
        "min_l1": min_l1,
        "min_l2": min_l2,
        "terms_cap": terms_cap,
        "train_fraction": train_fraction,
        "random_seed": random_seed,
        "refit_on_all_data": refit_on_all_data,
        "spline_smoothing": spline_smoothing,
        "max_spline_degree": max_spline_degree,
        "decay_a": decay_a,
    }

def hysteresis_anchor_from_previous_shells(previous_shell_fits, required_previous_count=3):
    """Return the oldest/largest cutoff when the previous selected cutoffs are nonincreasing."""
    if len(previous_shell_fits) < required_previous_count:
        return None

    previous_shells = list(previous_shell_fits[-required_previous_count:])
    max_l1_values = [int(shell_fit["max_l1"]) for shell_fit in previous_shells]
    max_l2_values = [int(shell_fit["max_l2"]) for shell_fit in previous_shells]

    max_l1_nonincreasing = all(
        max_l1_values[index] >= max_l1_values[index + 1]
        for index in range(len(max_l1_values) - 1)
    )
    max_l2_nonincreasing = all(
        max_l2_values[index] >= max_l2_values[index + 1]
        for index in range(len(max_l2_values) - 1)
    )
    if not (max_l1_nonincreasing and max_l2_nonincreasing):
        return None

    anchor_shell = previous_shells[0]
    return {
        "max_l1": int(anchor_shell["max_l1"]),
        "max_l2": int(anchor_shell["max_l2"]),
        "source_radii": [float(shell_fit["R"]) for shell_fit in previous_shells],
        "source_max_l1_values": max_l1_values,
        "source_max_l2_values": max_l2_values,
    }

def candidate_receives_hysteresis_tolerance(candidate, hysteresis_anchor):
    if hysteresis_anchor is None:
        return False
    return (
        int(candidate["max_l1"]) >= int(hysteresis_anchor["max_l1"])
        and int(candidate["max_l2"]) >= int(hysteresis_anchor["max_l2"])
    )

def select_minimal_basis_for_shell_truncated_svd_with_hysteresis(
    r_data,
    max_l1_grid,
    max_l2_grid,
    relative_rmse_cutoff=0.01,
    svd_threshold=1e-8,
    threshold_mode="relative",
    min_l1=8,
    min_l2=4,
    terms_cap=None,
    train_fraction=0.80,
    random_seed=202605702,
    refit_on_all_data=True,
    hysteresis_anchor=None,
    hysteresis_relative_tolerance=0.30,
):
    """Choose a max-l1/max-l2 basis, with optional hysteresis on the RMSE cutoff."""
    r_value = float(r_data[0, 0])
    r_training_data, r_testing_data, r_training_indices, r_testing_indices = create_train_test_split(
        r_data,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )

    candidates = basis_cutoff_candidates(
        max_l1_grid,
        max_l2_grid,
        min_l1=min_l1,
        min_l2=min_l2,
        terms_cap=terms_cap,
    )
    if not candidates:
        raise ValueError("No basis candidates satisfy the requested l1/l2 minima.")

    tried_results = []
    best_result = None
    chosen_result = None
    passed_cutoff = False
    passed_base_cutoff = False

    for candidate in candidates:
        score = fit_and_score_truncated_svd(
            r_training_data,
            r_testing_data,
            candidate["basis_terms"],
            svd_threshold=svd_threshold,
            threshold_mode=threshold_mode,
        )
        score.update(candidate)

        hysteresis_eligible = candidate_receives_hysteresis_tolerance(candidate, hysteresis_anchor)
        active_relative_rmse_cutoff = float(relative_rmse_cutoff)
        if hysteresis_eligible:
            active_relative_rmse_cutoff *= 1.0 + float(hysteresis_relative_tolerance)

        score["hysteresis_eligible"] = hysteresis_eligible
        score["active_relative_rmse_cutoff"] = active_relative_rmse_cutoff
        score["passes_base_cutoff"] = bool(
            numpy.isfinite(score["test_relative_rmse"])
            and score["test_relative_rmse"] <= relative_rmse_cutoff
        )
        score["passes_active_cutoff"] = bool(
            numpy.isfinite(score["test_relative_rmse"])
            and score["test_relative_rmse"] <= active_relative_rmse_cutoff
        )
        tried_results.append(score)

        if best_result is None or score["test_relative_rmse"] < best_result["test_relative_rmse"]:
            best_result = score

        if score["passes_active_cutoff"]:
            chosen_result = score
            passed_cutoff = True
            passed_base_cutoff = score["passes_base_cutoff"]
            break

    if chosen_result is None:
        chosen_result = best_result

    used_hysteresis = bool(
        chosen_result.get("hysteresis_eligible", False)
        and chosen_result.get("passes_active_cutoff", False)
        and not chosen_result.get("passes_base_cutoff", False)
    )

    final_fit_data = r_data if refit_on_all_data else r_training_data
    final_fit = fit_basis_truncated_svd(
        final_fit_data,
        chosen_result["basis_terms"],
        svd_threshold=svd_threshold,
        threshold_mode=threshold_mode,
    )
    coefficient_by_term = dict(zip(chosen_result["basis_terms"], final_fit["coefficients"]))

    return {
        "R": r_value,
        "basis_terms": chosen_result["basis_terms"],
        "coefficient_by_term": coefficient_by_term,
        "coefficients": final_fit["coefficients"],
        "final_fit": final_fit,
        "selection_result": chosen_result,
        "tried_results": tried_results,
        "passed_cutoff": passed_cutoff,
        "passed_base_cutoff": passed_base_cutoff,
        "used_hysteresis": used_hysteresis,
        "hysteresis_anchor": hysteresis_anchor,
        "hysteresis_relative_tolerance": hysteresis_relative_tolerance,
        "active_relative_rmse_cutoff": chosen_result.get("active_relative_rmse_cutoff", relative_rmse_cutoff),
        "max_l1": chosen_result["max_l1"],
        "max_l2": chosen_result["max_l2"],
        "term_count": chosen_result["term_count"],
        "training_rank": chosen_result["rank"],
        "discarded_singular_count": chosen_result["discarded_singular_count"],
        "svd_cutoff": chosen_result["svd_cutoff"],
        "retained_condition_number": chosen_result["retained_condition_number"],
        "full_condition_number": chosen_result["full_condition_number"],
        "max_abs_coefficient": final_fit["max_abs_coefficient"],
        "coefficient_l2_norm": final_fit["coefficient_l2_norm"],
        "test_rmse": chosen_result["test_rmse"],
        "test_relative_rmse": chosen_result["test_relative_rmse"],
        "train_rmse": chosen_result["train_rmse"],
        "train_relative_rmse": chosen_result["train_relative_rmse"],
        "training_indices": r_training_indices,
        "testing_indices": r_testing_indices,
        "refit_on_all_data": refit_on_all_data,
    }

def select_minimal_bases_by_radius_truncated_svd_with_hysteresis(
    all_data,
    max_l1_grid,
    max_l2_grid,
    relative_rmse_cutoff=0.01,
    svd_threshold=1e-8,
    threshold_mode="relative",
    min_l1=8,
    min_l2=4,
    terms_cap=None,
    train_fraction=0.80,
    random_seed=202605702,
    refit_on_all_data=True,
    hysteresis_previous_count=3,
    hysteresis_relative_tolerance=0.30,
    verbose=True,
):
    shell_fits = {}
    previous_shell_fits = []

    for radius_index, r_value in enumerate(numpy.unique(all_data[:, 0])):
        r_data = all_data[all_data[:, 0] == r_value]
        shell_seed = None if random_seed is None else int(random_seed) + radius_index
        hysteresis_anchor = hysteresis_anchor_from_previous_shells(
            previous_shell_fits,
            required_previous_count=hysteresis_previous_count,
        )
        shell_fit = select_minimal_basis_for_shell_truncated_svd_with_hysteresis(
            r_data,
            max_l1_grid,
            max_l2_grid,
            relative_rmse_cutoff=relative_rmse_cutoff,
            svd_threshold=svd_threshold,
            threshold_mode=threshold_mode,
            min_l1=min_l1,
            min_l2=min_l2,
            terms_cap=terms_cap,
            train_fraction=train_fraction,
            random_seed=shell_seed,
            refit_on_all_data=refit_on_all_data,
            hysteresis_anchor=hysteresis_anchor,
            hysteresis_relative_tolerance=hysteresis_relative_tolerance,
        )
        shell_fits[float(r_value)] = shell_fit
        previous_shell_fits.append(shell_fit)

        if verbose:
            if shell_fit["used_hysteresis"]:
                status = "hysteresis passed"
            elif shell_fit["passed_base_cutoff"]:
                status = "passed"
            else:
                status = "best available"
            print(
                f"R = {r_value:g}: {status}, "
                f"test relative RMSE = {shell_fit['test_relative_rmse']:.6g}, "
                f"active cutoff = {shell_fit['active_relative_rmse_cutoff']:.6g}, "
                f"terms = {shell_fit['term_count']}, "
                f"max_l1 = {shell_fit['max_l1']}, max_l2 = {shell_fit['max_l2']}, "
                f"rank = {shell_fit['training_rank']}, "
                f"discarded singular values = {shell_fit['discarded_singular_count']}, "
                f"max abs coefficient = {shell_fit['max_abs_coefficient']:.6g}"
            )

    return shell_fits

def build_minimized_truncated_svd_hysteresis_pes_model(
    all_data,
    max_l1_grid,
    max_l2_grid,
    relative_rmse_cutoff=0.01,
    svd_threshold=1e-8,
    threshold_mode="relative",
    min_l1=8,
    min_l2=4,
    terms_cap=None,
    train_fraction=0.80,
    random_seed=202605702,
    refit_on_all_data=True,
    hysteresis_previous_count=3,
    hysteresis_relative_tolerance=0.30,
    spline_smoothing=0.0,
    max_spline_degree=3,
    decay_a=1.0,
    verbose=True,
):
    shell_fits = select_minimal_bases_by_radius_truncated_svd_with_hysteresis(
        all_data,
        max_l1_grid,
        max_l2_grid,
        relative_rmse_cutoff=relative_rmse_cutoff,
        svd_threshold=svd_threshold,
        threshold_mode=threshold_mode,
        min_l1=min_l1,
        min_l2=min_l2,
        terms_cap=terms_cap,
        train_fraction=train_fraction,
        random_seed=random_seed,
        refit_on_all_data=refit_on_all_data,
        hysteresis_previous_count=hysteresis_previous_count,
        hysteresis_relative_tolerance=hysteresis_relative_tolerance,
        verbose=verbose,
    )
    Q_union, coefficient_models = build_radial_coefficient_models(
        shell_fits,
        spline_smoothing=spline_smoothing,
        max_spline_degree=max_spline_degree,
    )

    return {
        "shell_fits": shell_fits,
        "Q_union": Q_union,
        "coefficient_models": coefficient_models,
        "relative_rmse_cutoff": relative_rmse_cutoff,
        "svd_threshold": svd_threshold,
        "threshold_mode": threshold_mode,
        "min_l1": min_l1,
        "min_l2": min_l2,
        "terms_cap": terms_cap,
        "train_fraction": train_fraction,
        "random_seed": random_seed,
        "refit_on_all_data": refit_on_all_data,
        "hysteresis_previous_count": hysteresis_previous_count,
        "hysteresis_relative_tolerance": hysteresis_relative_tolerance,
        "spline_smoothing": spline_smoothing,
        "max_spline_degree": max_spline_degree,
        "decay_a": decay_a,
    }

def radial_switch_c2(R, R_cutoff, decay_a=1.0):
    R_array = numpy.asarray(R, dtype=float)
    switched = numpy.where(
        R_array <= R_cutoff,
        1.0,
        numpy.exp(-float(decay_a) * (R_array - R_cutoff) ** 4),
    )
    return float(switched) if switched.shape == () else switched

def build_c2_radial_coefficient_models(
    shell_fits,
    spline_smoothing=None,
    artificial_zero_weight=0.5,
):
    """Build globally cubic radial coefficient models using zeros for inactive terms."""
    radii = numpy.asarray(sorted(shell_fits), dtype=float)
    if radii.size < 4:
        raise ValueError("C2 radial coefficient models require at least four radii for cubic splines.")
    if artificial_zero_weight <= 0 or artificial_zero_weight > 1:
        raise ValueError("artificial_zero_weight must be in the interval (0, 1].")

    Q_union = sorted(
        {term for shell_fit in shell_fits.values() for term in shell_fit["basis_terms"]},
        key=lambda term: (sum(term), term[0], term[1], term[2]),
    )
    coefficient_models = {}

    for term in Q_union:
        coefficients = []
        active_mask = []

        for r_value in radii:
            coefficient = shell_fits[float(r_value)]["coefficient_by_term"].get(term)
            is_active = coefficient is not None
            coefficients.append(float(coefficient) if is_active else 0.0)
            active_mask.append(is_active)

        coefficients = numpy.asarray(coefficients, dtype=float)
        active_mask = numpy.asarray(active_mask, dtype=bool)
        weights = numpy.where(active_mask, 1.0, float(artificial_zero_weight))

        coefficient_model = scipy.interpolate.UnivariateSpline(
            radii,
            coefficients,
            w=weights,
            k=3,
            s=spline_smoothing,
            ext=0,
        )

        active_radii = radii[active_mask]
        coefficient_models[term] = {
            "term": term,
            "radii": radii,
            "active_radii": active_radii,
            "coefficients": coefficients,
            "active_mask": active_mask,
            "weights": weights,
            "coefficient_model": coefficient_model,
            "spline_degree": 3,
            "R_cutoff": float(active_radii.max()),
        }

    return Q_union, coefficient_models

def build_c2_pes_model_from_shell_fits(
    shell_fits,
    decay_a=1.0,
    spline_smoothing=None,
    artificial_zero_weight=0.5,
):
    Q_union, coefficient_models = build_c2_radial_coefficient_models(
        shell_fits,
        spline_smoothing=spline_smoothing,
        artificial_zero_weight=artificial_zero_weight,
    )

    return {
        "shell_fits": shell_fits,
        "Q_union": Q_union,
        "coefficient_models": coefficient_models,
        "decay_a": decay_a,
        "decay_power": 4,
        "spline_smoothing": spline_smoothing,
        "artificial_zero_weight": artificial_zero_weight,
        "max_spline_degree": 3,
    }

def evaluate_c2_pes(model, R, theta1, theta2, phi, angles_in_degrees=True):
    """Evaluate the C2-in-R PES for scalar R and scalar or array-valued angles."""
    if numpy.ndim(R) != 0:
        raise NotImplementedError("evaluate_c2_pes currently expects scalar R.")

    theta1_values = numpy.asarray(theta1, dtype=float)
    theta2_values = numpy.asarray(theta2, dtype=float)
    phi_values = numpy.asarray(phi, dtype=float)
    if angles_in_degrees:
        theta1_values = numpy.deg2rad(theta1_values)
        theta2_values = numpy.deg2rad(theta2_values)
        phi_values = numpy.deg2rad(phi_values)

    value = numpy.zeros_like(theta1_values, dtype=float)
    for term in model["Q_union"]:
        L, l1, l2 = term
        coefficient_info = model["coefficient_models"][term]
        coefficient = float(coefficient_info["coefficient_model"](R))
        switch = radial_switch_c2(R, coefficient_info["R_cutoff"], decay_a=model["decay_a"])
        value = value + switch * coefficient * angular_basis(L, l1, l2, theta1_values, theta2_values, phi_values)

    return float(value) if value.shape == () else value


# --- Dataset-oriented model classes ---


@dataclass
class PESDataset:
    """Container for PES rows in the notebook format: R, theta1, theta2, phi, energy."""

    data: numpy.ndarray
    name: str = "PES"

    def __post_init__(self):
        self.data = numpy.asarray(self.data, dtype=float)
        if self.data.ndim != 2 or self.data.shape[1] < 5:
            raise ValueError("data must be a 2D array with columns R, theta1, theta2, phi, energy.")

    @property
    def radii(self):
        return numpy.asarray(sorted(numpy.unique(self.data[:, 0]), reverse=True), dtype=float)

    def radius_data(self, radius):
        return radius_subset(self.data, radius)

    def grouped_by_radius(self):
        return grouped_radius_data(self.data)

    def validate_common_angular_grid(self):
        _, radius_data_by_r = self.grouped_by_radius()
        return validate_common_angular_grid(radius_data_by_r)


@dataclass
class TrainTestConfig:
    train_fraction: float = DEFAULT_TRAIN_FRACTION
    random_seed: int = DEFAULT_RANDOM_SEED
    initial_test_point_count: int = RADIUS_TO_RADIUS_INITIAL_TEST_POINT_COUNT


@dataclass
class BasisSelectionConfig:
    max_l1_grid: object = None
    max_l2_grid: object = None
    relative_rmse_cutoff: float = DEFAULT_RELATIVE_RMSE_CUTOFF
    svd_threshold: float = DEFAULT_SVD_THRESHOLD
    threshold_mode: str = DEFAULT_THRESHOLD_MODE

    def resolved_max_l1_grid(self):
        return ACQUISITION_SWEEP_MAX_L1_GRID if self.max_l1_grid is None else self.max_l1_grid

    def resolved_max_l2_grid(self):
        return ACQUISITION_SWEEP_MAX_L2_GRID if self.max_l2_grid is None else self.max_l2_grid


@dataclass
class AcquisitionConfig:
    gamma: float = RADIUS_TO_RADIUS_GAMMA
    required_angle_points_degrees: object = None
    alpha: float = ACQUISITION_ALPHA
    beta: float = ACQUISITION_BETA
    min_peak_separation_degrees: float = ACQUISITION_MIN_PEAK_SEPARATION_DEGREES
    candidate_grid: object = None
    initial_fit_point_count: int = RADIUS_TO_RADIUS_INITIAL_FIT_POINT_COUNT

    def resolved_required_angle_points(self):
        if self.required_angle_points_degrees is None:
            return ACQUISITION_REQUIRED_ANGLE_POINTS_DEGREES
        return numpy.asarray(self.required_angle_points_degrees, dtype=float)

    def resolved_candidate_grid(self):
        if self.candidate_grid is None:
            return make_angular_candidate_grid()
        return numpy.asarray(self.candidate_grid, dtype=float)


@dataclass
class SecondPassConfig:
    term_growth_fraction: float = SECOND_PASS_TERM_GROWTH_FRACTION
    term_growth_min_increase: int = SECOND_PASS_TERM_GROWTH_MIN_INCREASE
    max_condition_number: float = SECOND_PASS_MAX_CONDITION_NUMBER
    min_extra_points: int = SECOND_PASS_MIN_EXTRA_POINTS


class HysteresisAcquisitionModel:
    """Run the radius-to-radius hysteresis/acquisition model for any dataset in notebook format."""

    def __init__(
        self,
        dataset,
        train_test_config=None,
        basis_config=None,
        acquisition_config=None,
        second_pass_config=None,
        use_second_pass=True,
    ):
        self.dataset = dataset if isinstance(dataset, PESDataset) else PESDataset(dataset)
        self.train_test_config = train_test_config or TrainTestConfig()
        self.basis_config = basis_config or BasisSelectionConfig()
        self.acquisition_config = acquisition_config or AcquisitionConfig()
        self.second_pass_config = second_pass_config or SecondPassConfig()
        self.use_second_pass = bool(use_second_pass)
        self.result = None

    def fit(self):
        acquisition = self.acquisition_config
        basis = self.basis_config
        train_test = self.train_test_config
        if self.use_second_pass:
            self.result = run_radius_to_radius_hysteresis_second_pass_experiment(
                self.dataset.data,
                gamma=acquisition.gamma,
                train_fraction=train_test.train_fraction,
                random_seed=train_test.random_seed,
                candidate_grid=acquisition.resolved_candidate_grid(),
                initial_fit_point_count=acquisition.initial_fit_point_count,
                initial_test_point_count=train_test.initial_test_point_count,
                required_angle_points_degrees=acquisition.resolved_required_angle_points(),
                acquisition_alpha=acquisition.alpha,
                acquisition_beta=acquisition.beta,
                min_peak_separation_degrees=acquisition.min_peak_separation_degrees,
                relative_rmse_cutoff=basis.relative_rmse_cutoff,
                svd_threshold=basis.svd_threshold,
                threshold_mode=basis.threshold_mode,
                growth_fraction_threshold=self.second_pass_config.term_growth_fraction,
                growth_min_terms=self.second_pass_config.term_growth_min_increase,
                max_condition_number=self.second_pass_config.max_condition_number,
                min_extra_points=self.second_pass_config.min_extra_points,
            )
        else:
            self.result = run_radius_to_radius_hysteresis_acquisition_experiment(
                self.dataset.data,
                gamma=acquisition.gamma,
                train_fraction=train_test.train_fraction,
                random_seed=train_test.random_seed,
                candidate_grid=acquisition.resolved_candidate_grid(),
                initial_fit_point_count=acquisition.initial_fit_point_count,
                initial_test_point_count=train_test.initial_test_point_count,
                required_angle_points_degrees=acquisition.resolved_required_angle_points(),
                acquisition_alpha=acquisition.alpha,
                acquisition_beta=acquisition.beta,
                min_peak_separation_degrees=acquisition.min_peak_separation_degrees,
                relative_rmse_cutoff=basis.relative_rmse_cutoff,
                svd_threshold=basis.svd_threshold,
                threshold_mode=basis.threshold_mode,
            )
        return self

    def require_fit(self):
        if self.result is None:
            raise RuntimeError("Call fit() before requesting model outputs.")
        return self.result

    def transition_summary(self):
        result = self.require_fit()
        if self.use_second_pass:
            result_for_audit = dict(result)
            result_for_audit["all_data_for_audit"] = self.dataset.data
            return second_pass_transition_audit(result_for_audit)
        return result["transitions"]

    def coefficient_table(
        self,
        missing_value=YUMI_MISSING_COEFFICIENT_VALUE,
        apply_normalization=YUMI_APPLY_COEFFICIENT_NORMALIZATION,
    ):
        return collect_yumi_coefficient_table(
            self.require_fit(),
            missing_value=missing_value,
            apply_normalization=apply_normalization,
        )

    def save_yumi_coefficients(
        self,
        output_path=YUMI_COEFFICIENT_OUTPUT_PATH,
        title=None,
        apply_normalization=YUMI_APPLY_COEFFICIENT_NORMALIZATION,
    ):
        title = title or f"{self.dataset.name} PES fitted coefficients"
        return save_yumi_coefficients(
            self.require_fit(),
            output_path=output_path,
            title=title,
            apply_normalization=apply_normalization,
        )

    def isotropic_coefficients(self, term=(0, 0, 0)):
        return isotropic_coefficient_series_from_result(self.require_fit(), term=term)


class ShellRadialPESModel:
    """Build and evaluate shell-wise radial PES models from data in notebook format."""

    def __init__(self, dataset, model_builder=build_minimized_truncated_svd_hysteresis_pes_model):
        self.dataset = dataset if isinstance(dataset, PESDataset) else PESDataset(dataset)
        self.model_builder = model_builder
        self.model = None

    def fit(self, **builder_kwargs):
        self.model = self.model_builder(self.dataset.data, **builder_kwargs)
        return self

    def require_model(self):
        if self.model is None:
            raise RuntimeError("Call fit() before evaluating the model.")
        return self.model

    def evaluate(self, R, theta1, theta2, phi):
        model = self.require_model()
        if model.get("radial_model_type") == "c2_cubic_spline_with_quartic_decay":
            return evaluate_c2_pes(model, R, theta1, theta2, phi)
        return evaluate_minimized_pes(model, R, theta1, theta2, phi)
