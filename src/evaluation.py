from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

MAX_EXACT_GROUPS = 20
DEFAULT_MONTE_CARLO_REPEATS = 100_000
DEFAULT_RANDOMIZATION_SEED = 42


def _coerce_correct(values: pd.Series) -> pd.Series:
    """Return a strict boolean correctness series from in-memory or CSV data."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false"})
    if invalid.any():
        examples = sorted(normalized[invalid].unique())[:3]
        raise ValueError(f"Invalid is_correct values: {examples}")
    return normalized.eq("true")


def _paired_group_differences(
    predictions: pd.DataFrame,
    left_slug: str,
    right_slug: str,
    group_column: str,
) -> tuple[pd.Series, pd.Series, np.ndarray, int]:
    required = {"sample_id", "model_slug", "is_correct", group_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")

    left = predictions[predictions["model_slug"].eq(left_slug)].set_index("sample_id")
    right = predictions[predictions["model_slug"].eq(right_slug)].set_index("sample_id")
    if left.empty or right.empty:
        raise ValueError("Both compared models must have prediction rows")
    if set(left.index) != set(right.index):
        raise ValueError("Paired models do not cover the same validation rows")

    left = left.sort_index()
    right = right.loc[left.index]
    if not left[group_column].astype(str).equals(right[group_column].astype(str)):
        raise ValueError("Paired prediction rows disagree on group identity")

    left_correct = _coerce_correct(left["is_correct"])
    right_correct = _coerce_correct(right["is_correct"])
    grouped = pd.DataFrame(
        {
            "group": left[group_column].astype(str),
            "left_correct": left_correct.astype(int),
            "right_correct": right_correct.astype(int),
        },
        index=left.index,
    ).groupby("group", sort=True)[["left_correct", "right_correct"]].sum()
    if len(grouped) < 2:
        raise ValueError("At least two independent groups are required")
    differences = (grouped["left_correct"] - grouped["right_correct"]).to_numpy(
        dtype=np.int64
    )
    return left_correct, right_correct, differences, len(grouped)


def _common_result(
    *,
    left_slug: str,
    right_slug: str,
    group_column: str,
    left_correct: pd.Series,
    right_correct: pd.Series,
    group_differences: np.ndarray,
    method: str,
    assignments: int,
    p_value: float,
    randomization_seed: int | None,
) -> dict[str, object]:
    observed_difference = int(group_differences.sum())
    left_row = left_correct.to_numpy(dtype=bool)
    right_row = right_correct.to_numpy(dtype=bool)
    left_only_rows = int(np.sum(left_row & ~right_row))
    right_only_rows = int(np.sum(~left_row & right_row))
    total_rows = len(left_correct)
    return {
        "left_model_slug": left_slug,
        "right_model_slug": right_slug,
        "method": method,
        "group_column": group_column,
        "independent_groups": int(len(group_differences)),
        "paired_rows": int(total_rows),
        "left_correct_total": int(left_correct.sum()),
        "right_correct_total": int(right_correct.sum()),
        "observed_correct_difference": observed_difference,
        "observed_accuracy_difference": float(observed_difference / total_rows),
        "left_better_groups": int(np.sum(group_differences > 0)),
        "right_better_groups": int(np.sum(group_differences < 0)),
        "tied_groups": int(np.sum(group_differences == 0)),
        "left_correct_right_wrong_rows": left_only_rows,
        "left_wrong_right_correct_rows": right_only_rows,
        "row_discordant_predictions": left_only_rows + right_only_rows,
        "randomization_assignments": int(assignments),
        "randomization_seed": np.nan if randomization_seed is None else int(randomization_seed),
        "grouped_two_sided_p_value": float(p_value),
    }


def exact_grouped_paired_randomization(
    predictions: pd.DataFrame,
    left_slug: str,
    right_slug: str,
    *,
    group_column: str = "image_id",
) -> dict[str, object]:
    """Exact two-sided sign-flip test over complete independent image groups."""
    left_correct, right_correct, differences, group_count = _paired_group_differences(
        predictions, left_slug, right_slug, group_column
    )
    if group_count > MAX_EXACT_GROUPS:
        raise ValueError(
            f"Exact enumeration supports at most {MAX_EXACT_GROUPS} groups; got {group_count}"
        )

    observed_absolute = abs(int(differences.sum()))
    assignments = 2**group_count
    extreme = 0
    for signs in product((-1, 1), repeat=group_count):
        randomized = int(np.dot(differences, np.asarray(signs, dtype=np.int64)))
        if abs(randomized) >= observed_absolute:
            extreme += 1
    return _common_result(
        left_slug=left_slug,
        right_slug=right_slug,
        group_column=group_column,
        left_correct=left_correct,
        right_correct=right_correct,
        group_differences=differences,
        method="exact_image_group_sign_flip",
        assignments=assignments,
        p_value=float(extreme / assignments),
        randomization_seed=None,
    )


def monte_carlo_grouped_paired_randomization(
    predictions: pd.DataFrame,
    left_slug: str,
    right_slug: str,
    *,
    group_column: str = "image_id",
    repeats: int = DEFAULT_MONTE_CARLO_REPEATS,
    seed: int = DEFAULT_RANDOMIZATION_SEED,
) -> dict[str, object]:
    """Deterministic Monte Carlo sign-flip test over complete image groups.

    The plus-one correction keeps the estimated p-value valid and non-zero:
    ``(extreme + 1) / (repeats + 1)``.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    left_correct, right_correct, differences, _ = _paired_group_differences(
        predictions, left_slug, right_slug, group_column
    )
    observed_absolute = abs(int(differences.sum()))
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = repeats
    batch_size = min(10_000, repeats)
    while remaining:
        current = min(batch_size, remaining)
        signs = rng.integers(0, 2, size=(current, len(differences)), dtype=np.int8)
        signs = signs * 2 - 1
        randomized = signs.astype(np.int64) @ differences
        extreme += int(np.sum(np.abs(randomized) >= observed_absolute))
        remaining -= current
    p_value = float((extreme + 1) / (repeats + 1))
    return _common_result(
        left_slug=left_slug,
        right_slug=right_slug,
        group_column=group_column,
        left_correct=left_correct,
        right_correct=right_correct,
        group_differences=differences,
        method="monte_carlo_image_group_sign_flip",
        assignments=repeats,
        p_value=p_value,
        randomization_seed=seed,
    )


def grouped_paired_randomization(
    predictions: pd.DataFrame,
    left_slug: str,
    right_slug: str,
    *,
    group_column: str = "image_id",
    monte_carlo_repeats: int = DEFAULT_MONTE_CARLO_REPEATS,
    seed: int = DEFAULT_RANDOMIZATION_SEED,
) -> dict[str, object]:
    """Use exact enumeration for small samples and Monte Carlo for larger samples."""
    group_count = predictions[predictions["model_slug"].eq(left_slug)][
        group_column
    ].nunique()
    if group_count <= MAX_EXACT_GROUPS:
        return exact_grouped_paired_randomization(
            predictions, left_slug, right_slug, group_column=group_column
        )
    return monte_carlo_grouped_paired_randomization(
        predictions,
        left_slug,
        right_slug,
        group_column=group_column,
        repeats=monte_carlo_repeats,
        seed=seed,
    )
