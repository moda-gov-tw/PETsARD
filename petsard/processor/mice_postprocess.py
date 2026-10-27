"""Student A v3.1 A5 policy and domain checks for the optional MICE path.

Only a candidate table reaches this module. The caller restores observed values
from its masked transform input and checks the original DataFrame contract.
"""

import numpy as np
import pandas as pd

from petsard.exceptions import ConfigError

SUMMARY_SUFFIXES = ("_first", "_min", "_max", "_mean", "_last", "_n", "_amt")


def _summary_bases(columns):
    return sorted(
        {
            column[: -len(suffix)]
            for column in columns
            for suffix in SUMMARY_SUFFIXES
            if column.endswith(suffix)
        }
    )


def postprocess_mice(candidate: pd.DataFrame) -> pd.DataFrame:
    """Apply the qualified MICE-only A5 routing before observed restoration."""
    result = candidate.copy(deep=True)
    for base in _summary_bases(result.columns):
        low, high = f"{base}_min", f"{base}_max"
        if low not in result or high not in result:
            continue
        actual_low = np.minimum(result[low], result[high])
        actual_high = np.maximum(result[low], result[high])
        result[low], result[high] = actual_low, actual_high
        for suffix in ("_mean", "_first", "_last"):
            column = f"{base}{suffix}"
            if column in result:
                result[column] = result[column].clip(
                    lower=result[low], upper=result[high]
                )

    for column in result:
        if column.endswith("_n"):
            result[column] = result[column].clip(lower=0).round().astype(int)

    if "icu_stay_24h" in result:
        result["icu_stay_24h"] = (
            result["icu_stay_24h"].clip(lower=0, upper=1).round().astype(int)
        )
    return result


def audit_mice(
    result: pd.DataFrame,
    mask: pd.DataFrame,
    features: list[str],
    public_min: pd.Series,
    public_max: pd.Series,
) -> dict[str, int]:
    """Fail closed on the v3.1 feature-domain guarantees.

    Public-range checks apply only to newly filled cells. Unchanged observed
    local values are allowed to lie outside the Public cohort's range.
    """
    values = result[features].to_numpy(dtype=float, na_value=np.nan)
    qa = {
        "remaining_missing": int(result[features].isna().to_numpy().sum()),
        "non_finite": int((~np.isfinite(values)).sum()),
        "unexpected_negative": 0,
        "new_masked_public_range_violations": 0,
        "binary_violations": 0,
        "integer_violations": 0,
        "summary_order_violations": 0,
    }
    for column in features:
        numeric = result[column].to_numpy(dtype=float, na_value=np.nan)
        selected = mask[column].to_numpy(dtype=bool)
        if public_min[column] >= 0:
            qa["unexpected_negative"] += int((numeric < 0).sum())
        qa["new_masked_public_range_violations"] += int(
            (
                selected
                & (
                    (numeric < public_min[column] - 0.001)
                    | (numeric > public_max[column] + 0.001)
                )
            ).sum()
        )
        if column.endswith("_n"):
            qa["integer_violations"] += int(
                (~np.isclose(numeric, np.round(numeric), rtol=0, atol=0)).sum()
            )
    if "icu_stay_24h" in features:
        qa["binary_violations"] = int((~result["icu_stay_24h"].isin([0, 1])).sum())
    for base in _summary_bases(features):
        low, high = f"{base}_min", f"{base}_max"
        if low not in result or high not in result:
            continue
        qa["summary_order_violations"] += int((result[low] > result[high]).sum())
        for suffix in ("_mean", "_first", "_last"):
            column = f"{base}{suffix}"
            if column in result:
                qa["summary_order_violations"] += int(
                    (
                        (result[column] < result[low]) | (result[column] > result[high])
                    ).sum()
                )
    if any(qa.values()):
        raise ConfigError(f"MICE output violates the qualified domain: {qa}")
    return qa
