"""Artificial, truth-free contract tests for the opt-in Student A v3.1 MICE path."""

import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import KNNImputer, SimpleImputer

from petsard.exceptions import ConfigError, UnableToFollowMetadataError
from petsard.metadater import SchemaMetadater
from petsard.processor import CompositePreprocessor
from petsard.processor.mice_postprocess import postprocess_mice
from petsard.processor.missing_compensation import MissingCompensation

FEATURES = ["lab_min", "lab_max", "lab_mean", "lab_n", "icu_stay_24h"]
MODULE = Path(__file__).parents[2] / "petsard/processor/missing_compensation.py"


def frames():
    rng = np.random.default_rng(42)
    size = 48
    public = pd.DataFrame(
        {
            "record_id": [f"public-{i}" for i in range(size)],
            "lab_min": rng.uniform(1, 3, size).astype("float32"),
            "lab_max": rng.uniform(5, 7, size).astype("float32"),
            "lab_mean": rng.uniform(3, 5, size).astype("float32"),
            "lab_n": rng.integers(1, 8, size).astype("float32"),
            "icu_stay_24h": rng.integers(0, 2, size).astype("float32"),
            "late": rng.uniform(10, 20, size).astype("float32"),
            "y_malignancy": rng.integers(0, 2, size).astype("int8"),
        }
    )
    public.loc[2, "lab_mean"] = np.nan
    local = public.iloc[:8].copy(deep=True)
    local.index = [8, 7, 6, 5, 4, 3, 2, 1]
    local["record_id"] = [f"local-{i}" for i in range(len(local))]
    local.loc[8, "lab_mean"] = np.nan
    local.loc[7, "icu_stay_24h"] = np.nan
    local.loc[6, "lab_n"] = np.nan
    local.loc[5, "late"] = np.nan
    return public, local


def config(method="mice"):
    return {
        "missing_compensation": {"method": method},
        "feature_columns": FEATURES,
        "fit_source": {"id": "public-artificial", "role": "public"},
        "transform_source": "local-masked",
        "key_columns": ["record_id"],
        "target_columns": ["y_malignancy"],
    }


@pytest.fixture(scope="module")
def fitted_mice():
    public, local = frames()
    processor = MissingCompensation(config()).fit(fit_data=public)
    return public, local, processor


def test_mice_exact_estimator_external_fit_and_schema(fitted_mice):
    public, local, processor = fitted_mice
    before = local.copy(deep=True)
    output = processor.transform(local)
    report = processor.get_report()
    imputer = processor._imputer
    assert type(imputer).__name__ == "IterativeImputer"
    assert imputer.max_iter == 3 and imputer.random_state == 42
    assert type(imputer.estimator).__name__ == "RandomForestRegressor"
    assert imputer.estimator.get_params()["n_estimators"] == 50
    assert imputer.estimator.get_params()["max_depth"] == 10
    assert imputer.estimator.get_params()["random_state"] == 42
    assert imputer.estimator.get_params()["n_jobs"] == -1
    assert report["fit_rows"] == len(public) and report["fit_count"] == 1
    assert report["fit_source"]["role"] == "public"
    expected_fingerprint = hashlib.sha256()
    expected_fingerprint.update("\0".join(FEATURES).encode())
    expected_fingerprint.update("\0".join(map(str, public[FEATURES].dtypes)).encode())
    expected_fingerprint.update(public[FEATURES].to_numpy(dtype=float).tobytes())
    assert (
        processor.get_strategy_state()["fingerprint"]
        == expected_fingerprint.hexdigest()
    )
    assert output.index.equals(local.index)
    assert output.columns.equals(local.columns)
    assert output.dtypes.equals(local.dtypes)
    pd.testing.assert_frame_equal(
        output.where(~local.isna()), local.where(~local.isna())
    )
    pd.testing.assert_frame_equal(
        output[["record_id", "y_malignancy"]], local[["record_id", "y_malignancy"]]
    )
    pd.testing.assert_frame_equal(local, before)
    assert output[FEATURES].notna().all().all()
    assert pd.isna(output.loc[5, "late"])  # downstream built-in missing may handle it
    assert not any(report["mice_qa"].values())
    with pytest.raises(ConfigError, match="frozen"):
        processor.fit(fit_data=local)


def test_mice_a5_routing_and_observed_restoration(fitted_mice):
    _, local, processor = fitted_mice
    raw = pd.DataFrame(
        {
            "lab_min": [6.0],
            "lab_max": [2.0],
            "lab_mean": [8.0],
            "lab_n": [-1.4],
            "icu_stay_24h": [0.8],
        }
    )
    repaired = postprocess_mice(raw)
    assert repaired.iloc[0].to_dict() == {
        "lab_min": 2.0,
        "lab_max": 6.0,
        "lab_mean": 6.0,
        "lab_n": 0.0,
        "icu_stay_24h": 1.0,
    }
    # A5 may propose changes to observed cells, but transform commits only
    # cells declared missing by the input mask.
    candidate = processor._imputer.transform(local[FEATURES])
    with patch.object(processor._imputer, "transform", return_value=candidate):
        output = processor.transform(local)
    pd.testing.assert_frame_equal(
        output.where(~local.isna()), local.where(~local.isna())
    )
    assert output.loc[7, "icu_stay_24h"] in (0, 1)
    assert output.loc[6, "lab_n"].is_integer()


def test_mice_rejects_private_or_missing_fit_and_indicator_features(tmp_path):
    public, local = frames()
    cfg = config()
    cfg["fit_source"]["role"] = "private"
    with pytest.raises(ConfigError):
        MissingCompensation(cfg)
    cfg = config()
    cfg["fit_data"] = {"filepath": str(tmp_path / "absent.csv")}
    with pytest.raises(UnableToFollowMetadataError):
        MissingCompensation(cfg).fit(data=local)
    cfg = config()
    cfg["feature_columns"] = ["lab_mean__missing"]
    with pytest.raises(ConfigError, match="indicators"):
        MissingCompensation(cfg)
    assert public["record_id"].iloc[0] != local["record_id"].iloc[0]


def test_mice_composite_and_frozen_split_strategy(tmp_path):
    public, train = frames()
    train = train.reset_index(drop=True)
    validation = train.copy(deep=True)
    validation["record_id"] = [f"validation-{i}" for i in range(len(validation))]
    control = train.copy(deep=True)
    control["record_id"] = [f"control-{i}" for i in range(len(control))]
    path = tmp_path / "public.csv"
    public.to_csv(path, index=False)
    custom = config()
    custom.update(
        {
            "name": "compensation",
            "method": "custom_method",
            "module_path": str(MODULE),
            "class_name": "MissingCompensation",
            "fit_data": {"filepath": str(path)},
            "fit_source": {"id": str(path), "role": "public"},
            "required_columns": FEATURES,
        }
    )
    stages = [
        {
            "name": "outlier",
            "method": "custom",
            "sequence": ["outlier"],
            "config": {"outlier": dict.fromkeys(train.columns)},
        },
        custom,
        {
            "name": "missing",
            "method": "custom",
            "sequence": ["missing"],
            "config": {
                "missing": {
                    c: "missing_mean" if c in FEATURES + ["late"] else None
                    for c in train
                }
            },
        },
    ]
    processor = CompositePreprocessor(
        {"stages": stages}, SchemaMetadater.from_data(train)
    )
    inputs = {"train": train, "validation": validation, "control": control}
    outputs = processor.fit_transform_splits(inputs)
    artifacts = processor.get_artifacts()
    assert artifacts["fit_count"] == 1 and artifacts["transform_count"] == 3
    assert set(artifacts["reports"]) == set(inputs)
    states = artifacts["strategy_states"]
    assert set(states) == set(inputs)
    assert len({state["fingerprint"] for state in states.values()}) == 1
    assert all(state["frozen"] and state["fit_count"] == 1 for state in states.values())
    for name, input_frame in inputs.items():
        assert outputs[name]["late"].notna().all()
        pd.testing.assert_series_equal(
            outputs[name]["record_id"], input_frame["record_id"]
        )
        pd.testing.assert_series_equal(
            outputs[name]["y_malignancy"], input_frame["y_malignancy"]
        )
        report = artifacts["reports"][name]
        assert report["fit_count"] == 1
        assert report["fit_source"] == {"id": str(path), "role": "public"}
        assert report["outside_mask_changes"] == 0
        assert not any(report["mice_qa"].values())
    assert processor._stages[1]["processor"]._impl._imputer is not None


def test_mice_is_opt_in_and_does_not_bleed_into_median_knn(fitted_mice):
    public, local, mice = fitted_mice
    from sklearn.impute import IterativeImputer

    with patch.object(
        IterativeImputer, "__init__", side_effect=AssertionError("MICE instantiated")
    ) as init:
        median = MissingCompensation(config("median")).fit(fit_data=public)
        knn = MissingCompensation(config("knn")).fit(fit_data=public)
        init.assert_not_called()
    assert type(median._imputer) is SimpleImputer
    assert type(knn._imputer) is KNNImputer
    assert type(mice._imputer).__name__ == "IterativeImputer"
    median_before = median.transform(local)
    knn_before = knn.transform(local)
    mice.transform(local)
    pd.testing.assert_frame_equal(median.transform(local), median_before)
    pd.testing.assert_frame_equal(knn.transform(local), knn_before)
    assert "mice_qa" not in median.get_report()
    assert "mice_qa" not in knn.get_report()


@pytest.mark.parametrize("option,value", [("n_neighbors", 5), ("max_iter", 4)])
def test_mice_rejects_configurable_protocol_changes(option, value):
    cfg = config()
    cfg["missing_compensation"][option] = value
    with pytest.raises(ConfigError):
        MissingCompensation(cfg)
