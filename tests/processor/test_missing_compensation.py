"""Contract tests use synthetic external and local data only."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import KNNImputer, SimpleImputer

from petsard.adapter import PreprocessorAdapter
from petsard.exceptions import ConfigError, UnfittedError
from petsard.metadater import SchemaMetadater
from petsard.processor.custom_preprocessor import CustomPreprocessor
from petsard.processor.missing_compensation import MissingCompensation


def frames():
    public = pd.DataFrame(
        {
            "record_id": range(10),
            "subject_id": range(10),
            "x": np.arange(10, dtype="float32"),
            "z": np.arange(10, dtype="float32") * 2,
            "y_malignancy": np.arange(10) % 2,
        }
    )
    local = pd.DataFrame(
        {
            "record_id": [20, 21, 22],
            "subject_id": [20, 21, 22],
            "x": np.array([np.nan, 1000.125, 2], dtype="float32"),
            "z": np.array([3, np.nan, 7], dtype="float32"),
            "y_malignancy": [1, 0, 1],
        },
        index=[9, 4, 4],
    )
    return public, local


def config(method="median"):
    return {
        "missing_compensation": {"method": method},
        "feature_columns": ["x", "z"],
        "fit_source": {"id": "public-fixture", "role": "public"},
        "transform_source": "local-fixture",
    }


@pytest.mark.parametrize("method", ["median", "knn"])
def test_external_fit_preservation_and_replay(method):
    public, local = frames()
    original = local.copy(deep=True)
    processor = MissingCompensation(config(method))
    with pytest.raises(UnfittedError):
        processor.transform(local)
    processor.fit(fit_data=public)
    result = processor.transform(transform_data=local, mask=local.isna())
    reference = (
        SimpleImputer(strategy="median")
        if method == "median"
        else KNNImputer(n_neighbors=5, weights="uniform", metric="nan_euclidean")
    )
    expected = reference.fit(public[["x", "z"]]).transform(local[["x", "z"]])
    np.testing.assert_array_equal(result[["x", "z"]].to_numpy(), expected)
    pd.testing.assert_frame_equal(
        result.where(~local.isna()), local.where(~local.isna()), check_exact=True
    )
    pd.testing.assert_frame_equal(result, processor.transform(local), check_exact=True)
    pd.testing.assert_frame_equal(local, original, check_exact=True)
    assert result.index.equals(local.index) and result.dtypes.equals(local.dtypes)
    assert processor.get_report()["outside_mask_changes"] == 0
    if method == "knn":
        assert processor.get_report()["parameters"]["n_neighbors"] == 5
    changed = local.copy()
    changed["y_malignancy"] = [0, 1, 0]
    pd.testing.assert_frame_equal(
        result[["x", "z"]], processor.transform(changed)[["x", "z"]]
    )
    with pytest.raises(ConfigError, match="frozen"):
        processor.fit(fit_data=local)


@pytest.mark.parametrize("method", ["median", "knn"])
def test_strategy_state_is_nonempty_frozen_and_privacy_safe(method):
    public, _ = frames()
    processor = MissingCompensation(config(method))
    assert processor.get_strategy_state() is None

    processor.fit(fit_data=public)
    state = processor.get_strategy_state()

    assert state == {
        "method": method,
        "frozen": True,
        "fit_count": 1,
        "fit_source": {"id": "public-fixture", "role": "public"},
        "fingerprint": state["fingerprint"],
    }
    assert len(state["fingerprint"]) == 64
    assert int(state["fingerprint"], 16) >= 0
    assert not any(
        isinstance(value, (list, tuple, pd.Series, pd.DataFrame, np.ndarray))
        for value in state.values()
    )


@pytest.mark.parametrize("method", ["median", "knn"])
def test_strategy_fingerprint_is_deterministic_tracks_fit_and_ignores_transform(method):
    public, local = frames()
    first = MissingCompensation(config(method)).fit(fit_data=public)
    second = MissingCompensation(config(method)).fit(fit_data=public.copy(deep=True))
    fingerprint = first.get_strategy_state()["fingerprint"]
    assert second.get_strategy_state()["fingerprint"] == fingerprint

    first.transform(local)
    changed_local = local.copy(deep=True)
    changed_local[["x", "z"]] = changed_local[["x", "z"]] + 123.5
    first.transform(changed_local)
    assert first.get_strategy_state()["fingerprint"] == fingerprint

    changed_public = public.copy(deep=True)
    changed_public["x"] = changed_public["x"] + 10
    changed = MissingCompensation(config(method)).fit(fit_data=changed_public)
    assert changed.get_strategy_state()["fingerprint"] != fingerprint


@pytest.mark.parametrize("method", ["median", "knn"])
def test_split_strategy_state_reuses_one_fit_without_refit(tmp_path, method):
    public, train = frames()
    validation = train.copy(deep=True)
    validation[["record_id", "subject_id"]] = [[30, 30], [31, 31], [32, 32]]
    control = train.copy(deep=True)
    control[["record_id", "subject_id"]] = [[40, 40], [41, 41], [42, 42]]
    path = tmp_path / "public.csv"
    public.to_csv(path, index=False)
    cfg = config(method)
    cfg.update(
        {
            "method": "custom_method",
            "class_name": "MissingCompensation",
            "module_path": str(
                Path(__file__).parents[2] / "petsard/processor/missing_compensation.py"
            ),
            "fit_data": {"filepath": str(path)},
            "fit_source": {"id": str(path), "role": "public"},
        }
    )
    processor = CustomPreprocessor(cfg)
    inputs = {"train": train, "validation": validation, "control": control}

    with patch.object(processor._impl, "fit", wraps=processor._impl.fit) as fit_spy:
        processor.fit(train)
        processor.transform_splits(inputs)

    artifacts = processor.get_artifacts()
    assert fit_spy.call_count == 1
    assert artifacts["fit_count"] == 1
    assert artifacts["transform_count"] == 3
    assert set(artifacts["strategy_states"]) == set(inputs)
    states = list(artifacts["strategy_states"].values())
    assert all(state["method"] == method for state in states)
    assert all(state["fit_count"] == 1 and state["frozen"] for state in states)
    assert len({state["fingerprint"] for state in states}) == 1


@pytest.mark.parametrize("method", ["median", "knn"])
def test_only_declared_mask_filled(method):
    public, local = frames()
    processor = MissingCompensation(config(method)).fit(fit_data=public)
    mask = local.isna()
    mask.iloc[1, mask.columns.get_loc("z")] = False
    result = processor.transform(local, mask=mask)
    assert pd.isna(result.iloc[1]["z"])
    assert result.iloc[0]["x"] == processor.transform(local).iloc[0]["x"]


@pytest.mark.parametrize(
    "column", ["record_id", "subject_id", "y_malignancy", "id", "subject_identifier"]
)
def test_reject_protected_features(column):
    cfg = config()
    cfg["feature_columns"] = [column]
    with pytest.raises(ConfigError, match="Identifiers"):
        MissingCompensation(cfg)


@pytest.mark.parametrize(
    "change",
    ["source", "role", "fit_missing", "fit_same", "id_overlap", "empty_feature"],
)
def test_fit_transform_isolation(change):
    public, local = frames()
    cfg = config()
    if change == "source":
        cfg["transform_source"] = cfg["fit_source"]["id"]
    if change == "role":
        cfg["fit_source"]["role"] = "private"
    with pytest.raises(ConfigError):
        p = MissingCompensation(cfg)
        if change == "fit_missing":
            p.fit(local)
        else:
            if change == "empty_feature":
                public["x"] = np.nan
            p.fit(fit_data=public)
            if change == "fit_same":
                p.transform(public.copy())
            elif change == "id_overlap":
                local.iloc[0, 0] = 0
                p.transform(local)


@pytest.mark.parametrize(
    "case", ["protected", "observed", "dtype", "alignment", "null_mask"]
)
def test_invalid_masks(case):
    public, local = frames()
    p = MissingCompensation(config()).fit(fit_data=public)
    mask = local.isna()
    if case == "protected":
        mask["record_id"] = True
    elif case == "observed":
        mask["x"] = True
    elif case == "dtype":
        mask = mask.astype(int)
    elif case == "alignment":
        mask = mask.reset_index(drop=True)
    else:
        mask = mask.astype("boolean")
        mask.iloc[0, 0] = pd.NA
    with pytest.raises(ConfigError):
        p.transform(local, mask=mask)


def test_fractional_integer_result_rejected():
    public, local = frames()
    local["x"] = pd.array([None, 1000, 2], dtype="Int64")
    with pytest.raises(ConfigError, match="Fractional"):
        MissingCompensation(config()).fit(fit_data=public).transform(local)


@pytest.mark.parametrize("method", ["median", "knn"])
def test_native_adapter_loads_external_only(tmp_path, method):
    public, local = frames()
    path = tmp_path / "public.csv"
    public.to_csv(path, index=False)
    cfg = config(method)
    cfg.update(
        {
            "method": "custom_method",
            "class_name": "MissingCompensation",
            "module_path": str(
                Path(__file__).parents[2] / "petsard/processor/missing_compensation.py"
            ),
            "fit_data": {"filepath": str(path)},
        }
    )
    adapter = PreprocessorAdapter(cfg)
    adapter.run({"data": local, "metadata": SchemaMetadater.from_data(local)})
    result = adapter.get_result()
    pd.testing.assert_frame_equal(
        result,
        MissingCompensation(config(method)).fit(fit_data=public).transform(local),
    )
    assert adapter.processor.get_artifacts()["reports"]["data"]["fit_rows"] == len(
        public
    )


@pytest.mark.parametrize(
    "option,value",
    [
        ("method", "unsupported"),
        ("n_neighbors", 3),
        ("weights", "distance"),
        ("metric", "euclidean"),
        ("scaler", True),
    ],
)
def test_reject_protocol_changes(option, value):
    cfg = config("knn")
    cfg["missing_compensation"][option] = value
    with pytest.raises(ConfigError):
        MissingCompensation(cfg)


@pytest.mark.parametrize("method", ["median", "knn"])
def test_native_composition_with_external_fit(tmp_path, method):
    from petsard.processor import CompositePreprocessor

    public, local = frames()
    path = tmp_path / "public.csv"
    public.to_csv(path, index=False)
    custom = config(method)
    custom.update(
        {
            "name": "compensation",
            "method": "custom_method",
            "module_path": str(
                Path(__file__).parents[2] / "petsard/processor/missing_compensation.py"
            ),
            "class_name": "MissingCompensation",
            "fit_data": {"filepath": str(path)},
            "key_columns": ["record_id", "subject_id"],
            "target_columns": ["y_malignancy"],
            "required_columns": ["x", "z"],
        }
    )
    stages = [
        {
            "name": "outlier",
            "method": "custom",
            "sequence": ["outlier"],
            "config": {"outlier": dict.fromkeys(local)},
        },
        custom,
        {
            "name": "missing",
            "method": "custom",
            "sequence": ["missing"],
            "config": {
                "missing": {
                    c: "missing_mean" if c in ["x", "z"] else None for c in local
                }
            },
        },
    ]
    processor = CompositePreprocessor(
        {"stages": stages}, SchemaMetadater.from_data(local)
    )
    actual = processor.fit_transform(local)
    expected = MissingCompensation(config(method)).fit(fit_data=public).transform(local)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(
        processor.transform(local), expected, check_exact=True
    )


def test_existing_builtin_sequence_unchanged():
    _, local = frames()
    cfg = {
        "method": "custom",
        "sequence": ["outlier", "missing"],
        "config": {
            "outlier": dict.fromkeys(local),
            "missing": {c: "missing_mean" if c in ["x", "z"] else None for c in local},
        },
    }
    adapter = PreprocessorAdapter(cfg)
    adapter.run({"data": local, "metadata": SchemaMetadater.from_data(local)})
    result = adapter.get_result()
    assert result["x"].notna().all() and result["z"].notna().all()
    assert adapter.processor.__class__.__name__ == "Processor"
    assert result["record_id"].tolist() == local["record_id"].tolist()


def test_same_file_source_alias_rejected(tmp_path):
    public, local = frames()
    path = tmp_path / "public.csv"
    public.to_csv(path, index=False)
    alias = tmp_path / "alias.csv"
    try:
        alias.symlink_to(path)
    except OSError as exc:
        pytest.skip(f"symbolic links are not available: {exc}")
    cfg = config()
    cfg.update({"fit_data": {"filepath": str(alias)}, "transform_source": str(path)})
    with pytest.raises(ConfigError, match="paths must be distinct"):
        MissingCompensation(cfg).fit(local)


def test_missing_protected_values_remain_unchanged():
    public, local = frames()
    local["y_unobserved"] = np.nan
    local["subject_id"] = pd.array([None, 21, 22], dtype="Int64")
    out = MissingCompensation(config()).fit(fit_data=public).transform(local)
    pd.testing.assert_frame_equal(
        out[["subject_id", "y_unobserved"]],
        local[["subject_id", "y_unobserved"]],
        check_exact=True,
    )


@pytest.mark.parametrize("key", ["key_columns", "target_columns", "protected_columns"])
@pytest.mark.parametrize("value", ["x", ["x"]])
def test_configured_protected_aliases_cannot_be_features(key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(ConfigError, match="features"):
        MissingCompensation(cfg)


@pytest.mark.parametrize("method", ["median", "knn"])
def test_configured_keys_targets_and_protected_values_are_untouched(method):
    public, local = frames()
    public["case_number"] = np.arange(len(public)) + 100
    local["case_number"] = [900, 901, 902]
    public["diagnosis"] = 1
    local["diagnosis"] = pd.array([1, None, 0], dtype="Int64")
    public["site"] = "shared-site"
    local["site"] = ["shared-site", None, "other-site"]
    cfg = config(method)
    cfg.update(
        {
            "key_columns": "case_number",
            "target_columns": "diagnosis",
            "protected_columns": "site",
        }
    )
    p = MissingCompensation(cfg).fit(fit_data=public)
    result = p.transform(local)
    columns = [
        "case_number",
        "diagnosis",
        "site",
        "record_id",
        "subject_id",
        "y_malignancy",
    ]
    pd.testing.assert_frame_equal(result[columns], local[columns], check_exact=True)
    assert p.get_report()["protected_changes"] == p.get_report()["label_changes"] == 0
    overlap = local.copy()
    overlap["case_number"] = [100, 901, 902]
    with pytest.raises(ConfigError, match="identity overlap"):
        p.transform(overlap)


@pytest.mark.parametrize("method", ["median", "knn"])
def test_missing_external_source_never_fits_local_data(tmp_path, method):
    from unittest.mock import patch

    _, local = frames()
    cfg = config(method)
    cfg["fit_data"] = {"filepath": str(tmp_path / "absent-public.csv")}
    p = MissingCompensation(cfg)
    with (
        patch.object(
            SimpleImputer, "fit", side_effect=AssertionError("local fit")
        ) as median_fit,
        patch.object(
            KNNImputer, "fit", side_effect=AssertionError("local fit")
        ) as knn_fit,
    ):
        with pytest.raises(Exception) as failure:
            p.fit(data=local)
        assert not isinstance(failure.value, AssertionError)
        median_fit.assert_not_called()
        knn_fit.assert_not_called()
    with pytest.raises(UnfittedError):
        p.transform(local)
