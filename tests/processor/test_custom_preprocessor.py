"""Tests for the stage-level custom Preprocessor extension point."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from petsard.exceptions import ConfigError
from petsard.executor import Executor
from petsard.processor import CustomPreprocessor

PLUGIN_SOURCE = '''
import pandas as pd


class ExamplePreprocessor:
    def __init__(self, config, metadata):
        self.offset = config.get("offset", 0)

    def fit(self, data):
        self.columns = list(data.columns)

    def transform(self, data):
        result = data.copy()
        result["value"] = result["value"] + self.offset
        return result
'''


SPLIT_PLUGIN_SOURCE = '''
import hashlib
import json


class FrozenSplitPreprocessor:
    def __init__(self, config, metadata):
        self.config = config
        self.fit_count = 0
        self.transform_count = 0
        self.fingerprint = None

    def fit(self, data):
        self.fit_count += 1
        payload = {
            "columns": list(data.columns),
            "train_values": data["value"].tolist(),
            "offset": self.config["offset"],
        }
        self.fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def transform(self, data):
        self.transform_count += 1
        result = data.copy(deep=True)
        result.loc[result["value"].isna(), "value"] = self.config["offset"]
        return result

    def get_report(self):
        return {
            "fit_count": self.fit_count,
            "transform_count": self.transform_count,
            "strategy_fingerprint": self.fingerprint,
        }

    def get_strategy_state(self):
        return {
            "frozen": True,
            "fingerprint": self.fingerprint,
            "fit_count": self.fit_count,
            "transform_count": self.transform_count,
        }
'''


ABSTAIN_PLUGIN_SOURCE = '''
import pandas as pd


class AbstainingPreprocessor:
    def __init__(self, config, metadata):
        self.fill_abstained = config.get("fill_abstained", False)
        self.log = pd.DataFrame()

    def fit(self, data):
        pass

    def transform(self, data):
        result = data.copy(deep=True)
        missing_positions = [
            position
            for position, missing in enumerate(data["value"].isna())
            if missing
        ]
        self.log = pd.DataFrame(
            [
                {
                    "row_position": position,
                    "column": "value",
                    "decision": "abstained",
                    "reason": "no_external_evidence",
                }
                for position in missing_positions
            ]
        )
        if self.fill_abstained:
            result.loc[result["value"].isna(), "value"] = 99
        return result

    def get_log(self):
        return self.log.copy(deep=True)

    def get_report(self):
        return {"abstained_count": len(self.log)}
'''


def _write_plugin(tmp_path: Path, source: str = PLUGIN_SOURCE) -> Path:
    path = tmp_path / "custom_preprocessor.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_loads_external_preprocessor_and_preserves_dataframe_boundary(tmp_path):
    plugin_path = _write_plugin(tmp_path)
    processor = CustomPreprocessor(
        config={
            "module_path": str(plugin_path),
            "class_name": "ExamplePreprocessor",
            "offset": 3,
        }
    )
    source = pd.DataFrame({"value": [1, 2]})

    processor.fit(source)
    result = processor.transform(source)

    assert result.equals(pd.DataFrame({"value": [4, 5]}))
    assert source.equals(pd.DataFrame({"value": [1, 2]}))


def test_irreversible_preprocessor_uses_identity_postprocessing(tmp_path):
    processor = CustomPreprocessor(
        config={
            "module_path": str(_write_plugin(tmp_path)),
            "class_name": "ExamplePreprocessor",
        }
    )
    transformed = pd.DataFrame({"value": [4, 5]})

    restored = processor.inverse_transform(transformed)

    assert restored.equals(transformed)
    assert restored is not transformed


def test_reversible_preprocessor_requires_inverse_transform(tmp_path):
    with pytest.raises(ConfigError, match="must implement inverse_transform"):
        CustomPreprocessor(
            config={
                "module_path": str(_write_plugin(tmp_path)),
                "class_name": "ExamplePreprocessor",
                "reversible": True,
            }
        )


def test_transform_rejects_non_dataframe_result(tmp_path):
    source = PLUGIN_SOURCE.replace("return result", "return result.to_dict()")
    processor = CustomPreprocessor(
        config={
            "module_path": str(_write_plugin(tmp_path, source)),
            "class_name": "ExamplePreprocessor",
        }
    )
    data = pd.DataFrame({"value": [1]})
    processor.fit(data)

    with pytest.raises(ConfigError, match="must return a pandas DataFrame"):
        processor.transform(data)


def test_executor_runs_custom_preprocessor_from_yaml(tmp_path):
    plugin_path = _write_plugin(tmp_path)
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"value": [1, 2]}).to_csv(input_path, index=False)
    config = {
        "Executor": {"log_output_type": "stdout", "log_level": "ERROR"},
        "Loader": {"load": {"filepath": str(input_path)}},
        "Preprocessor": {
            "external": {
                "method": "custom_method",
                "module_path": str(plugin_path),
                "class_name": "ExamplePreprocessor",
                "offset": 3,
            }
        },
    }

    executor = Executor(yaml.safe_dump(config, sort_keys=False))
    executor.run()

    result = next(iter(executor.get_result().values()))
    assert result.equals(pd.DataFrame({"value": [4, 5]}))
    assert type(executor.status.get_processor()).__name__ == "CustomPreprocessor"


def test_executor_fits_train_once_and_transforms_train_and_control(tmp_path):
    plugin_path = _write_plugin(tmp_path, SPLIT_PLUGIN_SOURCE)
    train_path = tmp_path / "train.csv"
    control_path = tmp_path / "control.csv"
    pd.DataFrame({"value": [1.0, None]}).to_csv(train_path, index=False)
    pd.DataFrame({"value": [999.0, None]}).to_csv(control_path, index=False)
    config = {
        "Executor": {"log_output_type": "stdout", "log_level": "ERROR"},
        "Splitter": {
            "split": {
                "method": "custom_data",
                "filepath": {
                    "ori": str(train_path),
                    "control": str(control_path),
                },
            }
        },
        "Preprocessor": {
            "external": {
                "method": "custom_method",
                "module_path": str(plugin_path),
                "class_name": "FrozenSplitPreprocessor",
                "offset": 7,
            }
        },
    }

    executor = Executor(yaml.safe_dump(config, sort_keys=False))
    executor.run()

    result = executor.status.get_result("Preprocessor")
    processor = executor.status.get_processor()
    assert set(result) == {"train", "validation"}
    assert result["train"]["value"].tolist() == [1.0, 7.0]
    assert result["validation"]["value"].tolist() == [999.0, 7.0]
    assert processor._impl.fit_count == 1
    assert processor._impl.transform_count == 2
    assert set(processor.get_split_reports()) == {"train", "validation"}
    states = processor.get_split_strategy_states()
    assert states["train"]["fingerprint"] == states["validation"]["fingerprint"]


def test_preserve_missing_abstention_is_verified_and_exposed(tmp_path):
    processor = CustomPreprocessor(
        config={
            "module_path": str(_write_plugin(tmp_path, ABSTAIN_PLUGIN_SOURCE)),
            "class_name": "AbstainingPreprocessor",
            "abstain_action": "preserve_missing",
        }
    )
    source = pd.DataFrame({"value": [1.0, None]}, index=[10, 20])

    processor.fit(source)
    completed = processor.transform(source)

    pd.testing.assert_frame_equal(completed, source)
    artifacts = processor.get_artifacts()
    summary = artifacts["abstention_summaries"]["data"]
    assert artifacts["abstain_action"] == "preserve_missing"
    assert artifacts["native_fallback_applied"] is False
    assert summary == {
        "action": "preserve_missing",
        "abstained_count": 1,
        "missing_preserved_count": 1,
        "all_abstained_cells_remain_missing": True,
        "native_fallback_applied": False,
        "reason_counts": {"no_external_evidence": 1},
        "column_counts": {"value": 1},
        "cells": [
            {
                "row_position": 1,
                "column": "value",
                "reason": "no_external_evidence",
            }
        ],
    }


def test_preserve_missing_rejects_hidden_fallback_write(tmp_path):
    processor = CustomPreprocessor(
        config={
            "module_path": str(_write_plugin(tmp_path, ABSTAIN_PLUGIN_SOURCE)),
            "class_name": "AbstainingPreprocessor",
            "abstain_action": "preserve_missing",
            "fill_abstained": True,
        }
    )
    source = pd.DataFrame({"value": [None]})
    processor.fit(source)

    with pytest.raises(ConfigError, match="preserve_missing was violated"):
        processor.transform(source)


def test_rejects_native_fallback_abstain_action(tmp_path):
    with pytest.raises(ConfigError, match="native fallback is not implemented"):
        CustomPreprocessor(
            config={
                "module_path": str(_write_plugin(tmp_path, ABSTAIN_PLUGIN_SOURCE)),
                "class_name": "AbstainingPreprocessor",
                "abstain_action": "native_fallback",
            }
        )


def test_tracked_abstention_requires_plugin_log(tmp_path):
    with pytest.raises(ConfigError, match="requires.*get_log"):
        CustomPreprocessor(
            config={
                "module_path": str(_write_plugin(tmp_path)),
                "class_name": "ExamplePreprocessor",
                "abstain_action": "preserve_missing",
            }
        )


def test_executor_stores_tracked_abstention_in_status_and_snapshot(tmp_path):
    plugin_path = _write_plugin(tmp_path, ABSTAIN_PLUGIN_SOURCE)
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"value": [1.0, None]}).to_csv(input_path, index=False)
    config = {
        "Executor": {"log_output_type": "stdout", "log_level": "ERROR"},
        "Loader": {"load": {"filepath": str(input_path)}},
        "Preprocessor": {
            "tracked": {
                "method": "custom_method",
                "module_path": str(plugin_path),
                "class_name": "AbstainingPreprocessor",
                "abstain_action": "preserve_missing",
            }
        },
    }

    executor = Executor(yaml.safe_dump(config, sort_keys=False))
    executor.run()

    artifacts = executor.get_preprocessor_artifacts("tracked")
    summary = artifacts["abstention_summaries"]["data"]
    assert summary["abstained_count"] == 1
    assert summary["all_abstained_cells_remain_missing"] is True
    assert artifacts["native_fallback_applied"] is False
    assert isinstance(artifacts["logs"]["data"], pd.DataFrame)
    snapshots = executor.status.get_snapshots("Preprocessor")
    assert snapshots[-1].context["abstain_action"] == "preserve_missing"
    assert snapshots[-1].context["abstained_count"] == 1
    assert snapshots[-1].context["native_fallback_applied"] is False
    completed = executor.status.get_result("Preprocessor")
    assert pd.isna(completed.loc[1, "value"])

    artifacts["abstention_summaries"]["data"]["abstained_count"] = 999
    fresh = executor.get_preprocessor_artifacts("tracked")
    assert fresh["abstention_summaries"]["data"]["abstained_count"] == 1
