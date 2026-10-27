from pathlib import Path

import pandas as pd
import pytest
import yaml

from petsard.adapter import PreprocessorAdapter
from petsard.exceptions import ConfigError
from petsard.executor import Executor
from petsard.metadater.metadater import SchemaMetadater
from petsard.processor import CompositePreprocessor

PLUGIN_SOURCE = """
import hashlib
import json
import pandas as pd


class AbstainingStage:
    def __init__(self, config, metadata):
        self.fit_count = 0
        self.transform_count = 0
        self.fit_rows = None
        self.fingerprint = None
        self.log = pd.DataFrame()

    def fit(self, data):
        self.fit_count += 1
        self.fit_rows = len(data)
        self.fingerprint = hashlib.sha256(
            json.dumps({"rows": len(data), "columns": list(data.columns)}, sort_keys=True).encode()
        ).hexdigest()

    def transform(self, data):
        self.transform_count += 1
        result = data.copy(deep=True)
        self.log = pd.DataFrame([
            {
                "row_position": position,
                "column": "value",
                "decision": "abstained",
                "reason": "mvp_fixture",
            }
            for position, missing in enumerate(data["value"].isna())
            if missing
        ])
        return result

    def get_log(self):
        return self.log.copy(deep=True)

    def get_report(self):
        return {
            "fit_count": self.fit_count,
            "transform_count": self.transform_count,
            "fit_rows": self.fit_rows,
        }

    def get_strategy_state(self):
        return {
            "frozen": True,
            "fingerprint": self.fingerprint,
            "fit_count": self.fit_count,
            "transform_count": self.transform_count,
        }


class SchemaChangingStage(AbstainingStage):
    def transform(self, data):
        result = super().transform(data)
        return result.assign(added_column=1)


class IndexChangingStage(AbstainingStage):
    def transform(self, data):
        result = super().transform(data)
        result.index = range(100, 100 + len(result))
        return result


class MetadataProbeSynthesizer:
    def __init__(self, config, metadata):
        self.metadata_columns = list(metadata.attributes)
        self.fit_columns = None
        self.data = None

    def fit(self, data):
        self.fit_columns = list(data.columns)
        self.data = data.copy(deep=True)

    def sample(self):
        return self.data.copy(deep=True)
"""


def _plugin(tmp_path: Path) -> Path:
    path = tmp_path / "composition_plugin.py"
    path.write_text(PLUGIN_SOURCE, encoding="utf-8")
    return path


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "key": ["a", "b", "c", "d"],
            "value": [1.0, None, 2.0, None],
            "score": [1.0, 2.0, 3.0, 100.0],
        }
    )


def _stages(plugin_path: Path) -> list[dict]:
    return [
        {
            "name": "native_outlier",
            "method": "custom",
            "sequence": ["outlier"],
            "config": {
                "outlier": {
                    "key": None,
                    "value": None,
                    "score": "outlier_iqr",
                }
            },
        },
        {
            "name": "missing_compensation",
            "method": "custom_method",
            "module_path": str(plugin_path),
            "class_name": "AbstainingStage",
            "abstain_action": "preserve_missing",
        },
        {
            "name": "native_missing_fallback",
            "method": "custom",
            "sequence": ["missing"],
            "config": {
                "missing": {
                    "key": None,
                    "value": "missing_mean",
                    "score": None,
                }
            },
        },
    ]


def _custom_stage(plugin_path: Path) -> dict:
    return {
        "name": "missing_compensation",
        "method": "custom_method",
        "module_path": str(plugin_path),
        "class_name": "AbstainingStage",
        "abstain_action": "preserve_missing",
    }


def _processor(
    tmp_path: Path, data: pd.DataFrame | None = None
) -> CompositePreprocessor:
    source = _data() if data is None else data
    metadata = SchemaMetadater.from_data(source)
    return CompositePreprocessor(
        config={"stages": _stages(_plugin(tmp_path))}, metadata=metadata
    )


def test_composes_outlier_custom_and_explicit_missing_fallback(tmp_path):
    processor = _processor(tmp_path)

    result = processor.fit_transform(_data())

    assert result["key"].tolist() == ["a", "b", "c"]
    assert result["value"].tolist() == [1.0, 1.5, 2.0]
    artifacts = processor.get_artifacts()
    assert [stage["processor_class"] for stage in artifacts["stages"]] == [
        "Processor",
        "CustomPreprocessor",
        "Processor",
    ]
    assert artifacts["fit_count"] == 1
    assert artifacts["transform_count"] == 1
    assert artifacts["reports"]["train"]["fit_rows"] == 3
    assert artifacts["native_fallback_applied"] is True
    assert (
        artifacts["abstention_summaries"]["train"]["resolved_by_later_stage_count"] == 1
    )


def test_splitter_fits_progressive_train_once_and_reuses_frozen_chain(tmp_path):
    processor = _processor(tmp_path)
    validation = pd.DataFrame(
        {"key": ["x", "y", "z"], "value": [5.0, None, 5.0], "score": [1.0, 2.0, 3.0]}
    )

    result = processor.fit_transform_splits(
        {"train": _data(), "validation": validation}
    )

    assert result["train"]["key"].tolist() == ["a", "b", "c"]
    assert result["validation"]["value"].tolist() == [5.0, 1.5, 5.0]
    custom = processor._stages[1]["processor"]
    assert custom._impl.fit_count == 1
    assert custom._impl.transform_count == 2
    artifacts = processor.get_artifacts()
    assert artifacts["fit_count"] == 1
    assert artifacts["transform_count"] == 2
    assert set(artifacts["stage_runs"]) == {"train", "validation"}
    states = artifacts["strategy_states"]
    assert states["train"]["fingerprint"] == states["validation"]["fingerprint"]
    assert (
        artifacts["abstention_summaries"]["validation"]["resolved_by_later_stage_count"]
        == 1
    )


@pytest.mark.parametrize("method", ["outlier_isolationforest", "outlier_lof"])
def test_global_outlier_fits_train_once_and_predicts_later_splits(tmp_path, method):
    train = pd.DataFrame(
        {
            "key": [f"train-{index}" for index in range(30)],
            "value": [float(index % 7) for index in range(30)],
            "score": [float(index) for index in range(29)] + [1000.0],
        }
    )
    validation = pd.DataFrame(
        {
            "key": [f"validation-{index}" for index in range(25)],
            "value": [float(index % 7) for index in range(25)],
            "score": [float(index) for index in range(25)],
        }
    )
    control = validation.assign(
        key=[f"control-{index}" for index in range(len(validation))]
    )
    custom_stage = _custom_stage(_plugin(tmp_path))
    custom_stage.pop("abstain_action")
    stages = [
        {
            "name": "native_outlier",
            "method": "custom",
            "sequence": ["outlier"],
            "config": {"outlier": {"key": None, "value": None, "score": method}},
        },
        custom_stage,
    ]
    processor = CompositePreprocessor(
        config={"stages": stages}, metadata=SchemaMetadater.from_data(train)
    )

    processor.fit_transform_splits(
        {"train": train, "validation": validation, "control": control}
    )

    native = processor._stages[0]["processor"]
    mediator = native._mediator["outlier"]
    assert mediator.fit_count == 1
    custom = processor._stages[1]["processor"]._impl
    assert custom.fit_count == 1
    assert custom.transform_count == 3
    states = processor.get_artifacts()["strategy_states"]
    assert len({state["fingerprint"] for state in states.values()}) == 1


def test_executor_runs_composite_preprocessor_from_yaml(tmp_path):
    input_path = tmp_path / "input.csv"
    _data().to_csv(input_path, index=False)
    config = {
        "Executor": {"log_output_type": "stdout", "log_level": "ERROR"},
        "Loader": {"load": {"filepath": str(input_path)}},
        "Preprocessor": {
            "composition": {
                "method": "composite",
                "stages": _stages(_plugin(tmp_path)),
            }
        },
    }

    executor = Executor(yaml.safe_dump(config, sort_keys=False))
    executor.run()

    result = next(iter(executor.get_result().values()))
    assert result["key"].tolist() == ["a", "b", "c"]
    assert result["value"].tolist() == [1.0, 1.5, 2.0]
    assert type(executor.status.get_processor()).__name__ == "CompositePreprocessor"
    artifacts = executor.get_preprocessor_artifacts("composition")
    assert artifacts["native_fallback_applied"] is True


def test_executor_composition_preserves_split_roles(tmp_path):
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    _data().to_csv(train_path, index=False)
    pd.DataFrame(
        {"key": ["x", "y", "z"], "value": [5.0, None, 5.0], "score": [1.0, 2.0, 3.0]}
    ).to_csv(validation_path, index=False)
    config = {
        "Executor": {"log_output_type": "stdout", "log_level": "ERROR"},
        "Splitter": {
            "split": {
                "method": "custom_data",
                "filepath": {"ori": str(train_path), "control": str(validation_path)},
            }
        },
        "Preprocessor": {
            "composition": {
                "method": "composite",
                "stages": _stages(_plugin(tmp_path)),
            }
        },
    }

    executor = Executor(yaml.safe_dump(config, sort_keys=False))
    executor.run()

    result = next(iter(executor.get_result().values()))
    assert set(result) == {"train", "validation"}
    assert result["validation"]["value"].tolist() == [5.0, 1.5, 5.0]
    artifacts = executor.get_preprocessor_artifacts("composition")
    assert artifacts["fit_count"] == 1
    assert artifacts["transform_count"] == 2


@pytest.mark.parametrize(
    ("stages", "message"),
    [
        ([], "only supports"),
        (
            [
                {"name": "one", "method": "custom", "sequence": ["missing"]},
                {"name": "two", "method": "custom", "sequence": ["outlier"]},
            ],
            "exactly one custom_method",
        ),
        (
            [
                {"name": "same", "method": "custom", "sequence": ["missing"]},
                {"name": "same", "method": "custom_method"},
            ],
            "Duplicate",
        ),
        (
            [
                {"name": "native", "method": "custom", "sequence": "missing"},
                {"name": "external", "method": "custom_method"},
            ],
            "non-empty sequence list",
        ),
        (
            [
                {"name": "external", "method": "custom_method"},
                {"name": "late_outlier", "method": "custom", "sequence": ["outlier"]},
            ],
            "only supports",
        ),
        (
            [
                {"name": "encoder", "method": "custom", "sequence": ["encoder"]},
                {"name": "external", "method": "custom_method"},
            ],
            "only built-in missing/outlier",
        ),
    ],
)
def test_rejects_invalid_composition(stages, message):
    with pytest.raises(ConfigError, match=message):
        CompositePreprocessor(config={"stages": stages})


@pytest.mark.parametrize(
    "sequences",
    [
        [],
        [["outlier"]],
        [["missing"]],
        [["outlier"], ["missing"]],
    ],
)
def test_accepts_only_canonical_optional_orderings(tmp_path, sequences):
    custom = _custom_stage(_plugin(tmp_path))
    stages = [custom]
    if ["outlier"] in sequences:
        stages.insert(
            0,
            {
                "name": "native_outlier",
                "method": "custom",
                "sequence": ["outlier"],
            },
        )
    if ["missing"] in sequences:
        stages.append(
            {
                "name": "native_missing",
                "method": "custom",
                "sequence": ["missing"],
            }
        )

    validated = CompositePreprocessor.validate_config({"stages": stages})

    assert [
        stage.get("sequence") for stage in validated if "sequence" in stage
    ] == sequences


def test_custom_only_composition_executes(tmp_path):
    processor = CompositePreprocessor(
        config={"stages": [_custom_stage(_plugin(tmp_path))]},
        metadata=SchemaMetadater.from_data(_data()),
    )

    result = processor.fit_transform(_data())

    pd.testing.assert_frame_equal(result, _data())
    assert processor.get_artifacts()["fit_count"] == 1


@pytest.mark.parametrize(
    "stages",
    [
        [
            {"name": "missing", "method": "custom", "sequence": ["missing"]},
            {"name": "external", "method": "custom_method"},
        ],
        [
            {
                "name": "mixed",
                "method": "custom",
                "sequence": ["outlier", "missing"],
            },
            {"name": "external", "method": "custom_method"},
        ],
        [
            {"name": "outlier", "method": "custom", "sequence": ["outlier"]},
            {"name": "missing", "method": "custom", "sequence": ["missing"]},
            {"name": "external", "method": "custom_method"},
        ],
        [
            {"name": "external", "method": "custom_method"},
            {"name": "missing", "method": "custom", "sequence": ["missing"]},
            {"name": "outlier", "method": "custom", "sequence": ["outlier"]},
        ],
    ],
)
def test_adapter_construction_rejects_noncanonical_ordering(stages):
    with pytest.raises(ConfigError, match="only supports"):
        PreprocessorAdapter({"method": "composite", "stages": stages})


def test_adapter_rejects_composite_top_level_sequence():
    with pytest.raises(ConfigError, match="top-level sequence"):
        PreprocessorAdapter(
            {"method": "composite", "stages": [{}, {}], "sequence": ["missing"]}
        )


def test_rejects_schema_changing_custom_stage(tmp_path):
    stages = _stages(_plugin(tmp_path))
    stages[1]["class_name"] = "SchemaChangingStage"
    processor = CompositePreprocessor(
        config={"stages": stages}, metadata=SchemaMetadater.from_data(_data())
    )

    with pytest.raises(ConfigError, match="preserve DataFrame shape"):
        processor.fit_transform(_data())


def test_rejects_index_changing_custom_stage(tmp_path):
    stages = [_custom_stage(_plugin(tmp_path))]
    stages[0]["class_name"] = "IndexChangingStage"
    processor = CompositePreprocessor(
        config={"stages": stages}, metadata=SchemaMetadater.from_data(_data())
    )

    with pytest.raises(ConfigError, match="preserve.*DataFrame index"):
        processor.fit_transform(_data())


def test_protects_constant_custom_key_and_target_columns(tmp_path):
    data = pd.DataFrame(
        {
            "key": ["same"] * 4,
            "value": [1.0, None, 1.0, None],
            "score": [1.0, 2.0, 3.0, 4.0],
        }
    )
    stages = _stages(_plugin(tmp_path))
    stages[1]["key_columns"] = ["key"]
    stages[1]["target_columns"] = ["value"]
    processor = CompositePreprocessor(
        config={"stages": stages}, metadata=SchemaMetadater.from_data(data)
    )

    result = processor.fit_transform(data)

    assert list(result.columns) == ["key", "value", "score"]
    assert result["key"].tolist() == ["same"] * 4
    assert result["value"].tolist() == [1.0, 1.0, 1.0, 1.0]
    for stage in processor._stages:
        assert list(stage["processor"]._metadata.attributes) == list(result.columns)


def test_unprotected_constant_removal_updates_metadata(tmp_path):
    data = _data().assign(unused_constant="unused")
    stages = _stages(_plugin(tmp_path))
    stages[1]["required_columns"] = ["key", "value"]
    processor = CompositePreprocessor(
        config={"stages": stages}, metadata=SchemaMetadater.from_data(data)
    )

    result = processor.fit_transform(data)

    assert "unused_constant" not in result
    assert list(processor._metadata.attributes) == list(result.columns)
    for run in processor.get_artifacts()["stage_runs"]["train"]:
        assert run["metadata_columns"] == run["output_columns"]


def test_downstream_synthesizer_receives_matching_dataframe_and_metadata(tmp_path):
    data = _data().assign(unused_constant="unused")
    input_path = tmp_path / "input.csv"
    data.to_csv(input_path, index=False)
    plugin_path = _plugin(tmp_path)
    stages = _stages(plugin_path)
    stages[1]["required_columns"] = ["key", "value"]
    config = {
        "Executor": {"log_output_type": "stdout", "log_level": "ERROR"},
        "Loader": {"load": {"filepath": str(input_path)}},
        "Preprocessor": {"composition": {"method": "composite", "stages": stages}},
        "Synthesizer": {
            "probe": {
                "method": "custom_method",
                "module_path": str(plugin_path),
                "class_name": "MetadataProbeSynthesizer",
            }
        },
    }

    executor = Executor(yaml.safe_dump(config, sort_keys=False))
    executor.run()

    implementation = executor.status.get_synthesizer()._impl._impl
    assert implementation.fit_columns == ["key", "value", "score"]
    assert implementation.metadata_columns == implementation.fit_columns
    assert list(executor.status.get_metadata("Preprocessor").attributes) == (
        implementation.fit_columns
    )


def test_artifacts_are_defensive_copies(tmp_path):
    processor = _processor(tmp_path)
    processor.fit_transform(_data())

    artifacts = processor.get_artifacts()
    artifacts["stages"][0]["name"] = "mutated"
    artifacts["logs"]["train"].loc[:, "decision"] = "mutated"

    fresh = processor.get_artifacts()
    assert fresh["stages"][0]["name"] == "native_outlier"
    assert set(fresh["logs"]["train"]["decision"]) == {"abstained"}
