"""Ordered composition of built-in and external preprocessing stages."""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from typing import Any

import pandas as pd

from petsard.exceptions import ConfigError
from petsard.metadater import Schema, SchemaMetadater
from petsard.processor.base import Processor
from petsard.processor.custom_preprocessor import CustomPreprocessor


class CompositePreprocessor:
    """Fit and execute an explicit chain of Preprocessor implementations.

    This MVP supports exactly one stage-level ``custom_method`` with optional
    built-in outlier before it and optional built-in missing fallback after it.
    Every stage is fitted on the train output produced by the preceding stage.
    The fitted chain is then reused, without refit, for validation/control.
    """

    BUILTIN_METHODS = frozenset({"default", "custom"})
    CUSTOM_METHOD = "custom_method"
    ORDERING_ERROR = (
        "CompositePreprocessor MVP only supports: "
        "[outlier] -> custom_method -> [missing]"
    )

    def __init__(self, config: dict, metadata: Schema = None) -> None:
        self.config = deepcopy(config)
        self._input_metadata = deepcopy(metadata)
        self._metadata = deepcopy(metadata)
        self._stage_specs = self.validate_config(self.config)
        self._configured_protected_columns = self._get_custom_required_columns(
            self._stage_specs
        )
        self._protected_columns = set(self._configured_protected_columns)
        self._stages: list[dict[str, Any]] = []
        self._is_fitted = False
        self._fit_count = 0
        self._transform_count = 0
        self._fit_output: pd.DataFrame | None = None
        self._logs: dict[str, pd.DataFrame] = {}
        self._reports: dict[str, dict] = {}
        self._strategy_states: dict[str, dict] = {}
        self._abstention_summaries: dict[str, dict] = {}
        self._stage_runs: dict[str, list[dict[str, Any]]] = {}
        self._native_fallback_applied = False

    @classmethod
    def validate_config(cls, config: Any) -> list[dict[str, Any]]:
        """Validate the complete MVP shape without requiring data or metadata."""
        if not isinstance(config, dict):
            raise ConfigError("Composite Preprocessor configuration must be a mapping.")
        return cls._validate_stages(config.get("stages"))

    @classmethod
    def _validate_stages(cls, stages: Any) -> list[dict[str, Any]]:
        if not isinstance(stages, list) or not 1 <= len(stages) <= 3:
            raise ConfigError(cls.ORDERING_ERROR)

        validated: list[dict[str, Any]] = []
        names: set[str] = set()
        custom_count = 0
        custom_position = None
        for position, raw_stage in enumerate(stages):
            if not isinstance(raw_stage, dict):
                raise ConfigError(
                    f"Composite Preprocessor stage {position} must be a mapping."
                )
            stage = deepcopy(raw_stage)
            name = stage.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ConfigError(
                    f"Composite Preprocessor stage {position} requires a name."
                )
            if name in names:
                raise ConfigError(f"Duplicate composite Preprocessor stage: {name}")
            names.add(name)

            method = str(stage.get("method", "custom")).lower()
            if method not in cls.BUILTIN_METHODS | {cls.CUSTOM_METHOD}:
                raise ConfigError(
                    f"Unsupported composite Preprocessor stage method: {method}"
                )
            stage["method"] = method
            if method == cls.CUSTOM_METHOD:
                custom_count += 1
                custom_position = position
                if "sequence" in stage:
                    raise ConfigError(
                        "A custom_method composite stage cannot define a built-in "
                        "sequence."
                    )
            elif "sequence" not in stage:
                raise ConfigError(
                    f"Built-in composite stage '{name}' requires an explicit sequence."
                )
            elif not isinstance(stage["sequence"], list) or not stage["sequence"]:
                raise ConfigError(
                    f"Built-in composite stage '{name}' requires a non-empty "
                    "sequence list."
                )
            elif not set(stage["sequence"]).issubset({"missing", "outlier"}):
                raise ConfigError(
                    "The composition MVP supports only built-in missing/outlier stages."
                )
            validated.append(stage)

        if custom_count != 1:
            raise ConfigError(
                "The composition MVP requires exactly one custom_method stage."
            )
        assert custom_position is not None
        before_custom = validated[:custom_position]
        after_custom = validated[custom_position + 1 :]
        if len(before_custom) > 1 or any(
            stage["sequence"] != ["outlier"] for stage in before_custom
        ):
            raise ConfigError(cls.ORDERING_ERROR)
        if len(after_custom) > 1 or any(
            stage["sequence"] != ["missing"] for stage in after_custom
        ):
            raise ConfigError(cls.ORDERING_ERROR)
        return validated

    @classmethod
    def _get_custom_required_columns(cls, stages: list[dict[str, Any]]) -> set[str]:
        custom = next(stage for stage in stages if stage["method"] == cls.CUSTOM_METHOD)
        protected: set[str] = set()
        for key in (
            "row_id_column",
            "key_columns",
            "target_columns",
            "required_columns",
        ):
            value = custom.get(key)
            if isinstance(value, str):
                protected.add(value)
            elif isinstance(value, list | tuple | set):
                protected.update(item for item in value if isinstance(item, str))
        return protected

    def _metadata_for_stage(self, spec: dict[str, Any], metadata: Schema) -> Schema:
        """Protect custom-required columns from built-in constant removal."""
        prepared = deepcopy(metadata)
        if spec["method"] in self.BUILTIN_METHODS:
            for column in self._protected_columns:
                if column in prepared.attributes:
                    prepared.attributes[column].is_constant = False
        return prepared

    @staticmethod
    def _align_metadata(metadata: Schema, data: pd.DataFrame) -> Schema:
        """Rebuild field metadata from the actual stage output schema."""
        unknown = [
            column for column in data.columns if column not in metadata.attributes
        ]
        if unknown:
            raise ConfigError(
                "Composite stage produced columns absent from metadata: "
                + ", ".join(unknown)
            )
        aligned = deepcopy(metadata)
        recalculated = SchemaMetadater.from_data(
            data,
            enable_stats=aligned.enable_stats,
            base_schema=aligned,
            id=aligned.id,
            name=aligned.name,
            description=aligned.description,
        )
        aligned.attributes = recalculated.attributes
        aligned.stats = recalculated.stats
        return aligned

    @staticmethod
    def _assert_metadata_matches_data(
        metadata: Schema, data: pd.DataFrame, stage_name: str
    ) -> None:
        metadata_columns = list(metadata.attributes)
        data_columns = list(data.columns)
        if metadata_columns != data_columns:
            raise ConfigError(
                f"Composite stage '{stage_name}' metadata schema does not match "
                f"its DataFrame columns: metadata={metadata_columns}, "
                f"data={data_columns}"
            )

    def _assert_custom_required_columns(
        self, data: pd.DataFrame, stage_name: str
    ) -> None:
        missing = sorted(self._protected_columns - set(data.columns))
        if missing:
            raise ConfigError(
                f"Composite stage '{stage_name}' removed custom-required columns: "
                + ", ".join(missing)
            )

    def _resolve_protected_columns(self, data: pd.DataFrame) -> None:
        """Resolve implicit missing targets before any built-in stage runs."""
        self._protected_columns = set(self._configured_protected_columns)
        custom = next(
            stage
            for stage in self._stage_specs
            if stage["method"] == self.CUSTOM_METHOD
        )
        if not custom.get("target_columns"):
            self._protected_columns.update(
                column for column in data.columns if data[column].isna().any()
            )

    def _build_stage(
        self, spec: dict[str, Any], metadata: Schema
    ) -> Processor | CustomPreprocessor:
        method = spec["method"]
        if method == self.CUSTOM_METHOD:
            custom_config = {
                key: deepcopy(value)
                for key, value in spec.items()
                if key not in {"name", "method", "sequence"}
            }
            return CustomPreprocessor(config=custom_config, metadata=metadata)

        if method == "default":
            processor_config = {}
        else:
            processor_config = deepcopy(spec.get("config", {}))
        return Processor(
            metadata=metadata,
            config=processor_config,
            freeze_global_outlier=True,
        )

    @staticmethod
    def _fit_stage(
        processor: Processor | CustomPreprocessor,
        spec: dict[str, Any],
        data: pd.DataFrame,
    ) -> None:
        if isinstance(processor, CustomPreprocessor):
            processor.fit(data=data)
        else:
            processor.fit(data=data, sequence=deepcopy(spec["sequence"]))

    @staticmethod
    def _transform_stage(
        processor: Processor | CustomPreprocessor, data: pd.DataFrame
    ) -> pd.DataFrame:
        transformed = processor.transform(data=data)
        if not isinstance(transformed, pd.DataFrame):
            raise ConfigError(
                "Every composite Preprocessor stage must return a pandas DataFrame."
            )
        return transformed

    def fit(self, data: pd.DataFrame) -> None:
        """Fit each stage on the preceding stage's transformed train output."""
        if not isinstance(data, pd.DataFrame):
            raise ConfigError("Composite Preprocessor fit(data) requires a DataFrame.")

        self._reset_runtime_state()
        self._resolve_protected_columns(data)
        current = data.copy(deep=True)
        current_metadata = deepcopy(self._input_metadata)
        for spec in self._stage_specs:
            stage_metadata = self._metadata_for_stage(spec, current_metadata)
            processor = self._build_stage(spec, stage_metadata)
            self._fit_stage(processor, spec, current)
            before = current
            current = self._transform_stage(processor, current)
            if isinstance(processor, CustomPreprocessor):
                self._assert_custom_boundary(before, current)
            self._assert_custom_required_columns(current, spec["name"])
            current_metadata = self._align_metadata(processor._metadata, current)
            self._assert_metadata_matches_data(current_metadata, current, spec["name"])
            processor._metadata = deepcopy(current_metadata)
            self._stages.append(
                {
                    "name": spec["name"],
                    "method": spec["method"],
                    "sequence": deepcopy(spec.get("sequence")),
                    "processor": processor,
                }
            )
            self._capture_stage_artifacts(spec["name"], processor, "train")
            self._record_stage_run(spec, processor, "train", before, current)

        self._metadata = current_metadata
        self._fit_output = current.copy(deep=True)
        self._is_fitted = True
        self._fit_count += 1
        self._transform_count += 1
        self._finalize_abstentions("train", current)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the chain and return its already-computed train output."""
        self.fit(data)
        assert self._fit_output is not None
        return self._fit_output.copy(deep=True)

    def transform(
        self, data: pd.DataFrame, *, split_name: str = "data"
    ) -> pd.DataFrame:
        """Transform one frame with every already-fitted stage."""
        if not self._is_fitted:
            raise ConfigError(
                "Composite Preprocessor must be fitted before transform()."
            )
        if not isinstance(data, pd.DataFrame):
            raise ConfigError(
                "Composite Preprocessor transform(data) requires a DataFrame."
            )

        current = data.copy(deep=True)
        for stage, spec in zip(self._stages, self._stage_specs, strict=True):
            before = current
            processor = stage["processor"]
            current = self._transform_stage(processor, current)
            if isinstance(processor, CustomPreprocessor):
                self._assert_custom_boundary(before, current)
            self._assert_custom_required_columns(current, stage["name"])
            self._assert_metadata_matches_data(
                processor._metadata, current, stage["name"]
            )
            self._capture_stage_artifacts(stage["name"], processor, split_name)
            self._record_stage_run(spec, processor, split_name, before, current)
        self._transform_count += 1
        self._finalize_abstentions(split_name, current)
        return current

    def fit_transform_splits(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Fit on train, then reuse the fitted chain for all other branches."""
        if not isinstance(data, dict) or "train" not in data:
            raise ConfigError(
                "Splitter-aware composite preprocessing requires a train split."
            )
        if any(not isinstance(frame, pd.DataFrame) for frame in data.values()):
            raise ConfigError("All composite Preprocessor splits must be DataFrames.")

        transformed = {"train": self.fit_transform(data["train"])}
        transformed.update(
            {
                name: self.transform(frame, split_name=name)
                for name, frame in data.items()
                if name != "train"
            }
        )
        return transformed

    @staticmethod
    def _assert_custom_boundary(before: pd.DataFrame, after: pd.DataFrame) -> None:
        if before.shape != after.shape:
            raise ConfigError(
                "A custom_method composite stage must preserve DataFrame shape."
            )
        if list(before.columns) != list(after.columns):
            raise ConfigError(
                "A custom_method composite stage must preserve column order."
            )
        if not before.index.equals(after.index):
            raise ConfigError(
                "A custom_method composite stage must preserve the DataFrame index."
            )

    def _capture_stage_artifacts(
        self,
        stage_name: str,
        processor: Processor | CustomPreprocessor,
        split_name: str,
    ) -> None:
        if not isinstance(processor, CustomPreprocessor):
            return
        artifacts = processor.get_artifacts()
        source_name = "data"
        logs = artifacts.get("logs", {})
        reports = artifacts.get("reports", {})
        states = artifacts.get("strategy_states", {})
        if source_name in logs:
            self._logs[split_name] = logs[source_name].copy(deep=True)
        if source_name in reports:
            self._reports[split_name] = deepcopy(reports[source_name])
        if source_name in states:
            self._strategy_states[split_name] = deepcopy(states[source_name])

    def _record_stage_run(
        self,
        spec: dict[str, Any],
        processor: Processor | CustomPreprocessor,
        split_name: str,
        before: pd.DataFrame,
        after: pd.DataFrame,
    ) -> None:
        self._stage_runs.setdefault(split_name, []).append(
            {
                "name": spec["name"],
                "method": spec["method"],
                "processor_class": type(processor).__name__,
                "sequence": deepcopy(spec.get("sequence")),
                "input_shape": list(before.shape),
                "output_shape": list(after.shape),
                "input_columns": list(before.columns),
                "output_columns": list(after.columns),
                "metadata_columns": list(processor._metadata.attributes),
            }
        )

    def _finalize_abstentions(self, split_name: str, final_data: pd.DataFrame) -> None:
        audit = self._logs.get(split_name)
        if audit is None or "decision" not in audit:
            return
        abstained = audit.loc[audit["decision"] == "abstained"]
        resolved = 0
        remaining = 0
        untrackable = 0
        for row in abstained.itertuples(index=False):
            position = getattr(row, "row_position", None)
            column = getattr(row, "column", None)
            if (
                not isinstance(position, Integral)
                or position < 0
                or position >= len(final_data)
                or column not in final_data.columns
            ):
                untrackable += 1
            elif pd.isna(final_data.iloc[position][column]):
                remaining += 1
            else:
                resolved += 1
        self._native_fallback_applied = self._native_fallback_applied or resolved > 0
        self._abstention_summaries[split_name] = {
            "action": "preserve_missing_at_custom_stage_boundary",
            "abstained_count": len(abstained),
            "missing_preserved_count": len(abstained),
            "resolved_by_later_stage_count": resolved,
            "remaining_missing_count": remaining,
            "untrackable_after_later_stage_count": untrackable,
            "all_abstained_cells_remain_missing": resolved == 0 and untrackable == 0,
        }

    def _reset_runtime_state(self) -> None:
        self._stages = []
        self._fit_output = None
        self._logs = {}
        self._reports = {}
        self._strategy_states = {}
        self._abstention_summaries = {}
        self._stage_runs = {}
        self._native_fallback_applied = False
        self._is_fitted = False

    def get_artifacts(self) -> dict[str, Any]:
        """Return detached composition, custom audit, and fallback evidence."""
        return {
            "schema_version": 1,
            "processor_type": "composite",
            "fit_count": self._fit_count,
            "transform_count": self._transform_count,
            "stages": [
                {
                    "name": stage["name"],
                    "method": stage["method"],
                    "processor_class": type(stage["processor"]).__name__,
                    "sequence": deepcopy(stage["sequence"]),
                }
                for stage in self._stages
            ],
            "stage_runs": deepcopy(self._stage_runs),
            "abstain_action": "preserve_missing_at_custom_stage_boundary",
            "constant_column_behavior": "protect_custom_required_columns",
            "native_fallback_applied": self._native_fallback_applied,
            "abstention_summaries": deepcopy(self._abstention_summaries),
            "logs": {name: log.copy(deep=True) for name, log in self._logs.items()},
            "reports": deepcopy(self._reports),
            "strategy_states": deepcopy(self._strategy_states),
        }

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted stage inverses in reverse order."""
        if not self._is_fitted:
            raise ConfigError(
                "Composite Preprocessor must be fitted before inverse_transform()."
            )
        current = data.copy(deep=True)
        for stage in reversed(self._stages):
            current = stage["processor"].inverse_transform(current)
        return current
