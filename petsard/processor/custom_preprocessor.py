"""Dynamic loader wrapper for stage-level custom preprocessors."""

from __future__ import annotations

import logging
import sys
from copy import deepcopy
from numbers import Integral

import pandas as pd

from petsard.exceptions import ConfigError
from petsard.metadater import Schema
from petsard.utils import load_external_module


class CustomPreprocessor:
    """Load a DataFrame preprocessor from an external Python module.

    The interface mirrors ``CustomSynthesizer`` while keeping preprocessing a
    stage-level operation.  A custom implementation must accept ``config`` and
    ``metadata`` and implement ``fit(data)`` and ``transform(data)``.
    """

    REQUIRED_METHODS: dict[str, list[str]] = {
        "__init__": ["config", "metadata"],
        "fit": ["data"],
        "transform": ["data"],
    }
    ABSTAIN_ACTION_PRESERVE_MISSING = "preserve_missing"

    def __init__(self, config: dict, metadata: Schema = None) -> None:
        self._logger = logging.getLogger(f"PETsARD.{self.__class__.__name__}")
        self.config = deepcopy(config)
        self._metadata = metadata
        self.reversible = bool(self.config.get("reversible", False))
        self.abstain_action = self.config.get("abstain_action")
        if self.abstain_action not in {
            None,
            self.ABSTAIN_ACTION_PRESERVE_MISSING,
        }:
            raise ConfigError(
                "Custom Preprocessor currently supports only "
                "abstain_action: preserve_missing; native fallback is not "
                "implemented."
            )
        self._is_fitted = False
        self._fit_count = 0
        self._transform_count = 0
        self._split_logs: dict[str, pd.DataFrame] = {}
        self._split_reports: dict[str, dict] = {}
        self._split_strategy_states: dict[str, dict] = {}
        self._split_abstention_summaries: dict[str, dict] = {}

        for key in ("module_path", "class_name"):
            if key not in self.config:
                raise ConfigError(
                    f"{key.replace('_', ' ').capitalize()} ({key}) is not "
                    "provided in the configuration."
                )

        _, preprocessor_class = load_external_module(
            module_path=self.config["module_path"],
            class_name=self.config["class_name"],
            logger=self._logger,
            required_methods=self.REQUIRED_METHODS,
            search_paths=sys.path,
        )

        if self.reversible and not callable(
            getattr(preprocessor_class, "inverse_transform", None)
        ):
            raise ConfigError(
                "A reversible custom preprocessor must implement "
                "inverse_transform(data)."
            )

        self._impl = preprocessor_class(
            config=deepcopy(self.config), metadata=metadata
        )
        if self.abstain_action is not None and not callable(
            getattr(self._impl, "get_log", None)
        ):
            raise ConfigError(
                "A tracked abstain_action requires the custom preprocessor "
                "to implement get_log()."
            )

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the external implementation."""
        if not isinstance(data, pd.DataFrame):
            raise ConfigError("Custom preprocessor fit(data) requires a DataFrame.")
        self._impl.fit(data=data)
        self._is_fitted = True
        self._fit_count += 1

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data and enforce the PETsARD DataFrame boundary."""
        self._reset_split_artifacts()
        return self._transform_one(data, "data")

    def _transform_one(
        self, data: pd.DataFrame, split_name: str
    ) -> pd.DataFrame:
        """Transform one DataFrame and capture its trace artifacts."""
        if not self._is_fitted:
            raise ConfigError("Custom preprocessor must be fitted before transform().")
        if not isinstance(data, pd.DataFrame):
            raise ConfigError(
                "Custom preprocessor transform(data) requires a DataFrame."
            )
        result = self._impl.transform(data=data)
        if not isinstance(result, pd.DataFrame):
            raise ConfigError(
                "Custom preprocessor transform(data) must return a pandas DataFrame."
            )
        self._transform_count += 1
        self._capture_split_artifacts(split_name)
        self._track_abstentions(split_name, data, result)
        return result

    def transform_splits(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Transform every split with the one already-fitted implementation.

        ``fit`` is deliberately not called here.  The adapter fits once on
        ``train`` and then sends train plus validation/control through this
        method, preventing later split truth from changing plugin state.
        """
        if not self._is_fitted:
            raise ConfigError(
                "Custom preprocessor must be fitted on train before split transform."
            )
        if not isinstance(data, dict) or "train" not in data:
            raise ConfigError(
                "Splitter-aware custom preprocessing requires a train split."
            )
        unsupported = [
            name for name, frame in data.items() if not isinstance(frame, pd.DataFrame)
        ]
        if unsupported:
            raise ConfigError(
                "All custom preprocessor splits must be DataFrames: "
                + ", ".join(unsupported)
            )

        self._reset_split_artifacts()
        transformed: dict[str, pd.DataFrame] = {}
        frozen_fingerprint = None
        for split_name, frame in data.items():
            transformed[split_name] = self._transform_one(frame, split_name)
            strategy = self._split_strategy_states.get(split_name, {})
            fingerprint = strategy.get("fingerprint")
            if fingerprint is not None:
                if frozen_fingerprint is None:
                    frozen_fingerprint = fingerprint
                elif fingerprint != frozen_fingerprint:
                    raise ConfigError(
                        "Custom preprocessor strategy changed between data splits."
                    )
        return transformed

    def get_split_logs(self) -> dict[str, pd.DataFrame]:
        """Return detached per-split plugin audit logs when available."""
        return {name: log.copy(deep=True) for name, log in self._split_logs.items()}

    def get_split_reports(self) -> dict[str, dict]:
        """Return detached per-split plugin reports when available."""
        return deepcopy(self._split_reports)

    def get_split_strategy_states(self) -> dict[str, dict]:
        """Return detached strategy identities observed after each transform."""
        return deepcopy(self._split_strategy_states)

    def get_abstention_summaries(self) -> dict[str, dict]:
        """Return PETsARD-verified abstention behavior for each transformed split."""
        return deepcopy(self._split_abstention_summaries)

    def get_artifacts(self) -> dict:
        """Return public, detached artifacts suitable for PETsARD Status."""
        return {
            "schema_version": 1,
            "processor_type": "custom_method",
            "abstain_action": self.abstain_action,
            "native_fallback_applied": False,
            "fit_count": self._fit_count,
            "transform_count": self._transform_count,
            "abstention_summaries": self.get_abstention_summaries(),
            "logs": self.get_split_logs(),
            "reports": self.get_split_reports(),
            "strategy_states": self.get_split_strategy_states(),
        }

    def _reset_split_artifacts(self) -> None:
        self._split_logs = {}
        self._split_reports = {}
        self._split_strategy_states = {}
        self._split_abstention_summaries = {}

    def _capture_split_artifacts(self, split_name: str) -> None:
        get_log = getattr(self._impl, "get_log", None)
        if callable(get_log):
            log = get_log()
            if isinstance(log, pd.DataFrame):
                self._split_logs[split_name] = log.copy(deep=True)

        get_report = getattr(self._impl, "get_report", None)
        if callable(get_report):
            report = get_report()
            if isinstance(report, dict):
                self._split_reports[split_name] = deepcopy(report)

        get_strategy_state = getattr(self._impl, "get_strategy_state", None)
        if callable(get_strategy_state):
            strategy = get_strategy_state()
            if isinstance(strategy, dict):
                self._split_strategy_states[split_name] = deepcopy(strategy)

    def _track_abstentions(
        self,
        split_name: str,
        before: pd.DataFrame,
        after: pd.DataFrame,
    ) -> None:
        if self.abstain_action is None:
            return

        if before.shape != after.shape:
            raise ConfigError(
                "Tracked abstention requires transform to preserve DataFrame shape."
            )
        if list(before.columns) != list(after.columns):
            raise ConfigError(
                "Tracked abstention requires transform to preserve column order."
            )
        if not before.index.equals(after.index):
            raise ConfigError(
                "Tracked abstention requires transform to preserve DataFrame index."
            )

        audit = self._split_logs.get(split_name)
        if audit is None:
            raise ConfigError(
                "Tracked abstention requires get_log() to return a DataFrame."
            )
        required = {"row_position", "column", "decision", "reason"}
        missing_columns = sorted(required - set(audit.columns))
        if missing_columns:
            raise ConfigError(
                "Custom preprocessor abstention log is missing columns: "
                + ", ".join(missing_columns)
            )

        abstained = audit.loc[audit["decision"] == "abstained"].copy()
        cells = []
        for row in abstained.itertuples(index=False):
            position = row.row_position
            column = row.column
            if (
                not isinstance(position, Integral)
                or position < 0
                or position >= len(before)
                or column not in before.columns
            ):
                raise ConfigError(
                    "Custom preprocessor abstention log identifies an invalid cell."
                )
            column_position = before.columns.get_loc(column)
            if not pd.isna(before.iat[int(position), column_position]):
                raise ConfigError(
                    "Custom preprocessor marked an originally observed cell as "
                    "abstained."
                )
            if not pd.isna(after.iat[int(position), column_position]):
                raise ConfigError(
                    "abstain_action: preserve_missing was violated: an abstained "
                    "cell received a value."
                )
            cells.append(
                {
                    "row_position": int(position),
                    "column": column,
                    "reason": row.reason,
                }
            )

        reason_counts = {
            str(reason): int(count)
            for reason, count in abstained["reason"].value_counts(dropna=False).items()
        }
        column_counts = {
            str(column): int(count)
            for column, count in abstained["column"].value_counts(dropna=False).items()
        }
        summary = {
            "action": self.ABSTAIN_ACTION_PRESERVE_MISSING,
            "abstained_count": len(abstained),
            "missing_preserved_count": len(abstained),
            "all_abstained_cells_remain_missing": True,
            "native_fallback_applied": False,
            "reason_counts": reason_counts,
            "column_counts": column_counts,
            "cells": cells,
        }
        self._split_abstention_summaries[split_name] = summary
        if len(abstained):
            self._logger.warning(
                "Custom Preprocessor preserved %d abstained missing cells in "
                "split '%s'; no native fallback was applied.",
                len(abstained),
                split_name,
            )

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reverse a reversible plugin, or preserve an irreversible result."""
        if self.reversible:
            result = self._impl.inverse_transform(data=data)
            if not isinstance(result, pd.DataFrame):
                raise ConfigError(
                    "Custom preprocessor inverse_transform(data) must return a "
                    "pandas DataFrame."
                )
            return result
        return deepcopy(data)
