"""External-only structured compensation for the custom preprocessor interface."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.impute import KNNImputer, SimpleImputer

from petsard.exceptions import ConfigError, UnfittedError
from petsard.loader import Loader


def protected_name(name):
    """Conservative default; domain-specific identifiers must also be configured."""
    name = name.lower()
    return (
        name.startswith("y_")
        or name == "id"
        or name.endswith("_id")
        or name in {"record", "subject", "record_identifier", "subject_identifier"}
    )


class MissingCompensation:
    """Use explicit external fit data, never the adapter's local ``data``.

    Direct API: ``fit(fit_data=public)`` then
    ``transform(transform_data=local, mask=mask)``. In native pipelines,
    ``fit(data=local)`` loads the configured external file with PETsARD Loader.
    An omitted mask declares only feature NaNs missing at this stage boundary.
    """

    def __init__(self, config, metadata=None):
        self.config = deepcopy(config)
        self._metadata = metadata
        spec = self.config.get("missing_compensation", {})
        self.method = spec.get("method")
        if self.method not in {"median", "knn", "mice"}:
            raise ConfigError("missing_compensation.method must be median, knn or mice")
        allowed_options = (
            {"method"}
            if self.method == "mice"
            else {"method", "n_neighbors", "weights", "metric"}
        )
        if set(spec) - allowed_options:
            raise ConfigError("Unsupported missing compensation option")
        if self.method != "mice" and (
            spec.get("n_neighbors", 5) != 5
            or spec.get("weights", "uniform") != "uniform"
            or spec.get("metric", "nan_euclidean") != "nan_euclidean"
        ):
            raise ConfigError("Preview KNN requires 5 / uniform / nan_euclidean")
        self.features = self.config.get("feature_columns")
        if (
            not isinstance(self.features, list)
            or not self.features
            or not all(isinstance(c, str) for c in self.features)
            or len(set(self.features)) != len(self.features)
        ):
            raise ConfigError("feature_columns must be an explicit unique list")
        configured = {}
        for key in (
            "protected_columns",
            "key_columns",
            "target_columns",
            "row_id_column",
        ):
            names = self.config.get(key, [])
            if isinstance(names, str):
                names = [names]
            if not isinstance(names, (list, tuple, set)) or not all(
                isinstance(name, str) and name for name in names
            ):
                raise ConfigError(f"{key} must be column names or one column name")
            configured[key] = set(names)
        self.protected = set().union(*configured.values())
        self._identity_columns = configured["key_columns"] | configured["row_id_column"]
        self._target_columns = configured["target_columns"]
        if any(protected_name(c) or c in self.protected for c in self.features):
            raise ConfigError("Identifiers and y_* labels cannot be imputer features")
        if self.method == "mice" and any(
            "__missing" in c or c == "vitals_missing" for c in self.features
        ):
            raise ConfigError("MICE cannot model missingness indicators")
        self.fit_source = self.config.get("fit_source", {})
        self.transform_source = self.config.get("transform_source")
        if (
            self.fit_source.get("role") not in {"public", "external"}
            or not isinstance(self.fit_source.get("id"), str)
            or not self.fit_source["id"].strip()
            or not isinstance(self.transform_source, str)
            or not self.transform_source.strip()
            or self.fit_source["id"] == self.transform_source
        ):
            raise ConfigError(
                "Distinct fit_source.id (public/external) and transform_source required"
            )
        self._imputer = None
        self._report = {}

    def _median_knn_strategy_fingerprint(self, values):
        """Hash the fitted strategy without publishing row-level fit values."""
        parameters = (
            {"strategy": "median"}
            if self.method == "median"
            else {
                "n_neighbors": 5,
                "weights": "uniform",
                "metric": "nan_euclidean",
            }
        )
        contract = {
            "schema_version": 1,
            "method": self.method,
            "parameters": parameters,
            "feature_columns": list(self.features),
            "feature_dtypes": list(map(str, values.dtypes)),
            "fit_source": deepcopy(self.fit_source),
            "fit_rows": len(values),
        }
        fitted_state = (
            self._imputer.statistics_
            if self.method == "median"
            else values.to_numpy(dtype=float)
        )
        fitted_state = np.asarray(fitted_state, dtype="<f8", order="C")
        missing = np.isnan(fitted_state)
        canonical_values = np.where(missing, 0.0, fitted_state).astype(
            "<f8", copy=False
        )

        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                contract,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(missing.astype(np.uint8).tobytes(order="C"))
        digest.update(canonical_values.tobytes(order="C"))
        return digest.hexdigest()

    def _features(self, frame):
        if (
            not isinstance(frame, pd.DataFrame)
            or not frame.columns.is_unique
            or not all(isinstance(c, str) for c in frame.columns)
            or frame.empty
            or not set(self.features).issubset(frame.columns)
        ):
            raise ConfigError("Nonempty DataFrame with unique feature columns required")
        values = frame[self.features]
        if any(not is_numeric_dtype(t) or is_bool_dtype(t) for t in values.dtypes):
            raise ConfigError("Compensation features must have numeric dtypes")
        array = values.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(array).any():
            raise ConfigError("Infinite imputer feature values are unsupported")
        return values

    def fit(self, data=None, *, fit_data=None):
        """Fit once on explicit public/external data; local data never trains it."""
        if self._imputer is not None:
            raise ConfigError("Compensation is frozen after fit; create a new instance")
        if fit_data is None:
            loader_config = self.config.get("fit_data")
            if not isinstance(loader_config, dict) or not loader_config.get("filepath"):
                raise ConfigError(
                    "Explicit fit_data or external fit_data.filepath required"
                )
            path = Path(loader_config["filepath"]).expanduser().resolve()
            if path == Path(self.transform_source).expanduser().resolve():
                raise ConfigError("Fit and transform paths must be distinct")
            fit_data, _ = Loader(**loader_config).load()
        values = self._features(fit_data)
        if values.isna().all().any():
            raise ConfigError("External fit features cannot be entirely missing")
        self._fit_values = values.copy(deep=True)
        if self.method == "mice":
            self._public_min = values.min().astype(float)
            self._public_max = values.max().astype(float)
        self._fit_ids = {
            c: set(fit_data[c].dropna())
            for c in fit_data.columns
            if (protected_name(c) or c in self._identity_columns)
            and not c.lower().startswith("y_")
            and c not in self._target_columns
        }
        if data is not None:
            self._check_separation(data)
        if self.method == "mice":
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer

            imputer = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
                ),
                max_iter=3,
                random_state=42,
            )
        else:
            imputer = (
                SimpleImputer(strategy="median")
                if self.method == "median"
                else KNNImputer(
                    n_neighbors=5, weights="uniform", metric="nan_euclidean"
                )
            )
        imputer.fit(values)
        self._imputer = imputer
        if self.method == "mice":
            digest = hashlib.sha256()
            digest.update("\0".join(self.features).encode())
            digest.update("\0".join(map(str, values.dtypes)).encode())
            digest.update(values.to_numpy(dtype=float).tobytes())
            self._strategy_fingerprint = digest.hexdigest()
        else:
            self._strategy_fingerprint = self._median_knn_strategy_fingerprint(values)
        self._report = {
            "method": self.method,
            "fit_source": deepcopy(self.fit_source),
            "fit_rows": len(values),
            "feature_columns": list(self.features),
            "parameters": (
                {"strategy": "median"}
                if self.method == "median"
                else (
                    {"n_neighbors": 5, "weights": "uniform", "metric": "nan_euclidean"}
                    if self.method == "knn"
                    else {
                        "estimator": "RandomForestRegressor",
                        "n_estimators": 50,
                        "max_depth": 10,
                        "n_jobs": -1,
                        "estimator_random_state": 42,
                        "max_iter": 3,
                        "random_state": 42,
                    }
                )
            ),
            "fit_count": 1,
        }
        if self.method == "mice":
            self._report["strategy_fingerprint"] = self._strategy_fingerprint
        return self

    def _check_separation(self, frame):
        values = self._features(frame)
        if values.reset_index(drop=True).equals(
            self._fit_values.reset_index(drop=True)
        ):
            raise ConfigError("Transform data duplicates external fit features")
        for column, identities in self._fit_ids.items():
            # Labels are protected values, not identity keys.
            if column in frame and not column.lower().startswith("y_"):
                if identities.intersection(frame[column].dropna()):
                    raise ConfigError(f"Fit/transform identity overlap: {column}")

    def transform(self, data=None, *, transform_data=None, mask=None):
        if self._imputer is None:
            raise UnfittedError("Fit external data before transform")
        if data is not None and transform_data is not None:
            raise ConfigError("Supply only data or transform_data")
        frame = transform_data if transform_data is not None else data
        self._check_separation(frame)
        if mask is None:
            mask = pd.DataFrame(False, index=frame.index, columns=frame.columns)
            mask[self.features] = frame[self.features].isna()
        if (
            not isinstance(mask, pd.DataFrame)
            or not mask.index.equals(frame.index)
            or not mask.columns.equals(frame.columns)
            or mask.isna().any().any()
            or not all(is_bool_dtype(t) for t in mask.dtypes)
        ):
            raise ConfigError("Mask must be a complete boolean table aligned to input")
        if mask.drop(columns=self.features).any().any():
            raise ConfigError("Mask cannot include protected or non-feature columns")
        if (mask & frame.notna()).any().any():
            raise ConfigError("Declared missing cells must already be masked to NaN")
        candidate = self._imputer.transform(frame[self.features])
        if candidate.shape != (len(frame), len(self.features)):
            raise ConfigError("Imputer changed feature dimensionality")
        if self.method == "mice":
            from petsard.processor.mice_postprocess import audit_mice, postprocess_mice

            candidate = postprocess_mice(
                pd.DataFrame(candidate, index=frame.index, columns=self.features)
            ).to_numpy()
        result = frame.copy(deep=True)
        for i, column in enumerate(self.features):
            selected = mask[column].to_numpy(dtype=bool)
            if not selected.any():
                continue
            values = candidate[selected, i]
            if (
                pd.api.types.is_integer_dtype(frame[column].dtype)
                and not np.equal(values, np.trunc(values)).all()
            ):
                raise ConfigError(
                    f"Fractional compensation cannot preserve integer dtype: {column}"
                )
            try:
                cast = pd.Series(values).astype(frame[column].dtype)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ConfigError(f"Cannot preserve dtype for {column}") from exc
            if not np.isfinite(cast.to_numpy(dtype=float)).all():
                raise ConfigError(f"Non-finite compensation for {column}")
            if pd.api.types.is_integer_dtype(
                frame[column].dtype
            ) and not np.array_equal(cast.to_numpy(dtype=float), values):
                raise ConfigError(
                    f"Fractional compensation cannot preserve integer dtype: {column}"
                )
            result.iloc[np.flatnonzero(selected), result.columns.get_loc(column)] = (
                cast.array
            )
        if not result.dtypes.equals(frame.dtypes) or not result.where(~mask).equals(
            frame.where(~mask)
        ):
            raise ConfigError(
                "Compensation violated schema or outside-mask preservation"
            )
        if (result.isna() & mask).any().any():
            raise ConfigError("Declared missing cells remain unfilled")
        mice_qa = (
            audit_mice(result, mask, self.features, self._public_min, self._public_max)
            if self.method == "mice"
            else None
        )
        self._report.update(
            {
                "transform_source": self.transform_source,
                "masked_cells_filled": int(mask.to_numpy().sum()),
                "outside_mask_changes": 0,
                "protected_changes": 0,
                "label_changes": 0,
                "transform_role": "local_masked",
            }
        )
        if mice_qa is not None:
            self._report["mice_qa"] = mice_qa
        return result

    def get_report(self):
        return deepcopy(self._report)

    def get_strategy_state(self):
        """Expose a privacy-safe identity for the frozen Public-fit strategy."""
        if self._imputer is None:
            return None
        return {
            "method": self.method,
            "frozen": True,
            "fit_count": 1,
            "fit_source": deepcopy(self.fit_source),
            "fingerprint": self._strategy_fingerprint,
        }
