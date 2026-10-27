---
title: "Missing-Data Compensation"
type: docs
weight: 646
prev: docs/petsard-yaml/preprocessor-yaml/discretizing
next: docs/petsard-yaml/preprocessor-yaml
---

Fills the missing values of numeric feature columns with a Median, KNN or MICE strategy that is fitted on public or external reference data and then applied to the local data.

## Background

The built-in [missing value handling]({{< ref "/docs/petsard-yaml/preprocessor-yaml/missing-handling" >}}) computes its statistics from the data it processes. Missing-data compensation learns the imputation strategy from a separate reference dataset, such as a published dataset or an external dataset with the same features, and applies the fitted strategy to the local data. The local data is only transformed; it does not train the strategy.

- **External fit**: the strategy is fitted once on reference data declared as `public` or `external`.
- **Frozen strategy**: one fitted strategy transforms every dataset and every Splitter branch, identified by a SHA-256 fingerprint.
- **Targeted filling**: only the missing cells of the listed `feature_columns` are written. Observed values, identifiers, labels and other protected columns keep their exact values.
- **Schema preservation**: the output keeps the row count, column order, index and dtypes of the input.
- **Explicit activation**: the module runs only when a Preprocessor experiment uses `method: custom_method` with `class_name: MissingCompensation`. The `default` and `custom` preprocessing methods work as before.

The module is `petsard/processor/missing_compensation.py`, with the MICE post-processing in `petsard/processor/mice_postprocess.py`. Both files are installed with PETsARD from the source code, as described in [Extended Features Setup]({{< ref "/docs/installation/extended-features" >}}). A step-by-step walkthrough with example data is given in [Missing-Data Compensation with Median, KNN and MICE]({{< ref "/docs/getting-started/missing-compensation" >}}).

## Usage Examples

The examples use a membership dataset with the identifier `member_id`, the categorical column `region`, five numeric features and the label `churned`. `reference_members.csv` is the reference data and `local_members.csv` is the local data to compensate.

### Example 1: Median Compensation

```yaml
Loader:
  local_members:
    filepath: local_members.csv

Preprocessor:
  compensation:
    method: custom_method
    module_path: petsard/processor/missing_compensation.py
    class_name: MissingCompensation
    missing_compensation:
      method: median
    feature_columns: [age, tenure_months, annual_income, monthly_spend, store_visits]
    key_columns: [member_id]
    target_columns: [churned]
    protected_columns: [region]
    fit_data:
      filepath: reference_members.csv
    fit_source:
      id: reference-members-2026
      role: public
    transform_source: local-members-2026

Reporter:
  save_compensated:
    method: save_data
    source: Preprocessor
```

### Example 2: KNN Compensation

Only the `missing_compensation` block changes. The three KNN keys are optional; when present, they carry the fixed KNN parameters shown here.

```yaml
    missing_compensation:
      method: knn
      n_neighbors: 5
      weights: uniform
      metric: nan_euclidean
```

### Example 3: MICE Compensation

```yaml
    missing_compensation:
      method: mice
```

### Example 4: Compensation with Splitter

With a Splitter before the Preprocessor, the strategy is fitted once and the same fitted strategy transforms both the `train` and the `validation` branch.

```yaml
Loader:
  local_members:
    filepath: local_members.csv

Splitter:
  holdout:
    num_samples: 1
    train_split_ratio: 0.8

Preprocessor:
  compensation:
    method: custom_method
    module_path: petsard/processor/missing_compensation.py
    class_name: MissingCompensation
    missing_compensation:
      method: median
    feature_columns: [age, tenure_months, annual_income, monthly_spend, store_visits]
    key_columns: [member_id]
    target_columns: [churned]
    protected_columns: [region]
    fit_data:
      filepath: reference_members.csv
    fit_source:
      id: reference-members-2026
      role: public
    transform_source: local-members-2026
```

### Example 5: Chaining with Outlier and Missing Value Handling

`method: composite` runs a built-in outlier stage, the compensation stage and a built-in missing value stage in one Preprocessor experiment. See [Chaining with Other Preprocessing Steps](#chaining-with-other-preprocessing-steps).

```yaml
Preprocessor:
  compensation_chain:
    method: composite
    stages:
      - name: outlier
        method: custom
        sequence: [outlier]
        config:
          outlier:
            member_id: null
            region: null
            age: null
            tenure_months: null
            annual_income: outlier_iqr
            monthly_spend: null
            store_visits: null
            churned: null
      - name: compensation
        method: custom_method
        module_path: petsard/processor/missing_compensation.py
        class_name: MissingCompensation
        missing_compensation:
          method: mice
        feature_columns: [age, tenure_months, annual_income, monthly_spend, store_visits]
        key_columns: [member_id]
        target_columns: [churned]
        protected_columns: [region]
        fit_data:
          filepath: reference_members.csv
        fit_source:
          id: reference-members-2026
          role: public
        transform_source: local-members-2026
      - name: missing
        method: custom
        sequence: [missing]
        config:
          missing:
            member_id: null
            region: missing_mode
            age: null
            tenure_months: null
            annual_income: null
            monthly_spend: null
            store_visits: null
            churned: null
```

## Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `method` | `string` | Yes | `custom_method` |
| `module_path` | `string` | Yes | `petsard/processor/missing_compensation.py`. A relative path is resolved from the working directory and from each entry of the Python path, so with an editable installation of PETsARD this value works from any working directory. An absolute path is also accepted |
| `class_name` | `string` | Yes | `MissingCompensation` |
| `missing_compensation` | `dict` | Yes | Compensation method. `method` is `median`, `knn` or `mice`. With `knn`, the optional keys `n_neighbors` (`5`), `weights` (`uniform`) and `metric` (`nan_euclidean`) carry the fixed values in parentheses. With `mice`, the block contains `method` only |
| `feature_columns` | `list` | Yes | Ordered list of unique numeric columns that the strategy models and fills. Each column exists in both the reference and the local data and has a numeric, non-boolean dtype |
| `key_columns` | `list` or `string` | No | Record identifiers. They are protected, and their values are distinct between the reference and the local data. Default: none |
| `row_id_column` | `string` or `list` | No | Row identifier, handled in the same way as `key_columns`. Default: none |
| `target_columns` | `list` or `string` | No | Labels. They are protected and keep their values. Default: none |
| `protected_columns` | `list` or `string` | No | Other columns that keep their values. Default: none |
| `fit_data` | `dict` | Yes | Loader settings of the reference data. `filepath` is required; other Loader parameters, such as `schema`, may be added |
| `fit_source` | `dict` | Yes | Declaration of the reference data: `id` is a non-empty identifier, and `role` is `public` or `external` |
| `transform_source` | `string` | Yes | Non-empty identifier of the local data, different from `fit_source.id`. When it is a file path, it points to a file other than `fit_data.filepath` |

Columns listed in `key_columns`, `row_id_column`, `target_columns` and `protected_columns` are kept out of `feature_columns`.

## Methods

| `method` | Imputer (scikit-learn) | Fixed parameters | Fitted state |
|---|---|---|---|
| `median` | `SimpleImputer` | `strategy="median"` | The median of each feature in the reference data |
| `knn` | `KNNImputer` | `n_neighbors=5`, `weights="uniform"`, `metric="nan_euclidean"` | The reference feature matrix; a missing cell receives the mean of the 5 nearest reference rows |
| `mice` | `IterativeImputer` with `RandomForestRegressor` | Estimator: `n_estimators=50`, `max_depth=10`, `random_state=42`, `n_jobs=-1`; imputer: `max_iter=3`, `random_state=42` | Chained random-forest models that predict each feature from the other features |

The parameters are fixed, so repeated runs on the same data give the same result. The MICE random forest uses all CPU cores (`n_jobs=-1`).

## Processing Logic

### 1. Fit (once per strategy)

1. The reference data is loaded from `fit_data` with the PETsARD Loader.
2. The `feature_columns` are read in the listed order. Each feature is numeric and finite and has at least one observed value in the reference data.
3. The identifier values of the reference data are recorded for the separation check (see [Identifiers, Labels and Protected Columns](#identifiers-labels-and-protected-columns)).
4. The imputer of the selected method is fitted, frozen and identified by a SHA-256 fingerprint.

The local data passed through the pipeline takes part only in the separation check.

### 2. Transform

1. The missing cells of `feature_columns` in the local data form the compensation mask. Missing cells of other columns stay as they are, ready for a later stage.
2. The fitted imputer predicts candidate values, and only the masked cells receive them. Observed cells keep their exact values.
3. Each filled value takes the dtype of its column. A column stored as an integer type receives whole numbers; store a column as a float type when fractional values are expected.
4. The result is verified: dtypes are unchanged, every cell outside the mask is unchanged, and every masked cell holds a value.
5. The report is updated with the number of filled cells and the verification counts.

### 3. MICE Post-processing and Verification

The MICE method applies the following rules to its predictions before writing them back. The rules rely on column-name suffixes:

- For columns that share a base name with the suffixes `_min` and `_max` (for example `price_min` and `price_max`), the two values are ordered so that the minimum does not exceed the maximum. Columns of the same base with the suffixes `_mean`, `_first` and `_last` are clipped to that range.
- Columns ending in `_n` hold counts and are rounded to non-negative integers.
- A column named `icu_stay_24h` is a binary indicator; it is clipped to 0–1 and rounded to 0 or 1.

After writing back, MICE verifies the compensated table and records the counts in `mice_qa`. A completed run has every count equal to `0`:

| Key | What is counted |
|---|---|
| `remaining_missing` | Missing cells left in `feature_columns` |
| `non_finite` | Non-finite values in `feature_columns` |
| `unexpected_negative` | Negative values in features whose reference values are all non-negative |
| `new_masked_public_range_violations` | Filled cells outside the reference range of their feature by more than `0.001` |
| `binary_violations` | Values outside `{0, 1}` in the binary indicator column `icu_stay_24h` |
| `integer_violations` | Non-integer values in `_n` columns |
| `summary_order_violations` | Rows where `_min` exceeds `_max`, or `_mean`, `_first` or `_last` lies outside the `_min` to `_max` range |

The range check applies to newly filled cells, so observed local values outside the reference range are kept as they are. Columns whose names contain `__missing`, or the column named `vitals_missing`, mark missingness and are kept out of the MICE `feature_columns`.

## Identifiers, Labels and Protected Columns

- **Identifier names**: columns named `id`, `record`, `subject`, `record_identifier` or `subject_identifier`, columns ending in `_id`, and columns starting with `y_` (case-insensitive) are treated as identifiers or labels and are kept out of `feature_columns`.
- **Separation of the two datasets**: the reference data and the local data are distinct datasets. The values of identifier columns (`key_columns`, `row_id_column` and identifier-named columns other than labels) do not overlap between them, and the local feature table differs from the reference feature table.
- **Naming shared codes**: use names ending in `_id` for record identifiers only. A category code that appears in both datasets, such as a store code, works best with a name such as `store_code`.
- **Labels**: values of `target_columns` and `y_*` columns are protected values, and the same label values may appear in both datasets.

## Working with Splitter

When a Splitter precedes the Preprocessor, the compensation stage receives all Splitter branches:

- The strategy is fitted once, and the same fitted instance transforms `train` and `validation`.
- Every branch reports the same fingerprint with `fit_count: 1`; PETsARD confirms that the strategy stays identical across the branches.
- The Synthesizer receives the compensated `train` branch. The Evaluator uses the compensated `train` branch as the original data and the compensated `validation` branch as the control data, so both reflect the same strategy.

## Chaining with Other Preprocessing Steps

`method: composite` runs PETsARD built-in stages and the compensation stage as one ordered chain within a single Preprocessor experiment. The supported orders are:

```text
custom_method
outlier -> custom_method
custom_method -> missing
outlier -> custom_method -> missing
```

- `stages` lists one to three stages. Each stage has a unique `name`.
- The compensation stage uses `method: custom_method` with the parameters of this page.
- A built-in stage uses `method: custom` with a per-column `config`, and a `sequence` of exactly `[outlier]` before the compensation stage or `[missing]` after it.
- In a built-in stage, list every column and set `null` for the columns that the stage leaves unchanged; a column that is not listed uses the default processor of its type.
- The outlier stage treats outliers before compensation. The missing stage fills the columns that compensation does not model, such as a categorical column.
- Each stage is fitted on the output of the previous stage. With a Splitter, the whole chain is fitted on `train` once and the same fitted stages transform `validation`.
- The custom-stage keys `key_columns`, `target_columns`, `row_id_column` and the optional `required_columns` list columns that the built-in stages keep during constant-column handling. When `target_columns` is not set, columns with missing values in the training data are kept as well.
- The compensation stage keeps the shape, column order and index of its input.

Separate Preprocessor experiment names run as separate branches; stages are chained inside one `method: composite` experiment.

## Outputs

### Compensated Data

The result of a `custom_method` experiment is a DataFrame with the same rows, columns, order, index and dtypes as its input, with the masked feature cells filled. In a `composite` experiment, the built-in stages before and after the compensation stage process the data as configured. With a Splitter, the result contains the `train` and `validation` branches. A Reporter with `method: save_data` and `source: Preprocessor` writes the data, for example `petsard_Loader[local_members]_Preprocessor[compensation].csv`. The saved file serves as the Loader input of a synthesis workflow with the default Preprocessor, which encodes categorical columns such as `region` for the synthesizer, as shown in the [tutorial]({{< ref "/docs/getting-started/missing-compensation" >}}).

### Artifacts

`Executor.get_preprocessor_artifacts()` returns the compensation records of a Preprocessor experiment:

```python
from petsard import Executor

executor = Executor(config="missing-compensation.yaml")
executor.run()

artifacts = executor.get_preprocessor_artifacts("compensation")
report = artifacts["reports"]["data"]
strategy = artifacts["strategy_states"]["data"]
```

| Key | Content |
|---|---|
| `processor_type` | `custom_method`, or `composite` for a chained experiment |
| `fit_count` | Number of fits; `1` |
| `transform_count` | Number of transformed datasets or branches |
| `reports` | Compensation report of each dataset or branch |
| `strategy_states` | Strategy identity of each dataset or branch |
| `stages`, `stage_runs` | Composite only: the stage order, and the shapes and columns of every stage run |

`reports` and `strategy_states` are keyed by `data` for a single `custom_method` experiment, by `train` for a `composite` experiment, and by `train` and `validation` when a Splitter precedes the Preprocessor.

### Report Fields

| Field | Content |
|---|---|
| `method` | `median`, `knn` or `mice` |
| `fit_source` | `id` and `role` of the reference data |
| `fit_rows` | Number of rows of the reference data |
| `feature_columns` | The ordered feature list |
| `parameters` | The effective imputer parameters |
| `fit_count` | `1` |
| `strategy_fingerprint` | MICE only: the strategy fingerprint |
| `transform_source` | Identifier of the local data |
| `masked_cells_filled` | Number of filled cells |
| `outside_mask_changes` | Changed cells outside the mask; `0` |
| `protected_changes` | Changed cells in protected columns; `0` |
| `label_changes` | Changed label cells; `0` |
| `transform_role` | `local_masked` |
| `mice_qa` | MICE only: the verification counts listed above |

### Strategy State Fields

| Field | Content |
|---|---|
| `method` | `median`, `knn` or `mice` |
| `frozen` | `true` |
| `fit_count` | `1` |
| `fit_source` | `id` and `role` of the reference data |
| `fingerprint` | SHA-256 digest of the fitted strategy |

The Median fingerprint covers the fitted medians, and the KNN fingerprint covers the reference feature matrix; both also cover the method, the parameters, the ordered features and their dtypes, the reference identity and the number of reference rows. The MICE fingerprint covers the ordered features, their dtypes and the reference feature values. Only the digest is published, so the strategy state contains no row values.

## Python API

The class can also be used directly, which adds an explicit mask that selects the cells to fill:

```python
import pandas as pd
from petsard.processor.missing_compensation import MissingCompensation

reference = pd.read_csv("reference_members.csv")
local = pd.read_csv("local_members.csv")

config = {
    "missing_compensation": {"method": "knn"},
    "feature_columns": ["age", "tenure_months", "annual_income", "monthly_spend", "store_visits"],
    "key_columns": ["member_id"],
    "target_columns": ["churned"],
    "protected_columns": ["region"],
    "fit_source": {"id": "reference-members-2026", "role": "public"},
    "transform_source": "local-members-2026",
}

# Fill only the missing cells of annual_income
mask = pd.DataFrame(False, index=local.index, columns=local.columns)
mask["annual_income"] = local["annual_income"].isna()

compensation = MissingCompensation(config).fit(fit_data=reference)
result = compensation.transform(transform_data=local, mask=mask)
print(compensation.get_report())
print(compensation.get_strategy_state())
```

| Method | Description |
|---|---|
| `MissingCompensation(config)` | Creates the compensator. `config` takes the YAML keys of this page except `method`, `module_path` and `class_name`; `fit_data` is needed only when `fit()` reads the reference file itself |
| `fit(fit_data=reference)` | Fits the strategy on a reference DataFrame and returns the instance. Each instance is fitted once |
| `transform(transform_data=local, mask=mask)` | Returns the compensated DataFrame. The optional `mask` is a boolean DataFrame with the same index, columns and column order as `local`; `True` cells are already missing and lie in `feature_columns`. Without a mask, every missing cell of `feature_columns` is filled |
| `get_report()` | Returns the report: the fit information and the results of the latest transform |
| `get_strategy_state()` | Returns the strategy state |

One fitted instance transforms several datasets, for example a training set, a validation set and a holdout set, and all of them share the same fingerprint.

## Quick Reference

| Message | Resolution |
|---|---|
| `missing_compensation.method must be median, knn or mice` | Set `missing_compensation.method` to `median`, `knn` or `mice` |
| `Unsupported missing compensation option` | Keep `n_neighbors`, `weights` and `metric` for `knn` only, and only `method` for `mice` |
| `feature_columns must be an explicit unique list` | List each feature column once |
| `Identifiers and y_* labels cannot be imputer features` | Move identifier, label and protected columns from `feature_columns` to `key_columns`, `target_columns` or `protected_columns` |
| `Distinct fit_source.id (public/external) and transform_source required` | Give `fit_source.id` and `transform_source` different non-empty values, and set `fit_source.role` to `public` or `external` |
| `Explicit fit_data or external fit_data.filepath required` | Add `fit_data.filepath` with the path of the reference file |
| `Fit and transform paths must be distinct` | Point `fit_data.filepath` and `transform_source` to different files |
| `Compensation features must have numeric dtypes` | Keep numeric columns in `feature_columns`, or set a numeric type in the schema |
| `External fit features cannot be entirely missing` | Provide at least one observed value of every feature in the reference data |
| `Fit/transform identity overlap: <column>` | Use distinct identifier values in the two datasets, and rename shared codes without the `_id` suffix |
| `Transform data duplicates external fit features` | Use a local dataset that differs from the reference data |
| `Fractional compensation cannot preserve integer dtype: <column>` | Store the column as a float type |
| `Mask must be a complete boolean table aligned to input` | Build the mask from `local` with the same index and columns, filled with `True` and `False` |
| `Compensation is frozen after fit; create a new instance` | Create a new `MissingCompensation` instance for a new strategy |

## Notes

- Use reference data that is independent of the local data, such as a published dataset or an external dataset with the same features. Set `fit_source.role` to the actual role of the reference file; every report and strategy state records it.
- `feature_columns` defines exactly which columns are compensated. Columns outside the list keep their values, including their missing cells, which a later `missing` stage can handle.
- The compensated data keeps the dtypes of the input, so the following modules receive the same schema as without compensation.
- Fixed parameters and a frozen strategy make every run reproducible: the same reference data and local data give the same compensated values and the same fingerprint.
