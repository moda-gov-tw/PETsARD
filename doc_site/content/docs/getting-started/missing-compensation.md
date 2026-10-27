---
title: Missing-Data Compensation with Median, KNN and MICE
type: docs
weight: 160
prev: docs/getting-started/llm-text-synthesis
next: docs/getting-started
---

Fill the missing values of numeric feature columns with a strategy that is fitted on public or external reference data, and then use the compensated data in a synthesis workflow.

This tutorial uses a small membership dataset of a retail chain, created by a script, so every step can be reproduced. The complete parameter reference is in [Missing-Data Compensation (Preprocessor YAML)]({{< ref "/docs/petsard-yaml/preprocessor-yaml/missing-compensation" >}}).

{{< callout type="info" >}}
Prepare the environment with [Extended Features Setup]({{< ref "/docs/installation/extended-features" >}}) first, and run the commands of this page in the activated virtual environment. All files of this tutorial are created in the working directory.
{{< /callout >}}

## How It Works

1. **Reference data** (`reference_members.csv`): a public or external dataset with the same numeric features. The compensation strategy is fitted on this data once.
2. **Local data** (`local_members.csv`): the data to be compensated. It is loaded by the Loader, and its missing feature cells are filled with the fitted strategy.
3. **Compensated data**: the local data with the same rows, columns and dtypes, ready for synthesis or any other PETsARD workflow.

## Step 1: Prepare the Example Data

Save the following script as `make_members.py`:

```python
import numpy as np
import pandas as pd


def make_members(n, id_start, seed):
    rng = np.random.default_rng(seed)
    income = np.round(rng.normal(650, 180, n).clip(200, None), 1)
    spend = np.round(income * 0.004 + rng.normal(0, 0.4, n) + 1.0, 2)
    visits = np.round(spend * 2 + rng.normal(0, 1.0, n)).clip(0, None)
    return pd.DataFrame(
        {
            "member_id": np.arange(id_start, id_start + n),
            "region": rng.choice(["north", "central", "south", "east"], n),
            "age": rng.integers(20, 70, n),
            "tenure_months": rng.integers(1, 120, n),
            "annual_income": income,
            "monthly_spend": spend,
            "store_visits": visits,
            "churned": (rng.random(n) < 0.2).astype(int),
        }
    )


reference = make_members(300, 100001, seed=7)
local = make_members(120, 200001, seed=11)

rng = np.random.default_rng(2026)
for column, rate in {"annual_income": 0.15, "monthly_spend": 0.10, "store_visits": 0.10}.items():
    local.loc[rng.random(len(local)) < rate, column] = np.nan

reference.to_csv("reference_members.csv", index=False)
local.to_csv("local_members.csv", index=False)
print(local.isna().sum())
```

Run it:

```bash
python make_members.py
```

The script writes two files:

| File | Rows | Member numbers | Missing cells |
|---|---|---|---|
| `reference_members.csv` | 300 | 100001 to 100300 | None |
| `local_members.csv` | 120 | 200001 to 200120 | `annual_income` 12, `monthly_spend` 7, `store_visits` 11 (30 in total) |

Both files have the same columns:

| Column | Meaning | Role in compensation |
|---|---|---|
| `member_id` | Member number | Record identifier (`key_columns`) |
| `region` | Store region | Kept unchanged (`protected_columns`) |
| `age` | Age in years | Feature |
| `tenure_months` | Membership length in months | Feature |
| `annual_income` | Annual income, in thousands | Feature |
| `monthly_spend` | Average monthly spending, in thousands | Feature |
| `store_visits` | Store visits per month | Feature |
| `churned` | Whether the member left the program (0 or 1) | Label (`target_columns`) |

## Step 2: Write the YAML

Save the following configuration as `missing-compensation.yaml`:

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

### YAML Parameters Detailed Explanation

#### Loader

- **`local_members`**: experiment name, can be freely named.
- **`filepath`**: the local data to compensate.

#### Preprocessor

- **`compensation`**: experiment name. The same name is used to read the compensation records in Step 4.
- **`method: custom_method`**, **`module_path`** and **`class_name: MissingCompensation`**: load the missing-data compensation module.
- **`missing_compensation.method`**: the compensation method, `median`, `knn` or `mice`.
- **`feature_columns`**: the numeric columns that the strategy models and fills, in a fixed order.
- **`key_columns`**: the record identifier. Member numbers of the two files are distinct, as the module expects.
- **`target_columns`**: the label, kept unchanged.
- **`protected_columns`**: other columns kept unchanged.
- **`fit_data.filepath`**: the reference data used to fit the strategy.
- **`fit_source`**: the declaration of the reference data: an identifier (`id`) and its role (`public` or `external`).
- **`transform_source`**: an identifier of the local data, different from `fit_source.id`.

#### Reporter

- **`save_compensated`**: experiment name.
- **`method: save_data`** with **`source: Preprocessor`**: saves the compensated data as a CSV file.

## Step 3: Run

Save the following script as `run_compensation.py`:

```python
from petsard import Executor

executor = Executor(config="missing-compensation.yaml")
executor.run()

artifacts = executor.get_preprocessor_artifacts("compensation")
print(artifacts["reports"]["data"])
print(artifacts["strategy_states"]["data"])
```

Run it:

```bash
python run_compensation.py
```

## Step 4: Review the Results

### Compensated File

The Reporter writes `petsard_Loader[local_members]_Preprocessor[compensation].csv`. It has the same 120 rows and 8 columns as `local_members.csv`, and its 30 missing feature cells are filled. With the Median method, each filled cell holds the median of its feature in the reference data: `annual_income` 636.85, `monthly_spend` 3.48 and `store_visits` 7.0.

The following check confirms that every missing cell is filled and every observed value is unchanged:

```python
import pandas as pd

local = pd.read_csv("local_members.csv")
result = pd.read_csv("petsard_Loader[local_members]_Preprocessor[compensation].csv")

print(local.isna().sum().sum(), result.isna().sum().sum())  # 30 0
observed = local.notna()
print(local[observed].equals(result[observed]))              # True
```

### Compensation Report

`artifacts["reports"]["data"]` records the run:

```json
{
  "method": "median",
  "fit_source": {"id": "reference-members-2026", "role": "public"},
  "fit_rows": 300,
  "feature_columns": ["age", "tenure_months", "annual_income", "monthly_spend", "store_visits"],
  "parameters": {"strategy": "median"},
  "fit_count": 1,
  "transform_source": "local-members-2026",
  "masked_cells_filled": 30,
  "outside_mask_changes": 0,
  "protected_changes": 0,
  "label_changes": 0,
  "transform_role": "local_masked"
}
```

- `fit_rows: 300` and `fit_source` show that the strategy was fitted on the reference data.
- `masked_cells_filled: 30` matches the number of missing feature cells in the local data.
- `outside_mask_changes`, `protected_changes` and `label_changes` are `0`: observed values, `member_id`, `region` and `churned` are unchanged.

### Strategy State

`artifacts["strategy_states"]["data"]` identifies the fitted strategy:

```json
{
  "method": "median",
  "frozen": true,
  "fit_count": 1,
  "fit_source": {"id": "reference-members-2026", "role": "public"},
  "fingerprint": "880937a2601621048571180b8e74887f401ce76f5e818d440e81ab6513b88a7a"
}
```

The `fingerprint` is a SHA-256 digest of the fitted strategy. The same reference data and settings give the same fingerprint, so the value confirms which strategy produced a compensated file.

## Step 5: Switch to KNN or MICE

Change one line of `missing-compensation.yaml` and run Step 3 again:

{{< tabs items="KNN,MICE" >}}
{{< tab >}}
```yaml
    missing_compensation:
      method: knn
```

The report shows the fixed KNN parameters:

```json
"parameters": {"n_neighbors": 5, "weights": "uniform", "metric": "nan_euclidean"}
```
{{< /tab >}}
{{< tab >}}
```yaml
    missing_compensation:
      method: mice
```

The report shows the fixed MICE parameters, the strategy fingerprint and the verification counts in `mice_qa`, all of which are `0`:

```json
"parameters": {
  "estimator": "RandomForestRegressor",
  "n_estimators": 50,
  "max_depth": 10,
  "n_jobs": -1,
  "estimator_random_state": 42,
  "max_iter": 3,
  "random_state": 42
},
"mice_qa": {
  "remaining_missing": 0,
  "non_finite": 0,
  "unexpected_negative": 0,
  "new_masked_public_range_violations": 0,
  "binary_violations": 0,
  "integer_violations": 0,
  "summary_order_violations": 0
}
```
{{< /tab >}}
{{< /tabs >}}

Each method fills the same 30 cells. Median uses one value per feature, while KNN and MICE use the other features of each member, so the filled values differ from row to row. The first three missing `annual_income` cells of this example receive the following values (rounded to two decimals):

| `member_id` | Median | KNN | MICE |
|---|---|---|---|
| 200024 | 636.85 | 571.30 | 508.87 |
| 200059 | 636.85 | 631.96 | 315.29 |
| 200064 | 636.85 | 690.98 | 758.10 |

{{< callout type="info" >}}
Every run writes the file name of Step 4. To keep the results of several methods side by side, give each run its own Preprocessor experiment name, for example `compensation_knn`, and read its records with `executor.get_preprocessor_artifacts("compensation_knn")`.
{{< /callout >}}

## Step 6: Synthesize the Compensated Data

The compensated file is a regular CSV file, so it serves as the input of any PETsARD workflow. Save the following configuration as `synthesis.yaml`:

```yaml
Loader:
  compensated:
    filepath: "petsard_Loader[local_members]_Preprocessor[compensation].csv"

Preprocessor:
  default:
    method: 'default'

Synthesizer:
  default:
    method: 'default'

Postprocessor:
  default:
    method: 'default'

Reporter:
  save_synthetic:
    method: 'save_data'
    source: 'Postprocessor'
```

Run it with the same three lines of Python:

```python
from petsard import Executor

executor = Executor(config="synthesis.yaml")
executor.run()
```

The default Preprocessor applies the default processing sequence, which includes encoding `region` and scaling the numeric columns; the Synthesizer generates synthetic data with PETsARD Gaussian Copula; and the Postprocessor restores the original format. The synthetic data is saved as `petsard_Loader[compensated]_Preprocessor[default]_Synthesizer[default]_Postprocessor[default].csv`. Evaluation can be added as described in [Data Synthesis and Evaluation with Default Parameters]({{< ref "/docs/getting-started/default-synthesis-default-evaluation" >}}).

## Execution Flow

1. **Loader** loads `local_members.csv`.
2. **Preprocessor** loads `reference_members.csv`, fits the Median, KNN or MICE strategy once, and fills the missing cells of `feature_columns` in the local data.
3. **Reporter** saves the compensated data.
4. A second workflow loads the compensated file and synthesizes data with the default parameters.

## Advanced Usage

- **Splitter**: with a Splitter before the Preprocessor, one fitted strategy compensates both the `train` and the `validation` branch. See [Working with Splitter]({{< ref "/docs/petsard-yaml/preprocessor-yaml/missing-compensation#working-with-splitter" >}}).
- **Chaining**: `method: composite` runs a built-in outlier stage, the compensation stage and a built-in missing value stage in one experiment, for example to fill a categorical column after compensation. See [Chaining with Other Preprocessing Steps]({{< ref "/docs/petsard-yaml/preprocessor-yaml/missing-compensation#chaining-with-other-preprocessing-steps" >}}).
- **Explicit masks**: the Python API compensates only the cells selected by a boolean mask. See [Python API]({{< ref "/docs/petsard-yaml/preprocessor-yaml/missing-compensation#python-api" >}}).
- **Own data**: replace the two file paths, list the numeric columns of your data in `feature_columns`, and list identifiers, labels and other unchanged columns in `key_columns`, `target_columns` and `protected_columns`.
