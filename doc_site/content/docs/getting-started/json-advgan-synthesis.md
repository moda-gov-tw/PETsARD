---
title: JSON Data and AdvGAN Synthesis
type: docs
weight: 140
prev: docs/getting-started/external-synthesis-default-evaluation
next: docs/getting-started/llm-text-synthesis
---

Semi-structured data synthesis with JSON input and output.
This example loads JSON records, synthesizes them with the AdvGAN synthesizer, evaluates validity, fidelity, privacy and utility, and saves both the synthetic data and the evaluation report as JSON.

{{< callout type="info" >}}
This example uses the extended features of this edition. Prepare the environment with [Extended Features Setup](../../installation/extended-features), including the `opacus` package, and run the commands below from the source root.
{{< /callout >}}

## Demo Data

The example uses the benchmark dataset `customer-churn` (file `customer_churn.json`), which is downloaded to `benchmark/` under the working directory when the configuration runs: 1,200 customer records with 10 fields, stored as a JSON array of records. The data is generated for demonstration, with a learnable relationship between contract type, tenure, charges and `churn`.

```json
[
  {
    "age": 66,
    "region": "East",
    "contract_type": "Month-to-month",
    "payment_method": "Bank-transfer",
    "tenure_months": 39,
    "num_products": 2,
    "monthly_charges": 38.82,
    "total_charges": 1570.96,
    "has_support": "No",
    "churn": "Yes"
  }
]
```

| Field | Type | Values |
|---|---|---|
| `age` | Integer | 18 to 80 |
| `region` | Category | `Central`, `East`, `North`, `South`, `West` |
| `contract_type` | Category | `Month-to-month`, `One-year`, `Two-year` |
| `payment_method` | Category | `Bank-transfer`, `Credit-card`, `E-check`, `Mailed-check` |
| `tenure_months` | Integer | 1 to 72 |
| `num_products` | Integer | 1 to 4 |
| `monthly_charges` | Decimal | 19.04 to 117.99 |
| `total_charges` | Decimal | 19.66 to 8,309.74 |
| `has_support` | Category | `No`, `Yes` |
| `churn` | Category | `No`, `Yes` (target field) |

## YAML Configuration

Save the following configuration as `json-advgan-synthesis.yaml` in the source root:

```yaml
Loader:
  load_json:
    filepath: benchmark://customer-churn
Splitter:
  holdout:
    num_samples: 1
    train_split_ratio: 0.8
Preprocessor:
  default:
    method: 'default'
Synthesizer:
  advgan:
    method: 'AdvGAN'
    sigma: 1.0
    outlier: 10
    epochs: 5
    sample_num_rows: 960
Postprocessor:
  default:
    method: 'default'
Evaluator:
  diagnostic:
    method: sdmetrics-diagnosticreport
  quality:
    method: sdmetrics-qualityreport
  singling_out:
    method: anonymeter-singlingout
    n_attacks: 100
    n_cols: 3
  classification_utility:
    method: mlutility
    task_type: classification
    target: churn
Reporter:
  save_synthetic:
    method: save_data
    source: Postprocessor
    output: churn_advgan
    type: 'json'
  evaluation_report:
    method: save_report
    granularity: global
    output: churn_advgan
    type: 'json'
```

## Running the Example

Save the following script as `run_json_advgan.py` in the source root:

```python
from petsard import Executor

exec = Executor(config="json-advgan-synthesis.yaml")
exec.run()
```

{{< tabs items="Windows,macOS / Linux" >}}
{{< tab >}}
```bat
python -X utf8 run_json_advgan.py
```
{{< /tab >}}
{{< tab >}}
```bash
python run_json_advgan.py
```
{{< /tab >}}
{{< /tabs >}}

During training, the terminal shows the number of training rows after preprocessing and the device in use, followed by the progress of the evaluation reports:

```
The number of data: <number of training rows>
Using device: cpu
```

## YAML Parameters Detailed Explanation

### Loader (Data Loading Module)

- **`load_json`**: Experiment name, can be freely named
- **`filepath`**: Data file path
  - Value: `benchmark://customer-churn`
  - Description: Downloads `customer_churn.json` through the [benchmark dataset protocol](../../petsard-yaml/loader-yaml/benchmark-protocol) and verifies its SHA-256. The `.json` extension selects the JSON reader automatically. The file is a top-level array of records; each object is one row and each key is one column
  - Nested JSON and JSON Lines are prepared as described in [JSON Format](../../petsard-yaml/loader-yaml/json-format)

### Splitter (Data Splitting Module)

- **`holdout`**: Experiment name, can be freely named
- **`num_samples`**: `1`, one train and control split
- **`train_split_ratio`**: `0.8`, so 960 records form the training set and 240 records form the control set
- Description: The control set is used by the privacy and utility evaluators (`anonymeter-singlingout` and `mlutility`), which compare the synthetic data with records that were not used for training

### Preprocessor (Data Preprocessing Module)

- **`default`**: Experiment name, can be freely named
- **`method`**: `default`
- Description: Handles missing values, handles outliers, encodes categorical fields and scales numeric fields. AdvGAN trains on a fully numeric table, so the Preprocessor is part of every AdvGAN pipeline

### Synthesizer (Synthetic Data Generation Module)

- **`advgan`**: Experiment name, can be freely named
- **`method`**: `AdvGAN`, the improved AdvGAN synthesizer whose discriminator is trained with DP-SGD through Opacus
- **`sigma`**: `1.0`, the DP-SGD noise multiplier of the discriminator (default `1.0`); a larger value gives stronger privacy protection
- **`outlier`**: `10`, the weight of the boundary penalty (default `10`); a smaller value lets the generator produce more values near the extremes of each field
- **`epochs`**: `5`, the number of training epochs (default `200`); the example uses 5 epochs so that it completes quickly
- **`sample_num_rows`**: `960`, the number of synthetic records, equal to the size of the training set

For the algorithm and all parameters, see [AdvGAN Synthesizer](../../petsard-yaml/synthesizer-yaml/advgan).

### Postprocessor (Data Postprocessing Module)

- **`default`**: Experiment name, can be freely named
- **`method`**: `default`
- Description: Reverses scaling and encoding, so the synthetic records return to original units with the original category labels such as `Month-to-month` and `Yes`

### Evaluator (Data Evaluation Module)

| Experiment | `method` | Aspect | Main parameters |
|---|---|---|---|
| `diagnostic` | `sdmetrics-diagnosticreport` | Data validity and structure | None |
| `quality` | `sdmetrics-qualityreport` | Fidelity: column shapes and column pair trends | None |
| `singling_out` | `anonymeter-singlingout` | Privacy: singling-out risk, using the control set | `n_attacks: 100`, `n_cols: 3` |
| `classification_utility` | `mlutility` | Utility: models trained on synthetic data and tested on the control set | `task_type: classification`, `target: churn` |

### Reporter (Output Module)

- **`save_synthetic`**: Saves the output of `Postprocessor` with `method: save_data`
- **`evaluation_report`**: Saves the global evaluation scores with `method: save_report` and `granularity: global`
- **`output`**: `churn_advgan`, the file name prefix of both outputs
- **`type`**: `json`, writes both outputs as JSON arrays of records; without `type`, the outputs are written as CSV

## Execution Flow

1. **Loader** reads `customer_churn.json` with the JSON reader
2. **Splitter** splits the data into a training set (80%) and a control set (20%)
3. **Preprocessor** turns the training set into a fully numeric table
4. **Synthesizer** trains AdvGAN with a DP-SGD discriminator and generates 960 synthetic records
5. **Postprocessor** restores original units and category labels
6. **Evaluator** runs the validity, fidelity, privacy and utility evaluations
7. **Reporter** saves the synthetic data and the evaluation report as JSON

## Output Files

The two files are written to the source root:

| File | Content |
|---|---|
| `churn_advgan_Loader[load_json]_Splitter[holdout_[1-1]]_Preprocessor[default]_Synthesizer[advgan]_Postprocessor[default].json` | 960 synthetic records with the same 10 fields as the input |
| `churn_advgan_Reporter[global].json` | Global evaluation scores of all four evaluators |

The synthetic data keeps the layout of the input file, so it can be loaded again by Loader or used by other JSON tools directly:

```json
[
    {
        "age":38,
        "region":"North",
        "contract_type":"Two-year",
        "payment_method":"Credit-card",
        "tenure_months":54,
        "num_products":1,
        "monthly_charges":28.45,
        "total_charges":7017.77,
        "has_support":"No",
        "churn":"No"
    }
]
```

In the evaluation report, each record lists the experiment names of every module, followed by score fields prefixed with the evaluator experiment name:

| Field | Meaning | Reference standard |
|---|---|---|
| `diagnostic_Score` | Validity diagnosis score | Should approach 1.0 |
| `quality_Score` | Fidelity score, with `quality_Column Shapes` and `quality_Column Pair Trends` as its components | Above 0.75 |
| `singling_out_risk` | Singling-out risk, with its confidence interval in `singling_out_risk_CI_btm` and `singling_out_risk_CI_top` | Below 0.09 |
| `classification_utility_ori`, `classification_utility_syn` | Score of a model trained on the original training set and on the synthetic data, both tested on the control set; `classification_utility_metric` names the metric | Compare `_syn` with `_ori`; `classification_utility_diff` is `_syn` minus `_ori` |

For the meaning of each score, see [Data Synthesis and Evaluation with Default Parameters](../default-synthesis-default-evaluation).

## Adjusting the Example

- **Full training**: Increase `epochs` toward the default `200`
- **Privacy and utility balance**: Adjust `sigma`; a larger value strengthens the privacy protection of the discriminator
- **Extreme values**: Adjust `outlier`; a smaller value lets more values near the extremes of each field be synthesized
- **Your own data**: Point `filepath` to your JSON file, set `sample_num_rows` to the size of its training set, and add a `schema` to fix field types (see [JSON Format](../../petsard-yaml/loader-yaml/json-format#using-schema-with-json))
- **Comparison with another synthesizer**: Add a second experiment, such as `gaussian-copula` with `method: 'default'`, under `Synthesizer`; every downstream module then produces one result for each synthesizer
- **CSV output**: Remove `type: 'json'` from the Reporter blocks

## Advanced Usage

- [JSON Format](../../petsard-yaml/loader-yaml/json-format)
- [AdvGAN Synthesizer](../../petsard-yaml/synthesizer-yaml/advgan)
- [Save Data](../../petsard-yaml/reporter-yaml/save-data)
- [Evaluator YAML Parameter Guide](../../petsard-yaml/evaluator-yaml/)
