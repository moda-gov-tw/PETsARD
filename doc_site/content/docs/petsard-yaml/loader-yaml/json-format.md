---
title: "JSON Format"
type: docs
weight: 622
prev: docs/petsard-yaml/loader-yaml/benchmark-protocol
next: docs/petsard-yaml/loader-yaml
---

Loader reads semi-structured JSON data directly. When the `filepath` ends in `.json`, Loader selects the JSON reader automatically, so no extra parameter is needed.

## Usage Examples

The examples use the benchmark dataset `customer-churn` (file `customer_churn.json`), which is downloaded to `benchmark/` under the working directory when the configuration runs. To read your own data, set `filepath` to the path of a `.json` file.

### Basic Loading

```yaml
Loader:
  load_json:
    filepath: benchmark://customer-churn
```

### Using Schema File

```yaml
Loader:
  load_json_with_schema:
    filepath: benchmark://customer-churn
    schema: customer_churn_schema.yaml
```

### JSON In, JSON Out

Combine the JSON reader with `type: 'json'` in the Reporter to keep the whole pipeline in JSON:

```yaml
Loader:
  load_json:
    filepath: benchmark://customer-churn
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
    method: save_data
    source: Postprocessor
    type: 'json'
```

The synthetic data is written to `petsard_Loader[load_json]_Preprocessor[default]_Synthesizer[default]_Postprocessor[default].json`. For the `type` option, see [Save Data](../../reporter-yaml/save-data).

## Input Format

The JSON file contains a top-level array of objects (the `records` layout):

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
  },
  {
    "age": 51,
    "region": "North",
    "contract_type": "One-year",
    "payment_method": "Bank-transfer",
    "tenure_months": 28,
    "num_products": 2,
    "monthly_charges": 49.64,
    "total_charges": 1440.55,
    "has_support": "No",
    "churn": "No"
  }
]
```

| JSON element | Loaded as |
|---|---|
| Each object in the array | One row (record) |
| Each key | One column; the key is the column name |
| Number, string, boolean | The cell value; types are inferred or taken from the schema |
| `null`, or a key absent from a record | A missing value (`pd.NA`) |

- Files are read as UTF-8. Column names and values in any language, including Traditional Chinese, are kept as they are through to the JSON output.
- The file extension is matched case-insensitively, so `.json` and `.JSON` both select the JSON reader.
- Represent missing values as `null`. The Preprocessor then handles them in the same way as missing values from CSV files.

## Preparing JSON Data

Loader works on flat records, in which every value is a scalar. Prepare other JSON layouts with pandas before loading.

### Flattening Nested JSON

Nested objects are expanded into flat columns with `pandas.json_normalize`. Each nested key path becomes one column name:

```json
[
  {
    "age": 66,
    "region": "East",
    "contract": {"type": "Month-to-month", "tenure_months": 39},
    "billing": {"payment_method": "Bank-transfer", "monthly_charges": 38.82},
    "churn": "Yes"
  }
]
```

```python
import json
import pandas as pd

with open("customers_nested.json", encoding="utf-8") as f:
    records = json.load(f)

flat = pd.json_normalize(records, sep="_")
flat.to_json("customers_flat.json", orient="records", force_ascii=False, indent=2)
```

The flattened file has the columns `age`, `region`, `churn`, `contract_type`, `contract_tenure_months`, `billing_payment_method` and `billing_monthly_charges`, and is loaded with `filepath: customers_flat.json`.

For arrays nested inside a record, use the `record_path` and `meta` arguments of `pandas.json_normalize` to produce one row per array element, or aggregate the array into scalar columns (for example a count or a total) before saving.

### Converting JSON Lines

A JSON Lines file (`.jsonl`, one object per line) is converted into a records array and saved with the `.json` extension:

```python
import pandas as pd

pd.read_json("data.jsonl", lines=True).to_json(
    "data.json", orient="records", force_ascii=False, indent=2
)
```

### Converting CSV to JSON

```python
import pandas as pd

pd.read_csv("data.csv").to_json(
    "data.json", orient="records", force_ascii=False, indent=2
)
```

## Using Schema with JSON

Schema works with JSON files in the same way as with CSV files. It fixes the type of each column, marks categorical columns and sets numeric precision. The following schema describes the customer churn data; save it as `customer_churn_schema.yaml` in the working directory to use it with the schema example above:

```yaml
id: customer-churn
name: "Customer Churn"
description: "Customer contract and churn records"

attributes:
  age:
    type: integer
  region:
    type: string
    category: true
  contract_type:
    type: string
    category: true
  payment_method:
    type: string
    category: true
  tenure_months:
    type: integer
  num_products:
    type: integer
  monthly_charges:
    type: float
    type_attr:
      precision: 2
  total_charges:
    type: float
    type_attr:
      precision: 2
  has_support:
    type: string
    category: true
  churn:
    type: string
    category: true
```

- `string` and `float` columns are read with the declared type; `integer` columns are converted when the data is aligned with the schema.
- Columns present in the data but not declared in the schema are inferred and added to the schema automatically.
- Without a schema, Loader infers the type, category flag and precision of every column from the data.

For all schema options, see [Schema YAML](../../../schema-yaml/).

## Parameters

The JSON reader uses the following Loader parameters:

| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `filepath` | `string` | N/A | Path of the JSON file; the `.json` extension selects the JSON reader | `benchmark://customer-churn` |
| `schema` | `string\|dict` | `null` | Data structure definition, as a YAML file path or an inline dict | `customer_churn_schema.yaml` |

## How It Works

1. **Format detection**: Loader reads the extension of `filepath`; `.json` maps to the JSON reader `LoaderPandasJson`
2. **Reading**: The file is read with `pandas.read_json`, and the column types declared in the schema are applied
3. **Missing values**: `null` values are stored as `pd.NA`
4. **Schema processing**: The schema is inferred from the data, or merged with the supplied schema, and the data is aligned to it
5. **Downstream use**: The resulting table is passed to Splitter, Preprocessor and Synthesizer exactly like data loaded from CSV

## Related Information

- **Saving JSON output**: Set `type: 'json'` in the Reporter, see [Save Data](../../reporter-yaml/save-data)
- **Semi-structured synthesis**: The AdvGAN synthesizer works directly on JSON-sourced data, see [AdvGAN Synthesizer](../../synthesizer-yaml/advgan)
- **End-to-end tutorial**: [JSON Data and AdvGAN Synthesis](../../../getting-started/json-advgan-synthesis)

## Notes

- Use a top-level array of flat objects; flatten nested objects and convert JSON Lines first, as shown above
- Keep the same set of keys in every record; a key missing from a record is loaded as a missing value
- File paths support both relative and absolute paths
- On Windows, run Python in UTF-8 mode (`python -X utf8`) when YAML or schema files contain non-ASCII characters
