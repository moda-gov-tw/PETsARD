---
title: "Text Data Evaluation"
type: docs
weight: 697
prev: docs/petsard-yaml/evaluator-yaml/privacy-mpuccs
next: docs/petsard-yaml/reporter-yaml
---

The text evaluation module evaluates synthetic free text, such as the output of the [LLM Text Synthesizer]({{< ref "/docs/petsard-yaml/synthesizer-yaml/llm-synthesizer" >}}), with the standard PETsARD evaluators. PETsARD evaluators work on columns: `secret`, `target`, `aux_cols` and the scored fields all refer to column names. The module therefore first converts the text into typed columns with a parser that you provide, and then runs any number of PETsARD evaluators on the parsed data and merges their results into one report.

The module is loaded through the standard `custom_method` interface from `petsard/evaluator/text_evaluator.py` and needs no changes to the PETsARD core.

## Evaluator Classes

PETsARD reads the required inputs from the evaluator class, so the module provides two classes. Choose the class by the evaluators you nest inside it:

| Class | Required data | Nested evaluators |
|---|---|---|
| `TextEvaluator` | Original data (`ori`) and synthetic data (`syn`) | `sdmetrics-diagnosticreport`, `sdmetrics-qualityreport`, `mpuccs` |
| `TextEvaluatorWithControl` | `ori`, `syn` and control data (`control`); the YAML includes a Splitter | `anonymeter-singlingout`, `anonymeter-linkability`, `anonymeter-inference`, `mlutility` |

The control data is original data held out from synthesis. Anonymeter compares its attacks on the training data with the same attacks on the control data, and MLUtility trains on `ori` and on `syn` and tests both models on `control`.

## Usage Examples

The examples evaluate synthetic blood pressure records generated from the benchmark dataset `bloodpressure-text` (see [LLM Text Synthesizer]({{< ref "/docs/petsard-yaml/synthesizer-yaml/llm-synthesizer" >}})) with the parser that ships with the source code, `petsard/synthesizer/LLM_synth/utils/bloodpressure_text_parser.py`; its rules are shown in [Writing a Text Parser](#writing-a-text-parser). Run the examples from the PETsARD source root.

### Validity and Fidelity

```yaml
Loader:
  load_bloodpressure:
    filepath: benchmark://bloodpressure-text
Synthesizer:
  llm-synth:
    method: custom_method
    module_path: petsard/synthesizer/LLM_synthesizer.py
    class_name: LLM_synthesizer
    sample_num_rows: 50
    epochs: 1
    api_type: lmstudio
    api_url: "http://localhost:1234/v1"
    model: openai/gpt-oss-20b
    num_procs: 4
Evaluator:
  text_eval:
    method: custom_method
    module_path: petsard/evaluator/text_evaluator.py
    class_name: TextEvaluator
    text_parser: petsard/synthesizer/LLM_synth/utils/bloodpressure_text_parser.py   # relative to the working directory
    output_dir: ./llm_synth_output/parsed   # parsed tables, for inspection
    Evaluator:                              # nested PETsARD evaluators
      validity_check:
        method: sdmetrics-diagnosticreport
      fidelity_check:
        method: sdmetrics-qualityreport
Reporter:
  save_synth:
    method: save_data
    source: Synthesizer
    output: llm_text_result
  save_eval:
    method: save_report
    granularity:
      - global
      - columnwise
    output: llm_text_result
```

### Privacy Risk and Utility

Evaluators that need control data use `TextEvaluatorWithControl` together with a Splitter:

```yaml
Loader:
  load_bloodpressure:
    filepath: benchmark://bloodpressure-text
Splitter:
  holdout:
    num_samples: 1
    train_split_ratio: 0.8
Synthesizer:
  llm-synth:
    method: custom_method
    module_path: petsard/synthesizer/LLM_synthesizer.py
    class_name: LLM_synthesizer
    sample_num_rows: 80          # about the size of the training split
    epochs: 1
    api_type: lmstudio
    api_url: "http://localhost:1234/v1"
    model: openai/gpt-oss-20b
    num_procs: 4
Evaluator:
  text_eval:
    method: custom_method
    module_path: petsard/evaluator/text_evaluator.py
    class_name: TextEvaluatorWithControl
    text_parser: petsard/synthesizer/LLM_synth/utils/bloodpressure_text_parser.py
    output_dir: ./llm_synth_output/parsed
    Evaluator:
      singling_out_risk:
        method: anonymeter-singlingout
      linkability_risk:
        method: anonymeter-linkability
        aux_cols:
          - [gender, age]
          - [systolic, diastolic]
      inference_risk:
        method: anonymeter-inference
        secret: bp_label
      classification_utility:
        method: mlutility
        task_type: classification
        target: bp_label
Reporter:
  save_eval:
    method: save_report
    granularity:
      - global
    output: llm_text_privacy
```

Every key of a nested evaluator other than `method` is passed to that evaluator unchanged, so the arguments are the same as in [Privacy Protection Assessment]({{< ref "/docs/petsard-yaml/evaluator-yaml/privacy" >}}), [Data Fidelity Assessment]({{< ref "/docs/petsard-yaml/evaluator-yaml/fidelity" >}}) and [Data Utility Assessment]({{< ref "/docs/petsard-yaml/evaluator-yaml/utility" >}}). An `aux_cols` value written as a list of two lists is converted to the pair of lists that Anonymeter expects.

## Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `method` | `string` | Required | Fixed value: `custom_method` |
| `module_path` | `string` | Required | `petsard/evaluator/text_evaluator.py`, resolved from the working directory or the Python path (the source root after an editable install) |
| `class_name` | `string` | Required | `TextEvaluator` or `TextEvaluatorWithControl` |
| `text_parser` | `string` | Required | Path of the parser `.py` file, relative to the working directory |
| `text_parser_entry` | `string` | Optional | Name of the parser function; default `text_parser` |
| `Evaluator` | `dict` | Required | Nested block of PETsARD evaluators, written like the top-level Evaluator block: `{name: {method: ..., other arguments}}` |
| `output_dir` | `string` | Optional | Folder for the parsed tables `parsed_ori.csv`, `parsed_syn.csv` and, with `TextEvaluatorWithControl`, `parsed_control.csv` (UTF-8 with BOM) |

Write the arguments of each evaluator inside the nested `Evaluator` block, next to its `method`. The module reads only the keys listed above at the top level and logs any other top-level key with the message `Ignoring top-level keys`. The nested block is named `Evaluator`; the name `eval_method` is reserved by the `custom_method` interface.

## Writing a Text Parser

The parser is a plain Python file with one entry function:

```python
def text_parser(df: pd.DataFrame) -> pd.DataFrame:
    ...
```

The parser contract:

1. **Input**: one table at a time (`ori`, `syn` or `control`) with the columns of the synthesized data. The same function parses every table, so all of them are converted by identical rules.
2. **Output**: a `pd.DataFrame` with exactly one row per input row, preferably on the same index. Each column is one extracted field. Use numeric types for numbers and short labels for categories, and return `None` or `NaN` for a field that is absent in a row.
3. **Columns to return**: return the structured fields only. The evaluators score every column they receive, and the extracted fields are what describes the data in tabular form.
4. **Type alignment**: after parsing, the module aligns the type of each shared column across the tables (integer and float become `float64`, dates become `datetime64[ns]`, mixed types become `object`) and orders the columns like `ori`, so a field that is missing in some rows of one table compares cleanly with the others.

The following parser extracts five fields from the blood pressure records: the sex, the age, the systolic and diastolic pressures and the interpretation (偏高 high or 正常 normal).

```python
"""Text parser for blood pressure records."""

import re

import pandas as pd

SOURCE_COLUMN = "text"

AGE_RE = re.compile(r"(\d{1,3})\s*歲")
BP_RE = re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})")
GENDER_RULES = [
    ("女", re.compile(r"性別[=：:]?\s*女|女性|[（(]\s*女")),
    ("男", re.compile(r"性別[=：:]?\s*男|男性|[（(]\s*男")),
]
LABEL_RULES = [
    ("偏高", re.compile(r"偏高")),
    ("正常", re.compile(r"正常")),
]


def _first_label(value: str, rules) -> str | None:
    for label, pattern in rules:
        if pattern.search(value):
            return label
    return None


def _parse_row(value) -> dict:
    record = {"gender": None, "age": None, "systolic": None,
              "diastolic": None, "bp_label": None}
    if not isinstance(value, str) or not value.strip():
        return record

    record["gender"] = _first_label(value, GENDER_RULES)
    record["bp_label"] = _first_label(value, LABEL_RULES)
    if m := AGE_RE.search(value):
        record["age"] = int(m.group(1))
    if m := BP_RE.search(value):
        record["systolic"], record["diastolic"] = int(m.group(1)), int(m.group(2))
    return record


def text_parser(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row of typed fields for every input row."""
    parsed = pd.DataFrame([_parse_row(v) for v in df[SOURCE_COLUMN]], index=df.index)
    for column in ("age", "systolic", "diastolic"):
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
    return parsed
```

The bundled `petsard/synthesizer/LLM_synth/utils/bloodpressure_text_parser.py` extracts the same five fields by the same rules, and the examples above point to that file. On the 101 sample records the parser fills all five fields in every row. To adapt it to other text, change `SOURCE_COLUMN` and the regular expressions.

## Processing Flow

1. **Parse**: the parser runs on each input table; the module checks that it returns a DataFrame with the same number of rows.
2. **Align types**: shared columns receive a common type and the column order of `ori`.
3. **Report coverage**: the log records, for every table, how many rows have a value in each column, in the form `syn: 50 rows -> gender=50/50, age=50/50, ...`.
4. **Save parsed tables**: when `output_dir` is set, the parsed tables are written for inspection.
5. **Evaluate**: each nested evaluator runs on its own copy of the parsed tables; the log records `<name> (<method>): ok` for each completed evaluator.
6. **Merge**: the results of all nested evaluators are combined per granularity (`global`, `columnwise`, `pairwise`, `details`), with an added `evaluator` column naming the nested evaluator of each row.

The nested evaluators run independently, so the results of every completed evaluator are kept in the report. When a nested evaluator reports an error, the log records the message and the `global` table lists that evaluator with the message in an `error` column, which makes a misconfigured argument easy to locate.

## Reading the Results

The Reporter adds the outer evaluation name as a column prefix. With the outer name `text_eval`, the column `text_eval_evaluator` names the nested evaluator of each row (for example `validity_check` or `fidelity_check`), and the other columns are the report columns of that evaluator, read in the same way as in the corresponding evaluation pages. A granularity that none of the nested evaluators produces is left out of the report.

Recommended checks:

1. **Parse coverage**: in the log, confirm that every field is filled in most rows of each table. A field that is empty in every row is reported with the warning `columns empty in every row`; refine the parser for that field before reading its scores.
2. **Parsed tables**: open `parsed_ori.csv` and `parsed_syn.csv` and compare a few rows with the original text.
3. **Scores**: read validity, fidelity, privacy risk and utility with the thresholds in [Evaluator YAML]({{< ref "/docs/petsard-yaml/evaluator-yaml" >}}). For privacy and utility, set the synthesizer's `sample_num_rows` close to the size of the training split.
