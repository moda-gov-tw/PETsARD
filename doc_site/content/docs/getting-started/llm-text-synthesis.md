---
title: Local LLM Text Synthesis and Text Evaluation
type: docs
weight: 150
prev: docs/getting-started/json-advgan-synthesis
next: docs/getting-started/missing-compensation
---

This tutorial generates synthetic free text with the LLM text synthesizer running on a local large language model (LLM) in LM Studio, and then evaluates the result with the text evaluation module. It uses a sample of blood pressure records, and all inference stays on your own machine.

You will:

1. Confirm the environment and the LM Studio server.
2. Review the blood pressure sample data.
3. Run a small first synthesis of 5 records.
4. Run a full synthesis with a variation and selection round.
5. Evaluate the synthetic text by parsing it into columns.

## Step 1: Prepare the Environment

Follow [Extended Features Setup]({{< ref "/docs/installation/extended-features" >}}) to install PETsARD from the source code in editable mode, install the packages of the LLM text synthesizer, and set up LM Studio. In short, from the source root (the folder that contains `pyproject.toml`) with the virtual environment activated:

```bash
pip install -e ".[all,advgan,llm]"
```

In LM Studio, download a model (this tutorial uses `google/gemma-4-12b-qat`; `openai/gpt-oss-20b` also works), load it with a context length of at least 32,768 tokens, set Max Concurrent Predictions to 2, and start the local server. Then confirm the server and the model identifier:

{{< tabs items="Windows,macOS / Linux" >}}
{{< tab >}}
```bat
curl.exe http://localhost:1234/v1/models
```
{{< /tab >}}
{{< tab >}}
```bash
curl http://localhost:1234/v1/models
```
{{< /tab >}}
{{< /tabs >}}

The response lists the `id` of every available model. Use one of these identifiers as `model` in the YAML below.

{{< callout type="info" >}}
Run every command in this tutorial from the source root. The synthesizer module path and the parser path in the YAML files are relative to the working directory; the sample data is downloaded to `benchmark/` under the working directory through the benchmark dataset protocol.
{{< /callout >}}

## Step 2: Look at the Sample Data

The sample data is the benchmark dataset `bloodpressure-text` (file `real_bloodpressure2.csv`). It has one column, `text`, and 101 rows. Each row is one Chinese sentence giving a person's name, sex, age, blood pressure reading and its interpretation, for example:

```text
患者為45歲男性，姓名：陳柏宇，血壓138/88偏高
黃志豪是一位38歲男性，量測血壓為118/76，正常
```

The names and readings are fictitious. The sentence patterns vary from record to record. Readings with a systolic pressure of 130 mmHg or above are labelled 偏高 (high, 62 records) and the others 正常 (normal, 39 records).

To use your own data, prepare a CSV file with one record per row. A file with a single text column is synthesized as shown here; for a text column next to structured columns, see `main_column_name` in [LLM Text Synthesizer]({{< ref "/docs/petsard-yaml/synthesizer-yaml/llm-synthesizer" >}}).

## Step 3: Run a First Synthesis

Save the following as `llm_text_quickstart.yaml` in the source root:

```yaml
Loader:
  load_bloodpressure:
    filepath: benchmark://bloodpressure-text
Synthesizer:
  llm-synth:
    method: custom_method
    module_path: petsard/synthesizer/LLM_synthesizer.py
    class_name: LLM_synthesizer
    sample_num_rows: 5
    epochs: 0
    api_type: lmstudio
    api_url: "http://localhost:1234/v1"
    model: google/gemma-4-12b-qat
    num_procs: 2
Reporter:
  save_synth:
    method: save_data
    source: Synthesizer
    output: llm_quickstart
```

Run it with the `Executor`:

```python
from petsard import Executor

exec = Executor(config="llm_text_quickstart.yaml")
exec.run()
```

or directly from the command line:

{{< tabs items="Windows,macOS / Linux" >}}
{{< tab >}}
```bat
python -X utf8 -c "from petsard import Executor; Executor('llm_text_quickstart.yaml').run()"
```
{{< /tab >}}
{{< tab >}}
```bash
python -c "from petsard import Executor; Executor('llm_text_quickstart.yaml').run()"
```
{{< /tab >}}
{{< /tabs >}}

### YAML Parameters

- **`Loader`**: loads the sample CSV; the `text` column is read as one free-text field per row.
- **`Synthesizer`**:
  - `method: custom_method`, `module_path` and `class_name` load the LLM text synthesizer.
  - `sample_num_rows: 5`: number of synthetic records to return.
  - `epochs: 0`: runs the few-shot initial generation only. Each record is produced by one LLM call that shows the model `fewshot_n_examples` (default 10) training records as examples.
  - `api_type: lmstudio` and `api_url`: send the prompts to the local LM Studio server.
  - `model`: the identifier returned by `/v1/models`.
  - `num_procs: 2`: parallel requests, matched with Max Concurrent Predictions in LM Studio.
- **`Reporter`**: saves the synthetic records. The synthesizer passes its output straight to the Reporter, so `source` is `Synthesizer`.

### Check the Output

The run creates:

| Location | Content |
|---|---|
| `llm_quickstart_Loader[load_bloodpressure]_Synthesizer[llm-synth].csv` | The 5 synthetic records, with the `text` column |
| `llm_synth_output/0/samples.csv` | Records of the initial generation |
| `llm_synth_output/0/prompt_generation.jsonl` | Prompts sent and texts generated |
| `llm_synth_output/fid.csv` | FID between the synthetic and the training texts in embedding space; lower is closer |
| `LOG_FILES/MainSyntheticData_<YYYYMMDD_HHMMSS>.txt` | Prompts and responses of every LLM call |

The following command prints the number of records, the number of records identical to a training record, and the number of records that still contain the mask symbol `<MASK>`. The expected result is `5 0 0`:

```bash
python -c "import pandas as pd; r=pd.read_csv('benchmark/real_bloodpressure2.csv'); s=pd.read_csv('llm_quickstart_Loader[load_bloodpressure]_Synthesizer[llm-synth].csv'); print(len(s), int(s['text'].isin(r['text']).sum()), int(s['text'].str.contains('<MASK>', regex=False).sum()))"
```

Open the CSV file and confirm that each record gives a name, sex, age, blood pressure reading and interpretation. The LLM log in `LOG_FILES/` shows every prompt and response; a call that still has no response after `max_retries` attempts is recorded with the text `failed after`. Setting `exit_on_api_error: true` stops the run at such a call, which is convenient while confirming the connection settings.

## Step 4: Run the Full Synthesis

With the environment confirmed, generate more records and add one variation and selection round. Save the following as `llm_text_synthesis.yaml`:

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
    model: google/gemma-4-12b-qat
    num_procs: 2
    additional_info: >-
      Each record describes one patient's blood pressure reading. Keep the
      wording of the examples, and label a reading as high when the systolic
      pressure is 130 mmHg or above and as normal otherwise.
Reporter:
  save_synth:
    method: save_data
    source: Synthesizer
    output: llm_text_result
```

What changes:

- `epochs: 1` adds one round after the initial generation. Each candidate receives `lookahead_degree` (default 2) variants, in which 10% of the words (`variation_degree_schedule`, default 0.1) are masked with `<MASK>` and filled in by the LLM. Every training record then votes for its nearest candidate in sentence-embedding space, Gaussian noise (`noise_multiplier`, default 8.0) is added to the votes, and the candidates are selected and varied once more.
- `additional_info` adds domain rules to every prompt.
- The run takes about `50 × (1 + 1 × (2 + 1)) = 200` LLM calls.

Run it in the same way as Step 3, with `llm_text_synthesis.yaml`. The synthetic records are saved as `llm_text_result_Loader[load_bloodpressure]_Synthesizer[llm-synth].csv`, and each round has its own folder in `llm_synth_output/` (`0/`, `1/` and `1_all/`).

## Step 5: Evaluate the Synthetic Text

PETsARD evaluators work on columns, so the text is first parsed into typed fields. The source code includes a parser for the blood pressure text, `petsard/synthesizer/LLM_synth/utils/bloodpressure_text_parser.py`, which extracts the sex (`gender`), age (`age`), systolic pressure (`systolic`), diastolic pressure (`diastolic`) and interpretation (`bp_label`) of every record. How to write a parser is described in [Text Data Evaluation]({{< ref "/docs/petsard-yaml/evaluator-yaml/text-evaluation" >}}).

Then add the following `Evaluator` block to `llm_text_synthesis.yaml` and replace its `Reporter` block with the one below, which also saves the evaluation report:

```yaml
Evaluator:
  text_eval:
    method: custom_method
    module_path: petsard/evaluator/text_evaluator.py
    class_name: TextEvaluator
    text_parser: petsard/synthesizer/LLM_synth/utils/bloodpressure_text_parser.py
    output_dir: ./llm_synth_output/parsed
    Evaluator:
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

- `class_name: TextEvaluator` evaluates the original and the synthetic data.
- `text_parser` points to the parser; the same function parses both tables.
- The nested `Evaluator` block lists PETsARD evaluators with their usual arguments; here, the validity diagnosis and the fidelity report.
- `output_dir` saves the parsed tables `parsed_ori.csv` and `parsed_syn.csv` for inspection.

Run the file again. To evaluate the records of Step 4 without generating them again, add `continue: true` to the Synthesizer block; the run then reuses the finished generations in `llm_synth_output/`.

### Read the Results

| Location | Content |
|---|---|
| `llm_text_result_Reporter[global].csv` | One row per nested evaluator; `text_eval_evaluator` names it (`validity_check`, `fidelity_check`) and `text_eval_Score` holds its overall score |
| `llm_text_result_Reporter[columnwise].csv` | Scores of each parsed field (`gender`, `age`, `systolic`, `diastolic`, `bp_label`) |
| `llm_synth_output/parsed/parsed_ori.csv`, `parsed_syn.csv` | The parsed tables |

1. **Parsing**: the log lists how many rows of each table have a value in every field, for example `syn: 50 rows -> gender=50/50, age=50/50, ...`. On the 101 sample records the parser fills all five fields in every row.
2. **Validity**: a score close to 1.0 means the parsed fields of the synthetic text stay within the categories and ranges of the original data.
3. **Fidelity**: a score of 0.75 or higher means the distributions and pairwise trends of the parsed fields follow the original data.

### Privacy Risk and Utility

To measure privacy risk and machine-learning utility, hold out control data with a Splitter and switch the class to `TextEvaluatorWithControl`:

```yaml
Splitter:
  holdout:
    num_samples: 1
    train_split_ratio: 0.8
```

```yaml
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
      inference_risk:
        method: anonymeter-inference
        secret: bp_label
      classification_utility:
        method: mlutility
        task_type: classification
        target: bp_label
```

Set `sample_num_rows` close to the size of the training split (80 records here). The complete configuration and the parser contract are described in [Text Data Evaluation]({{< ref "/docs/petsard-yaml/evaluator-yaml/text-evaluation" >}}).

## Adjusting the Synthesis

| Goal | Setting |
|---|---|
| Follow the training data more closely | Describe field rules in `additional_info`; set `epochs` to 1 or more |
| Increase diversity | `variation_degree_schedule: 0.25`, `temperature: 1.1`, `select_syn_mode: prob` |
| Strengthen privacy protection | Raise `noise_multiplier` to between 12 and 16 and use `fewshot_n_examples: 5` |
| Reduce memory use | `feature_extractor_batch_size: 128` |
| Shorten the run | Lower `sample_num_rows`, `epochs` or `lookahead_degree` |

All parameters are listed in [LLM Text Synthesizer]({{< ref "/docs/petsard-yaml/synthesizer-yaml/llm-synthesizer" >}}).

{{< callout type="warning" >}}
**Data protection**: this tutorial keeps all prompts on the local LM Studio server. With `api_type` set to `openai`, `anthropic` or `gemini`, the prompts, including the training records used as examples, are sent to that external service, so use these API types for non-sensitive data only. The files in `llm_synth_output/` and `LOG_FILES/` contain training text; store them with the same care as the training data.
{{< /callout >}}
