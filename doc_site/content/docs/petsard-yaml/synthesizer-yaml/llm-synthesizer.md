---
title: "LLM Text Synthesizer"
type: docs
weight: 657
prev: docs/petsard-yaml/synthesizer-yaml/advgan
next: docs/petsard-yaml/synthesizer-yaml
---

The LLM text synthesizer generates synthetic free-text records with a large language model (LLM). It is designed for datasets in which each row is one self-contained piece of text, such as customer feedback, survey answers, support tickets, or system and network logs. The synthesizer prompts the model with examples drawn from the training data, refines the candidates over several rounds, and selects them with a differentially private nearest-neighbor vote, so no model fine-tuning is required.

The synthesizer is loaded through the standard `custom_method` interface (see [Custom Synthesis Method]({{< ref "/docs/petsard-yaml/synthesizer-yaml/custom-method" >}})). It runs with a local model served by [LM Studio](https://lmstudio.ai/) or with the OpenAI, Anthropic and Gemini APIs.

{{< callout type="info" >}}
For a step-by-step walkthrough, including LM Studio setup and evaluation of the generated text, see [Local LLM Text Synthesis and Text Evaluation]({{< ref "/docs/getting-started/llm-text-synthesis" >}}).
{{< /callout >}}

## Requirements

The environment is prepared as described in [Extended Features Setup]({{< ref "/docs/installation/extended-features" >}}):

1. PETsARD installed from the source root in editable mode, together with the synthesizer's dependencies (extra `llm`):

   ```bash
   pip install -e ".[all,advgan,llm]"
   ```

2. An LLM endpoint:
   - **Local inference (recommended)**: LM Studio with a model loaded and its local server started (default address `http://localhost:1234/v1`).
   - **Cloud inference**: an API key for OpenAI, Anthropic or Google Gemini.
3. With `compute_fid: true` (the default), the sentence-embedding model `all-MiniLM-L6-v2` is downloaded from Hugging Face on the first run and cached locally.

## Usage Examples

Run the examples from the PETsARD source root; the module paths in the examples are relative to it.

### Synthesizing Whole Records

The example data is the benchmark dataset `bloodpressure-text` (file `real_bloodpressure2.csv`, downloaded when the configuration runs). It has a single `text` column with 101 rows. Each row is one Chinese sentence giving a person's name, sex, age, blood pressure reading and its interpretation (high or normal).

```yaml
Loader:
  load_bloodpressure:
    filepath: benchmark://bloodpressure-text
Synthesizer:
  llm-synth:
    method: custom_method
    module_path: petsard/synthesizer/LLM_synthesizer.py
    class_name: LLM_synthesizer
    sample_num_rows: 20            # number of synthetic records returned
    epochs: 1                      # initial generation + 1 variation/selection round
    api_type: lmstudio             # local inference
    api_url: "http://localhost:1234/v1"
    model: openai/gpt-oss-20b      # identifier of the model loaded in LM Studio
    num_procs: 4                   # match LM Studio's Max Concurrent Predictions
    additional_info: >-
      Each record describes one patient's blood pressure reading. Keep the
      wording of the examples, and label a reading as high when the systolic
      pressure is 130 mmHg or above and as normal otherwise.
Reporter:
  save_synth:
    method: save_data
    source: Synthesizer
    output: llm_synth_result
```

The synthetic records are saved as `llm_synth_result_Loader[load_bloodpressure]_Synthesizer[llm-synth].csv`, with the same columns as the input.

### Synthesizing One Text Column

When the data has a free-text column next to structured attributes, set `main_column_name` to generate only that column. For example, with a file `customer_feedback.csv` that has the columns `channel`, `rating` and `feedback`:

```yaml
Loader:
  load_feedback:
    filepath: customer_feedback.csv
Synthesizer:
  llm-feedback:
    method: custom_method
    module_path: petsard/synthesizer/LLM_synthesizer.py
    class_name: LLM_synthesizer
    main_column_name: feedback     # only this column is generated
    sample_num_rows: 50
    epochs: 1
    api_type: lmstudio
    api_url: "http://localhost:1234/v1"
    model: google/gemma-4-12b-qat
    num_procs: 4
Reporter:
  save_synth:
    method: save_data
    source: Synthesizer
    output: llm_feedback_result
```

Each generated `feedback` text is modeled on one training record (the query example described below). The `channel` and `rating` values of that record are carried over, so every synthetic text stays paired with the attributes of the record it was modeled on.

### Using a Cloud API

Set `api_type` to `openai`, `anthropic` or `gemini` and choose a model. Set the API key in the environment variable `PETSARD_LLM_API_KEY` before running PETsARD (for example `export PETSARD_LLM_API_KEY=...` in bash or `$env:PETSARD_LLM_API_KEY = "..."` in PowerShell), so that the key stays out of the YAML file:

```yaml
    api_type: openai
    model: gpt-4o
```

{{< callout type="warning" >}}
**Data protection**: with `api_type` set to `openai`, `anthropic` or `gemini`, every prompt, including the training records used as few-shot examples, is sent to that external service. Use these API types for non-sensitive data, and use `lmstudio` to keep sensitive data on your own machine. The files `prompt_generation.jsonl` and the logs in `log_folder` also contain training text; store them with the same care as the training data. With `log_prompts: false`, the log records only the length of each text and `prompt_generation.jsonl` is not written.
{{< /callout >}}

## How It Works

1. **Fit**: the synthesizer stores the training rows as the pool of private examples. In whole-record mode each row is serialized as one quoted CSV line and the column names are passed as a header; with `main_column_name`, only the text of that column is used.
2. **Initial generation (epoch 0)**: for every target record, `fewshot_n_examples` training records are drawn at random. The first ones form the *support set*, which shows the structure, style and length of the data; the last one is the *query set*, whose settings (for example record type and structure) the new record follows. The LLM writes one new record per call. The text of `additional_info` is added to every prompt inside `<AdditionalInfo>` tags, and the model can call two random-number tools (`random_random`, `random_uniform`) when it needs numeric values.
3. **Variation and selection rounds** (repeated `epochs` times):
   1. *Lookahead variation*: for each candidate, `lookahead_degree` variants are produced by replacing a fraction `variation_degree_schedule` of its words with `<MASK>` and asking the LLM to fill the blanks. CJK characters are masked one character at a time; punctuation and other structural symbols are kept as they are.
   2. *Embedding*: sentence embeddings (`feature_extractor`, default `all-MiniLM-L6-v2`) are computed for the training records and for each candidate, averaged over its lookahead variants.
   3. *Differentially private nearest-neighbor histogram*: every training record votes for its nearest candidate(s) (`nn_mode`, `num_nearest_neighbor`), and Gaussian noise with standard deviation `noise_multiplier × √num_nearest_neighbor` is added to the vote counts.
   4. *Selection*: `N / combine_divide_L` candidates are kept, either those with the highest noisy votes (`select_syn_mode: rank`) or by sampling in proportion to the votes (`select_syn_mode: prob`).
   5. *Next pool*: the kept candidates are varied again with the same mask-and-fill step to form the candidates of the next round.
4. **Output**: the final pool is converted back into a table with the original columns, empty generations are removed, and the first `sample_num_rows` records are returned to PETsARD.

With `combine_divide_L` at its default of 1, the number of LLM calls is about `sample_num_rows × (1 + epochs × (lookahead_degree + 1))`. For example, 20 records with `epochs: 1` and the default `lookahead_degree: 2` take about 80 calls.

## Parameters

### PETsARD Keys

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | `string` | Required | Fixed value: `custom_method` |
| `module_path` | `string` | Required | `petsard/synthesizer/LLM_synthesizer.py`, resolved from the working directory or the Python path (the source root after an editable install) |
| `class_name` | `string` | Required | `LLM_synthesizer` |
| `sample_num_rows` | `integer` | Number of training rows | Number of synthetic records returned. The internal schedule is sized automatically from this value and any surplus is trimmed. Set it explicitly, since each record takes several LLM calls |

### Data

| Parameter | Type | Default | Description |
|---|---|---|---|
| `main_column_name` | `string` | Not set | Generate only this text column; the other columns are carried over from the training record each text was modeled on. When not set, whole records are generated and parsed back into the original columns |

### LLM API

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_type` | `string` | None (required) | `lmstudio`, `openai`, `anthropic` or `gemini`. A configuration without `api_type` stops with `ValueError` |
| `api_url` | `string` | Not set | LM Studio server URL, for example `http://localhost:1234/v1`. Accepted only with `api_type: lmstudio`; `/v1` is appended when missing |
| `api_key` | `string` | Not set | API key of the cloud service. When not set, the key is read from the environment variable `PETSARD_LLM_API_KEY`. With `lmstudio` and no `api_url`, this value is used as the server URL; when both are unset, `http://localhost:1234/v1` is used. For an LM Studio server behind an authenticated endpoint, put the key here and the URL in `api_url` |
| `model` | `string` | Provider default | Model for the initial generation. Defaults: `gpt-4o` (openai), `claude-sonnet-4-20250514` (anthropic), `gemini-2.5-flash` (gemini). With `lmstudio`, set it to the identifier of the loaded model as listed by `/v1/models` |
| `variation_api_type`, `variation_api_key`, `variation_api_url`, `variation_model` | `string` | Initial-generation values | API settings of the variation (mask-and-fill) stage |
| `post_processing_api_type`, `post_processing_api_key`, `post_processing_api_url`, `post_processing_model` | `string` | Initial-generation values | API settings of the post-processing agent |

### Generation and Selection

| Parameter | Type | Default | Description |
|---|---|---|---|
| `epochs` | `integer` | `1` | Variation and selection rounds after the initial generation. `0` runs the few-shot initial generation only |
| `lookahead_degree` | `integer` | `2` | Lookahead variants per candidate before voting; `0` votes on the candidates themselves |
| `variation_degree_schedule` | `float` or `list` | `0.1` | Fraction of words masked in a variation, between 0 and 1. A single value applies to every round; a list gives one value per round (length `epochs + 1`) |
| `noise_multiplier` | `float` | `8.0` | Multiplier of the Gaussian noise added to the nearest-neighbor votes; larger values give stronger privacy protection |
| `compute_fid` | `boolean` | `true` | Compute sentence embeddings for the differentially private selection and record FID in `fid.csv`. Keep `true` for semantic selection; `false` skips the embeddings and selects with uniform votes, which suits quick pipeline checks |
| `feature_extractor` | `string` | `all-MiniLM-L6-v2` | Sentence-transformers model used for the embeddings |
| `feature_extractor_batch_size` | `integer` | `1024` | Batch size of the embedding computation; lower it (for example to `128`) to reduce memory use |
| `select_syn_mode` | `string` | `rank` | `rank` keeps the candidates with the highest noisy votes; `prob` samples candidates in proportion to their votes |
| `nn_mode` | `string` | `L2` | Distance used for the vote: `L2`, `IP` (inner product) or `cos_sim` |
| `num_nearest_neighbor` | `integer` | `1` | Number of nearest candidates each training record votes for |
| `count_threshold` | `float` | `0.0` | Threshold subtracted from the noisy vote counts before selection; counts at or below it become zero |
| `combine_divide_L` | `integer` | `1` | In each round, keep `N / combine_divide_L` candidates and refill the pool with `combine_divide_L - 1` variations of each kept candidate |
| `donnot_keep_last_iter` | `boolean` | `false` | With `combine_divide_L` above 1, `true` fills all `combine_divide_L` slots with new variations instead of carrying the kept candidate over |
| `init_combine_divide_L` | `integer` | `1` | In epoch 0, generate `N / init_combine_divide_L` seed records with the LLM and add `init_combine_divide_L - 1` variations of each |
| `lookahead_self` | `boolean` | `false` | Include the candidate itself together with its lookahead variants when computing its embedding |
| `num_samples_schedule` | `integer` or `list` | Automatic | Candidate pool size per round; computed from `sample_num_rows` when omitted |
| `num_private_samples` | `integer` | `-1` | Maximum number of training rows used as private examples; `-1` uses all rows |

### LLM Calls

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fewshot_n_examples` | `integer` | `10` | Training records included as examples in each prompt |
| `additional_info` | `string` | `""` | Domain rules added to every prompt inside `<AdditionalInfo>` tags, for example field formats or value ranges |
| `temperature` | `float` | `1.0` | Sampling temperature; higher values give more diverse text |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold |
| `max_token_limit` | `integer` | `100000` | Maximum output tokens per call. Reasoning models (for example `openai/gpt-oss-20b`) spend part of this budget on reasoning, so keep the value large |
| `num_procs` | `integer` | `8` | Parallel LLM requests. For a local LM Studio server, 2 to 4 is recommended, matched with LM Studio's Max Concurrent Predictions |
| `sleep_time` | `integer` | `1` | Seconds to wait before each call; also the base of the retry back-off |
| `max_retries` | `integer` | `3` | Attempts per LLM call, including the first; a call is retried when it raises an error or returns an empty response |
| `retry_wait_time` | `float` | `15` | Seconds to wait before each retry (fixed) |
| `exit_on_api_error` | `boolean` | `false` | `true` stops the run when a call still fails after `max_retries` attempts |
| `seed` | `integer` | `42` | Random seed for Python, NumPy and PyTorch |
| `no_cuda` | `boolean` | `false` | `true` sets the synthesizer's compute device to CPU |

### Run Control and Logs

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `string` | `./llm_synth_output` | Folder for the per-round artifacts listed below |
| `continue` | `boolean` | `false` | `true` resumes a previous run from `output_dir/<round>/checkpoint.csv`; finished generations are reused and empty ones are generated again |
| `save_syn_mode` | `string` | `selected` | Records written to `output_dir/<round>/samples.csv`: `selected` (kept candidates), `one_var` (first variation of each kept candidate) or `all` (all new variations). The records returned to PETsARD are the same in every mode |
| `log_folder` | `string` | `./LOG_FILES` | Folder for the LLM interaction log |
| `log_prompts` | `boolean` | `true` | `true` writes the text of prompts and responses to the `log_folder` log and to `prompt_generation.jsonl`; `false` records only the length of each text in the log and does not write `prompt_generation.jsonl`. Generation and results are the same with either setting |
| `verbose` | `integer` | `0` | `0` writes to the log file only; `1` also prints the run steps and every prompt and response; `2` also prints the system prompts |

The keys `do_sample`, `variation_type`, `dynamic_len`, `min_target_word`, `word_var_scale` and `max_token_word_scale` are accepted for compatibility with Aug-PE configurations and have no effect on generation.

## Outputs

Paths are relative to the working directory.

| Location | Content |
|---|---|
| `<Reporter output>_Loader[...]_Synthesizer[...].csv` | Final synthetic records, saved by the Reporter (`source: Synthesizer`) |
| `<output_dir>/0/samples.csv` | Records of the initial generation, with a `unique_id` column that identifies the query example |
| `<output_dir>/<round>/samples.csv` | Records saved in each round according to `save_syn_mode` |
| `<output_dir>/<round>_all/samples.csv` | All candidates produced in each round (`epochs` of 1 or more) |
| `<output_dir>/<round>/prompt_generation.jsonl` | Prompts sent and texts generated in each round; written with `log_prompts: true` (the default) |
| `<output_dir>/<round>/checkpoint.csv` | Every finished generation, written as it completes; used by `continue: true` |
| `<output_dir>/fid.csv` | One line per round and run: `<round> <FID> <F1> <precision> <recall>`. Read the FID column; lower values mean the synthetic texts are closer to the training texts in embedding space |
| `<log_folder>/MainSyntheticData_<YYYYMMDD_HHMMSS>.txt` | System prompts, prompts and responses of every LLM call; only their lengths with `log_prompts: false` |

`fid.csv` is appended on each run, while the other files of a round are rewritten. To keep the results of several runs side by side, give each run its own `output_dir` and `log_folder`.

## Recommended Settings

| Goal | Setting |
|---|---|
| Confirm the environment | `sample_num_rows: 5`, `epochs: 0` |
| Follow the training data more closely | Describe field rules in `additional_info`; set `epochs` to 1 or more |
| Increase diversity | `variation_degree_schedule: 0.25`, `temperature: 1.1`, `select_syn_mode: prob` |
| Strengthen privacy protection | Raise `noise_multiplier` to between 12 and 16, use `fewshot_n_examples: 5`, and run locally with `lmstudio` |
| Reduce memory use | `feature_extractor_batch_size: 128` |
| Shorten the run | Lower `sample_num_rows`, `epochs` or `lookahead_degree` |

## Execution Notes

- Load the model in LM Studio with a context length of at least 32,768 tokens, and confirm the server and model identifier with `curl http://localhost:1234/v1/models` (on Windows, `curl.exe`) before running.
- Set `model`, and `variation_model` or `post_processing_model` when used, exactly to an identifier returned by `/v1/models`.
- The synthesizer returns its records directly to the Reporter; set the Reporter's `source` to `Synthesizer`.
- Evaluate the generated text with the [Text Data Evaluation]({{< ref "/docs/petsard-yaml/evaluator-yaml/text-evaluation" >}}) module, which parses the text into columns before running the PETsARD evaluators.

## References

- Xie, C., et al. (2024). Differentially Private Synthetic Data via Foundation Model APIs 2: Text. Proceedings of the 41st International Conference on Machine Learning (ICML 2024). https://arxiv.org/abs/2403.01749
- The Private Evolution workflow of this synthesizer is adapted from Aug-PE (https://github.com/AI-secure/aug-pe, Apache License 2.0); the adapted files and the modifications of this edition are listed in `petsard/synthesizer/LLM_synth/NOTICE`.
