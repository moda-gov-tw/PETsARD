---
title: "Differentially Private Synthesizers"
type: docs
weight: 658
prev: docs/petsard-yaml/synthesizer-yaml/sdv-methods
next: docs/petsard-yaml/synthesizer-yaml
math: true
---

PETsARD provides three differentially private synthesizers. The whole flow, from reading the data to writing the synthetic data, satisfies $(\varepsilon, \delta)$ differential privacy, and every run writes a privacy report:

| Synthesizer | Data | Configuration |
|---|---|---|
| DP Gaussian copula | Tables (numeric, categorical) | `method: DPflow_Copula` |
| Updated DP AdvGAN | Tables (numeric, categorical) | `method: DPflow_AdvGAN` |
| DP LLM text synthesizer | Text, optionally with short columns such as labels | `method: custom_method`, `module_path: petsard/synthesizer/DP_LLM_synthesizer.py`, `class_name: DP_LLM_synthesizer` |

{{< callout type="info" >}}
`DPflow_AdvGAN` needs the `advgan` extra (opacus, dp-accounting). The DP LLM text synthesizer needs the `llm` extra and a model served locally by LM Studio or another OpenAI-compatible server. See [Extended Features](../../../installation/extended-features).
{{< /callout >}}

{{< callout type="info" >}}
The synthetic output of the initial methods (the initial AdvGAN and the initial LLM synthesizer) does not carry a differential privacy guarantee covering the whole flow, so they have been removed from this version; the three synthesizers on this page are the updated methods. For research comparison, see the [`initial-methods` branch](https://github.com/moda-gov-tw/PETsARD/tree/initial-methods).
{{< /callout >}}

## Before You Start

1. **The unit of protection is one row.** The guarantee covers adding or removing any one row. If one person can have several rows (several visits, several transactions), first reduce the data to one row per person with a public rule, for example keeping one random row per person; the guarantee is then per person.
2. **Public information.** Column names, types, the possible values of categorical columns and the row count are treated as public. Take them from the public definition of the data (a data dictionary or a format agreed with the data owner) rather than from statistics of the private data, so that the guarantee covers every step.
3. **No Preprocessor.** The DP synthesizers handle types and scales themselves, so every step, from reading the data to writing the output, stays within the privacy accounting.
4. **Missing values.** The tabular synthesizers work on complete data. Handle missing values first with a fixed rule, for example treat a missing value as one more category, or fill numeric columns with a fixed value. Tables read from JSON files are used in the same way; a JSON `null` is a missing value.
5. **Seed.** `seed` is optional. Set it to reproduce a run; leave it out for a real release, so the noise comes from the operating system and cannot be recomputed from the code and configuration.
6. **Budgets add up.** Every synthesis run on the same data spends the budget again; for all results that are used or published, the $\varepsilon$ values add up.
7. **Examples.** The `tutorial/` folder has runnable versions of the three examples on this page: `run_dp1_copula.py`, `run_dp2_advgan.py` and `run_dp3_llm.py`.

## Column Declaration (column_spec)

The tabular synthesizers take `column_spec`: a YAML or JSON file, a Markdown type table, or a list written in the configuration. Each column has:

| Key | Meaning |
|---|---|
| `name` | Column name, in the same order as the data |
| `type` | `num` (numeric) or `cat` (categorical) |
| `values` | Required for categorical columns: every possible value, text or numbers |
| `integer` | Optional: `true` for numeric columns that only hold whole numbers; the output is rounded |

Example (from `demo/petsard-yaml/synthesizer-yaml/adult-income_column_spec.yaml`):

```yaml
columns:
  - {name: age, type: num, integer: true}
  - name: workclass
    type: cat
    values: [Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked, "?"]
  - name: income
    type: cat
    values: ["<=50K", ">50K"]
```

A Markdown type table (headers `name`/`欄位名稱`, `type`/`型別`, `values`/`可能值`, optionally `no`/`序號` and `integer`/`整數`) can be used directly as `column_spec`, or converted and checked first:

```bash
python -m petsard.synthesizer.DPflow_column_spec types.md -o column_spec.yaml --names column_names.txt
```

## DP Gaussian Copula (DPflow_Copula)

Releases noisy per-column histograms and the second-moment matrix of normal scores with the Gaussian mechanism, then generates data from these two released results only. The budget is split in $\mu^2$ terms, 20% histograms and 80% correlation matrix, and accounted exactly under Gaussian differential privacy.

```yaml
Loader:
  load_adult:
    filepath: benchmark://adult-income
Synthesizer:
  dp_copula:
    method: DPflow_Copula
    column_spec: demo/petsard-yaml/synthesizer-yaml/adult-income_column_spec.yaml
    epsilon: 8
    delta: 1.0e-5
    seed: 0
    privacy_report: ./dp_copula_adult-income_privacy_report.json
Reporter:
  save_synthetic:
    method: save_data
    source: Synthesizer
    output: dp_copula_adult-income
```

Full example: `demo/petsard-yaml/synthesizer-yaml/DPflow_Copula_adult-income.yaml` (run from the repository root).

| Parameter | Meaning |
|---|---|
| `column_spec` (required) | Column declaration, see above |
| `epsilon` (required) | $\varepsilon$ of the whole flow; `.inf` runs without noise as a reference for comparison |
| `delta` | Default `1.0e-5`; keep it well below 1 / number of rows |
| `sample_num_rows` | Rows to generate; default: as many as the training data |
| `seed` | Optional, see "Before You Start" |
| `privacy_report` | Optional path of the privacy report (JSON) |
| `k` | Histogram bins; default 512 below 5,000 rows, otherwise 2,048 |
| `two_stage` | Default `false`; `true` first locates each column's range on a public grid and then refines it, for columns with narrow ranges |
| `round_integers` | Default `true`; columns declared as integers are written as integers |

## Updated DP AdvGAN (DPflow_AdvGAN)

The output is a DP seed plus a perturbation from the generator. The seeds come from noisy histograms and a noisy copula; the real data reaches the generator only through a discriminator trained with DP-SGD, and sampling never reads a real row. The budget is split in $\mu^2$ terms between histograms, copula and DP-SGD. The DP-SGD noise multiplier is solved so that the accounted $\varepsilon$ does not exceed the target; the privacy report gives the value reached.

This method keeps the whole flow in the privacy accounting, handles types through `column_spec`, and is used without Preprocessor and Postprocessor.

```yaml
Synthesizer:
  dp_advgan:
    method: DPflow_AdvGAN
    column_spec: demo/petsard-yaml/synthesizer-yaml/adult-income_column_spec.yaml
    epsilon: 8
    delta: 1.0e-5
    epochs: 5
    batch_size: 256
    sample_num_rows: 48842
    keep_from_seed: [income]
    share_histograms: 0.1
    share_copula: 0.6
```

Full example: `demo/petsard-yaml/synthesizer-yaml/DPflow_AdvGAN_adult-income.yaml`.

| Parameter | Meaning |
|---|---|
| `column_spec`, `epsilon`, `delta`, `seed`, `privacy_report` | As for DPflow_Copula |
| `epochs` | Training epochs, default 50 |
| `batch_size` | Batch size, default 64; keep it well below the row count |
| `sample_num_rows` | Rows to generate, default 100; set it explicitly |
| `share_histograms`, `share_copula` | Budget split ($\mu^2$ shares), default 0.1 and 0.6; DP-SGD gets the remaining 0.3. Their sum must be below 1 |
| `keep_from_seed` | Optional: columns the generator leaves unchanged, taken from the seed, for example label columns |
| `k`, `two_stage`, `lr`, `round_integers` | Histogram bins (default 64), two-stage histograms (default `true`), learning rate (default 5e-4), integer output (default `true`) |

## DP LLM Text Synthesizer

The only step that reads the data is the release of a **differentially private data profile**: section structure, frequent words and phrases, length distribution, and the combinations of short columns such as labels. Every item goes through a Gaussian mechanism and a release threshold. The model only sees the released profile, never a data row. With `epochs: 0` the whole budget is spent on the profile, and generating text from it is post-processing that spends nothing more; with `epochs` above 0, a differentially private vote also selects synthetic texts.

```yaml
Loader:
  load_text:
    filepath: benchmark://bloodpressure-text
Synthesizer:
  dp_llm:
    method: custom_method
    module_path: petsard/synthesizer/DP_LLM_synthesizer.py
    class_name: DP_LLM_synthesizer
    epsilon: 8
    delta: 1.0e-5
    epochs: 0
    stop_after_epoch: 0
    num_samples_schedule: 20
    sample_num_rows: 20
    api_type: lmstudio
    api_url: "http://127.0.0.1:1234/v1"
    api_key: lm-studio
    model: openai/gpt-oss-20b
    output_dir: ./dp_llm_bloodpressure_output
```

Full example: `demo/petsard-yaml/synthesizer-yaml/dp-llm-synthesizer_bloodpressure.yaml`.

| Parameter | Meaning |
|---|---|
| `epsilon`, `delta` (required) | Budget of the whole flow; the profile's release threshold uses `profile_delta_s` (default 1e-6), which is part of `delta` |
| `api_type` (required) | `lmstudio`, `openai`, `anthropic` or `gemini`, with no default; `lmstudio` keeps prompts on the local server |
| `api_url`, `model`, `api_key` | Server URL and model ID; the key can come from the environment variable `PETSARD_LLM_API_KEY` instead of the configuration |
| `epochs` | Vote rounds, default `1`; `0` means no vote, generation from the profile only. The examples use `0` |
| `sample_num_rows`, `num_samples_schedule` | Rows to generate and candidates per round |
| `output_dir` | Folder for the profile (`profile.json`), the privacy report (`privacy_report.json`) and the results of each round |
| `require_released_profile` | With `true` and `continue: true`, the run uses the released `profile.json` in `output_dir` and stops if it is missing. With `epochs: 0`, this generates again from the same profile without spending more budget |
| `profile_joint_fields`, `profile_value_wording`, `hide_values`, `words_per_row` | Options for data with short columns such as labels: release the combinations of short values and the typical wording of each value, and let the model write only the long text columns |
| `variation_api_type`, `variation_api_url`, `variation_model`, `post_processing_api_type`, `post_processing_api_url`, `post_processing_model` | Separate servers or models for the variation and post-processing stages; by default these stages use `api_type`, `api_url` and `model` |
| `temperature`, `num_procs`, `sleep_time`, `max_retries`, `exit_on_api_error` | Sampling temperature (default 1.0), parallel requests (default 8; match Max Concurrent Predictions in LM Studio), seconds between calls (default 1), attempts for a failed call (default 3), and whether to stop when a call still fails (default `false`) |
| `feature_extractor`, `feature_extractor_batch_size` | Sentence-embedding model of the vote (default `all-MiniLM-L6-v2`) and its batch size (default 1024); used only with `epochs` of 1 or more or with `label_vote: true` |
| `continue` | Resume from `output_dir`; the released profile and the noisy votes already drawn are reused, never drawn again |
| `log_folder` | Folder for the log of every LLM call (default `./LOG_FILES`) |

Every release of a profile spends the budget once. If a profile is released again and both are used or published, the $\varepsilon$ values add up.

### Outputs

Paths are relative to the working directory. The synthesizer returns its rows directly to the Reporter, so set the Reporter's `source` to `Synthesizer`.

| Location | Content |
|---|---|
| `<output_dir>/profile.json` | The released differentially private data profile |
| `<output_dir>/privacy_report.json` | Privacy report |
| `<output_dir>/0/checkpoint.csv` | Finished generations of the initial round, reused by `continue: true` |
| `<output_dir>/0/prompt_generation.jsonl` | Prompts sent and texts generated in the initial round |
| `<output_dir>/epoch<t>_result.csv` | Rows after round `t`, written when `stop_after_epoch` is set |
| `<log_folder>/` | Prompts and responses of every LLM call |

To evaluate the synthetic text with the PETsARD evaluators, see [Text Data Evaluation]({{< ref "/docs/petsard-yaml/evaluator-yaml/text-evaluation" >}}).

## Privacy Report

Every run writes a privacy report (the tabular synthesizers to the path in `privacy_report`; the DP LLM synthesizer to `output_dir/privacy_report.json`). It gives the target and reached $\varepsilon$, $\delta$, the budget split, the noise scales, the row count and the seed source (the seed value when one was set, otherwise only "operating-system randomness").

## References

- Xiao, C., Li, B., Zhu, J.-Y., He, W., Liu, M., & Song, D. (2018). Generating Adversarial Examples with Adversarial Networks. Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI-18). https://arxiv.org/abs/1801.02610
- Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep Learning with Differential Privacy. Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS). https://doi.org/10.1145/2976749.2978318
- Yousefpour, A., et al. (2021). Opacus: User-Friendly Differential Privacy Library in PyTorch. arXiv:2109.12298. https://arxiv.org/abs/2109.12298
- Xie, C., et al. (2024). Differentially Private Synthetic Data via Foundation Model APIs 2: Text. Proceedings of the 41st International Conference on Machine Learning (ICML 2024). https://arxiv.org/abs/2403.01749
- The DP LLM text synthesizer adapts the Private Evolution workflow of Aug-PE (https://github.com/AI-secure/aug-pe, Apache License 2.0); the adapted files and the modifications of this edition are listed in `petsard/synthesizer/DP_LLM_synth/NOTICE`.
