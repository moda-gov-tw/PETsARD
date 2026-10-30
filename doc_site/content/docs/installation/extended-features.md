---
title: Extended Features Setup
type: docs
weight: 260
prev: docs/installation/dl-support-check
next: docs/installation
---

For environments that use the **extended features** added in this edition.

This edition adds three features to PETsARD: semi-structured JSON input and output with the AdvGAN synthesizer, a local-LLM text synthesizer with a text evaluator, and a missing-data compensation module offering Median, KNN and MICE. This page explains how to install PETsARD from the source code in editable mode, add the packages these features use, set up LM Studio for local large language model (LLM) inference, choose suitable hardware, and verify the installation.

## Feature Requirements

| Feature | Module | Additional packages | Additional software |
|---|---|---|---|
| JSON input and output | Loader, Reporter | None (uses pandas) | None |
| AdvGAN synthesizer | Synthesizer | `opacus` (installed with the `all` extra; also in extra `advgan`) | None; a CUDA GPU is used automatically when available |
| LLM text synthesizer | Synthesizer | sentence-transformers, faiss-cpu, tiktoken and the LLM SDKs (extra `llm`) | [LM Studio](https://lmstudio.ai) |
| Text evaluator | Evaluator | None (uses the built-in PETsARD evaluators) | None |
| Missing-data compensation | Preprocessor | None (uses scikit-learn) | None |

{{< callout type="info" >}}
`opacus` is installed with the `all` extra. PETsARD loads it only when `method: AdvGAN` is used; in an environment without it, the other synthesizers run as usual and selecting AdvGAN reports the missing package together with the install command.
{{< /callout >}}

## Step 1: Prepare Python

This edition supports Python 3.10 and 3.11 (`requires-python = ">=3.10,<3.12"` in `pyproject.toml`). Python 3.11 is recommended.

```bash
# Windows: list the installed Python versions and confirm that 3.11 runs
py -0
py -3.11 --version

# macOS / Linux
python3.11 --version
```

To use Python 3.10, replace `py -3.11` with `py -3.10` and `python3.11` with `python3.10` in the commands on this page.

## Step 2: Get the Source Code

```bash
git clone https://github.com/moda-gov-tw/PETsARD.git
cd PETsARD
```

Without Git, download the source code as a ZIP archive as described in [PyPI Install](../pypi-install) (Method 2-b). The folder that contains `pyproject.toml` is the **source root**; run all commands on this page from the source root unless stated otherwise.

## Step 3: Create a Virtual Environment

A virtual environment keeps the packages of PETsARD separate from other Python projects. The following commands create and activate an environment named `venv` in the source root.

{{< tabs items="Windows (Command Prompt),Windows (PowerShell),macOS / Linux" >}}
{{< tab >}}
```bat
py -3.11 -m venv venv
venv\Scripts\activate.bat
```
{{< /tab >}}
{{< tab >}}
```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

If the PowerShell execution policy blocks `Activate.ps1`, allow scripts for the current window and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```
{{< /tab >}}
{{< tab >}}
```bash
python3.11 -m venv venv
source venv/bin/activate
```
{{< /tab >}}
{{< /tabs >}}

When the environment is active, the prompt starts with `(venv)`. Run the remaining commands in the activated environment. In a new terminal window, activate the environment again from the source root; run `deactivate` to leave it.

## Step 4: Install PETsARD and the Extended-Feature Packages

```bash
python -m pip install -U pip
pip install -e ".[all,advgan,llm]"
```

`python -m pip install -U pip` installs the latest pip in the virtual environment. `pip install -e ".[all,advgan,llm]"` installs PETsARD in editable mode together with the following extras defined in `pyproject.toml`:

| Extra | What it installs |
|---|---|
| `all` | The complete dependency set: data handling (numpy, pandas, pyyaml), evaluation (anonymeter, sdmetrics, xgboost, scikit-learn), synthesis (torch), Jupyter and Excel support |
| `advgan` | Opacus, which the AdvGAN synthesizer uses to train its discriminator with differentially private stochastic gradient descent (DP-SGD); `all` also includes it |
| `llm` | Packages for the LLM text synthesizer: sentence-transformers, transformers, faiss-cpu, tiktoken, tqdm, and the openai, anthropic and google-genai SDKs |

Version ranges follow `pyproject.toml`, for example `torch>=2.13.0,<3`.

**Editable mode:** with `pip install -e`, Python loads `petsard` directly from the `petsard/` folder of the source root, and the source root is added to the Python path. `custom_method` entries such as `module_path: petsard/synthesizer/LLM_synthesizer.py` and `module_path: petsard/evaluator/text_evaluator.py` are therefore resolved from any working directory, while data paths such as `filepath` remain relative to the working directory. Changes to the source code take effect without reinstalling.

### GPU Acceleration (Optional)

All extended features run on a CPU. The AdvGAN synthesizer moves its model and data to the GPU automatically when a CUDA GPU is detected, and prints `Using device: cuda` (or `Using device: cpu`) at the start of training.

PyTorch from PyPI includes the CUDA runtime on Linux (x86_64). On Windows, install a CUDA build of PyTorch **before** running `pip install -e ".[all,advgan,llm]"`. Choose an index URL on the [PyTorch website](https://pytorch.org/get-started/locally/) that matches the CUDA version of the graphics driver and provides PyTorch 2.13.0 or later, for example CUDA 12.6 or CUDA 13.0:

```bash
# CUDA 12.6
pip install "torch>=2.13.0,<3" --index-url https://download.pytorch.org/whl/cu126

# CUDA 13.0
pip install "torch>=2.13.0,<3" --index-url https://download.pytorch.org/whl/cu130
```

A CUDA build within `torch>=2.13.0,<3` satisfies the version range of `pyproject.toml`, so it is kept when the extras are installed. Confirm GPU availability with the [Deep Learning Support Check](../dl-support-check).

## Step 5: Set Up LM Studio (LLM Text Synthesizer)

The LLM text synthesizer sends its prompts to the OpenAI-compatible API server of LM Studio on the local machine (default `http://localhost:1234/v1`), so all inference stays on the local machine.

### Install LM Studio and Download a Model

1. Download and install LM Studio from [https://lmstudio.ai](https://lmstudio.ai).
2. On the **Discover** page, search for and download the model to use. The example configurations of this edition use the following models:

| Model | Characteristics |
|---|---|
| `google/gemma-4-12b-qat` | Lightweight model for quick runs and tutorials |
| `openai/gpt-oss-20b` | Larger reasoning model; the model file is about 12.11 GB |

Any model loaded in LM Studio can be used. Set `model`, and optionally `variation_model` and `post_processing_model`, to its identifier.

### Load the Model and Start the Server

1. Load the model with a **Context Length** of at least 32,768 tokens; 50,000 tokens or more is recommended. Larger values use more memory.
2. Set **Max Concurrent Predictions** to match `num_procs` in the YAML; 2 to 4 is recommended for local inference.
3. On the **Developer** page, switch the server status from Stopped to Running, or run `lms server start` with the LM Studio command-line tool.
4. LM Studio displays the server address `http://127.0.0.1:1234`; PETsARD connects to it through `http://localhost:1234/v1`.

### Confirm the Server and the Model Identifier

```bash
# macOS / Linux
curl http://localhost:1234/v1/models

# Windows (Command Prompt and PowerShell)
curl.exe http://localhost:1234/v1/models
```

The response lists the `id` of every available model. The `model` value in the YAML must match one of them exactly. In Windows PowerShell, `curl` is an alias of `Invoke-WebRequest`, so call `curl.exe`.

### Connection Settings in YAML

```yaml
Synthesizer:
  llm-synth:
    method: custom_method
    module_path: petsard/synthesizer/LLM_synthesizer.py
    class_name: LLM_synthesizer
    api_type: lmstudio
    api_url: "http://localhost:1234/v1"
    model: google/gemma-4-12b-qat
    num_procs: 2
```

- `api_url` sets the LM Studio server. When `api_url` is not set, `api_type: lmstudio` reads the server URL from `api_key` (for example `api_key: "http://localhost:1234/v1"`), and the default server is `http://localhost:1234/v1`.
- The variation and post-processing stages reuse these settings unless `variation_api_url`, `variation_model`, `post_processing_api_url` or `post_processing_model` are set.
- `exit_on_api_error: true` stops the pipeline as soon as an API call still fails after `max_retries` attempts (default `false`), which makes connection settings quick to confirm.
- `api_type` also accepts `openai`, `anthropic` and `gemini` for hosted LLM services; set the API key of the provider in the environment variable `PETSARD_LLM_API_KEY` before running PETsARD, so that the key stays out of the YAML file.

The complete parameter list is given in [LLM Text Synthesizer](../../petsard-yaml/synthesizer-yaml/llm-synthesizer), and a runnable example in [Local LLM Text Synthesis](../../getting-started/llm-text-synthesis).

### Sentence-Embedding Model

With `compute_fid: true` (the default), the synthesizer computes sentence embeddings with the sentence-transformers model set in `feature_extractor` (default `all-MiniLM-L6-v2`). The model is downloaded from Hugging Face on the first run and cached locally, so keep the network connection available for the first run. To reduce memory usage, lower `feature_extractor_batch_size` (default `1024`) to `128`.

### Connecting from a Docker Container (Optional)

The extended features can also run inside the official PETsARD image, with the source root mounted into the container:

```bash
# macOS / Linux (bash, zsh)
docker run -it --rm -v "$(pwd)":/workspace -w /workspace ghcr.io/moda-gov-tw/petsard:latest bash

# Windows PowerShell
docker run -it --rm -v "${PWD}:/workspace" -w /workspace ghcr.io/moda-gov-tw/petsard:latest bash
```

The image already includes the `all`, `advgan` and `llm` packages of Step 4, so no installation is needed inside the container; PETsARD is loaded from the mounted source root `/workspace`. Within a container, `localhost` refers to the container itself; to reach LM Studio on the host:

1. Turn on **Serve on Local Network** on the **Developer** page of LM Studio.
2. Set `api_url: "http://host.docker.internal:1234/v1"` in the Synthesizer block.
3. Docker Desktop on Windows and macOS resolves `host.docker.internal` automatically; on Linux, add `--add-host=host.docker.internal:host-gateway` to `docker run`.

## Step 6: Missing-Data Compensation

Missing-data compensation needs no additional packages: it uses scikit-learn (`scikit-learn>=1.7.1,<2`), which the `all` extra of Step 4 installs. The module is `petsard/processor/missing_compensation.py` with the class `MissingCompensation`, enabled through `method: custom_method` in the Preprocessor, with `missing_compensation.method` set to one of the following:

| `method` | Imputer |
|---|---|
| `median` | `SimpleImputer` with the median strategy |
| `knn` | `KNNImputer` with 5 neighbors, `uniform` weights and the `nan_euclidean` distance |
| `mice` | `IterativeImputer` with a `RandomForestRegressor` (50 trees, maximum depth 10) and 3 iterations; the random forest uses all CPU cores |

The imputer parameters are fixed, so results are reproducible across runs. Configuration details are given in [Missing-Data Compensation](../../getting-started/missing-compensation).

## Hardware Recommendations

| Feature | CPU | RAM | GPU | Other requirements |
|---|---|---|---|---|
| JSON input and output, default synthesizer and evaluation | Any current CPU | Depends on the number of rows and columns | Not required | None |
| AdvGAN synthesizer (PyTorch, Opacus) | A CPU is sufficient | Depends on the number of rows and columns | Optional; used automatically when a CUDA GPU is detected | The batch size is fixed at 2; training time grows with the number of rows and `epochs` |
| LLM text synthesizer (local inference with LM Studio) | 4 cores or more | 16 GB or more; 32 GB or more without a GPU | 16 GB of VRAM or more recommended; 24 GB or more for longer contexts, larger batches or higher throughput; CPU-only inference is also supported | 15 GB or more of free disk space (`openai/gpt-oss-20b` is about 12.11 GB); model context length of at least 32,768 tokens; sentence embeddings and FAISS nearest-neighbor search can run on the CPU |
| Missing-data compensation (scikit-learn) | CPU; the MICE random forest uses all cores | Depends on the number of rows and features | Not required | None |

A verified CPU-only configuration for the LLM text synthesizer is a Windows 11 virtual machine with 32 GB of RAM and 12 virtual CPUs, running LM Studio 0.4.21 with `openai/gpt-oss-20b`.

## Verify the Installation

**1. Confirm the editable installation**

```bash
python -c "import petsard; print(petsard.__file__)"
```

The path points to `petsard/__init__.py` under the source root, for example `<source root>/petsard/__init__.py`. A path inside `site-packages` means that PETsARD was installed from PyPI; run `pip install -e ".[all,advgan,llm]"` again from the source root.

**2. Confirm the JSON loader, the AdvGAN synthesizer and the text evaluator**

```bash
python -c "from petsard import Executor; from petsard.loader.loader_pandas import LoaderPandasJson; from petsard.synthesizer.DP_AdvGAN import AdvGANSynthesizer; from petsard.evaluator.text_evaluator import TextEvaluator; print('OK')"
```

**3. Confirm the packages of the AdvGAN and LLM text synthesizers**

```bash
python -c "import opacus, sentence_transformers, faiss, tiktoken, openai, anthropic, google.genai; print('OK')"
```

**4. Confirm missing-data compensation**

```bash
python -c "from petsard.processor.missing_compensation import MissingCompensation; from sklearn.experimental import enable_iterative_imputer; from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer; print('OK')"
```

**5. Confirm the LM Studio server**

```bash
curl http://localhost:1234/v1/models
```

Commands 2 to 4 print `OK`, and command 5 returns the list of loaded models. The environment is then ready for the tutorials:

- [JSON Data and AdvGAN Synthesis](../../getting-started/json-advgan-synthesis)
- [Local LLM Text Synthesis](../../getting-started/llm-text-synthesis)
- [Missing-Data Compensation](../../getting-started/missing-compensation)

{{< callout type="info" >}}
PETsARD reads YAML and schema files with the default encoding of the operating system. On Windows, run Python in UTF-8 mode when configuration files contain non-ASCII characters: `python -X utf8 your_script.py`, or set the environment variable `PYTHONUTF8=1` (`set PYTHONUTF8=1` in Command Prompt, `$env:PYTHONUTF8 = "1"` in PowerShell).
{{< /callout >}}

## Quick Reference

| Message | Resolution |
|---|---|
| `ModuleNotFoundError: No module named 'opacus'` | Run `pip install -e ".[all,advgan,llm]"` from the source root (Step 4) |
| `ModuleNotFoundError: No module named 'tiktoken'`, `'faiss'`, `'sentence_transformers'` or `'google'` | Run `pip install -e ".[all,advgan,llm]"` from the source root (Step 4) |
| `Unsupported file extension: .json`, or `Module 'petsard/synthesizer/LLM_synthesizer.py' not found in any of the following locations` | Install PETsARD in editable mode from the source root (Step 4) and confirm the location with `petsard.__file__` |
| `codec can't decode byte` when a YAML or schema file is loaded on Windows | Run with `python -X utf8` or set `PYTHONUTF8=1` |
| The LLM log folder (`log_folder`, default `LOG_FILES` in the working directory) records `Failed to connect to LM Studio server` | Start the LM Studio server and confirm it with `curl http://localhost:1234/v1/models` (Step 5) |
| The LLM call reports a model error | Set `model`, `variation_model` and `post_processing_model` to an `id` returned by `/v1/models` |
| AdvGAN prints `Using device: cpu` on a machine with an NVIDIA GPU | Install a CUDA build of PyTorch ([GPU Acceleration](#gpu-acceleration-optional)) |
