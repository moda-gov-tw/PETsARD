---
title: 擴充功能安裝
type: docs
weight: 260
prev: docs/installation/dl-support-check
next: docs/installation
---

適用於使用本版新增**擴充功能**的環境。

本版為PETsARD新增三項功能：半結構化JSON資料讀取與輸出及AdvGAN合成器、本機LLM非結構化文字合成器及文字評測模組，以及提供Median、KNN、MICE三種方法的缺失資料補償模組。本文件說明以可編輯模式自原始碼安裝PETsARD、安裝各項功能所需套件、設定LM Studio以於本機執行大型語言模型(LLM)推論、選擇合適的硬體規格，以及驗證安裝結果的方式。

## 各功能的需求

| 功能 | 模組 | 額外套件 | 額外軟體 |
|---|---|---|---|
| JSON讀取與輸出 | Loader、Reporter | 無(使用pandas) | 無 |
| AdvGAN合成器 | Synthesizer | `opacus`(隨套件群組`all`安裝，亦包含於`advgan`) | 無；偵測到CUDA GPU時自動使用 |
| LLM文字合成器 | Synthesizer | sentence-transformers、faiss-cpu、tiktoken及LLM服務之SDK(套件群組`llm`) | [LM Studio](https://lmstudio.ai) |
| 文字評測模組 | Evaluator | 無(使用PETsARD內建評測器) | 無 |
| 缺失資料補償 | Preprocessor | 無(使用scikit-learn) | 無 |

{{< callout type="info" >}}
`opacus`隨套件群組`all`安裝。PETsARD僅於使用`method: AdvGAN`時載入；未安裝的環境中，其他合成器照常執行，選用AdvGAN時則提示缺少的套件及安裝指令。
{{< /callout >}}

## 步驟1：準備Python

本版支援Python 3.10與3.11(`pyproject.toml`設定`requires-python = ">=3.10,<3.12"`)，建議使用Python 3.11。

```bash
# Windows：列出已安裝的Python版本，並確認3.11可執行
py -0
py -3.11 --version

# macOS/Linux
python3.11 --version
```

使用Python 3.10時，將本文件指令中`py -3.11`改為`py -3.10`，`python3.11`改為`python3.10`。

## 步驟2：取得原始碼

```bash
git clone https://github.com/moda-gov-tw/PETsARD.git
cd PETsARD
```

未安裝Git時，可依[PyPI安裝](../pypi-install)方式2-b下載原始碼ZIP檔。含`pyproject.toml`的資料夾即**原始碼根目錄**，本文件指令除另有說明外，皆於原始碼根目錄執行。

## 步驟3：建立虛擬環境

虛擬環境可將PETsARD所需套件與電腦中其他Python專案隔離。下列指令於原始碼根目錄建立名為`venv`的虛擬環境並啟動。

{{< tabs items="Windows命令提示字元,Windows PowerShell,macOS/Linux" >}}
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

PowerShell執行原則不允許執行`Activate.ps1`時，先允許目前視窗執行指令碼，再重新啟動：

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

啟動後，命令提示字元前方顯示`(venv)`，後續指令皆於虛擬環境啟動狀態下執行。開啟新視窗時，於原始碼根目錄重新執行啟動指令；結束使用時執行`deactivate`。

## 步驟4：安裝PETsARD與擴充功能套件

```bash
python -m pip install -U pip
pip install -e ".[all,advgan,llm]"
```

`python -m pip install -U pip`將虛擬環境內pip更新至最新版本；`pip install -e ".[all,advgan,llm]"`以可編輯模式安裝PETsARD，並安裝`pyproject.toml`所定義之下列套件群組：

| 套件群組 | 安裝內容 |
|---|---|
| `all` | 完整相依套件：資料處理(numpy、pandas、pyyaml)、評測(anonymeter、sdmetrics、xgboost、scikit-learn)、合成(torch)、Jupyter及Excel讀寫 |
| `advgan` | Opacus，AdvGAN合成器以其對鑑別器執行差分隱私隨機梯度下降(DP-SGD)；`all`亦包含 |
| `llm` | LLM文字合成器所需套件：sentence-transformers、transformers、faiss-cpu、tiktoken、tqdm，以及openai、anthropic、google-genai的SDK |

版本範圍以`pyproject.toml`為準，例如`torch>=2.13.0,<3`。

**可編輯模式**：以`pip install -e`安裝時，Python直接載入原始碼根目錄下`petsard/`資料夾，並將原始碼根目錄加入Python路徑。因此`module_path: petsard/synthesizer/LLM_synthesizer.py`、`module_path: petsard/evaluator/text_evaluator.py`等`custom_method`設定，於任一工作目錄執行皆可自動解析；`filepath`等資料路徑則以工作目錄為基準。修改原始碼後亦不需重新安裝。

### 以GPU加速(選用)

各項擴充功能皆可以CPU執行。AdvGAN合成器偵測到CUDA GPU時，自動將模型與資料移至GPU運算，並於訓練開始時顯示`Using device: cuda`(以CPU運算時顯示`Using device: cpu`)。

Linux(x86_64)自PyPI安裝的PyTorch已含CUDA執行環境。Windows使用NVIDIA GPU時，於執行`pip install -e ".[all,advgan,llm]"`**之前**先安裝CUDA版PyTorch，依顯示卡驅動程式支援的CUDA版本，參考[PyTorch官方網站](https://pytorch.org/get-started/locally/)選擇提供PyTorch 2.13.0以上版本的套件索引，例如CUDA 12.6或CUDA 13.0：

```bash
# CUDA 12.6
pip install "torch>=2.13.0,<3" --index-url https://download.pytorch.org/whl/cu126

# CUDA 13.0
pip install "torch>=2.13.0,<3" --index-url https://download.pytorch.org/whl/cu130
```

符合`torch>=2.13.0,<3`之CUDA版PyTorch滿足`pyproject.toml`之版本範圍，安裝套件群組時即沿用該版本。安裝後可依[深度學習支援檢測](../dl-support-check)確認GPU狀態。

## 步驟5：設定LM Studio(LLM文字合成器)

LLM文字合成器將提示傳送至本機LM Studio的OpenAI相容API伺服器(預設`http://localhost:1234/v1`)進行推論，推論全程於本機完成。

### 安裝LM Studio與下載模型

1. 至[https://lmstudio.ai](https://lmstudio.ai)下載並安裝LM Studio。
2. 於**Discover**頁面搜尋並下載所需模型。本版範例設定檔使用下列模型：

| 模型 | 特性 |
|---|---|
| `google/gemma-4-12b-qat` | 輕量模型，適用於快速執行與教學範例 |
| `openai/gpt-oss-20b` | 較大型的推理模型，模型檔約12.11GB |

LM Studio中已載入的任一模型皆可使用，將`model`設為該模型名稱即可；`variation_model`與`post_processing_model`可視需要另行設定。

### 載入模型與啟動伺服器

1. 載入模型時，將**Context Length**(上下文長度)設為至少32,768 tokens，建議50,000 tokens以上；數值越大，記憶體用量越高。
2. **Max Concurrent Predictions**(同時處理的請求數)與YAML的`num_procs`相配合，本機推論建議設為2至4。
3. 於**Developer**頁面將伺服器狀態由Stopped切換為Running，或以LM Studio隨附的命令列工具執行`lms server start`。
4. LM Studio顯示伺服器位址`http://127.0.0.1:1234`，PETsARD即可由`http://localhost:1234/v1`連線。

### 確認伺服器與模型名稱

```bash
# macOS/Linux
curl http://localhost:1234/v1/models

# Windows(命令提示字元與PowerShell)
curl.exe http://localhost:1234/v1/models
```

回傳內容列出可使用的模型`id`，YAML的`model`須與其中之一完全一致。Windows PowerShell的`curl`為`Invoke-WebRequest`別名，因此以`curl.exe`呼叫。

### YAML的連線設定

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

- `api_url`指定LM Studio伺服器。未設定`api_url`時，`api_type: lmstudio`以`api_key`的值作為伺服器網址(例如`api_key: "http://localhost:1234/v1"`)，預設伺服器為`http://localhost:1234/v1`。
- 變異與後處理階段沿用上述設定；需分別指定時，設定`variation_api_url`、`variation_model`、`post_processing_api_url`或`post_processing_model`。
- 設定`exit_on_api_error: true`時，API呼叫經`max_retries`次重試仍失敗即停止執行(預設值為`false`)，便於快速確認連線設定。
- `api_type`亦可設為`openai`、`anthropic`或`gemini`以使用雲端LLM服務，該服務的API金鑰於執行PETsARD前設定於環境變數`PETSARD_LLM_API_KEY`，金鑰不寫入YAML檔。

完整參數請參見[LLM文字合成器](../../petsard-yaml/synthesizer-yaml/llm-synthesizer)，可執行範例請參見[本機LLM文字合成](../../getting-started/llm-text-synthesis)。

### 句向量模型

設定`compute_fid: true`(預設值)時，合成器以`feature_extractor`指定的sentence-transformers模型(預設`all-MiniLM-L6-v2`)計算句向量。該模型於首次執行時自Hugging Face下載並快取於本機，因此首次執行時保持網路連線。需降低記憶體用量時，可將`feature_extractor_batch_size`(預設`1024`)調整為`128`。

### 由Docker容器連線(選用)

擴充功能亦可於PETsARD官方映像檔中執行，將原始碼根目錄掛載至容器：

```bash
# macOS/Linux(bash、zsh)
docker run -it --rm -v "$(pwd)":/workspace -w /workspace ghcr.io/nics-dp/petsard:latest bash

# Windows PowerShell
docker run -it --rm -v "${PWD}:/workspace" -w /workspace ghcr.io/nics-dp/petsard:latest bash
```

進入容器後，於`/workspace`執行步驟4的安裝指令。容器內`localhost`指向容器本身，連線主機的LM Studio時依下列方式設定：

1. 於LM Studio的**Developer**頁面開啟**Serve on Local Network**。
2. 於YAML的Synthesizer區塊設定`api_url: "http://host.docker.internal:1234/v1"`。
3. Windows與macOS的Docker Desktop自動解析`host.docker.internal`；Linux主機於`docker run`加上`--add-host=host.docker.internal:host-gateway`。

## 步驟6：缺失資料補償

缺失資料補償不需額外套件，使用步驟4之套件群組`all`已安裝的scikit-learn(`scikit-learn>=1.7.1,<2`)。模組為`petsard/processor/missing_compensation.py`，類別為`MissingCompensation`，於Preprocessor以`method: custom_method`掛載，並以`missing_compensation.method`指定下列方法之一：

| `method` | 補值器 |
|---|---|
| `median` | `SimpleImputer`，採中位數策略 |
| `knn` | `KNNImputer`，5個鄰居、`uniform`權重及`nan_euclidean`距離 |
| `mice` | `IterativeImputer`搭配`RandomForestRegressor`(50棵樹、最大深度10)，迭代3次；隨機森林使用全部CPU核心 |

補值器參數固定，因此每次執行結果皆可重現。設定方式請參見[缺失資料補償](../../getting-started/missing-compensation)。

## 建議硬體規格

| 功能 | 中央處理器(CPU) | 系統記憶體(RAM) | 圖形處理器(GPU) | 其他需求 |
|---|---|---|---|---|
| JSON讀取與輸出、預設合成器及評測 | 一般CPU即可 | 依資料筆數與欄位數而定 | 不需要 | 無 |
| AdvGAN合成器(PyTorch、Opacus) | CPU即可運算 | 依資料筆數與欄位數而定 | 選用；偵測到CUDA GPU時自動使用 | 批次大小固定為2，訓練時間隨資料筆數與`epochs`增加 |
| LLM文字合成器(以LM Studio於本機推論) | 4核心以上 | 16GB以上；未配置GPU時建議32GB以上 | 建議顯示記憶體(VRAM)16GB以上；需較長上下文、大批次或較高推論吞吐量時，建議24GB以上；亦可僅以CPU推論 | 可用磁碟空間15GB以上(`openai/gpt-oss-20b`約12.11GB)；模型上下文長度至少32,768 tokens；句向量計算與FAISS最近鄰搜尋可於CPU執行 |
| 缺失資料補償(scikit-learn) | 以CPU運算；MICE的隨機森林使用全部核心 | 依資料筆數與特徵數而定 | 不需要 | 無 |

LLM文字合成器的CPU參考配置：Windows 11虛擬機(32GB RAM、12個虛擬CPU)，以LM Studio 0.4.21載入`openai/gpt-oss-20b`。

## 驗證安裝

**1. 確認可編輯模式安裝**

```bash
python -c "import petsard; print(petsard.__file__)"
```

顯示路徑為原始碼根目錄下`petsard/__init__.py`，例如`<原始碼根目錄>/petsard/__init__.py`。路徑位於`site-packages`時，表示載入者為自PyPI安裝的PETsARD，請於原始碼根目錄重新執行`pip install -e ".[all,advgan,llm]"`。

**2. 確認JSON讀取、AdvGAN合成器及文字評測模組**

```bash
python -c "from petsard import Executor; from petsard.loader.loader_pandas import LoaderPandasJson; from petsard.synthesizer.DP_AdvGAN import AdvGANSynthesizer; from petsard.evaluator.text_evaluator import TextEvaluator; print('OK')"
```

**3. 確認AdvGAN與LLM文字合成器相依套件**

```bash
python -c "import opacus, sentence_transformers, faiss, tiktoken, openai, anthropic, google.genai; print('OK')"
```

**4. 確認缺失資料補償**

```bash
python -c "from petsard.processor.missing_compensation import MissingCompensation; from sklearn.experimental import enable_iterative_imputer; from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer; print('OK')"
```

**5. 確認LM Studio伺服器**

```bash
curl http://localhost:1234/v1/models
```

指令2至4顯示`OK`，指令5回傳已載入的模型清單，即完成環境建置，可接續執行下列教學：

- [JSON資料與AdvGAN合成](../../getting-started/json-advgan-synthesis)
- [本機LLM文字合成](../../getting-started/llm-text-synthesis)
- [缺失資料補償](../../getting-started/missing-compensation)

{{< callout type="info" >}}
PETsARD以作業系統預設編碼讀取YAML設定檔與Schema檔。於Windows執行含中文等非ASCII字元的設定檔時，以UTF-8模式執行Python：`python -X utf8 your_script.py`，或設定環境變數`PYTHONUTF8=1`(命令提示字元為`set PYTHONUTF8=1`，PowerShell為`$env:PYTHONUTF8 = "1"`)。
{{< /callout >}}

## 速查表

| 訊息 | 處理方式 |
|---|---|
| `ModuleNotFoundError: No module named 'opacus'` | 於原始碼根目錄執行`pip install -e ".[all,advgan,llm]"`(步驟4) |
| `ModuleNotFoundError: No module named 'tiktoken'`、`'faiss'`、`'sentence_transformers'`或`'google'` | 於原始碼根目錄執行`pip install -e ".[all,advgan,llm]"`(步驟4) |
| `Unsupported file extension: .json`，或`Module 'petsard/synthesizer/LLM_synthesizer.py' not found in any of the following locations` | 於原始碼根目錄以可編輯模式安裝PETsARD(步驟4)，並以`petsard.__file__`確認載入位置 |
| Windows載入YAML或Schema檔時顯示`codec can't decode byte` | 以`python -X utf8`執行，或設定`PYTHONUTF8=1` |
| LLM呼叫紀錄資料夾(`log_folder`，預設為工作目錄下`LOG_FILES`)記錄`Failed to connect to LM Studio server` | 啟動LM Studio伺服器，並以`curl http://localhost:1234/v1/models`確認(步驟5) |
| LLM呼叫顯示模型相關的錯誤 | 將`model`、`variation_model`及`post_processing_model`設為`/v1/models`回傳的`id` |
| 具NVIDIA GPU的電腦執行AdvGAN時顯示`Using device: cpu` | 安裝CUDA版PyTorch([以GPU加速](#以gpu加速選用)) |
