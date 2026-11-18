# PETsARD 教學範例 Tutorial Examples

本資料夾為PETsARD教學課程的實作範例，涵蓋結構化資料(CSV)與半結構化資料(JSON)的合成，以及三種差分隱私合成方法，其中dp-llm-synth用於非結構化資料(文字)。範例資料均為公開資料集或教學示範資料，可直接執行。

This folder contains the hands-on examples of the PETsARD tutorial course: synthesis of structured (CSV) and semi-structured (JSON) data, and three differentially private synthesizers, of which dp-llm-synth synthesizes unstructured text. All example data are public datasets or teaching samples.

> **註記 Note**：初版方法(初版AdvGAN、初版LLM合成器)之合成輸出不具涵蓋完整流程之差分隱私保證，已自本版本移除。本資料夾的差分隱私範例使用更新後方法 `DPflow_Copula`、`DPflow_AdvGAN` 及dp-llm-synth(`DP_LLM_synthesizer`)。
>
> The synthetic output of the initial methods (the initial AdvGAN and the initial LLM synthesizer) does not carry a differential privacy guarantee covering the whole flow, so they have been removed from this version. The differentially private examples in this folder use the updated methods `DPflow_Copula`, `DPflow_AdvGAN` and dp-llm-synth (`DP_LLM_synthesizer`).

## 安裝與執行 Installation and Run (Windows)

需要Python 3.10或3.11，建議使用Python 3.11.5以上版本(Python 3.10及3.11.0至3.11.4會安裝數個含已知弱點之相依套件舊版，見原始碼根目錄之`.trivyignore.yaml`)；可用 `python --version` 確認版本。以下指令在「命令提示字元」(CMD)中執行，假設本儲存庫已下載並解壓縮到 `C:\petsard\PETsARD-main`。

Requires Python 3.10 or 3.11; Python 3.11.5 or later is recommended (on Python 3.10 and 3.11.0-3.11.4, a few dependencies resolve to older releases with known vulnerabilities, see `.trivyignore.yaml` in the repository root). Check the version with `python --version`. Run in Command Prompt (CMD), assuming this repository is unzipped to `C:\petsard\PETsARD-main`.

```bat
cd C:\petsard\PETsARD-main
python -m venv petsard-env
petsard-env\Scripts\activate.bat
python -m pip install --upgrade pip
pip install ".[all]"
cd tutorial
python -X utf8 run_csv3_tw_demographic.py
```

macOS／Linux：以 `python3 -m venv petsard-env` 建立虛擬環境，並以 `source petsard-env/bin/activate` 啟用，其餘指令相同。

On macOS/Linux, create the environment with `python3 -m venv petsard-env` and activate it with `source petsard-env/bin/activate`; the other commands are the same.

所有範例都在 `tutorial` 資料夾中執行，合成結果寫入各範例的 `output` 資料夾。

Run every example from the `tutorial` folder. Results are written to each example's `output` folder.

## 範例清單 Examples

| 執行程式 Script | 資料型態 Data type | 範例資料 Data | 合成方法 Method | 結果位置 Output |
|---|---|---|---|---|
| `run_csv1_syn.py` | 結構化 CSV | `adult_income.csv`(48,842筆，15欄) | Gaussian Copula | `demo_csv/output` |
| `run_csv2_bank.py` | 結構化 CSV | `bank_marketing.csv`(4,521筆，16欄) | Gaussian Copula，含保真度及效用評估 | `demo_csv/output` |
| `run_csv3_tw_demographic.py` | 結構化 CSV | `tw_demographic.csv`(10,000筆，7個中文欄位) | Gaussian Copula | `demo_csv/output` |
| `run_csv4_adult_income.py` | 結構化 CSV | `adult_income.csv`(48,842筆，15欄) | Gaussian Copula，含保真度及效用評估 | `demo_csv/output` |
| `run_json1.py` | 半結構化 JSON | `test_json_data.json`(30筆，6欄) | Gaussian Copula | `demo_json/output` |
| `run_json2.py` | 半結構化 JSON | `adult-income.json`(48,842筆，15欄) | Gaussian Copula | `demo_json/output` |
| `run_json3.py` | 半結構化 JSON | `customer_churn.json`(1,200筆，10欄) | Gaussian Copula，含隱私、保真度及效用評估 | `demo_json/output` |

各執行程式使用的設定檔在 `demo_csv/configs` 及 `demo_json/configs` 中。非結構化資料(文字)的合成請見下方「差分隱私合成方法」的 `run_dp3_llm.py`。

The YAML configurations used by each script are in `demo_csv/configs` and `demo_json/configs`. For unstructured text, see `run_dp3_llm.py` under "Differentially Private Synthesizers" below.

`run_json1.py` 使用預設合成器(PETsARD Gaussian Copula)。差分隱私合成器(`DPflow_Copula`、`DPflow_AdvGAN`)也可以合成由JSON讀入的資料表，但需要公開的欄位宣告(column_spec)、資料不可有缺值，且不加Preprocessor；`test_json_data.json` 刻意保留缺值，用來測試Preprocessor的缺值處理，因此本範例使用預設合成器。差分隱私合成範例見下方 `run_dp1_copula.py` 與 `run_dp2_advgan.py`。

`run_json1.py` uses the default synthesizer (PETsARD Gaussian Copula). The differentially private synthesizers (`DPflow_Copula`, `DPflow_AdvGAN`) also take tables read from JSON, but they need a public column declaration (column_spec) and data without missing values, and run without a Preprocessor. `test_json_data.json` keeps planted missing values to test the missing-value handling of the Preprocessor, so this example uses the default synthesizer. See `run_dp1_copula.py` and `run_dp2_advgan.py` below for differentially private synthesis.

## 文字合成之事前準備 Preparing the Text Example (`run_dp3_llm.py`)

`run_dp3_llm.py` 以本機的LM Studio執行大型語言模型，執行前需完成：

`run_dp3_llm.py` uses a large language model served locally by LM Studio. Before running it:

1. 在儲存庫根目錄安裝套件群組 `llm`(版本範圍依 `pyproject.toml`) Install the `llm` extra from the repository root (version ranges follow `pyproject.toml`):

   ```bat
   cd C:\petsard\PETsARD-main
   pip install ".[all,llm]"
   ```

2. 安裝LM Studio(https://lmstudio.ai)，下載並載入模型 `openai/gpt-oss-20b`，開啟本機伺服器(預設 `http://localhost:1234`)。
   Install LM Studio (https://lmstudio.ai), download and load the model `openai/gpt-oss-20b`, and start the local server (default `http://localhost:1234`).

3. 回到 `tutorial` 資料夾，依下一節執行 `run_dp3_llm.py`。
   Then run `run_dp3_llm.py` from the `tutorial` folder as described in the next section.

## 差分隱私合成方法 Differentially Private Synthesizers

以下三個範例使用PETsARD的差分隱私合成器，同樣在 `tutorial` 資料夾執行。隱私預算(ε、δ)在 `demo_dp/configs` 的設定檔中設定，執行完成時畫面會顯示設定值與實際使用的ε。最近一次執行的隱私報告存在 `demo_dp/output`(dp-llm-synth在 `demo_dp/output/dp_llm_bloodpressure_output/privacy_report.json`)，重新執行會覆寫。同一份資料每合成一次就使用一次隱私預算，多次被使用或公開的結果，ε要相加。方法說明見文件網站「差分隱私合成器」頁(https://moda-gov-tw.github.io/PETsARD/zh-tw/docs/petsard-yaml/synthesizer-yaml/dp-synthesizers/)。

The three examples below use the differentially private synthesizers of PETsARD and also run from the `tutorial` folder. The privacy budget (ε, δ) is set in the YAML files in `demo_dp/configs`; each run prints the target and the spent ε. The privacy report of the latest run is written to `demo_dp/output` (for dp-llm-synth, `demo_dp/output/dp_llm_bloodpressure_output/privacy_report.json`) and is overwritten by the next run. Every synthesis run on the same data spends the budget again, and ε adds up over all results that are used or released. See the documentation page "Differentially Private Synthesizers" (https://moda-gov-tw.github.io/PETsARD/docs/petsard-yaml/synthesizer-yaml/dp-synthesizers/).

```bat
cd C:\petsard\PETsARD-main\tutorial
python -X utf8 run_dp1_copula.py
python -X utf8 run_dp2_advgan.py
python -X utf8 run_dp3_llm.py
```

| 執行程式 Script | 方法 Method | 範例資料 Data | 設定檔 Configuration |
|---|---|---|---|
| `run_dp1_copula.py` | DP Copula(`DPflow_Copula`) | `demo_csv/data/adult_income.csv` | `demo_dp/configs/dp_copula_adult_income.yaml` |
| `run_dp2_advgan.py` | 更新版DP AdvGAN Updated DP AdvGAN(`DPflow_AdvGAN`) | `demo_csv/data/adult_income.csv` | `demo_dp/configs/dp_advgan_adult_income.yaml` |
| `run_dp3_llm.py` | dp-llm-synth(DP LLM文字合成器，`DP_LLM_synthesizer`) | `demo_llm/real_bloodpressure2.csv` | `demo_dp/configs/dp_llm_bloodpressure.yaml` |

兩個結構化資料範例使用公開的欄位宣告 `demo_dp/configs/adult_income_column_spec.yaml`。`run_dp3_llm.py` 需要先完成上一節「文字合成之事前準備」的套件安裝與LM Studio設定；程式會先確認LM Studio已提供設定檔中的模型，未開啟時直接停止，不讀取資料，也不使用隱私預算。執行時畫面出現的Warning訊息為套件提示，不影響結果。

The two structured-data examples use the public column declaration `demo_dp/configs/adult_income_column_spec.yaml`. `run_dp3_llm.py` needs the packages and the LM Studio setup of the previous section; it first checks that LM Studio serves the configured model and stops without reading the data or spending budget if it does not. Warning messages printed during the runs come from the packages and do not affect the results.

## 教學課程錄影對照 Course Recording

課程錄影使用之初版方法範例已自本版本移除，請改用更新後方法範例(對應檔案如下表)。錄影中的其他範例(`run_csv1_syn.py` 至 `run_csv4_adult_income.py`、`run_json2.py`、`run_json3.py`)不變。

The examples of the initial methods used in the course recording have been removed from this version; use the examples of the updated methods in the table below instead. The other examples in the recording (`run_csv1_syn.py` to `run_csv4_adult_income.py`, `run_json2.py` and `run_json3.py`) are unchanged.

| 課程錄影中的初版方法範例 Initial-method example in the recording | 本版本對應檔案 Corresponding files in this version |
|---|---|
| `run_json1.py`：初版AdvGAN合成JSON資料(`demo_json/configs/test_json_advgan.yaml`) Initial AdvGAN on JSON data | JSON讀取與輸出 JSON input and output：`run_json1.py`(預設合成器 default synthesizer，`demo_json/configs/test_json.yaml`)；AdvGAN合成 AdvGAN synthesis：`run_dp2_advgan.py`(更新版DP AdvGAN Updated DP AdvGAN，`demo_dp/configs/dp_advgan_adult_income.yaml`) |
| `run_llm1.py`：初版LLM合成器合成文字(`demo_llm/llm_synth*.yaml`) Initial LLM synthesizer on text | `run_dp3_llm.py`(dp-llm-synth，`demo_dp/configs/dp_llm_bloodpressure.yaml`) |
| `demo_json/configs/default-synthesis.yaml`：初版AdvGAN合成 `adult-income.json` Initial AdvGAN on `adult-income.json` | `run_json2.py`(預設合成器 default synthesizer，`demo_json/configs/adult-income-json.yaml`)；差分隱私合成 differentially private synthesis：`run_dp1_copula.py`、`run_dp2_advgan.py` |

文字範例的額外套件改以套件群組 `llm` 安裝：`pip install ".[all,llm]"`(見「文字合成之事前準備」)。

The extra packages of the text example are now installed with the `llm` extra: `pip install ".[all,llm]"` (see "Preparing the Text Example").

## 再次使用 Next Time

每次重新開啟命令提示字元時，先啟用虛擬環境並進入 `tutorial` 資料夾：

Each time you open a new Command Prompt, activate the environment and change to the `tutorial` folder first:

```bat
cd C:\petsard\PETsARD-main
petsard-env\Scripts\activate.bat
cd tutorial
```
