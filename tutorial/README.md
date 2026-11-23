# PETsARD 教學範例 Tutorial Examples

本資料夾為PETsARD教學課程的實作範例，涵蓋結構化資料(CSV)與半結構化資料(JSON)的合成，以及三種差分隱私合成方法，其中dp-llm-synth用於非結構化資料(文字)。範例資料均為公開資料集或教學示範資料，可直接執行。

This folder contains the hands-on examples of the PETsARD tutorial course: synthesis of structured (CSV) and semi-structured (JSON) data, and three differentially private synthesizers, of which dp-llm-synth synthesizes unstructured text. All example data are public datasets or teaching samples.

> **註記 Note**：初版方法(初版AdvGAN、初版LLM合成器)之合成輸出不具涵蓋完整流程之差分隱私保證，已移至[`initial-methods`分支](https://github.com/moda-gov-tw/PETsARD/tree/initial-methods)(標籤`initial-methods-v1`)，僅供研究比較及重現以初版方法產生之結果。本資料夾的範例使用更新後方法 `DPflow_Copula`、`DPflow_AdvGAN` 及dp-llm-synth(`DP_LLM_synthesizer`)；課程錄影使用之初版方法範例見下方「教學課程錄影對照」。
>
> The synthetic output of the initial methods (the initial AdvGAN and the initial LLM synthesizer) does not carry a differential privacy guarantee covering the whole flow, so they have moved to the [`initial-methods` branch](https://github.com/moda-gov-tw/PETsARD/tree/initial-methods) (tag `initial-methods-v1`), kept for research comparison and for reproducing results produced with them. The examples in this folder use the updated methods `DPflow_Copula`, `DPflow_AdvGAN` and dp-llm-synth (`DP_LLM_synthesizer`); for the initial-method examples of the course recording, see "Course Recording" below.

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
| `run_json1.py` | 半結構化 JSON | `adult-income.json`(48,842筆，15欄) | 更新版DP AdvGAN Updated DP AdvGAN(`DPflow_AdvGAN`)，合成1,000筆，以JSON輸出 | `demo_json/output` |
| `run_json2.py` | 半結構化 JSON | `adult-income.json`(48,842筆，15欄) | Gaussian Copula | `demo_json/output` |
| `run_json3.py` | 半結構化 JSON | `customer_churn.json`(1,200筆，10欄) | Gaussian Copula，含隱私、保真度及效用評估 | `demo_json/output` |
| `run_llm1.py` | 非結構化 文字 Text | `real_bloodpressure2.csv`(101筆血壓問診文本) | dp-llm-synth(`DP_LLM_synthesizer`，與 `run_dp3_llm.py` 相同) | `demo_dp/output` |

各執行程式使用的設定檔在 `demo_csv/configs`、`demo_json/configs` 及 `demo_dp/configs` 中。`run_llm1.py` 執行 `run_dp3_llm.py`，需要先完成「文字合成之事前準備」。

The YAML configurations used by each script are in `demo_csv/configs`, `demo_json/configs` and `demo_dp/configs`. `run_llm1.py` runs `run_dp3_llm.py` and needs the preparation in "Preparing the Text Example".

`run_json1.py` 以更新版DP AdvGAN(`DPflow_AdvGAN`)合成由JSON讀入的資料表，合成資料與評估報告皆以JSON格式輸出(設定檔 `demo_json/configs/dp_advgan_json.yaml`)。隱私預算ε=8、δ=10⁻⁵涵蓋自讀入資料至輸出合成資料之整個流程，執行完成時畫面顯示設定值與實際使用的ε，隱私報告存在 `demo_json/output/dp_advgan_json_privacy_report.json`。差分隱私合成器需要公開的欄位宣告(本範例使用 `demo_dp/configs/adult_income_column_spec.yaml`)、資料不可有缺值，且不加Preprocessor。

`run_json1.py` synthesizes a table read from JSON with the updated DP AdvGAN (`DPflow_AdvGAN`) and writes the synthetic data and the evaluation report as JSON (configuration `demo_json/configs/dp_advgan_json.yaml`). The budget ε=8, δ=10⁻⁵ covers the whole flow from reading the data to writing the synthetic data; each run prints the target and the spent ε, and the privacy report is written to `demo_json/output/dp_advgan_json_privacy_report.json`. The differentially private synthesizers need a public column declaration (here `demo_dp/configs/adult_income_column_spec.yaml`) and data without missing values, and run without a Preprocessor.

## 課程單元與檔案對照 Course Map

下表對照「資料保護驗測機制培訓工作坊」教材(含數位課程)的頁碼與本資料夾的檔案。課程影片中的範例資料夾`C:\Users\petsard\Desktop\petsard`對應本資料夾`tutorial`；結構化資料(CSV)與JSON範例的檔案名稱與執行指令與影片相同。數位課程錄影之 `run_json1.py`、`run_llm1.py` 使用初版方法之範例，本資料夾之同名範例改以更新後方法執行，初版範例之所在分支見下方「教學課程錄影對照」。

The table maps the pages of the workshop slides (also used in the online course) to the files in this folder. The course video folder `C:\Users\petsard\Desktop\petsard` corresponds to this `tutorial` folder; the CSV and JSON examples keep the file names and commands of the video. In the recording, `run_json1.py` and `run_llm1.py` run initial-method examples; the scripts of the same name in this folder run the updated methods, and "Course Recording" below gives the branch of the initial examples.

| 課程單元(教材頁碼) Unit (slides) | 內容 Topic | 檔案位置 Files |
|---|---|---|
| 單元二(第59至66頁) Unit 2 (59-66) | 安裝Python、虛擬環境與PETsARD Installation | 本說明「安裝與執行」；「PETsARD環境建置操作說明」簡報 |
| 單元三(第71至108頁) Unit 3 (71-108) | 結構化與JSON資料的YAML設定 YAML for CSV and JSON | `demo_csv/configs`、`demo_csv/schemas`、`demo_json/configs` |
| 單元四(第109至133頁) Unit 4 (109-133) | 非結構化資料的LLM合成 LLM text synthesis | `demo_dp/configs/dp_llm_bloodpressure.yaml`(dp-llm-synth)；影片中的初版範例 `demo_llm/llm_synth*.yaml` 見「教學課程錄影對照」 |
| 單元六(第161至169頁) Unit 6 (161-169) | 結構化資料合成實作 CSV hands-on | `run_csv1_syn.py`、`run_csv2_bank.py`、`run_csv3_tw_demographic.py`、`run_csv4_adult_income.py`；資料`demo_csv/data`，結果`demo_csv/output` |
| 單元六(第170至175頁) Unit 6 (170-175) | JSON資料合成實作 JSON hands-on | `run_json1.py`(更新版DP AdvGAN)、`run_json2.py`、`run_json3.py`；資料`demo_json/data`，結果`demo_json/output` |
| 單元六(第176至181頁) Unit 6 (176-181) | 非結構化資料合成實作 Text hands-on | `run_llm1.py` 或 `run_dp3_llm.py`(dp-llm-synth，需先完成「文字合成之事前準備」)；資料`demo_llm/real_bloodpressure2.csv`，結果`demo_dp/output` |
| 附錄(第185至196頁) Appendix (185-196) | 從零安裝與差分隱私合成方法 Fresh install and DP synthesizers | `run_dp1_copula.py`、`run_dp2_advgan.py`、`run_dp3_llm.py`；設定`demo_dp/configs`，結果`demo_dp/output` |

## 文字合成之事前準備 Preparing the Text Example (`run_dp3_llm.py`, `run_llm1.py`)

`run_dp3_llm.py`(及執行相同內容的 `run_llm1.py`)以本機的LM Studio執行大型語言模型，執行前需完成：

`run_dp3_llm.py` (and `run_llm1.py`, which runs the same script) uses a large language model served locally by LM Studio. Before running it:

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

課程錄影使用之初版方法範例(初版AdvGAN、初版LLM合成器)保存於[`initial-methods`分支](https://github.com/moda-gov-tw/PETsARD/tree/initial-methods)(標籤`initial-methods-v1`)，供對照錄影及研究比較；初版方法之合成輸出不具涵蓋完整流程之差分隱私保證，需具差分隱私保證之合成資料請改用本資料夾之更新後方法範例。本資料夾之同名範例執行指令與錄影相同，改以更新後方法執行(對應如下表)；錄影中的其他範例(`run_csv1_syn.py` 至 `run_csv4_adult_income.py`、`run_json2.py`、`run_json3.py`)不變。

The initial-method examples used in the course recording (the initial AdvGAN and the initial LLM synthesizer) are kept in the [`initial-methods` branch](https://github.com/moda-gov-tw/PETsARD/tree/initial-methods) (tag `initial-methods-v1`) for reference and research comparison. Their synthetic output does not carry a differential privacy guarantee covering the whole flow; for synthetic data that needs such a guarantee, use the updated-method examples in this folder. The scripts of the same name in this folder keep the commands of the recording and run the updated methods (see the table); the other examples in the recording (`run_csv1_syn.py` to `run_csv4_adult_income.py`, `run_json2.py` and `run_json3.py`) are unchanged.

| 課程錄影中的初版方法範例 Initial-method example in the recording | 本資料夾對應檔案(更新後方法) Corresponding files in this folder (updated methods) |
|---|---|
| `run_json1.py`：初版AdvGAN合成JSON資料(`demo_json/configs/test_json_advgan.yaml`) Initial AdvGAN on JSON data | `run_json1.py`：更新版DP AdvGAN合成JSON資料 Updated DP AdvGAN on JSON data(`demo_json/configs/dp_advgan_json.yaml`) |
| `run_llm1.py`：初版LLM合成器合成文字(`demo_llm/llm_synth*.yaml`) Initial LLM synthesizer on text | `run_llm1.py`：dp-llm-synth合成文字 dp-llm-synth on text(與 `run_dp3_llm.py` 相同 same as `run_dp3_llm.py`，`demo_dp/configs/dp_llm_bloodpressure.yaml`) |
| `demo_json/configs/default-synthesis.yaml`：初版AdvGAN合成 `adult-income.json` Initial AdvGAN on `adult-income.json` | `run_json1.py`(更新版DP AdvGAN Updated DP AdvGAN)；`run_json2.py`(預設合成器 default synthesizer，`demo_json/configs/adult-income-json.yaml`) |

取得初版範例：於分支頁面點選「Code」再選「Download ZIP」下載(解壓縮後之資料夾為 `PETsARD-initial-methods`)，或執行下列指令。

Getting the initial-method examples: on the branch page choose "Code" and then "Download ZIP" (the unzipped folder is `PETsARD-initial-methods`), or run the command below.

```bat
git clone -b initial-methods https://github.com/moda-gov-tw/PETsARD.git PETsARD-initial-methods
```

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
