# PETsARD 教學範例 Tutorial Examples

本資料夾為PETsARD教學課程的實作範例，涵蓋結構化資料(CSV)、半結構化資料(JSON)及非結構化資料(文字)的合成。範例資料均為公開資料集或教學示範資料，可直接執行。

This folder contains the hands-on examples of the PETsARD tutorial course: synthesis of structured (CSV), semi-structured (JSON) and unstructured (text) data. All example data are public datasets or teaching samples.

## 安裝與執行 Installation and Run (Windows)

需要Python 3.10或3.11。以下指令在「命令提示字元」(CMD)中執行，假設本儲存庫已下載並解壓縮到 `C:\petsard\PETsARD-main`。

Requires Python 3.10 or 3.11. Run in Command Prompt (CMD), assuming this repository is unzipped to `C:\petsard\PETsARD-main`.

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
| `run_json1.py` | 半結構化 JSON | `test_json_data.json`(30筆，6欄) | AdvGAN | `demo_json/output` |
| `run_json2.py` | 半結構化 JSON | `adult-income.json`(48,842筆，15欄) | Gaussian Copula | `demo_json/output` |
| `run_json3.py` | 半結構化 JSON | `customer_churn.json`(1,200筆，10欄) | Gaussian Copula，含隱私、保真度及效用評估 | `demo_json/output` |
| `run_llm1.py` | 非結構化 文字 Text | `real_bloodpressure2.csv`(101筆血壓問診文本) | LLM文字合成器 LLM synthesizer | `demo_llm/output` |

各執行程式使用的設定檔在 `demo_csv/configs`、`demo_json/configs` 及 `demo_llm` 中。

The YAML configurations used by each script are in `demo_csv/configs`, `demo_json/configs` and `demo_llm`.

## 非結構化資料範例 Text Example (`run_llm1.py`)

`run_llm1.py` 以本機的LM Studio執行大型語言模型，執行前需完成：

`run_llm1.py` uses a large language model served locally by LM Studio. Before running it:

1. 在儲存庫根目錄安裝額外套件 Install the extra packages from the repository root:

   ```bat
   cd C:\petsard\PETsARD-main
   pip install -r petsard\synthesizer\LLM_synth\requirements.txt
   ```

2. 安裝LM Studio(https://lmstudio.ai)，下載並載入模型 `openai/gpt-oss-20b`，開啟本機伺服器(預設 `http://localhost:1234`)。
   Install LM Studio (https://lmstudio.ai), download and load the model `openai/gpt-oss-20b`, and start the local server (default `http://localhost:1234`).

3. 回到 `tutorial` 資料夾執行 Run from the `tutorial` folder:

   ```bat
   cd tutorial
   python -X utf8 run_llm1.py
   ```

## 再次使用 Next Time

每次重新開啟命令提示字元時，先啟用虛擬環境並進入 `tutorial` 資料夾：

Each time you open a new Command Prompt, activate the environment and change to the `tutorial` folder first:

```bat
cd C:\petsard\PETsARD-main
petsard-env\Scripts\activate.bat
cd tutorial
```
