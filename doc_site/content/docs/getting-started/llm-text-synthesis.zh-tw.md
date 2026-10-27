---
title: 本機LLM文字合成與文字評測
type: docs
weight: 150
prev: docs/getting-started/json-advgan-synthesis
next: docs/getting-started/missing-compensation
---

本文件說明以LM Studio於本機執行大型語言模型(LLM)搭配LLM文字合成器生成合成的自由文字，再以文字評測模組評估合成結果。範例使用血壓問診範例資料，全程於本機推論。

操作步驟如下：

1. 確認執行環境與LM Studio伺服器。
2. 認識血壓問診範例資料。
3. 以5筆紀錄執行第一次合成。
4. 加入變異與挑選輪次，執行完整合成。
5. 將合成文字解析為欄位並進行評測。

## 步驟1：準備執行環境

依[擴充功能安裝]({{< ref "/docs/installation/extended-features" >}})以可編輯模式自原始碼安裝PETsARD，安裝LLM文字合成器的相依套件，並完成LM Studio設定。於原始碼根目錄(含`pyproject.toml`資料夾)啟動虛擬環境後，安裝指令如下：

```bash
pip install -e ".[all,advgan,llm]"
```

於LM Studio下載模型(本文件使用`google/gemma-4-12b-qat`，亦可使用`openai/gpt-oss-20b`)，載入時將上下文長度設為32,768 tokens以上，Max Concurrent Predictions設為2，並啟動本機伺服器。接著確認伺服器狀態與模型名稱：

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

回傳內容列出可使用的模型`id`，下方YAML的`model`填入其中之一。

{{< callout type="info" >}}
本文件的指令皆於原始碼根目錄執行，YAML中的合成器模組與解析程式路徑皆相對於執行目錄；範例資料以基準資料集協議下載至執行目錄下之`benchmark/`。
{{< /callout >}}

## 步驟2：認識範例資料

範例資料為基準資料集`bloodpressure-text`(檔案`real_bloodpressure2.csv`)，含單一`text`欄位，共101筆，每列以1句中文描述1位受檢者之姓名、性別、年齡、血壓量測值及判讀結果，例如：

```text
患者為45歲男性，姓名：陳柏宇，血壓138/88偏高
黃志豪是一位38歲男性，量測血壓為118/76，正常
```

其中的姓名與數值均為虛構。各筆紀錄之句型不盡相同。收縮壓達130 mmHg者判讀為偏高(62筆)，未達者判讀為正常(39筆)。

使用自有資料時，準備每列1筆紀錄的CSV檔案。僅含單一文字欄位的檔案依本文件方式合成；文字欄位與結構化欄位並存時，請參見[LLM文字合成器]({{< ref "/docs/petsard-yaml/synthesizer-yaml/llm-synthesizer" >}})的`main_column_name`。

## 步驟3：執行第一次合成

將下列內容儲存為原始碼根目錄`llm_text_quickstart.yaml`：

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

以`Executor`執行：

```python
from petsard import Executor

exec = Executor(config="llm_text_quickstart.yaml")
exec.run()
```

或於命令列直接執行：

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

### YAML參數說明

- **`Loader`**：載入範例CSV，`text`欄位的每列為1段自由文字。
- **`Synthesizer`**：
  - `method: custom_method`、`module_path`及`class_name`載入LLM文字合成器。
  - `sample_num_rows: 5`：輸出的合成紀錄筆數。
  - `epochs: 0`：僅執行少樣本導引的初始生成，每筆紀錄由1次LLM呼叫產生，每次提示附帶`fewshot_n_examples`筆(預設10筆)訓練紀錄作為範例。
  - `api_type: lmstudio`與`api_url`：將提示送至本機LM Studio伺服器。
  - `model`：`/v1/models`所列的模型名稱。
  - `num_procs: 2`：平行請求數，與LM Studio的Max Concurrent Predictions相配合。
- **`Reporter`**：儲存合成紀錄。合成器將輸出直接交給Reporter，因此`source`設為`Synthesizer`。

### 檢查輸出

執行完成後產生下列檔案：

| 位置 | 內容 |
|---|---|
| `llm_quickstart_Loader[load_bloodpressure]_Synthesizer[llm-synth].csv` | 5筆合成紀錄，欄位為`text` |
| `llm_synth_output/0/samples.csv` | 初始生成紀錄 |
| `llm_synth_output/0/prompt_generation.jsonl` | 送出的提示與生成文字 |
| `llm_synth_output/fid.csv` | 合成文字與訓練文字於句向量空間的FID，數值越低代表越接近 |
| `LOG_FILES/MainSyntheticData_<YYYYMMDD_HHMMSS>.txt` | 每次LLM呼叫的提示與回覆 |

下列指令依序輸出合成紀錄筆數、與訓練紀錄完全相同的筆數，以及含遮罩符號`<MASK>`的筆數，預期結果為`5 0 0`：

```bash
python -c "import pandas as pd; r=pd.read_csv('benchmark/real_bloodpressure2.csv'); s=pd.read_csv('llm_quickstart_Loader[load_bloodpressure]_Synthesizer[llm-synth].csv'); print(len(s), int(s['text'].isin(r['text']).sum()), int(s['text'].str.contains('<MASK>', regex=False).sum()))"
```

開啟CSV檔案，確認每筆紀錄皆包含姓名、性別、年齡、血壓量測值及判讀結果。`LOG_FILES/`中的紀錄檔記錄每次呼叫的提示與回覆，呼叫嘗試`max_retries`次後仍未取得回應時，紀錄檔以`failed after`字樣標示。確認連線設定時，可設定`exit_on_api_error: true`，於此類呼叫發生時即停止執行。

## 步驟4：執行完整合成

確認執行環境後，增加合成筆數並加入1輪變異與挑選。將下列內容儲存為`llm_text_synthesis.yaml`：

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

與步驟3的差異如下：

- `epochs: 1`於初始生成後加入1輪。每筆候選產生`lookahead_degree`個(預設2個)變體，變體以`<MASK>`遮罩10%的字詞(`variation_degree_schedule`，預設0.1)，再由LLM填空。接著每筆訓練紀錄於句向量空間投票給最接近的候選，票數加入高斯雜訊(`noise_multiplier`，預設8.0)，再挑選候選並進行下一次變異。
- `additional_info`於每次提示附加領域規則。
- 本次執行約呼叫LLM`50 × (1 + 1 × (2 + 1)) = 200`次。

以步驟3的方式執行`llm_text_synthesis.yaml`。合成紀錄儲存為`llm_text_result_Loader[load_bloodpressure]_Synthesizer[llm-synth].csv`，各輪的中間產物位於`llm_synth_output/`的`0/`、`1/`及`1_all/`資料夾。

## 步驟5：評測合成文字

PETsARD評測方法以欄位為單位計算，因此先將文字解析為具型別的欄位。原始碼附有血壓文字之解析程式`petsard/synthesizer/LLM_synth/utils/bloodpressure_text_parser.py`，由每筆紀錄擷取性別(`gender`)、年齡(`age`)、收縮壓(`systolic`)、舒張壓(`diastolic`)及判讀結果(`bp_label`)。解析程式之撰寫方式見[文字資料評測]({{< ref "/docs/petsard-yaml/evaluator-yaml/text-evaluation" >}})。

接著於`llm_text_synthesis.yaml`加入下列`Evaluator`區塊，並將`Reporter`區塊替換為下列內容，同時輸出評測報告：

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

- `class_name: TextEvaluator`評測原始資料與合成資料。
- `text_parser`指定解析程式，兩份資料表皆由同一函式解析。
- 巢狀`Evaluator`區塊列出PETsARD評測方法及其參數，本例為有效性診斷與保真度評測。
- `output_dir`輸出解析後的資料表`parsed_ori.csv`與`parsed_syn.csv`，供核對使用。

再次執行此設定檔。沿用步驟4的合成紀錄進行評測時，於Synthesizer區塊加入`continue: true`，即直接使用`llm_synth_output/`中已完成的生成結果。

### 結果判讀

| 位置 | 內容 |
|---|---|
| `llm_text_result_Reporter[global].csv` | 每個巢狀評測方法1列，`text_eval_evaluator`欄標示名稱(`validity_check`、`fidelity_check`)，`text_eval_Score`欄為整體分數 |
| `llm_text_result_Reporter[columnwise].csv` | 各解析欄位(`gender`、`age`、`systolic`、`diastolic`、`bp_label`)的分數 |
| `llm_synth_output/parsed/parsed_ori.csv`、`parsed_syn.csv` | 解析後的資料表 |

1. **解析情形**：log列出各資料表每個欄位有值的列數，例如`syn: 50 rows -> gender=50/50, age=50/50, ...`。以101筆範例資料執行時，每列的五個欄位皆可完整擷取。
2. **有效性**：分數接近1.0，代表合成文字解析出的欄位皆落在原始資料類別與數值範圍內。
3. **保真度**：分數達0.75以上，代表解析欄位的分布與欄位間趨勢貼近原始資料。

### 隱私風險與實用性

評估隱私風險與機器學習實用性時，以Splitter保留控制資料，並將類別改為`TextEvaluatorWithControl`：

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

`sample_num_rows`設為與訓練集相當的筆數(本例為80筆)。完整設定與解析程式規範請參見[文字資料評測]({{< ref "/docs/petsard-yaml/evaluator-yaml/text-evaluation" >}})。

## 調整合成結果

| 目標 | 設定方式 |
|---|---|
| 提高與訓練資料的相似度 | 於`additional_info`寫明欄位規則；`epochs`設為1以上 |
| 提高多樣性 | `variation_degree_schedule: 0.25`、`temperature: 1.1`、`select_syn_mode: prob` |
| 加強隱私保護 | `noise_multiplier`調高為12至16，並設定`fewshot_n_examples: 5` |
| 降低記憶體用量 | `feature_extractor_batch_size: 128` |
| 縮短執行時間 | 調降`sample_num_rows`、`epochs`或`lookahead_degree` |

完整參數請參見[LLM文字合成器]({{< ref "/docs/petsard-yaml/synthesizer-yaml/llm-synthesizer" >}})。

{{< callout type="warning" >}}
**資料保護**：本文件的提示全程送至本機LM Studio伺服器處理。`api_type`設為`openai`、`anthropic`或`gemini`時，提示內容(含作為範例的訓練紀錄)傳送至該外部服務，適用於非機敏資料。`llm_synth_output/`與`LOG_FILES/`中的檔案含訓練資料文字，以與訓練資料相同的方式妥善保管。
{{< /callout >}}
