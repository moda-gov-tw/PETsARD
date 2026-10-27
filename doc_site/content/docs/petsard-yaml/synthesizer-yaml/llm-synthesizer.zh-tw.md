---
title: "LLM文字合成器"
type: docs
weight: 657
prev: docs/petsard-yaml/synthesizer-yaml/advgan
next: docs/petsard-yaml/synthesizer-yaml
---

LLM文字合成器以大型語言模型(LLM)生成合成的自由文字紀錄，適用於每列為1段完整文字資料，例如顧客意見、問卷回答、客服工單，以及系統或網路日誌。合成器以訓練資料中的紀錄作為範例提示模型，經多輪變異改寫候選文字，並以差分隱私最近鄰投票挑選候選，全程不需對模型進行微調。

本合成器以PETsARD標準的`custom_method`介面載入(見[自訂合成方法]({{< ref "/docs/petsard-yaml/synthesizer-yaml/custom-method" >}}))，可搭配[LM Studio](https://lmstudio.ai/)於本機提供的模型，或OpenAI、Anthropic及Gemini的API執行。

{{< callout type="info" >}}
包含LM Studio設定及合成文字評測的完整操作步驟，參見[本機LLM文字合成與文字評測]({{< ref "/docs/getting-started/llm-text-synthesis" >}})。
{{< /callout >}}

## 執行需求

執行環境依[擴充功能安裝]({{< ref "/docs/installation/extended-features" >}})建置：

1. 於原始碼根目錄以可編輯模式安裝PETsARD，並安裝本合成器相依套件(套件群組`llm`)：

   ```bash
   pip install -e ".[all,advgan,llm]"
   ```

2. LLM服務端點：
   - **本機推論(建議)**：LM Studio已載入模型並啟動本機伺服器(預設位址`http://localhost:1234/v1`)。
   - **雲端推論**：OpenAI、Anthropic或Google Gemini的API金鑰。
3. `compute_fid: true`(預設值)時，首次執行自Hugging Face下載句向量模型`all-MiniLM-L6-v2`並快取於本機，之後直接使用快取。

## 使用範例

範例於PETsARD原始碼根目錄執行，範例中的模組路徑皆相對於原始碼根目錄。

### 合成整筆紀錄

範例資料為基準資料集`bloodpressure-text`(檔案`real_bloodpressure2.csv`，執行時自動下載)，共101筆、單一`text`欄位，每列以1句中文描述1位受檢者之姓名、性別、年齡、血壓量測值及判讀結果(偏高或正常)。

```yaml
Loader:
  load_bloodpressure:
    filepath: benchmark://bloodpressure-text
Synthesizer:
  llm-synth:
    method: custom_method
    module_path: petsard/synthesizer/LLM_synthesizer.py
    class_name: LLM_synthesizer
    sample_num_rows: 20            # 輸出的合成紀錄筆數
    epochs: 1                      # 初始生成加1輪變異與挑選
    api_type: lmstudio             # 本機推論
    api_url: "http://localhost:1234/v1"
    model: openai/gpt-oss-20b      # LM Studio已載入的模型名稱
    num_procs: 4                   # 與LM Studio的Max Concurrent Predictions相配合
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

合成紀錄儲存為`llm_synth_result_Loader[load_bloodpressure]_Synthesizer[llm-synth].csv`，欄位與輸入資料相同。

### 合成單一文字欄位

資料同時含自由文字欄位與結構化屬性時，以`main_column_name`指定僅生成該文字欄位。以含`channel`、`rating`及`feedback`三個欄位的`customer_feedback.csv`為例：

```yaml
Loader:
  load_feedback:
    filepath: customer_feedback.csv
Synthesizer:
  llm-feedback:
    method: custom_method
    module_path: petsard/synthesizer/LLM_synthesizer.py
    class_name: LLM_synthesizer
    main_column_name: feedback     # 僅生成此欄位
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

每筆生成的`feedback`文字皆以1筆訓練紀錄(即下文所描述的查詢範例)為參照，並沿用該筆紀錄的`channel`與`rating`，使每筆合成文字與其參照紀錄的屬性維持對應。

### 使用雲端API

將`api_type`設為`openai`、`anthropic`或`gemini`並指定模型；API金鑰於執行PETsARD前設定於環境變數`PETSARD_LLM_API_KEY`(例如bash之`export PETSARD_LLM_API_KEY=...`或PowerShell之`$env:PETSARD_LLM_API_KEY = "..."`)，金鑰不寫入YAML檔：

```yaml
    api_type: openai
    model: gpt-4o
```

{{< callout type="warning" >}}
**資料保護**：`api_type`設為`openai`、`anthropic`或`gemini`時，每次提示內容(含作為少樣本範例的訓練紀錄)皆傳送至該外部服務，適用於非機敏資料；機敏資料以`lmstudio`於本機推論。`prompt_generation.jsonl`及`log_folder`中的紀錄檔亦含訓練資料文字，以與訓練資料相同方式妥善保管；設定`log_prompts: false`時，紀錄檔僅記錄各段文字之長度，且不寫入`prompt_generation.jsonl`。
{{< /callout >}}

## 運作方式

1. **學習(fit)**：合成器保存訓練資料的各列，作為私有範例來源。合成整筆紀錄時，每列轉為1行加引號的CSV字串，欄位名稱作為標頭；設定`main_column_name`時，僅使用該欄位文字。
2. **初始生成(第0輪)**：每筆目標紀錄隨機抽取`fewshot_n_examples`筆訓練紀錄，前面各筆為*支援集*，呈現資料結構、風格及長度；最後1筆為*查詢集*，新紀錄依循其設定(例如紀錄類型與結構)。LLM每次呼叫生成1筆新紀錄。`additional_info`內容以`<AdditionalInfo>`標籤附加於每次提示；模型需要數值時，可呼叫`random_random`與`random_uniform`兩項亂數工具。
3. **變異與挑選**(重複`epochs`輪)：
   1. *前瞻變異*：每筆候選依`variation_degree_schedule`比例，以`<MASK>`遮罩部分字詞，再由LLM填空，產生`lookahead_degree`個變體。中文、日文及韓文以單一字元為遮罩單位，標點與其他結構符號維持原樣。
   2. *句向量*：以句向量模型(`feature_extractor`，預設`all-MiniLM-L6-v2`)計算訓練紀錄的句向量，以及各候選的前瞻變體句向量平均值。
   3. *差分隱私最近鄰直方圖*：每筆訓練紀錄投票給最接近的候選(`nn_mode`、`num_nearest_neighbor`)，票數加入標準差為`noise_multiplier × √num_nearest_neighbor`的高斯雜訊。
   4. *挑選*：保留`N / combine_divide_L`筆候選，依加噪後票數由高至低保留(`select_syn_mode: rank`)，或依票數比例抽樣(`select_syn_mode: prob`)。
   5. *下一輪候選*：對保留的候選再以遮罩填空產生變體，作為下一輪的候選。
4. **輸出**：最終候選轉回與原始資料相同欄位的表格，移除空白的生成結果，取前`sample_num_rows`筆回傳PETsARD。

`combine_divide_L`維持預設值1時，LLM呼叫次數約為`sample_num_rows × (1 + epochs × (lookahead_degree + 1))`。例如合成20筆、`epochs: 1`、`lookahead_degree`為預設值2時，約呼叫80次。

## 參數說明

### PETsARD參數

| 參數 | 類型 | 預設值 | 說明 |
|---|---|---|---|
| `method` | `string` | 必要 | 固定值：`custom_method` |
| `module_path` | `string` | 必要 | `petsard/synthesizer/LLM_synthesizer.py`，由執行目錄或Python路徑(可編輯模式安裝後即原始碼根目錄)解析 |
| `class_name` | `string` | 必要 | `LLM_synthesizer` |
| `sample_num_rows` | `integer` | 訓練資料筆數 | 輸出的合成紀錄筆數；內部排程依此數值自動配置，超出的筆數於輸出前截除。每筆紀錄需多次呼叫LLM，建議明確設定 |

### 資料

| 參數 | 類型 | 預設值 | 說明 |
|---|---|---|---|
| `main_column_name` | `string` | 未設定 | 僅生成此文字欄位，其餘欄位沿用各筆文字所參照的訓練紀錄；未設定時生成整筆紀錄，再解析回原始欄位 |

### LLM API

| 參數 | 類型 | 預設值 | 說明 |
|---|---|---|---|
| `api_type` | `string` | 無(必填) | `lmstudio`、`openai`、`anthropic`或`gemini`；未設定時以`ValueError`停止執行 |
| `api_url` | `string` | 未設定 | LM Studio伺服器網址，例如`http://localhost:1234/v1`；僅適用於`api_type: lmstudio`，網址未以`/v1`結尾時自動補上 |
| `api_key` | `string` | 未設定 | 雲端服務的API金鑰；未設定時由環境變數`PETSARD_LLM_API_KEY`讀取。`lmstudio`未設定`api_url`時，以此值作為伺服器網址；兩者皆未設定時使用`http://localhost:1234/v1`。LM Studio伺服器位於需驗證的端點時，金鑰填入此參數，網址填入`api_url` |
| `model` | `string` | 依服務而定 | 初始生成使用的模型。預設值：`gpt-4o`(openai)、`claude-sonnet-4-20250514`(anthropic)、`gemini-2.5-flash`(gemini)。使用`lmstudio`時，設為`/v1/models`所列已載入模型名稱 |
| `variation_api_type`、`variation_api_key`、`variation_api_url`、`variation_model` | `string` | 沿用初始生成設定 | 變異(遮罩填空)階段的API設定 |
| `post_processing_api_type`、`post_processing_api_key`、`post_processing_api_url`、`post_processing_model` | `string` | 沿用初始生成設定 | 後處理代理的API設定 |

### 生成與挑選

| 參數 | 類型 | 預設值 | 說明 |
|---|---|---|---|
| `epochs` | `integer` | `1` | 初始生成後的變異與挑選輪數；設為`0`時僅執行少樣本導引的初始生成 |
| `lookahead_degree` | `integer` | `2` | 每筆候選於投票前產生的前瞻變體數；設為`0`時以候選本身投票 |
| `variation_degree_schedule` | `float`或`list` | `0.1` | 變異時遮罩的字詞比例，介於0至1；單一數值套用於各輪，清單則逐輪指定(長度為`epochs + 1`) |
| `noise_multiplier` | `float` | `8.0` | 最近鄰票數的高斯雜訊倍數；數值越大，隱私保護越強 |
| `compute_fid` | `boolean` | `true` | 計算句向量，供差分隱私挑選使用，並將FID記錄於`fid.csv`。以語意挑選候選時維持`true`；設為`false`時不計算句向量，以均等票數挑選，適合快速確認流程 |
| `feature_extractor` | `string` | `all-MiniLM-L6-v2` | 計算句向量的sentence-transformers模型 |
| `feature_extractor_batch_size` | `integer` | `1024` | 句向量計算的批次大小；調降(例如`128`)可減少記憶體用量 |
| `select_syn_mode` | `string` | `rank` | `rank`保留加噪票數最高候選；`prob`依票數比例抽樣候選 |
| `nn_mode` | `string` | `L2` | 投票使用距離：`L2`、`IP`(內積)或`cos_sim` |
| `num_nearest_neighbor` | `integer` | `1` | 每筆訓練紀錄投票的最近候選數 |
| `count_threshold` | `float` | `0.0` | 挑選前自加噪票數扣除的門檻值，票數不超過此值者設為0 |
| `combine_divide_L` | `integer` | `1` | 每輪保留`N / combine_divide_L`筆候選，並對每筆保留的候選產生`combine_divide_L - 1`個變體補足候選數 |
| `donnot_keep_last_iter` | `boolean` | `false` | `combine_divide_L`大於1時，設為`true`則`combine_divide_L`個名額皆由新變體填入，不沿用保留的候選 |
| `init_combine_divide_L` | `integer` | `1` | 第0輪由LLM生成`N / init_combine_divide_L`筆種子紀錄，再對每筆產生`init_combine_divide_L - 1`個變體 |
| `lookahead_self` | `boolean` | `false` | 計算候選的句向量時，將候選本身與其前瞻變體一併納入 |
| `num_samples_schedule` | `integer`或`list` | 自動 | 各輪候選數；未設定時依`sample_num_rows`計算 |
| `num_private_samples` | `integer` | `-1` | 作為私有範例的訓練資料筆數上限；`-1`使用全部資料 |

### LLM呼叫

| 參數 | 類型 | 預設值 | 說明 |
|---|---|---|---|
| `fewshot_n_examples` | `integer` | `10` | 每次提示附帶的訓練紀錄範例筆數 |
| `additional_info` | `string` | `""` | 以`<AdditionalInfo>`標籤附加於每次提示的領域規則，例如欄位格式或數值範圍 |
| `temperature` | `float` | `1.0` | 取樣溫度；數值越高，生成文字越多樣 |
| `top_p` | `float` | `1.0` | 核取樣(nucleus sampling)的機率門檻 |
| `max_token_limit` | `integer` | `100000` | 每次呼叫的輸出上限；推理型模型(例如`openai/gpt-oss-20b`)的推理過程占用輸出額度，維持較大值 |
| `num_procs` | `integer` | `8` | 平行請求數；本機LM Studio建議設為2至4，並與LM Studio的Max Concurrent Predictions相配合 |
| `sleep_time` | `integer` | `1` | 每次呼叫前的等待秒數，亦為重試間隔的基準 |
| `max_retries` | `integer` | `3` | 每次LLM呼叫的嘗試次數上限(含第1次)；呼叫發生錯誤或回應空白時重新嘗試 |
| `retry_wait_time` | `float` | `15` | 每次重試前等待之秒數(固定值) |
| `exit_on_api_error` | `boolean` | `false` | 設為`true`時，呼叫嘗試`max_retries`次後仍未取得回應，即停止執行 |
| `seed` | `integer` | `42` | Python、NumPy及PyTorch亂數種子 |
| `no_cuda` | `boolean` | `false` | 設為`true`時，合成器運算裝置設為CPU |

### 執行控制與紀錄

| 參數 | 類型 | 預設值 | 說明 |
|---|---|---|---|
| `output_dir` | `string` | `./llm_synth_output` | 各輪中間產物目錄(見下方輸出說明) |
| `continue` | `boolean` | `false` | 設為`true`時，由`output_dir/<輪次>/checkpoint.csv`接續先前執行，已完成的生成結果直接沿用，空白的生成結果重新生成 |
| `save_syn_mode` | `string` | `selected` | 寫入`output_dir/<輪次>/samples.csv`紀錄：`selected`(保留的候選)、`one_var`(每筆保留候選的第1個變體)或`all`(全部新變體)；回傳PETsARD的紀錄於各模式皆相同 |
| `log_folder` | `string` | `./LOG_FILES` | LLM互動紀錄目錄 |
| `log_prompts` | `boolean` | `true` | `true`時，提示與回覆之全文寫入`log_folder`紀錄檔及`prompt_generation.jsonl`；`false`時，紀錄檔僅記錄各段文字之長度，且不寫入`prompt_generation.jsonl`。兩種設定之生成過程與結果相同 |
| `verbose` | `integer` | `0` | `0`僅寫入紀錄檔；`1`另於終端機輸出執行步驟及每次呼叫的提示與回覆；`2`另輸出系統提示(system prompt) |

`do_sample`、`variation_type`、`dynamic_len`、`min_target_word`、`word_var_scale`及`max_token_word_scale`等參數為相容Aug-PE設定檔而保留，設定後不影響生成。

## 輸出

以下路徑相對於執行目錄。

| 位置 | 內容 |
|---|---|
| `<Reporter output>_Loader[...]_Synthesizer[...].csv` | 最終合成紀錄，由Reporter(`source: Synthesizer`)儲存 |
| `<output_dir>/0/samples.csv` | 初始生成紀錄，`unique_id`欄標示所參照的查詢範例 |
| `<output_dir>/<輪次>/samples.csv` | 各輪依`save_syn_mode`儲存的紀錄 |
| `<output_dir>/<輪次>_all/samples.csv` | 各輪產生的全部候選(`epochs`為1以上時產生) |
| `<output_dir>/<輪次>/prompt_generation.jsonl` | 各輪送出的提示與生成文字；於`log_prompts: true`(預設值)時寫入 |
| `<output_dir>/<輪次>/checkpoint.csv` | 逐筆寫入的已完成生成結果，供`continue: true`接續使用 |
| `<output_dir>/fid.csv` | 每次執行的每一輪記錄1行：`<輪次> <FID> <F1> <precision> <recall>`。判讀以FID欄為準，數值越低，代表合成文字與訓練文字的句向量分布越接近 |
| `<log_folder>/MainSyntheticData_<YYYYMMDD_HHMMSS>.txt` | 每次LLM呼叫的系統提示、提示及回覆；`log_prompts: false`時僅記錄文字長度 |

`fid.csv`於每次執行時追加內容，其他各輪檔案則依新執行的結果更新。並列比較多次執行結果時，為每次執行指定不同的`output_dir`與`log_folder`。

## 建議設定

| 目標 | 設定方式 |
|---|---|
| 確認執行環境 | `sample_num_rows: 5`、`epochs: 0` |
| 提高與訓練資料的相似度 | 於`additional_info`寫明欄位規則；`epochs`設為1以上 |
| 提高多樣性 | `variation_degree_schedule: 0.25`、`temperature: 1.1`、`select_syn_mode: prob` |
| 加強隱私保護 | `noise_multiplier`調高為12至16、`fewshot_n_examples: 5`，並以`lmstudio`於本機推論 |
| 降低記憶體用量 | `feature_extractor_batch_size: 128` |
| 縮短執行時間 | 調降`sample_num_rows`、`epochs`或`lookahead_degree` |

## 執行說明

- 於LM Studio載入模型時，上下文長度(Context Length)設為32,768 tokens以上；執行前以`curl http://localhost:1234/v1/models`(Windows使用`curl.exe`)確認伺服器狀態與模型名稱。
- `model`，以及有使用的`variation_model`、`post_processing_model`，設為與`/v1/models`所列名稱完全一致的模型名稱。
- 合成器將紀錄直接交給Reporter，Reporter的`source`設為`Synthesizer`。
- 合成文字以[文字資料評測]({{< ref "/docs/petsard-yaml/evaluator-yaml/text-evaluation" >}})模組評測，該模組先將文字解析為欄位，再執行PETsARD的評測方法。

## 參考文獻

- Xie, C., et al. (2024). Differentially Private Synthetic Data via Foundation Model APIs 2: Text. Proceedings of the 41st International Conference on Machine Learning (ICML 2024). https://arxiv.org/abs/2403.01749
- 本合成器之Private Evolution流程改寫自Aug-PE(https://github.com/AI-secure/aug-pe，Apache License 2.0)；改寫之檔案及本版之修改內容列於`petsard/synthesizer/LLM_synth/NOTICE`。
