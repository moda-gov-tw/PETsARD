---
title: "差分隱私合成器"
type: docs
weight: 658
prev: docs/petsard-yaml/synthesizer-yaml/sdv-methods
next: docs/petsard-yaml/synthesizer-yaml
math: true
---

PETsARD提供三種差分隱私合成器，從讀入資料到輸出合成資料的整個流程都滿足 $(\varepsilon, \delta)$ 差分隱私，每次執行都會產生隱私報告：

| 合成器 | 資料類型 | 設定方式 |
|---|---|---|
| DP高斯耦合 | 表格(數值、類別) | `method: DPflow_Copula` |
| 更新版DP AdvGAN | 表格(數值、類別) | `method: DPflow_AdvGAN` |
| DP LLM文字合成器 | 文字(可含短欄位，例如標籤) | `method: custom_method`，`module_path: petsard/synthesizer/DP_LLM_synthesizer.py`，`class_name: DP_LLM_synthesizer` |

{{< callout type="info" >}}
`DPflow_AdvGAN`需要`advgan`擴充套件(opacus、dp-accounting)；DP LLM文字合成器需要`llm`擴充套件，並在本機以LM Studio等OpenAI相容伺服器提供模型。安裝方式請見[擴充功能安裝](../../../installation/extended-features)。
{{< /callout >}}

{{< callout type="info" >}}
初版方法(初版AdvGAN、初版LLM合成器)之合成輸出不具涵蓋完整流程之差分隱私保證，已自本版本移除，本頁三種合成器為更新後方法；如需研究比較，請參考[`initial-methods`分支](https://github.com/moda-gov-tw/PETsARD/tree/initial-methods)。
{{< /callout >}}

## 使用前須知

1. **保護單位是「一筆資料列」**：保證對象是新增或移除任一筆資料。若同一個人在資料中可能有多筆(例如多次就診、多筆交易)，請先依公開規則整理成每人一筆，例如每人隨機保留一筆，使保證以「每個人」計算。
2. **公開資訊**：欄位名稱、型別、類別欄的可能值與資料筆數視為公開。這些資訊請取自資料的公開定義(資料字典、與資料提供者約定的格式)，而非由私有資料統計得出，使隱私保證涵蓋整個流程。
3. **不需Preprocessor**：差分隱私合成器自行處理型別與尺度，隱私帳目涵蓋整個流程，設定中不加Preprocessor。
4. **缺值**：表格合成器以完整資料為輸入，缺值請先以固定規則處理，例如把缺值當成類別欄的一個類別，或以固定值填補數值欄。由JSON檔讀入的資料表亦同，JSON的`null`即為缺值。
5. **亂數種子**：`seed`可省略。要重現結果時設定；正式釋出時不要設定，噪音改由作業系統亂數產生，無法由程式與設定檔重算。
6. **預算相加**：同一份資料每執行一次合成就花一次預算，被使用或公開的各次結果，$\varepsilon$要相加。
7. **範例**：`tutorial/`資料夾有本頁三個範例的可執行版本：`run_dp1_copula.py`、`run_dp2_advgan.py`與`run_dp3_llm.py`。

## 欄位宣告(column_spec)

表格合成器以`column_spec`宣告欄位，可以是YAML或JSON檔案路徑、Markdown型別說明表，或直接寫在設定中的清單。每一欄的設定：

| 鍵 | 說明 |
|---|---|
| `name` | 欄位名稱，順序與資料相同 |
| `type` | `num`(數值)或`cat`(類別) |
| `values` | 類別欄必填，列出所有可能的值，可以是文字或數字 |
| `integer` | 選填，數值欄只有整數時設為`true`，輸出會四捨五入為整數 |

範例(節錄自`demo/petsard-yaml/synthesizer-yaml/adult-income_column_spec.yaml`)：

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

若已有Markdown型別說明表(表頭含`欄位名稱`/`name`、`型別`/`type`、`可能值`/`values`，可另含`序號`、`整數`)，可直接作為`column_spec`，或先轉換檢查：

```bash
python -m petsard.synthesizer.DPflow_column_spec 型別說明.md -o column_spec.yaml --names 欄位清單.txt
```

## DP高斯耦合(DPflow_Copula)

以高斯機制釋出各欄加噪直方圖與正規分數的二階動差矩陣，再由這兩項已釋出的結果生成資料。預算以$\mu^2$分配：直方圖20%，相關矩陣80%，以高斯差分隱私精確計算。

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

完整範例：`demo/petsard-yaml/synthesizer-yaml/DPflow_Copula_adult-income.yaml`(於專案根目錄執行)。

| 參數 | 說明 |
|---|---|
| `column_spec`(必填) | 欄位宣告，見上一節 |
| `epsilon`(必填) | 整個流程的$\varepsilon$；`.inf`表示不加噪音，作為對照基準 |
| `delta` | 預設`1.0e-5`，應遠小於1/資料筆數 |
| `sample_num_rows` | 合成筆數，預設與訓練資料相同 |
| `seed` | 選填，見「使用前須知」 |
| `privacy_report` | 選填，隱私報告(JSON)的輸出路徑 |
| `k` | 直方圖格數，預設資料少於5,000筆時512、其餘2,048 |
| `two_stage` | 預設`false`；`true`時先以公開格線找出各欄範圍再細分，適合值域窄的欄位 |
| `round_integers` | 預設`true`，宣告為整數的欄位輸出整數 |

## 更新版DP AdvGAN(DPflow_AdvGAN)

輸出為「DP種子＋生成器擾動」：種子由加噪直方圖與加噪copula產生，真實資料只經由以DP-SGD訓練的鑑別器影響生成器，取樣時不讀取任何真實資料。預算以$\mu^2$分配給直方圖、copula與DP-SGD，DP-SGD的噪音倍數依目標求解，帳目上的$\varepsilon$不會超過設定值，實際值寫在隱私報告。

本方法的整個流程都在隱私帳目內，並依`column_spec`處理型別，不使用Preprocessor與Postprocessor。

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

完整範例：`demo/petsard-yaml/synthesizer-yaml/DPflow_AdvGAN_adult-income.yaml`。

| 參數 | 說明 |
|---|---|
| `column_spec`、`epsilon`、`delta`、`seed`、`privacy_report` | 同DPflow_Copula |
| `epochs` | 訓練週期數，預設50 |
| `batch_size` | 批次大小，預設64；應遠小於資料筆數 |
| `sample_num_rows` | 合成筆數，預設100，建議明確設定 |
| `share_histograms`、`share_copula` | 預算分配($\mu^2$比例)，預設0.1與0.6，DP-SGD得到其餘0.3；兩者相加須小於1 |
| `keep_from_seed` | 選填，生成器不改動、直接沿用種子值的欄位，例如標籤欄 |
| `k`、`two_stage`、`lr`、`round_integers` | 直方圖格數(預設64)、兩段式直方圖(預設`true`)、學習率(預設5e-4)、整數輸出(預設`true`) |

## DP LLM文字合成器

本合成器釋出**差分隱私資料輪廓**：段落結構、常見詞彙與片語、長度分布，以及短欄位(例如標籤)的組合；每一項都經高斯機制與釋出門檻。模型只看到已釋出的輪廓，不會看到任何資料列。投票輪數由`epochs`設定(預設1)：`epochs: 0`時輪廓是唯一讀取資料的步驟，全部預算花在輪廓，之後依輪廓生成文字屬於後處理，不再花預算；`epochs`大於0時，另以差分隱私投票挑選合成文字，投票的預算含在`epsilon`內。本頁範例設定`epochs: 0`。

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

完整範例：`demo/petsard-yaml/synthesizer-yaml/dp-llm-synthesizer_bloodpressure.yaml`。

| 參數 | 說明 |
|---|---|
| `epsilon`、`delta`(必填) | 整個流程的預算；資料輪廓的釋出門檻另用`profile_delta_s`(預設1e-6)，已含在`delta`內 |
| `api_type`(必填) | `lmstudio`、`openai`、`anthropic`或`gemini`，沒有預設值；`lmstudio`使提示只送到本機伺服器 |
| `api_url`、`model`、`api_key` | 伺服器網址、模型ID；金鑰可改用環境變數`PETSARD_LLM_API_KEY`，不必寫在設定檔 |
| `epochs` | 差分隱私投票輪數，預設`1`；`0`表示不投票，只依輪廓生成 |
| `sample_num_rows`、`num_samples_schedule` | 合成筆數與每輪候選數 |
| `output_dir` | 輪廓(`profile.json`)、隱私報告(`privacy_report.json`)與各輪結果的資料夾 |
| `require_released_profile` | 設為`true`時，`output_dir`必須已有釋出的`profile.json`，否則停止；需搭配`continue: true`，用同一份輪廓再生成時使用，`epochs: 0`時不會再花預算 |
| `profile_joint_fields`、`profile_value_wording`、`hide_values`、`words_per_row` | 資料含短欄位(例如標籤)時的選項：釋出短欄位組合、各值的常用詞彙，並讓模型只寫長文字欄 |
| `variation_api_type`、`variation_api_url`、`variation_model`、`post_processing_api_type`、`post_processing_api_url`、`post_processing_model` | 變異與後處理階段另用的伺服器或模型；未設定時沿用`api_type`、`api_url`與`model` |
| `temperature`、`num_procs`、`sleep_time`、`max_retries`、`exit_on_api_error` | 取樣溫度(預設1.0)、同時送出的請求數(預設8，與LM Studio的Max Concurrent Predictions相配合)、呼叫間隔秒數(預設1)、失敗呼叫的嘗試次數(預設3)，以及呼叫仍失敗時是否停止(預設`false`) |
| `feature_extractor`、`feature_extractor_batch_size` | 投票用的句向量模型(預設`all-MiniLM-L6-v2`)與批次大小(預設1024)；只在`epochs`為1以上或`label_vote: true`時使用 |
| `continue` | 自`output_dir`接續執行；已釋出的輪廓與已抽出的加噪投票直接沿用，不會重抽 |
| `check_server` | 預設`true`：由資料計算任何結果之前，每個`lmstudio`階段(初始生成、變異)須回應`GET <api_url>/models`並列出其模型，否則停止執行，不釋出輪廓；僅於OpenAI相容伺服器不提供`/models`端點時設為`false` |
| `log_folder` | 每次LLM呼叫紀錄的資料夾(預設`./LOG_FILES`) |

每釋出一次輪廓就花一次預算。若重新釋出，新舊輪廓都被使用或公開時，$\varepsilon$相加。

{{< callout type="info" >}}
執行前請先開啟LM Studio伺服器並載入模型。執行時會先確認伺服器(`check_server`)，伺服器或模型無法使用時，在由資料計算任何結果之前即停止，不花用預算。若執行中仍未產生任何有效資料列(例如伺服器在執行途中關閉)，會以錯誤訊息停止，此時`output_dir`中的輪廓已經釋出；要重新執行而不再釋出資料，請保留`output_dir`，並設定`continue: true`與`require_released_profile: true`。
{{< /callout >}}

### 輸出

路徑以工作目錄為基準。合成器將結果直接交給Reporter，Reporter的`source`設為`Synthesizer`。

| 位置 | 內容 |
|---|---|
| `<output_dir>/profile.json` | 已釋出的差分隱私資料輪廓 |
| `<output_dir>/privacy_report.json` | 隱私報告 |
| `<output_dir>/0/checkpoint.csv` | 初始生成已完成的結果，`continue: true`時沿用 |
| `<output_dir>/0/prompt_generation.jsonl` | 初始生成送出的提示與生成的文字 |
| `<output_dir>/epoch<t>_result.csv` | 設定`stop_after_epoch`時，第`t`輪後的結果 |
| `<log_folder>/` | 每次LLM呼叫的提示與回應 |

以PETsARD評測方法評估合成文字，請參見[文字資料評測]({{< ref "/docs/petsard-yaml/evaluator-yaml/text-evaluation" >}})。

## 隱私報告

每次執行都會輸出隱私報告(表格合成器寫入`privacy_report`指定的路徑；DP LLM寫入`output_dir/privacy_report.json`)，包含目標與實際的$\varepsilon$、$\delta$、預算分配、噪音大小、資料筆數，以及種子來源(有設定時記錄種子值，未設定時只記錄「作業系統亂數」)。

## 參考文獻

- Xiao, C., Li, B., Zhu, J.-Y., He, W., Liu, M., & Song, D. (2018). Generating Adversarial Examples with Adversarial Networks. Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI-18). https://arxiv.org/abs/1801.02610
- Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep Learning with Differential Privacy. Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS). https://doi.org/10.1145/2976749.2978318
- Yousefpour, A., et al. (2021). Opacus: User-Friendly Differential Privacy Library in PyTorch. arXiv:2109.12298. https://arxiv.org/abs/2109.12298
- Xie, C., et al. (2024). Differentially Private Synthetic Data via Foundation Model APIs 2: Text. Proceedings of the 41st International Conference on Machine Learning (ICML 2024). https://arxiv.org/abs/2403.01749
- DP LLM文字合成器之Private Evolution流程改寫自Aug-PE(https://github.com/AI-secure/aug-pe，Apache License 2.0)；改寫之檔案及本版之修改內容列於`petsard/synthesizer/DP_LLM_synth/NOTICE`。
