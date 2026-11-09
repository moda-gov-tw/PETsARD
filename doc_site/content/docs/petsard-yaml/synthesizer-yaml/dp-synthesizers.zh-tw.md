---
title: "差分隱私合成器"
type: docs
weight: 658
prev: docs/petsard-yaml/synthesizer-yaml/llm-synthesizer
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

## 使用前須知

1. **保護單位是「一筆資料列」**：保證對象是新增或移除任一筆資料。若同一個人在資料中可能有多筆(例如多次就診、多筆交易)，請先依公開規則整理成每人一筆，例如每人隨機保留一筆，使保證以「每個人」計算。
2. **公開資訊**：欄位名稱、型別、類別欄的可能值與資料筆數視為公開。這些資訊請取自資料的公開定義(資料字典、與資料提供者約定的格式)，而非由私有資料統計得出，使隱私保證涵蓋整個流程。
3. **不需Preprocessor**：差分隱私合成器自行處理型別與尺度，隱私帳目涵蓋整個流程，設定中不加Preprocessor。
4. **缺值**：表格合成器以完整資料為輸入，缺值請先以固定規則處理，例如把缺值當成類別欄的一個類別，或以固定值填補數值欄。
5. **亂數種子**：`seed`可省略。要重現結果時設定；正式釋出時不要設定，噪音改由作業系統亂數產生，無法由程式與設定檔重算。
6. **預算相加**：同一份資料每執行一次合成就花一次預算，被使用或公開的各次結果，$\varepsilon$要相加。

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

每釋出一次輪廓就花一次預算。若重新釋出，新舊輪廓都被使用或公開時，$\varepsilon$相加。

## 隱私報告

每次執行都會輸出隱私報告(表格合成器寫入`privacy_report`指定的路徑；DP LLM寫入`output_dir/privacy_report.json`)，包含目標與實際的$\varepsilon$、$\delta$、預算分配、噪音大小、資料筆數，以及種子來源(有設定時記錄種子值，未設定時只記錄「作業系統亂數」)。
