---
title: JSON資料與AdvGAN合成
type: docs
weight: 140
prev: docs/getting-started/external-synthesis-default-evaluation
next: docs/getting-started/llm-text-synthesis
---

以JSON格式讀取與輸出的半結構化資料合成。
本範例載入JSON紀錄，以AdvGAN合成器產生合成資料，評估有效性、保真度、隱私與效用，並將合成資料與評估報告皆以JSON格式儲存。

{{< callout type="info" >}}
本範例使用本版的擴充功能。請先依[擴充功能安裝](../../installation/extended-features)完成環境設定(包含`opacus`套件)，並於原始碼根目錄執行以下指令。
{{< /callout >}}

## 範例資料

本範例使用基準資料集`customer-churn`(檔案`customer_churn.json`)，執行時自動下載至執行目錄下之`benchmark/`。內容為1,200筆客戶紀錄、10個欄位，以JSON紀錄陣列儲存。此資料為示範用途所產生，合約類型、使用月數、費用與`churn`之間具有可學習的關聯。

```json
[
  {
    "age": 66,
    "region": "East",
    "contract_type": "Month-to-month",
    "payment_method": "Bank-transfer",
    "tenure_months": 39,
    "num_products": 2,
    "monthly_charges": 38.82,
    "total_charges": 1570.96,
    "has_support": "No",
    "churn": "Yes"
  }
]
```

| 欄位 | 類型 | 值域 |
|---|---|---|
| `age` | 整數 | 18至80 |
| `region` | 類別 | `Central`、`East`、`North`、`South`、`West` |
| `contract_type` | 類別 | `Month-to-month`、`One-year`、`Two-year` |
| `payment_method` | 類別 | `Bank-transfer`、`Credit-card`、`E-check`、`Mailed-check` |
| `tenure_months` | 整數 | 1至72 |
| `num_products` | 整數 | 1至4 |
| `monthly_charges` | 小數 | 19.04至117.99 |
| `total_charges` | 小數 | 19.66至8,309.74 |
| `has_support` | 類別 | `No`、`Yes` |
| `churn` | 類別 | `No`、`Yes`(目標欄位) |

## YAML設定

將下列設定儲存為原始碼根目錄`json-advgan-synthesis.yaml`：

```yaml
Loader:
  load_json:
    filepath: benchmark://customer-churn
Splitter:
  holdout:
    num_samples: 1
    train_split_ratio: 0.8
Preprocessor:
  default:
    method: 'default'
Synthesizer:
  advgan:
    method: 'AdvGAN'
    sigma: 1.0
    outlier: 10
    epochs: 5
    sample_num_rows: 960
Postprocessor:
  default:
    method: 'default'
Evaluator:
  diagnostic:
    method: sdmetrics-diagnosticreport
  quality:
    method: sdmetrics-qualityreport
  singling_out:
    method: anonymeter-singlingout
    n_attacks: 100
    n_cols: 3
  classification_utility:
    method: mlutility
    task_type: classification
    target: churn
Reporter:
  save_synthetic:
    method: save_data
    source: Postprocessor
    output: churn_advgan
    type: 'json'
  evaluation_report:
    method: save_report
    granularity: global
    output: churn_advgan
    type: 'json'
```

## 執行範例

將下列程式儲存為原始碼根目錄`run_json_advgan.py`：

```python
from petsard import Executor

exec = Executor(config="json-advgan-synthesis.yaml")
exec.run()
```

{{< tabs items="Windows,macOS/Linux" >}}
{{< tab >}}
```bat
python -X utf8 run_json_advgan.py
```
{{< /tab >}}
{{< tab >}}
```bash
python run_json_advgan.py
```
{{< /tab >}}
{{< /tabs >}}

訓練時，終端機顯示前處理後的訓練資料筆數與運算裝置，接著顯示各項評估報告的進度：

```
The number of data: <訓練資料筆數>
Using device: cpu
```

## YAML參數詳細說明

### Loader(資料載入模組)

- **`load_json`**：實驗名稱，可自由命名
- **`filepath`**：資料檔案路徑
  - 值：`benchmark://customer-churn`
  - 說明：以[基準資料集協議](../../petsard-yaml/loader-yaml/benchmark-protocol)下載`customer_churn.json`並核對SHA-256。副檔名`.json`即自動選用JSON讀取器。檔案最上層為紀錄陣列，每個物件為1筆資料列，每個鍵為1個欄位
  - 巢狀JSON與JSON Lines整理方式請參見[JSON格式](../../petsard-yaml/loader-yaml/json-format)

### Splitter(資料分割模組)

- **`holdout`**：實驗名稱，可自由命名
- **`num_samples`**：`1`，切分1組訓練集與控制集
- **`train_split_ratio`**：`0.8`，訓練集960筆、控制集240筆
- 說明：控制集供隱私與效用評估(`anonymeter-singlingout`與`mlutility`)使用，以未參與訓練的紀錄與合成資料比較

### Preprocessor(資料前處理模組)

- **`default`**：實驗名稱，可自由命名
- **`method`**：`default`
- 說明：處理缺失值與離群值、編碼類別欄位並縮放數值欄位。AdvGAN以完整的數值資料表訓練，因此AdvGAN流程皆包含Preprocessor

### Synthesizer(合成資料產生模組)

- **`advgan`**：實驗名稱，可自由命名
- **`method`**：`AdvGAN`，鑑別器透過Opacus以DP-SGD訓練的改良式AdvGAN合成器
- **`sigma`**：`1.0`，鑑別器DP-SGD雜訊倍數(預設`1.0`)；數值越大，隱私保護越強
- **`outlier`**：`10`，邊界懲罰權重(預設`10`)；數值越小，生成器產生越多接近各欄位極端值的資料
- **`epochs`**：`5`，訓練週期數(預設`200`)；範例以5個週期快速完成流程
- **`sample_num_rows`**：`960`，合成筆數，與訓練集筆數相同

演算法與完整參數請參見[AdvGAN合成器](../../petsard-yaml/synthesizer-yaml/advgan)。

### Postprocessor(資料後處理模組)

- **`default`**：實驗名稱，可自由命名
- **`method`**：`default`
- 說明：執行反縮放與反編碼，使合成紀錄還原為原始單位與原始類別標籤，例如`Month-to-month`與`Yes`

### Evaluator(資料評測模組)

| 實驗名稱 | `method` | 面向 | 主要參數 |
|---|---|---|---|
| `diagnostic` | `sdmetrics-diagnosticreport` | 資料有效性與結構 | 無 |
| `quality` | `sdmetrics-qualityreport` | 保真度：欄位分布與欄位配對趨勢 | 無 |
| `singling_out` | `anonymeter-singlingout` | 隱私：指認性風險，使用控制集 | `n_attacks: 100`、`n_cols: 3` |
| `classification_utility` | `mlutility` | 效用：以合成資料訓練模型，並以控制集測試 | `task_type: classification`、`target: churn` |

### Reporter(結果輸出模組)

- **`save_synthetic`**：以`method: save_data`儲存`Postprocessor`輸出
- **`evaluation_report`**：以`method: save_report`與`granularity: global`儲存整體評估分數
- **`output`**：`churn_advgan`，兩份輸出檔的檔名前綴
- **`type`**：`json`，兩份輸出皆以JSON紀錄陣列儲存；未設定`type`時輸出CSV

## 執行流程說明

1. **Loader**以JSON讀取器讀取`customer_churn.json`
2. **Splitter**將資料切分為訓練集(80%)與控制集(20%)
3. **Preprocessor**將訓練集轉換為完整的數值資料表
4. **Synthesizer**以DP-SGD鑑別器訓練AdvGAN，並產生960筆合成資料
5. **Postprocessor**還原原始單位與類別標籤
6. **Evaluator**執行有效性、保真度、隱私與效用評估
7. **Reporter**將合成資料與評估報告以JSON格式儲存

## 輸出檔案

兩份輸出檔寫入原始碼根目錄：

| 檔案 | 內容 |
|---|---|
| `churn_advgan_Loader[load_json]_Splitter[holdout_[1-1]]_Preprocessor[default]_Synthesizer[advgan]_Postprocessor[default].json` | 960筆合成紀錄，欄位與輸入資料的10個欄位相同 |
| `churn_advgan_Reporter[global].json` | 4項評估的整體分數 |

合成資料沿用輸入檔格式，可直接由Loader再次載入，或交由其他JSON工具使用：

```json
[
    {
        "age":38,
        "region":"North",
        "contract_type":"Two-year",
        "payment_method":"Credit-card",
        "tenure_months":54,
        "num_products":1,
        "monthly_charges":28.45,
        "total_charges":7017.77,
        "has_support":"No",
        "churn":"No"
    }
]
```

評估報告的每筆紀錄先列出各模組的實驗名稱，再列出以評估實驗名稱為前綴的分數欄位：

| 欄位 | 意義 | 參考標準 |
|---|---|---|
| `diagnostic_Score` | 有效性診斷分數 | 接近1.0 |
| `quality_Score` | 保真度分數，組成項目為`quality_Column Shapes`與`quality_Column Pair Trends` | 高於0.75 |
| `singling_out_risk` | 指認性風險，信賴區間為`singling_out_risk_CI_btm`與`singling_out_risk_CI_top` | 低於0.09 |
| `classification_utility_ori`、`classification_utility_syn` | 分別以原始訓練集與合成資料訓練模型，於控制集測試分數；`classification_utility_metric`為指標名稱 | 比較`_syn`與`_ori`；`classification_utility_diff`為`_syn`減`_ori` |

各項分數的意義請參見[以預設參數做資料合成與資料評測](../default-synthesis-default-evaluation)。

## 調整範例

- **完整訓練**：將`epochs`提高至預設值`200`
- **隱私與效用的平衡**：調整`sigma`；數值越大，鑑別器的隱私保護越強
- **極端值**：調整`outlier`；數值越小，合成越多接近各欄位極端值的資料
- **使用自有資料**：將`filepath`指向自有JSON檔，將`sample_num_rows`設為其訓練集筆數，並加入`schema`固定欄位型別(見[JSON格式](../../petsard-yaml/loader-yaml/json-format#搭配schema使用))
- **與其他合成器比較**：於`Synthesizer`下新增第2個實驗，例如`method: 'default'`的`gaussian-copula`，後續各模組即為每個合成器分別產生結果
- **輸出CSV**：移除Reporter區塊中的`type: 'json'`

## 進階使用

- [JSON格式](../../petsard-yaml/loader-yaml/json-format)
- [AdvGAN合成器](../../petsard-yaml/synthesizer-yaml/advgan)
- [儲存資料](../../petsard-yaml/reporter-yaml/save-data)
- [Evaluator YAML參數說明](../../petsard-yaml/evaluator-yaml/)
