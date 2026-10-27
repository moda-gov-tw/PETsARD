---
title: "JSON格式"
type: docs
weight: 622
prev: docs/petsard-yaml/loader-yaml/benchmark-protocol
next: docs/petsard-yaml/loader-yaml
---

Loader可直接讀取半結構化JSON資料。`filepath`副檔名為`.json`時，Loader自動選用JSON讀取器，不需另行設定參數。

## 使用範例

範例使用基準資料集`customer-churn`(檔案`customer_churn.json`)，執行時自動下載至執行目錄下之`benchmark/`。讀取自有資料時，將`filepath`設為`.json`檔案之路徑。

### 基本載入

```yaml
Loader:
  load_json:
    filepath: benchmark://customer-churn
```

### 使用表詮釋資料檔案

```yaml
Loader:
  load_json_with_schema:
    filepath: benchmark://customer-churn
    schema: customer_churn_schema.yaml
```

### JSON讀取與JSON輸出

搭配Reporter的`type: 'json'`，整個流程皆以JSON格式讀寫：

```yaml
Loader:
  load_json:
    filepath: benchmark://customer-churn
Preprocessor:
  default:
    method: 'default'
Synthesizer:
  default:
    method: 'default'
Postprocessor:
  default:
    method: 'default'
Reporter:
  save_synthetic:
    method: save_data
    source: Postprocessor
    type: 'json'
```

合成資料輸出為`petsard_Loader[load_json]_Preprocessor[default]_Synthesizer[default]_Postprocessor[default].json`。`type`的設定方式請參見[儲存資料](../../reporter-yaml/save-data)。

## 輸入格式

JSON檔的最上層為物件陣列(`records`格式)：

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
  },
  {
    "age": 51,
    "region": "North",
    "contract_type": "One-year",
    "payment_method": "Bank-transfer",
    "tenure_months": 28,
    "num_products": 2,
    "monthly_charges": 49.64,
    "total_charges": 1440.55,
    "has_support": "No",
    "churn": "No"
  }
]
```

| JSON元素 | 載入結果 |
|---|---|
| 陣列中的每個物件 | 1筆資料列(紀錄) |
| 每個鍵 | 1個欄位，鍵名即欄位名稱 |
| 數字、字串、布林值 | 儲存格的值，型別由資料推斷或依Schema設定 |
| `null`，或紀錄中未出現的鍵 | 缺失值(`pd.NA`) |

- 檔案以UTF-8讀取，欄位名稱與內容可使用任何語言(包含繁體中文)，並完整保留至JSON輸出。
- 副檔名不區分大小寫，`.json`與`.JSON`皆選用JSON讀取器。
- 缺失值以`null`表示，Preprocessor即以與CSV資料相同的方式處理。

## 準備JSON資料

Loader以單層紀錄為讀取單位，每個值皆為純量。其他結構的JSON資料，先以pandas整理後再載入。

### 展平巢狀JSON

巢狀物件以`pandas.json_normalize`展開為單層欄位，每條巢狀鍵路徑成為1個欄位名稱：

```json
[
  {
    "age": 66,
    "region": "East",
    "contract": {"type": "Month-to-month", "tenure_months": 39},
    "billing": {"payment_method": "Bank-transfer", "monthly_charges": 38.82},
    "churn": "Yes"
  }
]
```

```python
import json
import pandas as pd

with open("customers_nested.json", encoding="utf-8") as f:
    records = json.load(f)

flat = pd.json_normalize(records, sep="_")
flat.to_json("customers_flat.json", orient="records", force_ascii=False, indent=2)
```

展平後的檔案包含`age`、`region`、`churn`、`contract_type`、`contract_tenure_months`、`billing_payment_method`及`billing_monthly_charges`等欄位，以`filepath: customers_flat.json`載入。

紀錄中含有陣列時，以`pandas.json_normalize`的`record_path`與`meta`引數將陣列的每個元素展開為1筆資料列，或先將陣列彙總為純量欄位(例如筆數或總和)再儲存。

### 轉換JSON Lines

JSON Lines檔(`.jsonl`，每行1個物件)先轉為物件陣列，並以`.json`副檔名儲存：

```python
import pandas as pd

pd.read_json("data.jsonl", lines=True).to_json(
    "data.json", orient="records", force_ascii=False, indent=2
)
```

### 將CSV轉換為JSON

```python
import pandas as pd

pd.read_csv("data.csv").to_json(
    "data.json", orient="records", force_ascii=False, indent=2
)
```

## 搭配Schema使用

Schema用於JSON檔的方式與CSV檔相同，可固定各欄位的型別、標示類別欄位並設定數值精度。以下Schema描述客戶流失資料，儲存為執行目錄下之`customer_churn_schema.yaml`即可搭配上方的表詮釋資料範例使用：

```yaml
id: customer-churn
name: "Customer Churn"
description: "Customer contract and churn records"

attributes:
  age:
    type: integer
  region:
    type: string
    category: true
  contract_type:
    type: string
    category: true
  payment_method:
    type: string
    category: true
  tenure_months:
    type: integer
  num_products:
    type: integer
  monthly_charges:
    type: float
    type_attr:
      precision: 2
  total_charges:
    type: float
    type_attr:
      precision: 2
  has_support:
    type: string
    category: true
  churn:
    type: string
    category: true
```

- `string`與`float`欄位依宣告的型別讀取；`integer`欄位於資料與Schema對齊時轉換型別。
- 資料中存在而Schema未宣告的欄位，由系統自動推斷並加入Schema。
- 未提供Schema時，Loader依資料推斷各欄位的型別、類別標記與精度。

Schema的完整設定方式請參見[Schema YAML](../../../schema-yaml/)。

## 參數說明

JSON讀取器使用下列Loader參數：

| 參數 | 類型 | 預設值 | 說明 | 範例 |
|------|------|--------|------|------|
| `filepath` | `string` | 無 | JSON檔路徑，副檔名`.json`即選用JSON讀取器 | `benchmark://customer-churn` |
| `schema` | `string\|dict` | `null` | 資料結構定義，可為YAML檔路徑或內嵌dict | `customer_churn_schema.yaml` |

## 運作方式

1. **格式判斷**：Loader依`filepath`副檔名判斷格式，`.json`對應JSON讀取器`LoaderPandasJson`
2. **讀取資料**：以`pandas.read_json`讀取檔案，並套用Schema宣告的欄位型別
3. **缺失值**：`null`以`pd.NA`儲存
4. **Schema處理**：依資料推斷Schema，或與指定的Schema合併，再將資料與Schema對齊
5. **後續流程**：所得的資料表與CSV載入的資料相同，交由Splitter、Preprocessor及Synthesizer使用

## 相關說明

- **輸出JSON**：於Reporter設定`type: 'json'`，請參見[儲存資料](../../reporter-yaml/save-data)
- **半結構化資料合成**：AdvGAN合成器可直接合成JSON來源的資料，請參見[AdvGAN合成器](../../synthesizer-yaml/advgan)
- **完整教學**：[JSON資料與AdvGAN合成](../../../getting-started/json-advgan-synthesis)

## 使用要點

- JSON檔的最上層為單層物件的陣列；巢狀物件與JSON Lines依上述方式先行整理
- 每筆紀錄使用相同的鍵；紀錄中未出現的鍵載入為缺失值
- 檔案路徑支援相對路徑與絕對路徑
- 於Windows執行且YAML或Schema檔含非ASCII字元時，以UTF-8模式執行Python(`python -X utf8`)
