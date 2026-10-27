---
title: 以Median、KNN、MICE補償缺失資料
type: docs
weight: 160
prev: docs/getting-started/llm-text-synthesis
next: docs/getting-started
---

以公開或外部參考資料擬合補值策略，補償數值特徵欄位的缺失值，並將補償後資料用於合成流程。

本教學以程式產生一份零售連鎖通路的小型會員資料，每個步驟皆可完整重現。完整參數說明請參見[缺失資料補償(Preprocessor YAML)]({{< ref "/docs/petsard-yaml/preprocessor-yaml/missing-compensation" >}})。

{{< callout type="info" >}}
請先依[擴充功能安裝]({{< ref "/docs/installation/extended-features" >}})完成環境設定，並於已啟動的虛擬環境中執行本頁指令。本教學所有檔案皆建立於工作目錄。
{{< /callout >}}

## 運作方式

1. **參考資料**(`reference_members.csv`)：具相同數值特徵的公開或外部資料，補值策略以此資料擬合一次。
2. **本地資料**(`local_members.csv`)：需補償的資料，由Loader讀取，並以擬合完成的策略補償其特徵缺失格。
3. **補償後資料**：筆數、欄位及資料型別皆與本地資料相同，可直接用於合成或其他PETsARD流程。

## 步驟1：準備範例資料

將下列程式儲存為`make_members.py`：

```python
import numpy as np
import pandas as pd


def make_members(n, id_start, seed):
    rng = np.random.default_rng(seed)
    income = np.round(rng.normal(650, 180, n).clip(200, None), 1)
    spend = np.round(income * 0.004 + rng.normal(0, 0.4, n) + 1.0, 2)
    visits = np.round(spend * 2 + rng.normal(0, 1.0, n)).clip(0, None)
    return pd.DataFrame(
        {
            "member_id": np.arange(id_start, id_start + n),
            "region": rng.choice(["north", "central", "south", "east"], n),
            "age": rng.integers(20, 70, n),
            "tenure_months": rng.integers(1, 120, n),
            "annual_income": income,
            "monthly_spend": spend,
            "store_visits": visits,
            "churned": (rng.random(n) < 0.2).astype(int),
        }
    )


reference = make_members(300, 100001, seed=7)
local = make_members(120, 200001, seed=11)

rng = np.random.default_rng(2026)
for column, rate in {"annual_income": 0.15, "monthly_spend": 0.10, "store_visits": 0.10}.items():
    local.loc[rng.random(len(local)) < rate, column] = np.nan

reference.to_csv("reference_members.csv", index=False)
local.to_csv("local_members.csv", index=False)
print(local.isna().sum())
```

執行：

```bash
python make_members.py
```

程式產生兩個檔案：

| 檔案 | 筆數 | 會員編號 | 缺失格 |
|---|---|---|---|
| `reference_members.csv` | 300 | 100001至100300 | 無 |
| `local_members.csv` | 120 | 200001至200120 | `annual_income`12格、`monthly_spend`7格、`store_visits`11格(合計30格) |

兩個檔案的欄位相同：

| 欄位 | 意義 | 於補償中的角色 |
|---|---|---|
| `member_id` | 會員編號 | 紀錄識別碼(`key_columns`) |
| `region` | 門市區域 | 保留原值(`protected_columns`) |
| `age` | 年齡(歲) | 特徵 |
| `tenure_months` | 入會月數 | 特徵 |
| `annual_income` | 年收入(千元) | 特徵 |
| `monthly_spend` | 平均每月消費(千元) | 特徵 |
| `store_visits` | 每月到店次數 | 特徵 |
| `churned` | 是否退出會員方案(0或1) | 標籤(`target_columns`) |

## 步驟2：撰寫YAML

將下列設定儲存為`missing-compensation.yaml`：

```yaml
Loader:
  local_members:
    filepath: local_members.csv

Preprocessor:
  compensation:
    method: custom_method
    module_path: petsard/processor/missing_compensation.py
    class_name: MissingCompensation
    missing_compensation:
      method: median
    feature_columns: [age, tenure_months, annual_income, monthly_spend, store_visits]
    key_columns: [member_id]
    target_columns: [churned]
    protected_columns: [region]
    fit_data:
      filepath: reference_members.csv
    fit_source:
      id: reference-members-2026
      role: public
    transform_source: local-members-2026

Reporter:
  save_compensated:
    method: save_data
    source: Preprocessor
```

### YAML參數詳細說明

#### Loader(資料讀取模組)

- **`local_members`**：實驗名稱，可自由命名。
- **`filepath`**：需補償的本地資料。

#### Preprocessor(資料前處理模組)

- **`compensation`**：實驗名稱，步驟4以此名稱讀取補償紀錄。
- **`method: custom_method`**、**`module_path`**及**`class_name: MissingCompensation`**：載入缺失資料補償模組。
- **`missing_compensation.method`**：補償方法，為`median`、`knn`或`mice`。
- **`feature_columns`**：補值策略建模及補值的數值欄位，依固定順序列出。
- **`key_columns`**：紀錄識別碼，兩個檔案的會員編號互不重複，符合本模組設計。
- **`target_columns`**：標籤，保留原值。
- **`protected_columns`**：其他保留原值的欄位。
- **`fit_data.filepath`**：擬合補值策略的參考資料。
- **`fit_source`**：參考資料的宣告，含識別字串(`id`)及性質(`public`或`external`)。
- **`transform_source`**：本地資料的識別字串，與`fit_source.id`不同。

#### Reporter(輸出模組)

- **`save_compensated`**：實驗名稱。
- **`method: save_data`**搭配**`source: Preprocessor`**：將補償後資料儲存為CSV檔案。

## 步驟3：執行

將下列程式儲存為`run_compensation.py`：

```python
from petsard import Executor

executor = Executor(config="missing-compensation.yaml")
executor.run()

artifacts = executor.get_preprocessor_artifacts("compensation")
print(artifacts["reports"]["data"])
print(artifacts["strategy_states"]["data"])
```

執行：

```bash
python run_compensation.py
```

## 步驟4：檢視結果

### 補償後檔案

Reporter輸出`petsard_Loader[local_members]_Preprocessor[compensation].csv`，與`local_members.csv`同為120筆、8個欄位，30個特徵缺失格皆已補值。使用Median方法時，每個補值為該特徵於參考資料的中位數：`annual_income`為636.85，`monthly_spend`為3.48，`store_visits`為7.0。

下列程式確認所有缺失格皆已補值，且所有觀測值維持原值：

```python
import pandas as pd

local = pd.read_csv("local_members.csv")
result = pd.read_csv("petsard_Loader[local_members]_Preprocessor[compensation].csv")

print(local.isna().sum().sum(), result.isna().sum().sum())  # 30 0
observed = local.notna()
print(local[observed].equals(result[observed]))              # True
```

### 補償報告

`artifacts["reports"]["data"]`記錄本次執行：

```json
{
  "method": "median",
  "fit_source": {"id": "reference-members-2026", "role": "public"},
  "fit_rows": 300,
  "feature_columns": ["age", "tenure_months", "annual_income", "monthly_spend", "store_visits"],
  "parameters": {"strategy": "median"},
  "fit_count": 1,
  "transform_source": "local-members-2026",
  "masked_cells_filled": 30,
  "outside_mask_changes": 0,
  "protected_changes": 0,
  "label_changes": 0,
  "transform_role": "local_masked"
}
```

- `fit_rows: 300`及`fit_source`表示補值策略以參考資料擬合。
- `masked_cells_filled: 30`與本地資料的特徵缺失格數相符。
- `outside_mask_changes`、`protected_changes`及`label_changes`皆為`0`，表示觀測值、`member_id`、`region`及`churned`皆維持原值。

### 策略識別資訊

`artifacts["strategy_states"]["data"]`識別擬合完成的策略：

```json
{
  "method": "median",
  "frozen": true,
  "fit_count": 1,
  "fit_source": {"id": "reference-members-2026", "role": "public"},
  "fingerprint": "880937a2601621048571180b8e74887f401ce76f5e818d440e81ab6513b88a7a"
}
```

`fingerprint`為擬合策略的SHA-256摘要。相同參考資料與設定產生相同指紋，可據以確認補償後檔案所使用的策略。

## 步驟5：改用KNN或MICE

修改`missing-compensation.yaml`中的一行，再執行一次步驟3：

{{< tabs items="KNN,MICE" >}}
{{< tab >}}
```yaml
    missing_compensation:
      method: knn
```

報告列出KNN固定參數：

```json
"parameters": {"n_neighbors": 5, "weights": "uniform", "metric": "nan_euclidean"}
```
{{< /tab >}}
{{< tab >}}
```yaml
    missing_compensation:
      method: mice
```

報告列出MICE固定參數、策略指紋及`mice_qa`驗證計數，各計數皆為`0`：

```json
"parameters": {
  "estimator": "RandomForestRegressor",
  "n_estimators": 50,
  "max_depth": 10,
  "n_jobs": -1,
  "estimator_random_state": 42,
  "max_iter": 3,
  "random_state": 42
},
"mice_qa": {
  "remaining_missing": 0,
  "non_finite": 0,
  "unexpected_negative": 0,
  "new_masked_public_range_violations": 0,
  "binary_violations": 0,
  "integer_violations": 0,
  "summary_order_violations": 0
}
```
{{< /tab >}}
{{< /tabs >}}

三種方法補償相同的30個格位。Median每個特徵使用單一值，KNN與MICE則依據每位會員的其他特徵推估，因此各列補值各有不同。本範例前三個`annual_income`缺失格補值如下(四捨五入至小數第二位)：

| `member_id` | Median | KNN | MICE |
|---|---|---|---|
| 200024 | 636.85 | 571.30 | 508.87 |
| 200059 | 636.85 | 631.96 | 315.29 |
| 200064 | 636.85 | 690.98 | 758.10 |

{{< callout type="info" >}}
每次執行皆輸出步驟4的檔名。如需並列保存多種方法的結果，可為每次執行設定不同的Preprocessor實驗名稱，例如`compensation_knn`，並以`executor.get_preprocessor_artifacts("compensation_knn")`讀取其紀錄。
{{< /callout >}}

## 步驟6：合成補償後資料

補償後檔案為一般CSV檔案，可作為任何PETsARD流程輸入。將下列設定儲存為`synthesis.yaml`：

```yaml
Loader:
  compensated:
    filepath: "petsard_Loader[local_members]_Preprocessor[compensation].csv"

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
    method: 'save_data'
    source: 'Postprocessor'
```

以相同的三行Python程式執行：

```python
from petsard import Executor

executor = Executor(config="synthesis.yaml")
executor.run()
```

預設Preprocessor依預設處理序列處理資料，包括將`region`編碼及將數值欄位標準化；Synthesizer以PETsARD Gaussian Copula產生合成資料；Postprocessor再將資料還原為原始格式。合成資料儲存為`petsard_Loader[compensated]_Preprocessor[default]_Synthesizer[default]_Postprocessor[default].csv`。加入評測的方式請參見[以預設參數做資料合成與資料評測]({{< ref "/docs/getting-started/default-synthesis-default-evaluation" >}})。

## 執行流程

1. **Loader**讀取`local_members.csv`。
2. **Preprocessor**讀取`reference_members.csv`，擬合Median、KNN或MICE補值策略一次，並補償本地資料中`feature_columns`缺失格。
3. **Reporter**儲存補償後資料。
4. 第二個流程讀取補償後檔案，並以預設參數合成資料。

## 進階使用

- **Splitter**：Preprocessor之前設定Splitter時，同一組擬合完成的策略補償`train`與`validation`兩個分支，請參見[搭配Splitter]({{< ref "/docs/petsard-yaml/preprocessor-yaml/missing-compensation#搭配splitter" >}})。
- **串接**：`method: composite`於同一個實驗中依序執行內建離群值處理、補償及內建遺失值處理，例如於補償後補齊類別欄位，請參見[與其他前處理步驟串接]({{< ref "/docs/petsard-yaml/preprocessor-yaml/missing-compensation#與其他前處理步驟串接" >}})。
- **明確遮罩**：以Python API依布林遮罩僅補償指定的格位，請參見[Python API]({{< ref "/docs/petsard-yaml/preprocessor-yaml/missing-compensation#python-api" >}})。
- **使用自有資料**：替換兩個檔案路徑，於`feature_columns`列出資料的數值欄位，並將識別碼、標籤及其他保留原值的欄位分別列於`key_columns`、`target_columns`及`protected_columns`。
