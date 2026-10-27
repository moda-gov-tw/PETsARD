---
title: "缺失資料補償"
type: docs
weight: 646
prev: docs/petsard-yaml/preprocessor-yaml/discretizing
next: docs/petsard-yaml/preprocessor-yaml
---

以公開或外部參考資料擬合Median、KNN或MICE補值策略，再將擬合完成的策略套用於本地資料，補償數值特徵欄位的缺失值。

## 背景

PETsARD內建的[遺失值處理]({{< ref "/docs/petsard-yaml/preprocessor-yaml/missing-handling" >}})以所處理的資料本身計算統計值；缺失資料補償模組則以另一份參考資料(例如公開資料集，或具相同特徵的外部資料)學習補值策略，再將擬合完成的策略套用於本地資料。本地資料僅進行轉換，補值策略完全由參考資料學得。

- **外部擬合**：補值策略以宣告為`public`或`external`的參考資料擬合一次。
- **凍結策略**：同一組擬合完成的策略轉換所有資料集及Splitter的各分支，並以SHA-256指紋識別。
- **精準補值**：僅寫入`feature_columns`所列欄位的缺失格；觀測值、識別碼、標籤及其他受保護欄位皆完整保留原值。
- **結構保持**：輸出資料保持輸入資料的筆數、欄位順序、索引及資料型別。
- **明確啟用**：Preprocessor實驗設定`method: custom_method`及`class_name: MissingCompensation`時才執行本模組，`default`與`custom`前處理方式維持原有行為。

本模組為`petsard/processor/missing_compensation.py`，MICE後處理位於`petsard/processor/mice_postprocess.py`，兩者皆隨PETsARD原始碼安裝，安裝方式請參見[擴充功能安裝]({{< ref "/docs/installation/extended-features" >}})。以範例資料逐步操作的說明，參見[以Median、KNN、MICE補償缺失資料]({{< ref "/docs/getting-started/missing-compensation" >}})。

## 使用範例

以下範例使用一份會員資料，含識別碼`member_id`、類別欄位`region`、五個數值特徵及標籤`churned`。`reference_members.csv`為參考資料，`local_members.csv`為需補償的本地資料。

### 範例1：Median補償

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

### 範例2：KNN補償

僅需修改`missing_compensation`區塊。三個KNN參數為選填，填寫時即為下列固定參數值。

```yaml
    missing_compensation:
      method: knn
      n_neighbors: 5
      weights: uniform
      metric: nan_euclidean
```

### 範例3：MICE補償

```yaml
    missing_compensation:
      method: mice
```

### 範例4：搭配Splitter

Preprocessor之前設定Splitter時，補值策略擬合一次，並以同一組擬合完成的策略轉換`train`與`validation`兩個分支。

```yaml
Loader:
  local_members:
    filepath: local_members.csv

Splitter:
  holdout:
    num_samples: 1
    train_split_ratio: 0.8

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
```

### 範例5：串接離群值處理與遺失值處理

`method: composite`於同一個Preprocessor實驗中依序執行內建離群值處理、補償及內建遺失值處理，說明請參見[與其他前處理步驟串接](#與其他前處理步驟串接)。

```yaml
Preprocessor:
  compensation_chain:
    method: composite
    stages:
      - name: outlier
        method: custom
        sequence: [outlier]
        config:
          outlier:
            member_id: null
            region: null
            age: null
            tenure_months: null
            annual_income: outlier_iqr
            monthly_spend: null
            store_visits: null
            churned: null
      - name: compensation
        method: custom_method
        module_path: petsard/processor/missing_compensation.py
        class_name: MissingCompensation
        missing_compensation:
          method: mice
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
      - name: missing
        method: custom
        sequence: [missing]
        config:
          missing:
            member_id: null
            region: missing_mode
            age: null
            tenure_months: null
            annual_income: null
            monthly_spend: null
            store_visits: null
            churned: null
```

## 參數說明

| 參數 | 型別 | 必要 | 說明 |
|---|---|---|---|
| `method` | `string` | 是 | `custom_method` |
| `module_path` | `string` | 是 | `petsard/processor/missing_compensation.py`。相對路徑依工作目錄及Python路徑的各項目解析，以可編輯模式安裝PETsARD時，於任何工作目錄皆可使用此值；亦可填寫絕對路徑 |
| `class_name` | `string` | 是 | `MissingCompensation` |
| `missing_compensation` | `dict` | 是 | 補償方法。`method`為`median`、`knn`或`mice`。使用`knn`時，可選填`n_neighbors`(`5`)、`weights`(`uniform`)及`metric`(`nan_euclidean`)，其值為括號內的固定值；使用`mice`時，區塊僅含`method` |
| `feature_columns` | `list` | 是 | 補值策略建模及補值的數值欄位，依序列出且不重複。各欄位同時存在於參考資料與本地資料，資料型別為數值(非布林) |
| `key_columns` | `list`或`string` | 否 | 紀錄識別碼。列入者受保護，且參考資料與本地資料的值互不重複。預設：無 |
| `row_id_column` | `string`或`list` | 否 | 列識別碼，處理方式與`key_columns`相同。預設：無 |
| `target_columns` | `list`或`string` | 否 | 標籤欄位。列入者受保護並保留原值。預設：無 |
| `protected_columns` | `list`或`string` | 否 | 其他保留原值的欄位。預設：無 |
| `fit_data` | `dict` | 是 | 參考資料的Loader設定。`filepath`為必要，亦可加入`schema`等其他Loader參數 |
| `fit_source` | `dict` | 是 | 參考資料的宣告：`id`為非空白的識別字串，`role`為`public`或`external` |
| `transform_source` | `string` | 是 | 本地資料的非空白識別字串，與`fit_source.id`不同；填寫檔案路徑時，指向`fit_data.filepath`以外的檔案 |

`key_columns`、`row_id_column`、`target_columns`及`protected_columns`所列欄位，皆列於`feature_columns`之外。

## 補償方法

| `method` | 補值器(scikit-learn) | 固定參數 | 擬合內容 |
|---|---|---|---|
| `median` | `SimpleImputer` | `strategy="median"` | 參考資料各特徵的中位數 |
| `knn` | `KNNImputer` | `n_neighbors=5`、`weights="uniform"`、`metric="nan_euclidean"` | 參考資料的特徵矩陣；缺失格填入最近5筆參考資料的平均值 |
| `mice` | `IterativeImputer`搭配`RandomForestRegressor` | 估計器：`n_estimators=50`、`max_depth=10`、`random_state=42`、`n_jobs=-1`；補值器：`max_iter=3`、`random_state=42` | 以其他特徵預測各特徵的鏈式隨機森林模型 |

各方法的參數固定，相同資料重複執行可得到相同結果。MICE的隨機森林使用全部CPU核心(`n_jobs=-1`)。

## 處理邏輯

### 擬合(每組策略一次)

1. 以PETsARD Loader依`fit_data`讀取參考資料。
2. 依列出順序讀取`feature_columns`；各特徵皆為有限數值，且於參考資料中至少有一個觀測值。
3. 記錄參考資料的識別碼值，供資料區隔檢查使用(見[識別碼、標籤與受保護欄位](#識別碼標籤與受保護欄位))。
4. 擬合所選方法的補值器，凍結後以SHA-256指紋識別。

流程傳入的本地資料僅用於資料區隔檢查。

### 轉換

1. 本地資料中`feature_columns`的缺失格構成補償遮罩；其他欄位的缺失格維持原狀，供後續步驟處理。
2. 擬合完成的補值器產生候選值，僅寫入遮罩內的格位，觀測格完整保留原值。
3. 每個補值採用該欄位的資料型別。整數型別欄位填入整數值；預期補值含小數時，將該欄位設為浮點數型別。
4. 驗證結果：資料型別不變、遮罩外的每一格不變，且遮罩內每一格皆已填值。
5. 更新報告，記錄補值格數及各項驗證計數。

### MICE後處理與驗證

MICE方法於寫回前，依欄位名稱字尾對預測值套用下列規則：

- 以`_min`與`_max`為字尾且基底名稱相同的欄位(例如`price_min`與`price_max`)，排序兩值使最小值不大於最大值；同一基底以`_mean`、`_first`、`_last`為字尾的欄位，限制於該範圍內。
- 以`_n`結尾的欄位表示計數，四捨五入為非負整數。
- 名為`icu_stay_24h`之欄位為二元指標，限制於0至1並四捨五入為0或1。

寫回後，MICE驗證補償結果，並將計數記錄於`mice_qa`；執行完成時各計數皆為`0`：

| 鍵 | 計數內容 |
|---|---|
| `remaining_missing` | `feature_columns`中剩餘的缺失格 |
| `non_finite` | `feature_columns`中的非有限值 |
| `unexpected_negative` | 參考資料全為非負值的特徵中出現的負值 |
| `new_masked_public_range_violations` | 補值超出該特徵參考資料範圍`0.001`以上的格數 |
| `binary_violations` | 二元指標欄位`icu_stay_24h`中`{0, 1}`以外的值 |
| `integer_violations` | `_n`欄位中的非整數值 |
| `summary_order_violations` | `_min`大於`_max`，或`_mean`、`_first`、`_last`落於`_min`至`_max`範圍外的列數 |

範圍檢查適用於新補值的格位，本地資料中原有的觀測值即使超出參考資料範圍亦完整保留。名稱含`__missing`或名為`vitals_missing`的欄位為缺失指標，列於MICE的`feature_columns`以外。

## 識別碼、標籤與受保護欄位

- **識別碼名稱**：名稱為`id`、`record`、`subject`、`record_identifier`或`subject_identifier`的欄位、以`_id`結尾的欄位，以及以`y_`開頭的欄位(不分大小寫)，皆視為識別碼或標籤，列於`feature_columns`以外。
- **兩份資料的區隔**：參考資料與本地資料為兩份不同的資料集。識別碼欄位(`key_columns`、`row_id_column`及標籤以外的識別碼名稱欄位)的值於兩份資料間互不重複，且本地資料的特徵表與參考資料的特徵表不同。
- **共用代碼的命名**：`_id`字尾專用於紀錄識別碼；兩份資料共有的類別代碼(例如門市代碼)，建議命名為`store_code`等形式。
- **標籤**：`target_columns`及`y_*`欄位的值為受保護的值，兩份資料可含相同的標籤值。

## 搭配Splitter

Preprocessor之前設定Splitter時，補償步驟接收Splitter的所有分支：

- 補值策略擬合一次，並以同一個擬合完成的實例轉換`train`與`validation`。
- 各分支回報相同的指紋及`fit_count: 1`，PETsARD並確認各分支的策略一致。
- Synthesizer接收補償後的`train`分支。Evaluator以補償後的`train`分支為原始資料、補償後的`validation`分支為控制資料，兩者採用同一組策略。

## 與其他前處理步驟串接

`method: composite`於單一Preprocessor實驗中，將PETsARD內建步驟與補償步驟組成依序執行的處理鏈。支援的順序如下：

```text
custom_method
outlier -> custom_method
custom_method -> missing
outlier -> custom_method -> missing
```

- `stages`列出一至三個步驟，每個步驟具唯一的`name`。
- 補償步驟使用`method: custom_method`及本頁的參數。
- 內建步驟使用`method: custom`並以逐欄位的`config`設定，`sequence`於補償步驟之前為`[outlier]`，於補償步驟之後為`[missing]`。
- 內建步驟中列出所有欄位，並將該步驟不處理的欄位設為`null`；未列出的欄位使用其資料型別的預設處理器。
- 離群值步驟於補償前處理離群值；遺失值步驟處理補償模組未建模的欄位，例如類別欄位。
- 每個步驟以前一步驟的輸出擬合。搭配Splitter時，整條處理鏈以`train`擬合一次，再以同一組擬合完成的步驟轉換`validation`。
- 補償步驟的`key_columns`、`target_columns`、`row_id_column`及選填的`required_columns`所列欄位，於內建步驟的常數欄位處理中保留；未設定`target_columns`時，訓練資料中含缺失值的欄位亦一併保留。
- 補償步驟保持其輸入資料的形狀、欄位順序及索引。

不同Preprocessor實驗名稱各自為獨立分支，串接步驟皆置於同一個`method: composite`實驗中。

## 輸出

### 補償後資料

`custom_method`實驗結果為DataFrame，其列、欄、順序、索引及資料型別皆與輸入相同，遮罩內的特徵格已完成補值；`composite`實驗中，補償步驟前後的內建步驟依其設定處理資料；搭配Splitter時，結果含`train`與`validation`兩個分支。Reporter設定`method: save_data`及`source: Preprocessor`即可輸出資料，例如`petsard_Loader[local_members]_Preprocessor[compensation].csv`。輸出的檔案可作為合成流程的Loader輸入，並以預設Preprocessor將`region`等類別欄位編碼後交由合成器使用，操作方式請參見[快速入門]({{< ref "/docs/getting-started/missing-compensation" >}})。

### 處理紀錄

`Executor.get_preprocessor_artifacts()`回傳Preprocessor實驗的補償紀錄：

```python
from petsard import Executor

executor = Executor(config="missing-compensation.yaml")
executor.run()

artifacts = executor.get_preprocessor_artifacts("compensation")
report = artifacts["reports"]["data"]
strategy = artifacts["strategy_states"]["data"]
```

| 鍵 | 內容 |
|---|---|
| `processor_type` | `custom_method`；串接實驗為`composite` |
| `fit_count` | 擬合次數，為`1` |
| `transform_count` | 已轉換的資料集或分支數 |
| `reports` | 各資料集或分支的補償報告 |
| `strategy_states` | 各資料集或分支的策略識別資訊 |
| `stages`、`stage_runs` | 僅串接實驗：步驟順序，以及每次步驟執行的形狀與欄位 |

`reports`與`strategy_states`鍵，於單一`custom_method`實驗為`data`，於`composite`實驗為`train`；Preprocessor之前設定Splitter時為`train`與`validation`。

### 報告欄位

| 欄位 | 內容 |
|---|---|
| `method` | `median`、`knn`或`mice` |
| `fit_source` | 參考資料的`id`與`role` |
| `fit_rows` | 參考資料筆數 |
| `feature_columns` | 依序列出的特徵 |
| `parameters` | 實際使用的補值器參數 |
| `fit_count` | `1` |
| `strategy_fingerprint` | 僅MICE：策略指紋 |
| `transform_source` | 本地資料的識別字串 |
| `masked_cells_filled` | 補值格數 |
| `outside_mask_changes` | 遮罩外變動格數，為`0` |
| `protected_changes` | 受保護欄位變動格數，為`0` |
| `label_changes` | 標籤變動格數，為`0` |
| `transform_role` | `local_masked` |
| `mice_qa` | 僅MICE：前述驗證計數 |

### 策略識別欄位

| 欄位 | 內容 |
|---|---|
| `method` | `median`、`knn`或`mice` |
| `frozen` | `true` |
| `fit_count` | `1` |
| `fit_source` | 參考資料的`id`與`role` |
| `fingerprint` | 擬合策略的SHA-256摘要 |

Median指紋涵蓋擬合的中位數，KNN指紋涵蓋參考資料的特徵矩陣，兩者並涵蓋方法、參數、特徵順序及其資料型別、參考資料識別及參考資料筆數；MICE指紋涵蓋特徵順序、資料型別及參考資料的特徵值。對外僅公開摘要值，策略識別資訊不含任何資料列的值。

## Python API

亦可直接使用本類別，並以明確的遮罩指定補值格位：

```python
import pandas as pd
from petsard.processor.missing_compensation import MissingCompensation

reference = pd.read_csv("reference_members.csv")
local = pd.read_csv("local_members.csv")

config = {
    "missing_compensation": {"method": "knn"},
    "feature_columns": ["age", "tenure_months", "annual_income", "monthly_spend", "store_visits"],
    "key_columns": ["member_id"],
    "target_columns": ["churned"],
    "protected_columns": ["region"],
    "fit_source": {"id": "reference-members-2026", "role": "public"},
    "transform_source": "local-members-2026",
}

# 僅補償annual_income的缺失格
mask = pd.DataFrame(False, index=local.index, columns=local.columns)
mask["annual_income"] = local["annual_income"].isna()

compensation = MissingCompensation(config).fit(fit_data=reference)
result = compensation.transform(transform_data=local, mask=mask)
print(compensation.get_report())
print(compensation.get_strategy_state())
```

| 方法 | 說明 |
|---|---|
| `MissingCompensation(config)` | 建立補償器。`config`使用本頁的YAML參數，`method`、`module_path`及`class_name`除外；由`fit()`自行讀取參考資料檔案時，才需設定`fit_data` |
| `fit(fit_data=reference)` | 以參考資料DataFrame擬合策略並回傳實例，每個實例擬合一次 |
| `transform(transform_data=local, mask=mask)` | 回傳補償後的DataFrame。選填的`mask`為與`local`具相同索引、欄位及欄位順序的布林DataFrame，`True`的格位已為缺失值且位於`feature_columns`；未提供遮罩時，補償`feature_columns`的所有缺失格 |
| `get_report()` | 回傳報告，含擬合資訊及最近一次轉換的結果 |
| `get_strategy_state()` | 回傳策略識別資訊 |

同一個擬合完成的實例可轉換多份資料，例如訓練集、驗證集及保留集，各份資料共用相同的指紋。

## 訊息對照

| 訊息 | 處理方式 |
|---|---|
| `missing_compensation.method must be median, knn or mice` | 將`missing_compensation.method`設為`median`、`knn`或`mice` |
| `Unsupported missing compensation option` | `n_neighbors`、`weights`及`metric`僅用於`knn`，`mice`僅設定`method` |
| `feature_columns must be an explicit unique list` | 每個特徵欄位列出一次 |
| `Identifiers and y_* labels cannot be imputer features` | 將識別碼、標籤及受保護欄位自`feature_columns`移至`key_columns`、`target_columns`或`protected_columns` |
| `Distinct fit_source.id (public/external) and transform_source required` | `fit_source.id`與`transform_source`填入不同的非空白值，並將`fit_source.role`設為`public`或`external` |
| `Explicit fit_data or external fit_data.filepath required` | 於`fit_data.filepath`填入參考資料檔案路徑 |
| `Fit and transform paths must be distinct` | `fit_data.filepath`與`transform_source`指向不同檔案 |
| `Compensation features must have numeric dtypes` | `feature_columns`僅列數值欄位，或於schema將該欄位設為數值型別 |
| `External fit features cannot be entirely missing` | 參考資料的每個特徵至少提供一個觀測值 |
| `Fit/transform identity overlap: <column>` | 兩份資料使用互不重複的識別碼值，共用代碼則改用不以`_id`結尾名稱 |
| `Transform data duplicates external fit features` | 本地資料使用與參考資料不同的資料集 |
| `Fractional compensation cannot preserve integer dtype: <column>` | 將該欄位設為浮點數型別 |
| `Mask must be a complete boolean table aligned to input` | 以`local`的索引與欄位建立遮罩，並以`True`、`False`填滿 |
| `Compensation is frozen after fit; create a new instance` | 建立新的`MissingCompensation`實例以擬合新策略 |

## 使用說明

- 參考資料採用與本地資料相互獨立的資料，例如公開資料集或具相同特徵的外部資料。`fit_source.role`依參考資料的實際性質設定，每份報告及策略識別資訊皆記錄此值。
- `feature_columns`明確界定補償的欄位。清單以外的欄位保留原值及其缺失格，可交由後續`missing`步驟處理。
- 補償後資料保持輸入資料的資料型別，後續模組接收的結構與未補償時相同。
- 固定參數搭配凍結策略，使每次執行皆可重現：相同的參考資料與本地資料，產生相同補值結果及相同指紋。
