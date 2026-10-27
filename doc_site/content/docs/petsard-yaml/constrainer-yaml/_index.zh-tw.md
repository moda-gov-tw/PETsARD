---
title: "Constrainer YAML"
type: docs
weight: 670
prev: docs/petsard-yaml/postprocessor-yaml
next: docs/petsard-yaml/describer-yaml
---

資料約束是一種精細控制合成資料品質和一致性的機制，允許使用者透過多層次的規則定義資料的可接受範圍。`PETsARD` 提供四種主要的約束類型：遺失值群組約束、欄位約束、欄位組合約束和欄位比例約束。這些約束共同確保生成的合成資料不僅在統計特性上忠實於原始資料，更能符合特定的領域邏輯和業務規範。

Constrainer 模組支援兩種運作模式：**反覆抽樣模式**（resample）和**驗證檢查模式**（validate），可根據不同的使用場景自動選擇或手動指定。

## 使用範例

請點擊下方按鈕在 Colab 中執行範例：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moda-gov-tw/PETsARD/blob/main/demo/petsard-yaml/constrainer-yaml/constrainer.ipynb)
> **注意**：使用 Colab 請參考[執行環境設定](/PETsARD/zh-tw/docs/#colab-執行說明)。

### 反覆抽樣模式：內嵌約束配置

```yaml
Loader:
  load_benchmark_with_schema:
    filepath: benchmark://adult-income
    schema: benchmark://adult-income_schema
Synthesizer:
  default:
    method: default
Constrainer:
  inline_field_constraints:
    # 運作模式設定
    method: auto  # 運作模式，預設 'auto'（自動判斷：有 Synthesizer 且非 custom_data → resample）
    # 約束條件（與 constraints_yaml 擇一使用）
    field_constraints:      # 欄位約束條件，預設無
                            # 年齡介於 18 到 65 歲之間
      - "age >= 18 & age <= 65"
    # 抽樣參數（僅 resample 模式使用）
    target_rows: None        # 目標輸出筆數，選用（不設定或設為 None 時預設為輸入資料筆數）
    sampling_ratio: 10.0     # 每次採樣倍數，預設 10.0
    max_trials: 300          # 最大嘗試次數，預設 300
    verbose_step: 10         # 進度輸出間隔，預設 10
```

### 反覆抽樣模式：外部約束檔案（推薦）

**使用外部檔案的優勢**：
- ✅ 更好的可維護性：複雜的約束定義獨立管理
- ✅ 重複使用：同一組約束可在不同實驗中重複使用
- ✅ 版本控制：約束檔案可獨立進行版本管理
- ✅ 清晰的職責分離：主 YAML 專注於流程配置，約束檔案專注於資料規則
#### 約束檔案範例

以下是 [`adult-income_constraints.yaml`](demo/petsard-yaml/constrainer-yaml/adult-income_constraints.yaml:1) 的完整內容，展示了四種約束類型的實際應用：

```yaml
nan_groups:             # 空值處理規則，預設為無
                        # 當 workclass 為空值時刪除整筆資料
  workclass: 'delete'
                        # 當 occupation 為空值時將 income 設為空值
  occupation:
    'erase':
      - 'income'
                        # 當 age 為空值且 educational-num 有值時，將 educational-num 的值複製到 age
  age:
    'copy':
      'educational-num'
field_constraints:      # 欄位約束條件，預設為無
                        # 年齡介於 18 到 65 歲之間
  - "age >= 18 & age <= 65"
                        # 每週工作時數介於 20 到 60 小時之間
  - "hours-per-week >= 20 & hours-per-week <= 60"
field_combinations:     # 欄位值配對關係，預設為無
                        # 博士學歷只能配對高收入
                        # 碩士學歷可配對高收入或低收入
  -
    - education: income
    - Doctorate:
        - '>50K'
      Masters:
        - '>50K'
        - '<=50K'
field_proportions:      # 欄位比例維護，預設為無
                        # 維護教育程度分布，容許 10% 的誤差
  - fields: 'education'
    mode: 'all'
    tolerance: 0.1
                        # 維護收入分布，容許 5% 的誤差
  - fields: 'income'
    mode: 'all'
    tolerance: 0.05
                        # 維護工作類別遺失值比例，容許 3% 的誤差
  - fields: 'workclass'
    mode: 'missing'
    tolerance: 0.03
```

#### 約束條件詳解

##### 1. 遺失值群組約束（nan_groups）

**空值處理規則，預設為無**

- **`workclass: 'delete'`**
  - 🌐 **English**: Delete entire row when workclass is NA
  - 🇹🇼 **繁體中文**：當 `workclass` 欄位為空值時，刪除整筆資料
  - 💡 **說明**：此規則確保所有保留的資料都有完整的工作類別資訊

- **`occupation` 的 `erase` 規則**
  - 🌐 **English**: Set income to NA when occupation is NA
  - 🇹🇼 **繁體中文**：當 `occupation` 欄位為空值時，將 `income` 欄位設為空值
  - 💡 **說明**：建立職業與收入的關聯性，沒有職業資訊時收入資訊也不可靠

- **`age` 的 `copy` 規則**
  - 🌐 **English**: Copy value from educational-num to age when age is NA and educational-num has value
  - 🇹🇼 **繁體中文**：當 `age` 欄位為空值且 `educational-num` 有值時，將 `educational-num` 的值複製到 `age`
  - 💡 **說明**：這是一種補值策略，利用教育年數來估算年齡（僅作示範，實際應用需考慮合理性）

##### 2. 欄位約束（field_constraints）

**欄位約束條件，預設為無**

- **`"age >= 18 & age <= 65"`**
  - 🌐 **English**: Age between 18 and 65
  - 🇹🇼 **繁體中文**：年齡必須介於 18 到 65 歲之間
  - 💡 **說明**：限制資料集只包含工作年齡人口，符合勞動力統計的常見範圍

- **`"hours-per-week >= 20 & hours-per-week <= 60"`**
  - 🌐 **English**: Hours per week between 20 and 60
  - 🇹🇼 **繁體中文**：每週工作時數必須介於 20 到 60 小時之間
  - 💡 **說明**：排除兼職（<20 小時）和極端過勞（>60 小時）的案例，聚焦於標準就業型態

##### 3. 欄位組合約束（field_combinations）

**欄位值配對關係，預設為無**

- **教育與收入的配對規則**
  - 🌐 **English**:
    - Doctorate education can only have >50K income
    - Masters education can have >50K or <=50K income
  - 🇹🇼 **繁體中文**：
    - 博士學歷（`Doctorate`）只能配對高收入（`>50K`）
    - 碩士學歷（`Masters`）可以配對高收入（`>50K`）或低收入（`<=50K`）
  - 💡 **說明**：
    - 反映現實中的教育報酬規律：博士學歷通常對應較高收入
    - 碩士學歷則可能因領域、經驗等因素而有不同收入水準
    - **正面表列**：未列出的教育程度將被視為無效組合

##### 4. 欄位比例約束（field_proportions）

**欄位比例維護，預設為無**

- **教育程度分布維護**
  - 🌐 **English**: Maintain education distribution, 10% tolerance
  - 🇹🇼 **繁體中文**：維護 `education` 欄位的整體分布，容許 10% 的誤差
  - 💡 **說明**：
    - `mode: 'all'`：維護所有類別的比例
    - `tolerance: 0.1`：允許合成資料與原始資料的比例差異在 ±10% 內

- **收入分布維護**
  - 🌐 **English**: Maintain income distribution, 5% tolerance
  - 🇹🇼 **繁體中文**：維護 `income` 欄位的整體分布，容許 5% 的誤差
  - 💡 **說明**：更嚴格的容差（5%）確保收入階層的分布更接近原始資料

- **工作類別遺失值比例維護**
  - 🌐 **English**: Maintain workclass missing value proportion, 3% tolerance
  - 🇹🇼 **繁體中文**：維護 `workclass` 欄位的遺失值比例，容許 3% 的誤差
  - 💡 **說明**：
    - `mode: 'missing'`：僅維護遺失值（NA）的比例
    - `tolerance: 0.03`：嚴格控制遺失值比例，保持資料品質特徵

#### 在主配置中引用約束檔案

```yaml
Loader:
  load_benchmark_with_schema:
    filepath: benchmark://adult-income
    schema: benchmark://adult-income_schema
Synthesizer:
  default:
    method: default
Constrainer:
  external_constraints:
    method: auto
    constraints_yaml: adult-income_constraints.yaml  # 引用外部約束檔案
    target_rows: None
    sampling_ratio: 10.0
    max_trials: 300
```


{{< callout type="warning" >}}
**重要**：不可同時使用 `constraints_yaml` 和個別約束參數。
{{< /callout >}}

### 驗證檢查模式：單一資料來源

驗證單一資料來源的約束符合性。

#### Source 參數說明

`source` 參數用於指定要驗證的資料來源，支援以下格式：

**基本格式**：
- **單一模組**：`source: Loader`（使用該模組的預設輸出）
- **模組.鍵名**：`source: Splitter.ori`（指定模組的特定輸出）

**Splitter 特殊說明**：
- Splitter 有兩個輸出：`ori`（訓練集）和 `control`（驗證集）
- 底層儲存為 `train` 和 `validation`，但使用者可使用熟悉的名稱
- 支援的別名：
  - `Splitter.ori` → `Splitter.train`
  - `Splitter.control` → `Splitter.validation`

```yaml
Splitter:
  external_split:
    method: custom_data
    filepath:
      ori: benchmark://adult-income_ori
      control: benchmark://adult-income_control
    schema:
      ori: benchmark://adult-income_schema
      control: benchmark://adult-income_schema
Synthesizer:
  external_data:
    method: custom_data
    filepath: benchmark://adult-income
    schema: benchmark://adult-income_schema
Constrainer:
  inline_field_constraints:
    method: auto          # 自動選擇 validate 模式
    source: Splitter.ori  # 指定單一資料來源（如果只有一個來源則可選）
    constraints_yaml: adult-income_constraints.yaml
```

### 驗證檢查模式：多個資料來源

同時驗證多個資料來源的約束符合性（使用列表格式）：

```yaml
Loader:
  load_benchmark_with_schema:
    filepath: benchmark://adult-income
    schema: benchmark://adult-income_schema
Splitter:
  external_split:
    method: custom_data
    filepath:
      ori: benchmark://adult-income_ori
      control: benchmark://adult-income_control
    schema:
      ori: benchmark://adult-income_schema
      control: benchmark://adult-income_schema
Synthesizer:
  external_data:
    method: custom_data
    filepath: benchmark://adult-income_syn
    schema: benchmark://adult-income_schema
Constrainer:
  inline_field_constraints:
    method: auto  # 自動選擇 validate 模式
    source:       # 使用列表格式指定多個來源
      - Loader
      - Splitter.ori
      - Splitter.control
      - Synthesizer
    constraints_yaml: adult-income_constraints.yaml
```

## 主要參數

- **method** (`string`, 選用)
  - 運作模式
  - `auto`：根據流程自動判斷（有 Synthesizer 且非 custom_data → resample，其他 → validate）
  - `resample`：反覆抽樣模式
  - `validate`：驗證檢查模式
  - 預設值：`auto`

- **constraints_yaml** (`string`, 選用)
  - 外部約束檔案路徑
  - 與個別約束參數擇一使用

- **約束參數** (選用)
  - `nan_groups`：空值處理規則
  - `field_constraints`：欄位約束條件
  - `field_combinations`：欄位組合規則
  - `field_proportions`：欄位比例維護
  - 與 `constraints_yaml` 擇一使用

- **抽樣參數** (選用，僅 resample 模式使用)
  - `target_rows`：目標輸出筆數
  - `sampling_ratio`：每次採樣倍數（預設 10.0）
  - `max_trials`：最大嘗試次數（預設 300）
  - `verbose_step`：進度輸出間隔（預設 10）

## 運作模式

Constrainer 透過 `method` 參數控制運作模式（預設為 `auto` 自動判斷）：

### 模式選擇決策樹

method='auto' 時：

{{< mermaid-file file="content/docs/petsard-yaml/constrainer-yaml/mode-decision-tree.zh-tw.mmd" >}}

**圖例說明**：
- 🟢 **起點**：流程開始
- 🟠 **決策點**：條件判斷（菱形）
- 🔵 **模式節點**：選定的運作模式
- 🟣 **結果節點**：最終輸出結果

### 反覆抽樣 (Resample) 模式

反覆抽樣直到符合條件

**使用時機**：合成資料生成流程（有 Synthesizer 且非 custom_data）

**運作方式**：

{{< mermaid-file file="content/docs/petsard-yaml/constrainer-yaml/resample-flow.zh-tw.mmd" >}}

**圖例說明**：
- 🔵 **處理節點**：生成資料、套用約束
- 🟠 **決策點**：判斷筆數是否足夠
- 🟢 **結果節點**：最終輸出

**特點**：
- ✅ 自動重採樣直到獲得足夠的符合條件資料
- ✅ 過濾掉不符合的資料
- ✅ 記錄採樣次數
- 💡 可選配置 `target_rows`、`sampling_ratio` 等參數以優化效能

### 驗證檢查 (Validat)e 模式

驗證資料是否符合條件

**使用時機**：
- 使用 Synthesizer 的 `custom_data` 方法（外部資料檔案）
- 沒有 Synthesizer 的流程
- 想要檢查現有資料是否符合條件
- 手動指定 `method='validate'`

**運作方式**：

{{< mermaid-file file="content/docs/petsard-yaml/constrainer-yaml/validate-flow.zh-tw.mmd" >}}

**圖例說明**：
- 🔵 **輸入節點**：讀取資料
- 🟠 **處理節點**：檢查約束、記錄違規
- 🟢 **結果節點**：輸出資料與報告

**特點**：
- ✅ 保留所有資料（不刪除違規資料）
- ✅ 提供詳細的違規統計和記錄
- ✅ 輸出通過率、違規比例等分析
- ✅ 搭配 Reporter 可以輸出驗證報告
- ⚠️ 不使用 `target_rows`、`sampling_ratio` 等抽樣參數（即使設定也會被忽略）

**驗證報告格式**：

使用 `Reporter` 的 `SAVE_VALIDATION` 方法可將驗證結果輸出為 CSV 報告包含：

1. **`{output}_summary.csv`** - 總體統計摘要

| Metric | Value |
|--------|-------|
| total_rows | 1000 |
| passed_rows | 850 |
| failed_rows | 150 |
| pass_rate | 0.850000 |
| is_fully_compliant | False |

2. **`{output}_violations.csv`** - 各條件違規統計

| Constraint Type | Rule | Failed Count | Fail Rate | Violation Examples | Error Message |
|-----------------|------|--------------|-----------|-------------------|---------------|
| field_constraints | age >= 18 & age <= 65 | 80 | 0.080000 | 5, 12, 23 | |
| field_constraints | salary > 30000 | 40 | 0.040000 | 8, 15, 31 | |
| field_combinations | education-income | 30 | 0.030000 | 2, 9, 17 | |

3. **`{output}_details.csv`** - 詳細違規記錄（可選，最多顯示每條規則前 10 筆）

| Constraint Type | Rule | Violation Index | age | salary | education | income |
|-----------------|------|-----------------|-----|--------|-----------|--------|
| field_constraints | age >= 18 & age <= 65 | 1 | 15 | 35000 | HS-grad | <=50K |
| field_constraints | age >= 18 & age <= 65 | 2 | 16 | 42000 | Bachelors | <=50K |
| field_constraints | salary > 30000 | 1 | 25 | 28000 | Masters | <=50K |

## 約束類型

Constrainer 支援四種約束類型，執行順序固定：

```
nan_groups（空值處理）
  ↓
field_constraints（欄位約束）
  ↓
field_combinations（欄位組合）
  ↓
field_proportions（欄位比例）
```

詳細說明請參閱各專屬頁面

## 重要注意事項

{{< callout type="warning" >}}
**約束邏輯與配置規則**

- **AND 邏輯組合**：所有約束條件必須同時滿足，一筆資料須通過所有檢查才會被保留
- **執行順序固定**：`nan_groups` → `field_constraints` → `field_combinations` → `field_proportions`，不可調整
- **正面表列方式**：`field_combinations` 僅影響明確列出的值，未列出的組合將視為無效
- **欄位比例維護**：`field_proportions` 透過迭代移除過量資料同時保護代表性不足的群體
- **空值表示**：必須使用字串 `"pd.NA"`（大小寫敏感），避免使用 `None`、`null` 或 `np.nan`
- **YAML 字串格式**：字串值必須加引號，如 `"HS-grad"` 而非 `HS-grad`
- **配置擇一使用**：`constraints_yaml` 與個別約束參數（`field_constraints` 等）不可同時使用
- **模式限制**：Validate 模式會忽略抽樣參數（`target_rows`、`sampling_ratio` 等）
{{< /callout >}}