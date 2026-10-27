---
title: "文字資料評測"
type: docs
weight: 697
prev: docs/petsard-yaml/evaluator-yaml/privacy-mpuccs
next: docs/petsard-yaml/reporter-yaml
---

文字評測模組以PETsARD標準的評測方法評測合成的自由文字，例如[LLM文字合成器]({{< ref "/docs/petsard-yaml/synthesizer-yaml/llm-synthesizer" >}})的輸出。PETsARD評測方法以欄位為單位計算，`secret`、`target`、`aux_cols`及各項評分皆對應欄位名稱，因此本模組先以使用者提供的解析程式將文字轉為具型別的欄位，再對解析後的資料執行任意數量的PETsARD評測方法，並將結果合併為一份報告。

本模組以標準`custom_method`介面自`petsard/evaluator/text_evaluator.py`載入，不需修改PETsARD核心程式。

## 評測類別

PETsARD依評測類別取得所需的輸入資料，因此本模組提供兩個類別，依巢狀設定的評測方法選擇：

| 類別 | 所需資料 | 巢狀評測方法 |
|---|---|---|
| `TextEvaluator` | 原始資料(`ori`)與合成資料(`syn`) | `sdmetrics-diagnosticreport`、`sdmetrics-qualityreport`、`mpuccs` |
| `TextEvaluatorWithControl` | `ori`、`syn`及控制資料(`control`)，YAML包含Splitter | `anonymeter-singlingout`、`anonymeter-linkability`、`anonymeter-inference`、`mlutility` |

控制資料為未參與合成的原始資料。Anonymeter比較對訓練資料與對控制資料的相同攻擊結果；MLUtility分別以`ori`與`syn`訓練模型，再以`control`測試兩者。

## 使用範例

以下範例評測以基準資料集`bloodpressure-text`生成的合成血壓問診文字(見[LLM文字合成器]({{< ref "/docs/petsard-yaml/synthesizer-yaml/llm-synthesizer" >}}))，解析程式使用原始碼所附之`petsard/synthesizer/LLM_synth/utils/bloodpressure_text_parser.py`，其規則請參見下方「撰寫文字解析程式」一節。範例於PETsARD原始碼根目錄執行。

### 有效性與保真度

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
    model: openai/gpt-oss-20b
    num_procs: 4
Evaluator:
  text_eval:
    method: custom_method
    module_path: petsard/evaluator/text_evaluator.py
    class_name: TextEvaluator
    text_parser: petsard/synthesizer/LLM_synth/utils/bloodpressure_text_parser.py   # 相對於執行目錄
    output_dir: ./llm_synth_output/parsed   # 解析結果，供核對使用
    Evaluator:                              # 巢狀的PETsARD評測方法
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

### 隱私風險與實用性

需要控制資料的評測方法，使用`TextEvaluatorWithControl`並搭配Splitter：

```yaml
Loader:
  load_bloodpressure:
    filepath: benchmark://bloodpressure-text
Splitter:
  holdout:
    num_samples: 1
    train_split_ratio: 0.8
Synthesizer:
  llm-synth:
    method: custom_method
    module_path: petsard/synthesizer/LLM_synthesizer.py
    class_name: LLM_synthesizer
    sample_num_rows: 80          # 與訓練集筆數相當
    epochs: 1
    api_type: lmstudio
    api_url: "http://localhost:1234/v1"
    model: openai/gpt-oss-20b
    num_procs: 4
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
      linkability_risk:
        method: anonymeter-linkability
        aux_cols:
          - [gender, age]
          - [systolic, diastolic]
      inference_risk:
        method: anonymeter-inference
        secret: bp_label
      classification_utility:
        method: mlutility
        task_type: classification
        target: bp_label
Reporter:
  save_eval:
    method: save_report
    granularity:
      - global
    output: llm_text_privacy
```

巢狀評測方法中，`method`以外的各鍵皆原樣傳給該評測方法，參數寫法與[隱私保護力評測]({{< ref "/docs/petsard-yaml/evaluator-yaml/privacy" >}})、[資料保真度評測]({{< ref "/docs/petsard-yaml/evaluator-yaml/fidelity" >}})及[資料實用性評測]({{< ref "/docs/petsard-yaml/evaluator-yaml/utility" >}})相同。`aux_cols`以兩個清單組成的清單撰寫時，自動轉為Anonymeter所需的清單組。

## 參數說明

| 參數 | 類型 | 必要性 | 說明 |
|---|---|---|---|
| `method` | `string` | 必要 | 固定值：`custom_method` |
| `module_path` | `string` | 必要 | `petsard/evaluator/text_evaluator.py`，由執行目錄或Python路徑(可編輯模式安裝後即原始碼根目錄)解析 |
| `class_name` | `string` | 必要 | `TextEvaluator`或`TextEvaluatorWithControl` |
| `text_parser` | `string` | 必要 | 解析程式`.py`檔的路徑，相對於執行目錄 |
| `text_parser_entry` | `string` | 選用 | 解析函式名稱，預設為`text_parser` |
| `Evaluator` | `dict` | 必要 | 巢狀的PETsARD評測方法區塊，寫法與最外層的Evaluator區塊相同：`{名稱: {method: ..., 其他參數}}` |
| `output_dir` | `string` | 選用 | 解析結果的輸出目錄，輸出`parsed_ori.csv`、`parsed_syn.csv`，使用`TextEvaluatorWithControl`時另輸出`parsed_control.csv`(編碼為UTF-8 with BOM) |

各評測方法的參數寫在巢狀`Evaluator`區塊內，與其`method`並列。本模組於最外層讀取上表所列的參數，其他寫在最外層的鍵由log以`Ignoring top-level keys`訊息記錄。巢狀區塊名稱為`Evaluator`；`eval_method`為`custom_method`介面的保留名稱。

## 撰寫文字解析程式

解析程式為一般Python檔案，提供一個進入點函式：

```python
def text_parser(df: pd.DataFrame) -> pd.DataFrame:
    ...
```

解析程式規範如下：

1. **輸入**：每次傳入一個資料表(`ori`、`syn`或`control`)，欄位與合成資料相同。所有資料表皆由同一函式解析，因此以一致的規則轉換。
2. **輸出**：回傳`pd.DataFrame`，列數與輸入完全相同，建議沿用輸入的索引。每個欄位為一項擷取出的資訊；數值使用數值型別，類別使用簡短的標籤，某列缺少該項資訊時回傳`None`或`NaN`。
3. **回傳的欄位**：僅回傳結構化欄位。評測方法對收到的每個欄位計分，擷取出的欄位即以表格形式描述資料的內容。
4. **型別對齊**：解析後，本模組將各資料表共有欄位的型別對齊(整數與浮點數統一為`float64`，日期統一為`datetime64[ns]`，型別不同者統一為`object`)，並依`ori`的欄位順序排列，使某一資料表中部分列缺值的欄位亦可與其他資料表比較。

以下解析程式由血壓問診文字擷取五個欄位：性別、年齡、收縮壓、舒張壓及判讀結果(偏高或正常)。

```python
"""Text parser for blood pressure records."""

import re

import pandas as pd

SOURCE_COLUMN = "text"

AGE_RE = re.compile(r"(\d{1,3})\s*歲")
BP_RE = re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})")
GENDER_RULES = [
    ("女", re.compile(r"性別[=：:]?\s*女|女性|[（(]\s*女")),
    ("男", re.compile(r"性別[=：:]?\s*男|男性|[（(]\s*男")),
]
LABEL_RULES = [
    ("偏高", re.compile(r"偏高")),
    ("正常", re.compile(r"正常")),
]


def _first_label(value: str, rules) -> str | None:
    for label, pattern in rules:
        if pattern.search(value):
            return label
    return None


def _parse_row(value) -> dict:
    record = {"gender": None, "age": None, "systolic": None,
              "diastolic": None, "bp_label": None}
    if not isinstance(value, str) or not value.strip():
        return record

    record["gender"] = _first_label(value, GENDER_RULES)
    record["bp_label"] = _first_label(value, LABEL_RULES)
    if m := AGE_RE.search(value):
        record["age"] = int(m.group(1))
    if m := BP_RE.search(value):
        record["systolic"], record["diastolic"] = int(m.group(1)), int(m.group(2))
    return record


def text_parser(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row of typed fields for every input row."""
    parsed = pd.DataFrame([_parse_row(v) for v in df[SOURCE_COLUMN]], index=df.index)
    for column in ("age", "systolic", "diastolic"):
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
    return parsed
```

原始碼所附之`petsard/synthesizer/LLM_synth/utils/bloodpressure_text_parser.py`以相同規則擷取這五個欄位，上方範例即直接指定使用該檔案。以101筆範例資料執行時，每列的五個欄位皆可完整擷取。解析其他文字時，修改`SOURCE_COLUMN`與其中的規則運算式即可。

## 處理流程

1. **解析**：解析程式逐一處理各輸入資料表，本模組確認回傳的DataFrame列數與輸入相同。
2. **型別對齊**：共有欄位統一型別，並依`ori`欄位順序排列。
3. **解析涵蓋率**：log記錄各資料表每個欄位有值的列數，格式為`syn: 50 rows -> gender=50/50, age=50/50, ...`。
4. **輸出解析結果**：設定`output_dir`時，輸出解析後的資料表供核對。
5. **評測**：各巢狀評測方法以解析後資料表副本執行，完成時log記錄`<名稱> (<method>): ok`。
6. **合併**：依粒度(`global`、`columnwise`、`pairwise`、`details`)合併各巢狀評測方法的結果，並加入`evaluator`欄標示每列所屬的巢狀評測方法。

各巢狀評測方法獨立執行，已完成的評測結果皆保留於報告中。某一巢狀評測方法回報錯誤時，log記錄其訊息，`global`報表亦列出該評測方法並於`error`欄記錄訊息，便於定位需調整的參數。

## 結果判讀

Reporter以外層的評測名稱作為欄位前綴。外層名稱為`text_eval`時，`text_eval_evaluator`欄標示每列所屬的巢狀評測方法(例如`validity_check`或`fidelity_check`)，其餘欄位為該評測方法的報表欄位，判讀方式與對應的評測頁面相同。報告依巢狀評測方法實際產生的粒度輸出。

建議檢查步驟：

1. **解析涵蓋率**：於log確認各資料表的每個欄位於多數列皆有值。某欄位於全部資料列皆無值時，log以`columns empty in every row`訊息提示，此時先調整解析程式的對應規則，再判讀該欄位分數。
2. **解析結果**：開啟`parsed_ori.csv`與`parsed_syn.csv`，抽查數列並與原始文字比對。
3. **評測分數**：依[Evaluator YAML]({{< ref "/docs/petsard-yaml/evaluator-yaml" >}})建議標準判讀有效性、保真度、隱私風險及實用性。判讀隱私與實用性時，合成器的`sample_num_rows`設為與訓練集相當的筆數。
