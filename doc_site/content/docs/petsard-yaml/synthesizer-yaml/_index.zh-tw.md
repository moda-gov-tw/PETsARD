---
title: "Synthesizer YAML"
type: docs
weight: 650
prev: docs/petsard-yaml/preprocessor-yaml
next: docs/petsard-yaml/postprocessor-yaml
---

Synthesizer 模組負責生成合成資料，支援多種合成方法。

## 使用範例

請點擊下方按鈕在 Colab 中執行範例：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moda-gov-tw/PETsARD/blob/main/demo/petsard-yaml/synthesizer-yaml/synthesizer-yaml.ipynb)
> **注意**：使用 Colab 請參考[執行環境設定](/PETsARD/zh-tw/docs/#colab-執行說明)。

### 使用預設方法

```yaml
Loader:
  load_benchmark_with_schema:
    filepath: benchmark://adult-income
    schema: benchmark://adult-income_schema
Synthesizer:
  default_synthesis:
    method: default
```

### 多重實驗

可在同一個 YAML 中定義多個合成實驗：

```yaml
Loader:
  load_benchmark_with_schema:
    filepath: benchmark://adult-income
    schema: benchmark://adult-income_schema
Synthesizer:
  default_synthesis:
    method: default

  custom-method:
    method: custom_method
    module_path: custom-synthesis.py
    class_name: MySynthesizer_Shuffle
```

### 使用差分隱私合成器

差分隱私合成器不使用Preprocessor，欄位型別取自公開的欄位宣告`column_spec`：

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
```

完整參數請參見「[差分隱私合成器](dp-synthesizers)」文件。

## 主要參數

- **method** (`string`)
  - 合成方式
  - 使用 `method: default` 時，會自動使用 PETsARD 內建的 Gaussian Copula 方法作為預設合成方法。
  - 使用`method: DPflow_Copula`或`method: DPflow_AdvGAN`時，使用整個流程滿足差分隱私之合成器，請參見[差分隱私合成器](dp-synthesizers)。

### 支援的合成方法

本模組支援以下四種方式生成或載入合成資料：

1. **[PETsARD 內建方法](petsard-gaussian-copula)**
   - `petsard-gaussian-copula`：使用 Numba JIT 與 PyTorch 實現的高效高斯耦合合成器

2. **[SDV 整合方法](sdv-methods)**（可選功能）
   - **[內建整合](sdv-methods)**：直接使用 SDV 預設參數的快速方式（需額外安裝，僅供參考）
   - **[SDV Custom Methods](sdv-custom-methods)**：使用 `custom_method` 靈活控制 SDV 方法的所有參數（需額外安裝，僅供參考）

3. **[自訂合成方法](custom-method)**
   - 使用 `custom_method` 整合自行開發的合成演算法
   - **[差分隱私合成器](dp-synthesizers)**：`DPflow_Copula`、`DPflow_AdvGAN`與DP LLM文字合成器，從讀入資料到輸出的整個流程滿足差分隱私，並輸出隱私報告；欄位型別由`column_spec`宣告，不使用Preprocessor

4. **[外部資料載入](custom-data)**
   - 使用 `custom_data` 載入其他工具產生的合成資料進行評測

## 執行流程

1. **接收輸入**：從 Loader 或 Preprocessor 接收資料
2. **生成合成資料**：根據指定方法生成合成資料
3. **保持結構**：維持原始資料的欄位結構
4. **輸出結果**：將合成資料傳遞給後續模組
