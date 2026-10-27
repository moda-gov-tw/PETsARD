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

### 使用AdvGAN合成器

AdvGAN合成器置於Preprocessor與Postprocessor之間執行，合成筆數由`sample_num_rows`設定：

```yaml
Loader:
  load_json:
    filepath: benchmark://customer-churn
Preprocessor:
  default:
    method: 'default'
Synthesizer:
  advgan:
    method: 'AdvGAN'
    sample_num_rows: 1200
    sigma: 1.0
    outlier: 10
    epochs: 5
Postprocessor:
  default:
    method: 'default'
```

AdvGAN完整參數請參見「[AdvGAN合成器](advgan)」文件。

## 主要參數

- **method** (`string`)
  - 合成方式
  - 使用 `method: default` 時，會自動使用 PETsARD 內建的 Gaussian Copula 方法作為預設合成方法。
  - 使用`method: 'AdvGAN'`時，使用以DP-SGD訓練鑑別器的AdvGAN合成器。

### 支援的合成方法

本模組支援以下四種方式生成或載入合成資料：

1. **[PETsARD 內建方法](petsard-gaussian-copula)**
   - `petsard-gaussian-copula`：使用 Numba JIT 與 PyTorch 實現的高效高斯耦合合成器
   - **[AdvGAN](advgan)**(`AdvGAN`)：改良式AdvGAN合成器，鑑別器透過Opacus以DP-SGD訓練，適用於JSON資料等半結構化紀錄

2. **[SDV 整合方法](sdv-methods)**（可選功能）
   - **[內建整合](sdv-methods)**：直接使用 SDV 預設參數的快速方式（需額外安裝，僅供參考）
   - **[SDV Custom Methods](sdv-custom-methods)**：使用 `custom_method` 靈活控制 SDV 方法的所有參數（需額外安裝，僅供參考）

3. **[自訂合成方法](custom-method)**
   - 使用 `custom_method` 整合自行開發的合成演算法
   - **[LLM文字合成器](llm-synthesizer)**：以本機大型語言模型生成非結構化的合成文字，透過`custom_method`載入，設定`module_path: petsard/synthesizer/LLM_synthesizer.py`與`class_name: LLM_synthesizer`

4. **[外部資料載入](custom-data)**
   - 使用 `custom_data` 載入其他工具產生的合成資料進行評測

## 執行流程

1. **接收輸入**：從 Loader 或 Preprocessor 接收資料
2. **生成合成資料**：根據指定方法生成合成資料
3. **保持結構**：維持原始資料的欄位結構
4. **輸出結果**：將合成資料傳遞給後續模組
