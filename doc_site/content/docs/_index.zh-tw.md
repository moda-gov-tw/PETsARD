---
title: 文件
type: docs
weight: 0
prev: docs/developer-guide
next: docs/getting-started
---

`PETsARD`（隱私強化技術分析、研究與開發, /pəˈtɑrd/）是一套為了促進資料生成演算法及其評估過程而設計的 Python 程式庫。

其主要功能包括描述資料集、執行各種資料集生成算法，以及對隱私保護和效用進行測量。

## 設計架構

下圖為 `PETsARD` 模組設計與執行流程。

<p align="center"><img src="/PETsARD/images/PETsARD_design_zhtw.png"></p>

### 系統設計說明

PETsARD 採用模組化設計，透過 YAML 描述資料處理流程。系統整合多種合成演算法，內建隱私評估與品質評估，自動產生視覺化報告。

### 開箱即用

系統提供命令列介面與預設配置範本,支援批次處理與詳細日誌記錄。即使僅具備基礎資料知識的使用者，也能快速上手並有效應用隱私強化技術。

## 本版新增功能

本版將PETsARD適用範圍由結構化表格延伸至半結構化與非結構化資料，並於前處理新增缺失資料補償。既有設定檔可直接沿用，各項新功能於YAML中明確指定即可啟用。

{{< cards >}}
{{< card link="getting-started/json-advgan-synthesis" title="JSON資料與AdvGAN合成" icon="code" subtitle="以Loader與Reporter讀取及輸出JSON紀錄，並以AdvGAN合成器合成半結構化資料。" >}}
{{< card link="getting-started/llm-text-synthesis" title="本機LLM文字合成" icon="chat-alt-2" subtitle="以本機大型語言模型生成合成文字，並以文字評測模組評估合成結果。" >}}
{{< card link="getting-started/missing-compensation" title="缺失資料補償" icon="puzzle" subtitle="以公開或外部資料擬合Median、KNN或MICE補值策略，補償特徵欄位的缺失值。" >}}
{{< /cards >}}

| 功能 | 模組 | 啟用方式 |
|---|---|---|
| JSON讀取與輸出 | Loader、Reporter | `filepath`副檔名為`.json`；Reporter設定`type: 'json'` |
| AdvGAN合成器 | Synthesizer | `method: 'AdvGAN'` |
| LLM文字合成器 | Synthesizer | `method: custom_method`，並設定`class_name: LLM_synthesizer` |
| 文字評測模組 | Evaluator | `method: custom_method`，並設定`class_name: TextEvaluator`或`TextEvaluatorWithControl` |
| 缺失資料補償 | Preprocessor | `method: custom_method`，並設定`class_name: MissingCompensation`，以`missing_compensation.method`指定`median`、`knn`或`mice` |

使用上述功能前，請依[擴充功能安裝](installation/extended-features)完成環境設定。

## 使用指南

為方便使用者快速上手，我們準備了完整的[安裝指南](installation)與[最佳實踐](best-practices)，涵蓋環境檢查、資料治理、評測設計與屬性調整。建議循序漸進：安裝前確認環境相容性、資料準備階段遵循治理規範、配置實驗時依用途選定評測策略、執行後透過屬性調整優化效能。

<p align="center"><img src="/PETsARD/images/best-practice.zh-tw.png"></p>

## Colab 執行說明

Colab 目前預設的版本已經到 Python 3.12.12 (2025.10)，但 PETsARD 目前僅支援 Python 3.10 和 3.11。如果你想在 Colab 上執行 PETsARD，請依照下圖將 Runtime type 改成 Python 3.11.13 (2025.07)：

![如何更改 Colab Runtime Version](/images/colab-change-runtime.png)

詳細的 Colab runtime 版本資訊請參考：https://github.com/googlecolab/backend-info#202507

## 版本紀錄

- [Milestone](https://github.com/nics-dp/PETsARD/releases/latest)
  - 里程碑文件包含了 `PETsARD` 最新版本的詳細資訊。
- [Release Note](https://github.com/nics-dp/petsard/releases)
  - 發佈通知提供 `PETsARD` 每個版本的變更內容。

## 開源

歡迎提交拉取請求。對於重大變更，請先開啟一個議題來討論您想做的改變。並請確保適當地更新測試。