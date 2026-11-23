---
title: "LLM文字合成器(初版方法)"
type: docs
weight: 657
sidebar:
  exclude: true
prev: docs/petsard-yaml/synthesizer-yaml/sdv-methods
next: docs/petsard-yaml/synthesizer-yaml/dp-synthesizers
---

本頁原為初版LLM文字合成器(`class_name: LLM_synthesizer`)之參數說明。初版方法之合成輸出不具涵蓋完整流程之差分隱私保證：其差分隱私機制僅涵蓋最近鄰投票，且其提示含訓練資料之紀錄。初版方法已移至[initial-methods分支](https://github.com/moda-gov-tw/PETsARD/tree/initial-methods)(標籤`initial-methods-v1`)，僅供研究比較及重現以初版方法產生之結果；原說明頁保留於該分支：[llm-synthesizer.zh-tw.md](https://github.com/moda-gov-tw/PETsARD/blob/initial-methods/doc_site/content/docs/petsard-yaml/synthesizer-yaml/llm-synthesizer.zh-tw.md)。

{{< callout type="warning" >}}
需具差分隱私保證之合成資料，請勿使用初版方法。
{{< /callout >}}

請改用更新後方法：

1. DP LLM文字合成器dp-llm-synth：`method: custom_method`，`module_path: petsard/synthesizer/DP_LLM_synthesizer.py`，`class_name: DP_LLM_synthesizer`，說明見[差分隱私合成器]({{< ref "/docs/petsard-yaml/synthesizer-yaml/dp-synthesizers" >}})。
2. 設定`module_path: petsard/synthesizer/LLM_synthesizer.py`時，本版本以錯誤訊息停止，並指出替代方法。
