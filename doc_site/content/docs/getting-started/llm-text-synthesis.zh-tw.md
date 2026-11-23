---
title: "本機LLM文字合成與文字評測(初版方法)"
type: docs
weight: 150
sidebar:
  exclude: true
prev: docs/getting-started/external-synthesis-default-evaluation
next: docs/petsard-yaml/synthesizer-yaml/dp-synthesizers
---

本頁原為以初版LLM合成器搭配LM Studio本機模型合成文字之操作說明。初版方法之合成輸出不具涵蓋完整流程之差分隱私保證：初版LLM合成器之差分隱私機制僅涵蓋最近鄰投票，且其提示含訓練資料之紀錄。初版方法已移至[initial-methods分支](https://github.com/moda-gov-tw/PETsARD/tree/initial-methods)(標籤`initial-methods-v1`)，僅供研究比較及重現以初版方法產生之結果；原說明頁保留於該分支：[llm-text-synthesis.zh-tw.md](https://github.com/moda-gov-tw/PETsARD/blob/initial-methods/doc_site/content/docs/getting-started/llm-text-synthesis.zh-tw.md)。

{{< callout type="warning" >}}
需具差分隱私保證之合成資料，請勿使用初版方法。
{{< /callout >}}

請改用更新後方法：

1. 文字資料：DP LLM文字合成器dp-llm-synth(`method: custom_method`，`class_name: DP_LLM_synthesizer`)，其提示不含任何真實紀錄，整個流程滿足差分隱私，說明見[差分隱私合成器]({{< ref "/docs/petsard-yaml/synthesizer-yaml/dp-synthesizers" >}})。
2. 教學範例：`tutorial/run_llm1.py`(與`tutorial/run_dp3_llm.py`相同)以dp-llm-synth合成文字，見`tutorial/README.md`。
