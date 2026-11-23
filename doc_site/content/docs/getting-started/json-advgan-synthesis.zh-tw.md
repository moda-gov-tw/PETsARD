---
title: "JSON資料與AdvGAN合成(初版方法)"
type: docs
weight: 140
sidebar:
  exclude: true
prev: docs/getting-started/external-synthesis-default-evaluation
next: docs/petsard-yaml/synthesizer-yaml/dp-synthesizers
---

本頁原為以初版AdvGAN合成JSON資料之操作說明。初版方法之合成輸出不具涵蓋完整流程之差分隱私保證：初版AdvGAN之差分隱私機制(DP-SGD)僅涵蓋鑑別器之訓練。初版方法已移至[initial-methods分支](https://github.com/moda-gov-tw/PETsARD/tree/initial-methods)(標籤`initial-methods-v1`)，僅供研究比較及重現以初版方法產生之結果；原說明頁保留於該分支：[json-advgan-synthesis.zh-tw.md](https://github.com/moda-gov-tw/PETsARD/blob/initial-methods/doc_site/content/docs/getting-started/json-advgan-synthesis.zh-tw.md)。

{{< callout type="warning" >}}
需具差分隱私保證之合成資料，請勿使用初版方法。
{{< /callout >}}

請改用更新後方法：

1. 半結構化JSON資料：以`method: DPflow_AdvGAN`(更新版DP AdvGAN)或`method: DPflow_Copula`合成由JSON讀入之資料表，整個流程滿足差分隱私，說明見[差分隱私合成器]({{< ref "/docs/petsard-yaml/synthesizer-yaml/dp-synthesizers" >}})。
2. 教學範例：`tutorial/run_json1.py`以更新版DP AdvGAN合成JSON資料，見`tutorial/README.md`。
