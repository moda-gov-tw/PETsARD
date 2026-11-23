---
title: "AdvGAN合成器(初版方法)"
type: docs
weight: 656
sidebar:
  exclude: true
prev: docs/petsard-yaml/synthesizer-yaml/sdv-methods
next: docs/petsard-yaml/synthesizer-yaml/dp-synthesizers
---

本頁原為初版AdvGAN合成器(`method: AdvGAN`)之參數說明。初版方法之合成輸出不具涵蓋完整流程之差分隱私保證：其差分隱私機制(DP-SGD)僅涵蓋鑑別器之訓練。初版方法已移至[initial-methods分支](https://github.com/moda-gov-tw/PETsARD/tree/initial-methods)(標籤`initial-methods-v1`)，僅供研究比較及重現以初版方法產生之結果；原說明頁保留於該分支：[advgan.zh-tw.md](https://github.com/moda-gov-tw/PETsARD/blob/initial-methods/doc_site/content/docs/petsard-yaml/synthesizer-yaml/advgan.zh-tw.md)。

{{< callout type="warning" >}}
需具差分隱私保證之合成資料，請勿使用初版方法。
{{< /callout >}}

請改用更新後方法：

1. 更新版DP AdvGAN：`method: DPflow_AdvGAN`，整個流程滿足差分隱私，說明見[差分隱私合成器]({{< ref "/docs/petsard-yaml/synthesizer-yaml/dp-synthesizers" >}})。
2. 設定`method: AdvGAN`時，本版本以錯誤訊息停止，並指出替代方法。
