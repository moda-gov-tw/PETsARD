---
title: "AdvGAN Synthesizer (Initial Method)"
type: docs
weight: 656
sidebar:
  exclude: true
prev: docs/petsard-yaml/synthesizer-yaml/sdv-methods
next: docs/petsard-yaml/synthesizer-yaml/dp-synthesizers
---

This page used to describe the parameters of the initial AdvGAN synthesizer (`method: AdvGAN`). The synthetic output of the initial methods does not carry a differential privacy guarantee covering the whole flow: its differentially private mechanism (DP-SGD) covers only the training of the discriminator. The initial methods have moved to the [initial-methods branch](https://github.com/moda-gov-tw/PETsARD/tree/initial-methods) (tag `initial-methods-v1`) and are kept only for research comparison and for reproducing results produced with them. The original page is in that branch: [advgan.md](https://github.com/moda-gov-tw/PETsARD/blob/initial-methods/doc_site/content/docs/petsard-yaml/synthesizer-yaml/advgan.md).

{{< callout type="warning" >}}
Do not use the initial methods for synthetic data that needs a differential privacy guarantee.
{{< /callout >}}

Use the updated methods instead:

1. The updated DP AdvGAN, `method: DPflow_AdvGAN`; the whole flow is differentially private. See [Differentially Private Synthesizers]({{< ref "/docs/petsard-yaml/synthesizer-yaml/dp-synthesizers" >}}).
2. With `method: AdvGAN`, this version stops with an error message that names the replacement.
