---
title: "JSON Data and AdvGAN Synthesis (Initial Method)"
type: docs
weight: 140
sidebar:
  exclude: true
prev: docs/getting-started/external-synthesis-default-evaluation
next: docs/petsard-yaml/synthesizer-yaml/dp-synthesizers
---

This page used to describe the synthesis of JSON data with the initial AdvGAN. The synthetic output of the initial methods does not carry a differential privacy guarantee covering the whole flow: the differentially private mechanism of the initial AdvGAN (DP-SGD) covers only the training of the discriminator. The initial methods have moved to the [initial-methods branch](https://github.com/moda-gov-tw/PETsARD/tree/initial-methods) (tag `initial-methods-v1`) and are kept only for research comparison and for reproducing results produced with them. The original page is in that branch: [json-advgan-synthesis.md](https://github.com/moda-gov-tw/PETsARD/blob/initial-methods/doc_site/content/docs/getting-started/json-advgan-synthesis.md).

{{< callout type="warning" >}}
Do not use the initial methods for synthetic data that needs a differential privacy guarantee.
{{< /callout >}}

Use the updated methods instead:

1. Semi-structured JSON data: synthesize the table read from JSON with `method: DPflow_AdvGAN` (updated DP AdvGAN) or `method: DPflow_Copula`; the whole flow is differentially private. See [Differentially Private Synthesizers]({{< ref "/docs/petsard-yaml/synthesizer-yaml/dp-synthesizers" >}}).
2. Tutorial example: `tutorial/run_json1.py` synthesizes JSON data with the updated DP AdvGAN; see `tutorial/README.md`.
