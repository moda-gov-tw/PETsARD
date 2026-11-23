---
title: "LLM Text Synthesizer (Initial Method)"
type: docs
weight: 657
sidebar:
  exclude: true
prev: docs/petsard-yaml/synthesizer-yaml/sdv-methods
next: docs/petsard-yaml/synthesizer-yaml/dp-synthesizers
---

This page used to describe the parameters of the initial LLM text synthesizer (`class_name: LLM_synthesizer`). The synthetic output of the initial methods does not carry a differential privacy guarantee covering the whole flow: its differentially private mechanism covers only the nearest-neighbor vote, while its prompts contain records of the training data. The initial methods have moved to the [initial-methods branch](https://github.com/moda-gov-tw/PETsARD/tree/initial-methods) (tag `initial-methods-v1`) and are kept only for research comparison and for reproducing results produced with them. The original page is in that branch: [llm-synthesizer.md](https://github.com/moda-gov-tw/PETsARD/blob/initial-methods/doc_site/content/docs/petsard-yaml/synthesizer-yaml/llm-synthesizer.md).

{{< callout type="warning" >}}
Do not use the initial methods for synthetic data that needs a differential privacy guarantee.
{{< /callout >}}

Use the updated methods instead:

1. The DP LLM text synthesizer dp-llm-synth: `method: custom_method`, `module_path: petsard/synthesizer/DP_LLM_synthesizer.py`, `class_name: DP_LLM_synthesizer`. See [Differentially Private Synthesizers]({{< ref "/docs/petsard-yaml/synthesizer-yaml/dp-synthesizers" >}}).
2. With `module_path: petsard/synthesizer/LLM_synthesizer.py`, this version stops with an error message that names the replacement.
