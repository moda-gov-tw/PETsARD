---
title: "Local LLM Text Synthesis and Text Evaluation (Initial Method)"
type: docs
weight: 150
sidebar:
  exclude: true
prev: docs/getting-started/external-synthesis-default-evaluation
next: docs/petsard-yaml/synthesizer-yaml/dp-synthesizers
---

This page used to describe text synthesis with the initial LLM synthesizer and a local model in LM Studio. The synthetic output of the initial methods does not carry a differential privacy guarantee covering the whole flow: the differentially private mechanism of the initial LLM synthesizer covers only the nearest-neighbor vote, while its prompts contain records of the training data. The initial methods have moved to the [initial-methods branch](https://github.com/moda-gov-tw/PETsARD/tree/initial-methods) (tag `initial-methods-v1`) and are kept only for research comparison and for reproducing results produced with them. The original page is in that branch: [llm-text-synthesis.md](https://github.com/moda-gov-tw/PETsARD/blob/initial-methods/doc_site/content/docs/getting-started/llm-text-synthesis.md).

{{< callout type="warning" >}}
Do not use the initial methods for synthetic data that needs a differential privacy guarantee.
{{< /callout >}}

Use the updated methods instead:

1. Text data: the DP LLM text synthesizer dp-llm-synth (`method: custom_method`, `class_name: DP_LLM_synthesizer`); its prompts contain no real record and the whole flow is differentially private. See [Differentially Private Synthesizers]({{< ref "/docs/petsard-yaml/synthesizer-yaml/dp-synthesizers" >}}).
2. Tutorial example: `tutorial/run_llm1.py` (same as `tutorial/run_dp3_llm.py`) synthesizes text with dp-llm-synth; see `tutorial/README.md`.
