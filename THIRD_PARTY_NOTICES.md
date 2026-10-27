# Third-Party Notices

This document lists the upstream project, adapted source code, pretrained
models, public datasets, notable dependency licenses and documentation
components of this extended edition of PETsARD. The project itself is
distributed under the MIT License (`LICENSE`); the files adapted from Aug-PE
(section 2) remain under the Apache License 2.0, and the package metadata
declares the license expression `MIT AND Apache-2.0`.

## 1. Upstream Project

| Item | Details |
| --- | --- |
| Project | PETsARD (Privacy Enhancing Technologies Analysis, Research, and Development) |
| Source | https://github.com/nics-dp/petsard |
| Base revision | release 1.10.2-rc.5, commit `0a2ea52e38e42b05433e1023b6e42815eb5bf080` (dev branch, 2026-07-10) |
| License | MIT License, Copyright (c) 2024 PETsARD Contributors (`LICENSE`) |
| Extension | Extended by NTUST (National Taiwan University of Science and Technology) in 2026 |

## 2. Adapted Source Code

| Component | Location in this package | Origin | License |
| --- | --- | --- | --- |
| Aug-PE, Private Evolution for text (Xie et al., ICML 2024) | `petsard/synthesizer/LLM_synth/` (see `NOTICE` there) | https://github.com/AI-secure/aug-pe, commit `a3f292f6fec7427b6a6e6d982b83b68c1f0a86d5` | Apache License 2.0; full text in `petsard/synthesizer/LLM_synth/LICENSE-APACHE-2.0.txt` |
| DPSDA, Private Evolution for images (Lin et al., ICLR 2024) | basis of Aug-PE, as acknowledged by the Aug-PE authors | https://github.com/microsoft/DPSDA | MIT License, Copyright (c) Microsoft Corporation |

The files adapted from Aug-PE carry a header naming their origin and the
NTUST modification, and `petsard/synthesizer/LLM_synth/NOTICE` lists each
file with a summary of the changes.

`petsard/synthesizer/LLM_synth/synthetic_data/dpsda/metrics.py` is an
original NTUST implementation, written from the published definitions of the
Frechet distance (FID; Heusel et al., NeurIPS 2017) and of k-NN precision and
recall (Kynkaanniemi et al., NeurIPS 2019), and is distributed under the MIT
License of this project.

## 3. Pretrained Models

The models below are downloaded at run time (by `sentence-transformers`
from Hugging Face on first use) or loaded by the user in LM Studio or another
OpenAI-compatible server. Model weights are not included in this package.

| Model | Used by | License | Model card |
| --- | --- | --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | LLM synthesizer embeddings (default `feature_extractor`) | Apache-2.0 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| `sentence-transformers/all-mpnet-base-v2` | default model of `extract_features` in `LLM_synth/synthetic_data/feature_extractor.py` | Apache-2.0 | https://huggingface.co/sentence-transformers/all-mpnet-base-v2 |
| `openai/gpt-oss-20b` | LLM for the LLM example configurations in `demo/petsard-yaml/` | Apache-2.0 | https://huggingface.co/openai/gpt-oss-20b |
| Gemma 4 12B QAT (LM Studio ID `google/gemma-4-12b-qat`) | LLM for the LLM example configurations in `demo/petsard-yaml/` | Apache-2.0 | https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf |

## 4. Public Datasets

The Adult dataset is used under the Creative Commons Attribution 4.0
International license (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).
The benchmark files derived from it are published as GitHub Release assets of
this repository (tag `benchmark-datasets-v1`) and downloaded at run time; they
are not included in the package. They are a copy of the dataset and its
schema, a training and control split, synthetic data generated from the
training split, and a JSON conversion of the records with no values changed.

| Dataset | Citation | Benchmark datasets (files) |
| --- | --- | --- |
| Adult (Census Income) | Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20 | `adult-income` (`adult-income.csv`), `adult-income_schema` (`adult-income_schema.yaml`), `adult-income_ori` (`adult-income_ori.csv`), `adult-income_control` (`adult-income_control.csv`), `adult-income_syn` (`adult-income_syn.csv`), `adult-income-json` (`adult-income.json`) |

The other benchmark datasets are example datasets of PETsARD and datasets
prepared for the examples of the extended modules; they are listed in
`demo/data/README.md`.

## 5. Notable Dependency Licenses

Dependencies are installed from PyPI according to `pyproject.toml` and
`uv.lock` and are not bundled in this package. Most of them use permissive
licenses (MIT, BSD, Apache-2.0). The following licenses carry specific terms:

| Package | License | Notes |
| --- | --- | --- |
| `copulas` (DataCebo), installed with `sdmetrics` (MIT) | Business Source License 1.1 | The Additional Use Grant permits use of the software and its derivatives except in a Synthetic Data Creation Service, that is, a commercial offering that gives third parties access to its functionality; each version changes to the MIT License four years after its release |
| `sdv` (DataCebo), optional | Business Source License 1.1 | Optional component, not installed by default; the Additional Use Grant permits use except in a commercial Synthetic Data Service for third parties; each version changes to the MIT License four years after its release |
| NVIDIA CUDA library wheels (`nvidia-*`), installed on Linux only; `uv.lock` pins wheels for x86_64 and aarch64, pulled in by `torch` and `xgboost` | NVIDIA proprietary license (LicenseRef-NVIDIA-Proprietary) for most packages; `nvidia-nvtx` is licensed under Apache-2.0 | Provided by NVIDIA free of charge; not installed on Windows or macOS |
| `certifi`, `fqdn`, `tqdm` | Mozilla Public License 2.0 (`tqdm`: MPL-2.0 and MIT) | File-level license; used unmodified |
| `anonymeter` | Clear BSD License | BSD-style license without a patent grant |

Trivy generates a software bill of materials (SBOM) of the dependencies
pinned in `uv.lock`: `trivy fs --format cyclonedx --output sbom.cdx.json .`
writes a CycloneDX SBOM, and `.github/workflows/trivy-sbom-gen.yml` submits
the dependency list to the GitHub dependency graph (`dependency.sbom.json`).
The license of each installed package is listed by `pip-licenses` run in
the installed environment.

## 6. Documentation Site

| Component | License | Notes |
| --- | --- | --- |
| Hextra Hugo theme (https://github.com/imfing/hextra) | MIT License, Copyright (c) 2023 Xin | Hugo module referenced in `doc_site/go.mod`, fetched at build time |
| Mermaid (https://github.com/mermaid-js/mermaid) | MIT License | Loaded from the jsDelivr CDN by `doc_site/layouts/partials/custom/head-end.html` |
