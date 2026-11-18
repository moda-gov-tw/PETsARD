<p align="center"><img width=75% src="https://raw.githubusercontent.com/moda-gov-tw/PETsARD/main/.github/assets/PETsARD-logo.png"></p>

![Python 3.10](https://img.shields.io/badge/python-v3.10-blue.svg)
![Python 3.11](https://img.shields.io/badge/python-v3.11-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![PyPI - Status](https://img.shields.io/pypi/status/petsard)

`PETsARD` (Privacy Enhancing Technologies Analysis, Research, and Development, /pəˈtɑrd/) is a Python library for facilitating synthetic data generation and evaluation processes.

`PETsARD`（隱私強化技術分析、研究與開發）是一套為了促進合成資料生成與評估過程而設計的 Python 程式庫。

> **Note 註記**: The synthetic output of the initial methods (the initial AdvGAN and the initial LLM synthesizer) does not carry a differential privacy guarantee covering the whole flow, so they have been removed from this version. This version keeps the updated methods `DPflow_Copula`, `DPflow_AdvGAN` and `dp-llm-synth` (`DP_LLM_synthesizer`), whose whole flow is differentially private; see [Differentially Private Synthesizers](https://moda-gov-tw.github.io/PETsARD/docs/petsard-yaml/synthesizer-yaml/dp-synthesizers/). Install from the source code to use them; see [Extended Features Setup](https://moda-gov-tw.github.io/PETsARD/docs/installation/extended-features/).
>
> 初版方法(初版AdvGAN、初版LLM合成器)之合成輸出不具涵蓋完整流程之差分隱私保證，已自本版本移除。本版本保留整個流程皆滿足差分隱私之更新後方法`DPflow_Copula`、`DPflow_AdvGAN`及`dp-llm-synth`(`DP_LLM_synthesizer`)，說明見[差分隱私合成器](https://moda-gov-tw.github.io/PETsARD/zh-tw/docs/petsard-yaml/synthesizer-yaml/dp-synthesizers/)。使用時請自原始碼安裝，見[擴充功能安裝](https://moda-gov-tw.github.io/PETsARD/zh-tw/docs/installation/extended-features/)。

---

## **✨ Features 主要功能**

- 🔄 **Data Generation 資料生成**: Multiple synthetic data generation algorithms 多種合成資料生成演算法
- 🔒 **Privacy Evaluation 隱私評估**: Comprehensive privacy risk assessment 全面的隱私風險評估
- 📊 **Utility Metrics 效用指標**: Data quality and utility measurements 資料品質與效用測量
- 🎯 **Flexible Configuration 靈活配置**: YAML-based workflow configuration 基於 YAML 的工作流程配置
- 📦 **Benchmark Datasets 基準資料集**: Built-in benchmark dataset support 內建基準資料集支援

---

## **📚 Documentation 文件**

**Website 網站**: https://moda-gov-tw.github.io/PETsARD/

### [**📦 Installation 安裝**](https://moda-gov-tw.github.io/PETsARD/docs/installation/)
- PyPI installation PyPI 安裝
- Docker deployment Docker 部署
- Offline setup 離線設置


### Docker Compose on Windows or without an NVIDIA GPU Docker Compose：Windows 或無 NVIDIA GPU 環境

`compose.yml` targets Linux hosts with an NVIDIA GPU: it reserves the GPU and mounts the host files `/usr/bin/nvidia-smi` and `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1`. On Docker Desktop for Windows, or on a machine without an NVIDIA GPU, keep `compose.yml` as it is, create `compose.override.yml` next to it (Docker Compose 2.24 or later reads it automatically) and run `docker compose up -d` as usual.

`compose.yml` 預設適用於具 NVIDIA GPU 之 Linux 主機：會保留 GPU，並掛載主機之 `/usr/bin/nvidia-smi` 與 `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1`。於Windows之Docker Desktop或無NVIDIA GPU之電腦使用時，`compose.yml`不需修改，於同一資料夾新增`compose.override.yml`(Docker Compose 2.24以上版本會自動合併)，再照常執行`docker compose up -d`即可。

- Windows with an NVIDIA GPU (GPU still available in the container) Windows 且有 NVIDIA GPU（容器仍可使用 GPU）：

```yaml
services:
  petsard:
    volumes: !override
      - ./data:/app/data
```

- No NVIDIA GPU (CPU only) 無 NVIDIA GPU（只用 CPU）：

```yaml
services:
  petsard:
    volumes: !override
      - ./data:/app/data
    deploy: !reset {}
```

### [**🚀 Getting Started 入門指南**](https://moda-gov-tw.github.io/PETsARD/docs/getting-started/)
- Default synthesis workflow 預設合成流程
- Using external synthetic data 使用外部合成資料

### [**🎯 Evaluation Purpose 評估目的**](https://moda-gov-tw.github.io/PETsARD/docs/evaluation-purpose/)
- Experiment design 實驗設計
- Fidelity vs. utility 保真度與效用
- Privacy risk estimation 隱私風險估計

### [**⚙️ Data Property Adjustment 資料屬性調整**](https://moda-gov-tw.github.io/PETsARD/docs/data-property-adjustment/)
- Long-tail distribution handling 長尾分佈處理
- Time anchoring 時間錨定
- Uniform encoding 統一編碼

### [**📝 YAML Configuration YAML 配置**](https://moda-gov-tw.github.io/PETsARD/docs/petsard-yaml/)
- Executor, Loader, Splitter 執行器、載入器、分割器
- Preprocessor, Synthesizer, Postprocessor 前處理器、合成器、後處理器
- Evaluator, Constrainer, Reporter 評估器、約束器、報告器

### [**📋 Schema YAML 綱要配置**](https://moda-gov-tw.github.io/PETsARD/docs/schema-yaml/)
- Data types and logical types 資料型別與邏輯型別
- Attribute parameters 屬性參數
- Statistics configuration 統計配置

### [**🐍 Python API**](https://moda-gov-tw.github.io/PETsARD/docs/python-api/)
- Programmatic usage API 參考 API reference

### [**👨‍💻 Developer Guide 開發者指南**](https://moda-gov-tw.github.io/PETsARD/docs/developer-guide/)
- Development setup 開發環境設置
- Test coverage 測試覆蓋率

### [**📚 Glossary 詞彙表**](https://moda-gov-tw.github.io/PETsARD/docs/glossary/)
- Key terminology 關鍵術語

### [**⚠️ Error Handling 錯誤處理**](https://moda-gov-tw.github.io/PETsARD/docs/error-handling/)
- Common errors and solutions 常見錯誤與解決方案

---

## **🛠️ Development 開發**

### Requirements 需求

- Python 3.10 or 3.11 Python 3.10 或 3.11
- Python 3.11.5 or later is recommended: on Python 3.10 and 3.11.0-3.11.4, a few dependencies resolve to older releases with known vulnerabilities, because their fixed releases need a newer Python (see `.trivyignore.yaml`) 建議使用Python 3.11.5以上版本：於Python 3.10及3.11.0至3.11.4，數個相依套件會安裝含已知弱點之舊版，因其修正版需較新之Python(見`.trivyignore.yaml`)

### Quick Start 快速開始

```bash
# Install 安裝
pip install petsard

# Run tests 執行測試
pip install petsard[dev]
python -m pytest
```

The PyPI package does not include the differentially private synthesizers of this version. To use them, or to run the tests of this version, install from the source root (the folder that contains `pyproject.toml`); see [Extended Features Setup](https://moda-gov-tw.github.io/PETsARD/docs/installation/extended-features/) and [tutorial/README.md](tutorial/README.md).

PyPI套件不含本版本之差分隱私合成器。使用這些合成器或執行本版本之測試時，請於原始碼根目錄(含`pyproject.toml`之資料夾)安裝，見[擴充功能安裝](https://moda-gov-tw.github.io/PETsARD/zh-tw/docs/installation/extended-features/)及[tutorial/README.md](tutorial/README.md)。

```bash
# Install from the source root 自原始碼根目錄安裝
pip install -e ".[all,advgan,llm,dev]"

# Run tests 執行測試
python -m pytest
```

On Windows, set `PYTHONUTF8=1` before running the tests (Command Prompt: `set PYTHONUTF8=1`; PowerShell: `$env:PYTHONUTF8 = "1"`).

於Windows執行測試前，請先設定`PYTHONUTF8=1`(命令提示字元：`set PYTHONUTF8=1`；PowerShell：`$env:PYTHONUTF8 = "1"`)。

### Example Datasets 範例資料集

The examples load their data through the `benchmark://` protocol, for example `benchmark://adult-income` or `benchmark://customer-churn`. The benchmark datasets are published as GitHub Release assets of this repository (tag `benchmark-datasets-v1`, https://github.com/moda-gov-tw/PETsARD/releases/tag/benchmark-datasets-v1). Loader downloads each file automatically into `./benchmark/` of the working directory and verifies it against the SHA-256 value recorded in `petsard/loader/benchmark_datasets.yaml`. To work offline, place the files unchanged at `./benchmark/<filename>`, for example `./benchmark/customer_churn.json`. The datasets, their sources and licenses are listed in [demo/data/README.md](demo/data/README.md).

範例以`benchmark://`協議載入資料，例如`benchmark://adult-income`或`benchmark://customer-churn`。基準資料集以本專案之GitHub Release附件發布(標籤`benchmark-datasets-v1`，https://github.com/moda-gov-tw/PETsARD/releases/tag/benchmark-datasets-v1)，Loader自動將檔案下載至工作目錄之`./benchmark/`，並以`petsard/loader/benchmark_datasets.yaml`登記之SHA-256驗證。離線使用時，將檔案原樣放在`./benchmark/<檔名>`，例如`./benchmark/customer_churn.json`。資料集清單、來源及授權請見[demo/data/README.md](demo/data/README.md)。

---

## **📄 License 授權**

This project is licensed under MIT License. See [LICENSE](LICENSE) for details.

本專案採用 MIT License 授權。詳見 [LICENSE](LICENSE)。

---

## **🔗 Links 連結**

- **GitHub**: https://github.com/moda-gov-tw/PETsARD
- **Documentation 文件**: https://moda-gov-tw.github.io/PETsARD/
- **PyPI**: https://pypi.org/project/petsard/
- **Issues 問題追蹤**: https://github.com/moda-gov-tw/PETsARD/issues

---

## **📧 Contact 聯絡**

For questions or support: 如有問題或需要支援：
- Open an issue on GitHub 在 GitHub 開啟 issue
- Check the documentation 查看文件