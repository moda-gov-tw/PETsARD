<p align="center"><img width=75% src="https://raw.githubusercontent.com/moda-gov-tw/PETsARD/main/.github/assets/PETsARD-logo.png"></p>

![Python 3.10](https://img.shields.io/badge/python-v3.10-blue.svg)
![Python 3.11](https://img.shields.io/badge/python-v3.11-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![PyPI - Status](https://img.shields.io/pypi/status/petsard)

`PETsARD` (Privacy Enhancing Technologies Analysis, Research, and Development, /pəˈtɑrd/) is a Python library for facilitating synthetic data generation and evaluation processes.

`PETsARD`（隱私強化技術分析、研究與開發）是一套為了促進合成資料生成與評估過程而設計的 Python 程式庫。

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

### Quick Start 快速開始

```bash
# Install 安裝
pip install petsard

# Run tests 執行測試
pip install petsard[dev]
pytest
```

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