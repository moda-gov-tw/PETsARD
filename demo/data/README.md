# Benchmark Datasets 基準資料集

The example configurations in `demo/petsard-yaml/` and the examples in the documentation load their data through the `benchmark://` protocol, for example `filepath: benchmark://customer-churn`. The data files are not stored in the repository: they are published as GitHub Release assets of `moda-gov-tw/PETsARD` under the tag `benchmark-datasets-v1`. On first use, Loader downloads the file into `./benchmark/` of the working directory and verifies it against the SHA-256 value recorded in `petsard/loader/benchmark_datasets.yaml`. A verified file in `./benchmark/` is reused without downloading.

`demo/petsard-yaml/`之範例設定檔及說明文件中之範例，皆以`benchmark://`協議載入資料，例如`filepath: benchmark://customer-churn`。資料檔不存放於版本庫，而以`moda-gov-tw/PETsARD`之GitHub Release附件發布，標籤為`benchmark-datasets-v1`。Loader首次使用時將檔案下載至工作目錄之`./benchmark/`，並以`petsard/loader/benchmark_datasets.yaml`登記之SHA-256驗證；`./benchmark/`中已通過驗證之檔案直接重複使用，不再下載。

- Release page 發布頁面：https://github.com/moda-gov-tw/PETsARD/releases/tag/benchmark-datasets-v1
- Download URL 下載網址：`https://github.com/moda-gov-tw/PETsARD/releases/download/benchmark-datasets-v1/<filename>`

## Offline use 離線使用

Download the files from the Release page on a computer with Internet access, then place each file unchanged at `./benchmark/<filename>` in the working directory, for example `./benchmark/customer_churn.json`. Loader verifies its SHA-256 value and uses it without connecting to the network. The working directory is the folder in which Executor runs; the example notebooks change to `demo/`, and the LLM and text evaluation notebooks change to the source root.

於可連網之電腦自發布頁面下載檔案後，將檔案原樣放在工作目錄之`./benchmark/<檔名>`，例如`./benchmark/customer_churn.json`；Loader驗證SHA-256相符後即使用該檔案，不需連網。工作目錄為Executor執行時所在之資料夾，範例notebook切換至`demo/`，LLM及文字評測notebook切換至原始碼根目錄。

## Datasets 資料集

| `benchmark://` name 名稱 | File 檔案 | Bytes 位元組 | Rows x columns 筆數x欄位數 | Examples 使用之範例 | Source 來源 |
|---|---|---:|---|---|---|
| `adult-income` | `adult-income.csv` | 5,326,368 | 48,842 x 15 | Most configurations in `demo/petsard-yaml/` (loader, splitter, preprocessor, synthesizer, constrainer, describer, reporter, postprocessor) | (1) |
| `adult-income_schema` | `adult-income_schema.yaml` | 1,733 | Schema | Used with the Adult datasets in most configurations | (1) |
| `adult-income_ori` | `adult-income_ori.csv` | 4,221,305 | 39,073 x 15 | `evaluator-yaml/`, `describer-yaml/describer-compare.yaml`, `constrainer-yaml/validate-mode_*.yaml`, `reporter-yaml/reporter_save-report.yaml`, `splitter-yaml/custom-data.yaml` | (1) |
| `adult-income_control` | `adult-income_control.csv` | 1,056,383 | 9,769 x 15 | Same as `adult-income_ori` | (1) |
| `adult-income_syn` | `adult-income_syn.csv` | 4,212,684 | 39,073 x 15 | `evaluator-yaml/`, `describer-yaml/`, `constrainer-yaml/validate-mode_*.yaml`, `reporter-yaml/reporter_save-report.yaml`, `synthesizer-yaml/custom-data.yaml` | (1) |
| `adult-income-json` | `adult-income.json` | 21,102,177 | 48,842 x 15 | `loader-yaml/json-format_large-file.yaml`, `synthesizer-yaml/advgan_adult-income-json.yaml` | (1) |
| `taiwan-salary-statistics-300k` | `taiwan_salary_statistics_300k.csv` | 58,132,748 | 300,000 x 14 | Benchmark protocol documentation | (2) |
| `taiwan-salary-statistics-300k_schema` | `taiwan_salary_statistics_300k_schema.yaml` | 2,574 | Schema | Benchmark protocol documentation | (2) |
| `taiwan-salary-statistics-300k-no-di` | `taiwan_salary_statistics_300k_no_di.csv` | 46,598,897 | 300,000 x 13 | Benchmark protocol documentation | (2) |
| `taiwan-salary-statistics-300k-no-di_schema` | `taiwan_salary_statistics_300k_no_di_schema.yaml` | 1,730 | Schema | Benchmark protocol documentation | (2) |
| `best-practices_multi-table_companies` | `best-practices_multi-table_companies.csv` | 71,235 | 1,000 x 7 | Best practices (multi-table) | (3) |
| `best-practices_multi-table_applications` | `best-practices_multi-table_applications.csv` | 194,939 | 2,513 x 8 | Best practices (multi-table) | (3) |
| `best-practices_multi-table_tracking` | `best-practices_multi-table_tracking.csv` | 261,533 | 1,879 x 11 | Best practices (multi-table) | (3) |
| `best-practices_multi-table` | `best-practices_multi-table.csv` | 303,012 | 1,000 x 32 | `preprocessor-yaml/preprocessor_encoder-datediff.yaml`, `preprocessor-yaml/preprocessor_scaler-time-anchor-scaling.yaml` | (3) |
| `best-practices_categorical_high-cardinality` | `best-practices_categorical_high-cardinality.csv` | 1,568,943 | 10,000 x 19 | Best practices (categorical and high-cardinality data) | (3) |
| `customer-churn` | `customer_churn.json` | 339,284 | 1,200 x 10 | `loader-yaml/json-format_full-evaluation.yaml`; documentation of the JSON reader, the AdvGAN synthesizer and the Reporter JSON output | (4) |
| `tw-salary` | `tw_salary.json` | 445,376 | 2,000 x 8 | `loader-yaml/json-format_traditional-chinese.yaml` | (4) |
| `test-json-data` | `test_json_data.json` | 4,004 | 30 x 6 | `synthesizer-yaml/advgan.yaml` | (4) |
| `bloodpressure-text` | `real_bloodpressure2.csv` | 7,708 | 101 x 1 (`text`) | `synthesizer-yaml/llm-synthesizer*.yaml`, `evaluator-yaml/text-evaluation*.yaml`; documentation of the LLM text synthesizer and the text evaluation | (4) |

The SHA-256 value of every file is recorded in `petsard/loader/benchmark_datasets.yaml`.

各檔案之SHA-256值記錄於`petsard/loader/benchmark_datasets.yaml`。

## Sources and licenses 來源及授權

(1) Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20. Used under the Creative Commons Attribution 4.0 International license (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). `adult-income.csv` holds the 48,842 records of the dataset and `adult-income_schema.yaml` its schema. `adult-income_ori.csv` and `adult-income_control.csv` are a training and control split of these records, and `adult-income_syn.csv` is synthetic data generated from `adult-income_ori.csv` with the SDV Gaussian Copula synthesizer. `adult-income.json` holds the 48,842 records of `adult-income.csv` as a JSON array of records; no values were changed.

(1) 取自上述UCI Adult資料集，依CC BY 4.0授權使用。`adult-income.csv`為該資料集之48,842筆紀錄，`adult-income_schema.yaml`為其結構描述；`adult-income_ori.csv`與`adult-income_control.csv`為上述紀錄分割之訓練資料與對照資料，`adult-income_syn.csv`為以SDV Gaussian Copula合成器自`adult-income_ori.csv`產生之合成資料；`adult-income.json`為`adult-income.csv`之48,842筆紀錄轉換為JSON紀錄陣列，數值未變更。

(2) Original example dataset of PETsARD. As described in the benchmark protocol documentation, it is a simulated dataset modelled on occupational salary survey statistics, and its content does not involve any real individuals or legal entities. In the `no_di` version the name and ID columns are removed, the birth date is split into year and month-day, and the address is kept at the township level.

(2) PETsARD原有之範例資料集。依基準資料集協議說明文件，本資料為模擬職類別薪資調查統計之模擬資料集，資料內容不涉及任何真實存在之個人與法人單位；`no_di`版本已移除姓名及身分證字號欄位，出生日期拆分為年及月日，地址保留至鄉鎮市區。

(3) Original example datasets of PETsARD for the best practices documentation (multi-table, multi-timestamp, and categorical and high-cardinality data).

(3) PETsARD原有之範例資料集，供最佳實務說明文件使用(多表格、多時間戳記，以及類別型與高基數資料)。

(4) Datasets prepared by the NTUST project team for the JSON, AdvGAN and LLM text examples of the extended modules. The `real_` prefix of `real_bloodpressure2.csv` marks the original input that the LLM text synthesizer learns from, as opposed to its synthetic output. All names and values in these files are fictitious; they were generated for the examples and refer to no real person.

(4) 本計畫臺科大團隊為擴充模組的JSON、AdvGAN及LLM文字範例準備的資料集。`real_bloodpressure2.csv`檔名中的`real_`是指LLM文字合成器學習的原始輸入資料，以與合成輸出區別。這些檔案中的姓名與數值均為虛構，專為範例產生，不對應任何真實個人。

## Adding datasets 新增資料集

Files already attached to the Release are not replaced. A new dataset is attached to the Release and registered in `petsard/loader/benchmark_datasets.yaml` with its file name and SHA-256 value; updated data is uploaded under a new file name, so that earlier PETsARD versions can still download the original file. The steps are described in the developer guide (`doc_site/content/docs/developer-guide/benchmark-datasets.md`).

已附加於Release的檔案不予替換。新增資料集時，將檔案附加於Release，並在`petsard/loader/benchmark_datasets.yaml`登記檔名及SHA-256；資料更新時以新檔名上傳，使舊版PETsARD仍可下載原檔案。操作步驟請參見開發者指南(`doc_site/content/docs/developer-guide/benchmark-datasets.zh-tw.md`)。
