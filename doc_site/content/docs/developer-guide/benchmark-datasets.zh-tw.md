---
title: 基準資料集
type: docs
weight: 1110
prev: docs/developer-guide
next: docs/developer-guide/test-coverage
---

## 儲存位置

基準資料集以附件形式存放於 PETsARD GitHub repository 之 Release：
- **Repository**：`moda-gov-tw/PETsARD`
- **Release 標籤**：`benchmark-datasets-v1`
- **下載網址**：`https://github.com/moda-gov-tw/PETsARD/releases/download/benchmark-datasets-v1/<檔名>`

資料檔不納入 git 版本紀錄，因此不影響 repository 之大小；單一附件之上限為 2 GB。

## 可用資料集

系統會自動從 `petsard/loader/benchmark_datasets.yaml` 設定檔讀取可用的資料集清單。設定檔之 `base_url` 為下載網址之前綴，各資料集記錄名稱、檔名(`filename`)、存取類型(`access`，僅支援 `public`)及 SHA-256 雜湊值(`sha256`)。下載後之檔案須與登記之雜湊值相符，否則載入時回報錯誤。

## 新增資料集

1. 計算檔案之 SHA-256 雜湊值，例如 `sha256sum <檔名>`(Linux / macOS)或 PowerShell 之 `Get-FileHash <檔名> -Algorithm SHA256`。
2. 於 GitHub 開啟 `benchmark-datasets-v1` Release 之編輯頁面，將檔案加入附件。已上傳之附件不再覆蓋；資料更新時以新檔名上傳，使舊版 PETsARD 仍可下載原檔案。
3. 於 `benchmark_datasets.yaml` 新增一筆紀錄，填入名稱、檔名、`access: public` 及雜湊值。
4. 於 [基準資料集協議]({{< ref "/docs/petsard-yaml/loader-yaml/benchmark-protocol" >}})之可用資料集表格加入說明。