# Docker Related Files Docker 相關檔案

This directory contains Docker container related configuration files for the PETsARD project.

此目錄包含 PETsARD 專案的 Docker 容器相關配置檔案。

## File Descriptions 檔案說明

### [`entrypoint.sh`](entrypoint.sh)

Docker container entrypoint script for starting Jupyter Lab service.

Docker 容器的進入點腳本，用於啟動 Jupyter Lab 服務。

**Features 功能:**

- Start Jupyter Lab on 0.0.0.0:8888 as the non-root user `petsard` 以非 root 使用者 `petsard` 啟動 Jupyter Lab 在 0.0.0.0:8888
- Token authentication required: a random token is printed in the log (`docker compose logs petsard`), or set `JUPYTER_TOKEN` 須以 token 登入：隨機 token 印在 log（`docker compose logs petsard`），或以 `JUPYTER_TOKEN` 自訂
- Same-origin requests only 只接受同源請求

**Usage 使用方式:**

This file is automatically executed when the Docker container starts, no manual invocation required.

此檔案會在 Docker 容器啟動時自動執行，無需手動呼叫。

## Directory Structure 目錄結構

```
.release/docker/
├── README.md          # This documentation file 此說明檔案
└── entrypoint.sh      # Docker entrypoint script Docker 進入點腳本
```

## Related Files 相關檔案

- [`../../Dockerfile`](../../Dockerfile) - Main Docker build file 主要的 Docker 建置檔案
- [`../../compose.yml`](../../compose.yml) - Docker Compose configuration file Docker Compose 配置檔案

## Maintenance Notes 維護注意事項

- Docker image needs to be rebuilt after modifying `entrypoint.sh` 修改 `entrypoint.sh` 後需要重新建置 Docker 映像檔
- Ensure the script has execution permission (chmod +x) 確保腳本具有執行權限（chmod +x）
- Consider security settings when using in production environment 生產環境使用時請考慮安全性設定
- Do not re-add `--allow-root`, an empty token or `allow_origin=*`: anyone who can reach the port, or any web page the user visits, could run code in the container 請勿加回 `--allow-root`、空 token 或 `allow_origin=*`：任何連得到此埠的人或使用者瀏覽的網頁都可能在容器內執行程式
