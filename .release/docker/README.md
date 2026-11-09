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

## Docker Compose on Windows or without an NVIDIA GPU Docker Compose：Windows 或無 NVIDIA GPU 環境

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

## Maintenance Notes 維護注意事項

- Docker image needs to be rebuilt after modifying `entrypoint.sh` 修改 `entrypoint.sh` 後需要重新建置 Docker 映像檔
- Ensure the script has execution permission (chmod +x) 確保腳本具有執行權限（chmod +x）
- Consider security settings when using in production environment 生產環境使用時請考慮安全性設定
- Keep token authentication and same-origin checks enabled (no `--allow-root`, empty token or `allow_origin=*`) 請保留token認證與同源限制，不使用`--allow-root`、空token或`allow_origin=*`
