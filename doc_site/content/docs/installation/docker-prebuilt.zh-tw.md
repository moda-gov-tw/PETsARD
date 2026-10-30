---
title: Docker 預建
type: docs
weight: 210
prev: docs/installation
next: docs/installation/pypi-install
---

適用於**有網路連線**且**支援 Docker** 的環境。

> 如果您的環境**無網路連線**，請參閱 [Docker 離線部署](../docker-offline-deployment) 方式。

## 方式：使用預建 Docker 容器（推薦）

PETsARD 提供預先建置的 Docker 容器，無需本地 Python 環境設定即可使用。

```bash
# 拉取最新版本
docker pull ghcr.io/moda-gov-tw/petsard:latest

# 運行互動式容器
docker run -it --rm ghcr.io/moda-gov-tw/petsard:latest
```

**可用標籤：**
- `latest`、`main` - 最新版本（來自 main 分支）
- 版本標籤（例如 `v1.10.1`、`v1.10`、`v1`）- 正式發布版本，於推送版本標籤時產生
