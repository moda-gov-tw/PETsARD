---
title: Docker Prebuilt
type: docs
weight: 210
prev: docs/installation
next: docs/installation/pypi-install
---

Suitable for environments **with network connection** and **Docker support**.

> If your environment is **without network connection**, please refer to [Docker Offline Deployment](../docker-offline-deployment) method.

## Method: Using Pre-built Docker Container (Recommended)

PETsARD provides pre-built Docker containers, allowing you to use it without local Python environment setup.

```bash
# Pull the latest version
docker pull ghcr.io/moda-gov-tw/petsard:latest

# Run interactive container
docker run -it --rm ghcr.io/moda-gov-tw/petsard:latest
```

**Available Tags:**
- `latest`, `main` - Latest version (from main branch)
- Version tags (e.g. `v1.10.1`, `v1.10`, `v1`) - Released versions, published when a version tag is pushed
