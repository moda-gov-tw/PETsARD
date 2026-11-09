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

## Docker Compose on Windows or without an NVIDIA GPU

When you build and start the container with the `compose.yml` in the project root, note that it targets Linux hosts with an NVIDIA GPU: it reserves the GPU and mounts the host files `/usr/bin/nvidia-smi` and `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1`. Docker Desktop on Windows does not have these two files, and a machine without an NVIDIA GPU cannot satisfy the GPU reservation, so `docker compose up -d` stops with a container start error. Keep `compose.yml` as it is, create `compose.override.yml` next to it (Docker Compose 2.24 or later reads it automatically) and run `docker compose up -d` as usual.

Windows with an NVIDIA GPU (GPU still available in the container):

```yaml
services:
  petsard:
    volumes: !override
      - ./data:/app/data
```

No NVIDIA GPU (CPU only):

```yaml
services:
  petsard:
    volumes: !override
      - ./data:/app/data
    deploy: !reset {}
```
