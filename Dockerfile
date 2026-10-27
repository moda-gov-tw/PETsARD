# syntax=docker/dockerfile:1
# check=error=true

# Python 3.11 base image (PETsARD supports Python 3.10 and 3.11)
# Python 3.11 基礎映像檔（PETsARD 支援 Python 3.10 與 3.11）
FROM python:3.11.17-slim-trixie

# Expose the GPU to the container
# 讓容器使用 GPU
ENV NVIDIA_VISIBLE_DEVICES=all \
	NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Apply Debian security updates
# 套用 Debian 安全性更新
RUN apt-get update \
	&& DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY petsard/ ./petsard/
COPY .release/docker/entrypoint.sh ./
# COPY demo/ ./demo/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
# RUN pip install --no-cache-dir -e .
# Install the versions locked in uv.lock, then reinstall the CUDA 13 NCCL used by torch
# 依 uv.lock 鎖定的版本安裝，再重新安裝 torch 使用的 CUDA 13 版 NCCL
RUN pip install --no-cache-dir uv==0.13.5 \
	&& uv export --frozen --only-group ds --no-hashes --no-emit-project -o /tmp/constraints.txt \
	&& pip install --no-cache-dir -c /tmp/constraints.txt --group ds \
	&& pip install --no-cache-dir --no-deps --force-reinstall -c /tmp/constraints.txt nvidia-nccl-cu13 \
	&& pip uninstall -y uv \
	&& rm /tmp/constraints.txt # petsard default + jupyter

# Create necessary directories for Jupyter
# 為 Jupyter 創建必要的目錄
ENV JUPYTER_CONFIG_DIR=/app/.jupyter JUPYTER_DATA_DIR=/app/.local/share/jupyter
RUN mkdir -p /app/.local/share/jupyter /app/.jupyter

# Run as a non-root user
# 以非 root 使用者執行
RUN useradd --create-home --shell /bin/bash petsard \
	&& chown -R petsard:petsard /app
USER petsard

EXPOSE 8888

# Define build arguments for labels
ARG BUILD_DATE
ARG VCS_REF

# Metadata labels
LABEL maintainer="matheme.justyn@gmail.com" \
	description="PETsARD Production Environment" \
	com.nvidia.volumes.needed="nvidia_driver" \
	org.opencontainers.image.source="https://github.com/moda-gov-tw/PETsARD" \
	org.opencontainers.image.documentation="https://moda-gov-tw.github.io/PETsARD/" \
	org.opencontainers.image.licenses="MIT AND Apache-2.0" \
	org.opencontainers.image.created=${BUILD_DATE} \
	org.opencontainers.image.revision=${VCS_REF} \
	org.opencontainers.image.title="PETsARD Development Environment" \
	org.opencontainers.image.description="Full development environment with Jupyter Lab, all dev tools, and PETsARD"

# Health check: the Jupyter server started by entrypoint.sh answers on its local port
# (/api returns the server version and needs no token)
# 健康檢查：確認 entrypoint.sh 啟動的 Jupyter 服務在本機埠回應(/api 不需 token)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
	CMD ["python", "-c", "import urllib.request as u; u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:8888/api', timeout=5)"]

# Start Jupyter Lab by default (compose.yml runs the same script as its entrypoint)
# 預設以 entrypoint.sh 啟動 Jupyter Lab(compose.yml 的 entrypoint 為同一腳本)
CMD ["bash", "/app/entrypoint.sh"]
