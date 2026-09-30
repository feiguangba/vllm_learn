FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/exercises \
    JUPYTER_TOKEN=vllm_learn \
    JUPYTER_PORT=8888 \
    STREAMLIT_PORT=8501

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
# 拆成两层：requirements 变动少可缓存；torch 大下载失败不连坐
RUN pip install --no-cache-dir --retries 5 --timeout 60 -r /tmp/requirements.txt
# cu128 轮子自带 CUDA 运行库，sm_120 (RTX 50 系) 兼容；仅运行时需要宿主机 NVIDIA 驱动
RUN pip install --no-cache-dir --retries 5 --timeout 60 torch --index-url https://download.pytorch.org/whl/cu128

COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 8888 8501

ENTRYPOINT ["/entrypoint.sh"]
