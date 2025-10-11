# syntax=docker/dockerfile:1.6

# Python version must be 3.12 !!
FROM python:3.12-slim-bookworm

# Install all dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake g++ ccache git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -U pip setuptools wheel \
 && python -m pip install "pip<24.1" \
 && pip install --no-cache-dir -r requirements.txt 


COPY src/ /app/src

# Compile c++ code
ARG BUILD_TYPE=Release
ENV CCACHE_DIR=/ccache CCACHE_MAXSIZE=5G PATH="/usr/lib/ccache:${PATH}"

RUN --mount=type=cache,target=/ccache \
    mkdir -p /app/src/build && \
    cmake -S /app/src/cpp -B /app/src/build -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DCMAKE_C_COMPILER_LAUNCHER=ccache \
          -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
          -DCMAKE_CXX_FLAGS="-Wno-error -Wno-error=maybe-uninitialized -Wno-maybe-uninitialized" \
 && cmake --build /app/src/build -j"$(nproc)"

COPY checkpoints /app/checkpoints

# Inference service API
COPY webapp.py /app/webapp.py

COPY static /app/static

RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "webapp:app", "--host", "0.0.0.0", "--port", "8000"]