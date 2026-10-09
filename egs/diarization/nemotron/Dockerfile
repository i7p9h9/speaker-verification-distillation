# Nemotron-3-Diarization (Sortformer) inference via Hugging Face Transformers.
# CUDA 12.8 wheels are required for Blackwell GPUs (RTX 50xx).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# Transformers main with nemotron3_diarization, tested commit.
ARG TRANSFORMERS_REF=4cc2aa84301c9aa210b5513dab9fefae03981f6e

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf-cache

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "git+https://github.com/huggingface/transformers@${TRANSFORMERS_REF}" accelerate soundfile librosa tqdm

WORKDIR /app
COPY infer.py /app/infer.py

ENTRYPOINT ["python", "/app/infer.py"]
