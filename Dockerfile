# ── Base image ───────────────────────────────────────────────────────────────
FROM runpod/pytorch:1.0.3-cu1281-torch260-ubuntu2204

WORKDIR /app

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        "runpod>=1.7.0" \
        "diffusers>=0.32.0" \
        "transformers>=4.46.0" \
        "accelerate>=1.0.0" \
        "huggingface_hub>=0.24.0" \
        "boto3>=1.34.0" \
        "hf-transfer" \
        "sentencepiece" \
        "ftfy" \
        "opencv-python-headless" \
        "pillow" \
        "numpy"

# hf-transfer active at runtime for fast user-triggered downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# ── Pre-download model weights ───────────────────────────────────────────────
# HF_TRANSFER build-arg lets each build environment opt in/out:
#   RunPod build servers (lots of RAM, fast CDN) → HF_TRANSFER=1 (default)
#   GitHub Actions (7 GB RAM)                    → HF_TRANSFER=0 via --build-arg
ARG HF_TRANSFER=1
RUN HF_HUB_ENABLE_HF_TRANSFER=${HF_TRANSFER} python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Wan-AI/Wan2.2-I2V-A14B-Diffusers')"

# ── Application code ─────────────────────────────────────────────────────────
COPY handler.py storage.py loras.py wan_core.py ./

CMD ["python", "-u", "handler.py"]
