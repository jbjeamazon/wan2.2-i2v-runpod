"""
RunPod Serverless Handler — Wan2.2 Image-to-Video (I2V)
Model: Wan-AI/Wan2.2-I2V-A14B-Diffusers

Queue-based worker (/run, /runsync). Generation takes minutes, so the queue
endpoint type is the right fit — submit with /run and poll /status.

Pipeline setup and generation live in wan_core, shared with local/server.py so
both backends behave identically.
"""

import base64
import os
import tempfile

import runpod
import torch

import loras
import storage
import wan_core

# ---------------------------------------------------------------------------
# Configuration (from environment variables)
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
DEFAULT_RESOLUTION = os.environ.get("RESOLUTION", "480p")

# VRAM_MODE is the modern control; ENABLE_CPU_OFFLOAD is kept for existing
# deployments and maps onto it.
if os.environ.get("VRAM_MODE"):
    VRAM_MODE = os.environ["VRAM_MODE"]
elif os.environ.get("ENABLE_CPU_OFFLOAD", "false").lower() == "true":
    VRAM_MODE = "balanced"
else:
    VRAM_MODE = "auto"

# /run caps payloads at 10 MB and base64 adds ~33%. Stay under it with room
# for the rest of the JSON body.
MAX_INLINE_VIDEO_BYTES = 7 * 1024 * 1024

# ---------------------------------------------------------------------------
# Model loading — runs once at worker startup
# ---------------------------------------------------------------------------
pipe = wan_core.load_pipeline(MODEL_ID, HF_TOKEN, VRAM_MODE)

print(f"[wan2.2-i2v] LoRAs available: {loras.available() or 'none'}")
print(f"[wan2.2-i2v] Video delivery: {'S3 URL' if storage.is_configured() else 'inline base64'}")


def deliver_video(path: str) -> dict:
    """Return the finished MP4 as a URL when possible, inline base64 otherwise."""
    if storage.is_configured():
        return {"video_url": storage.upload_video(path)}

    size = os.path.getsize(path)
    if size > MAX_INLINE_VIDEO_BYTES:
        raise ValueError(
            f"Video is {size / 1e6:.1f} MB, too large to return inline "
            f"(RunPod caps /run payloads at 10 MB and base64 adds ~33%). "
            f"Configure S3_BUCKET / S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY to "
            f"receive a download URL instead, or lower num_frames / resolution."
        )

    with open(path, "rb") as f:
        return {"video_base64": base64.b64encode(f.read()).decode("utf-8")}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    job_input = job["input"]

    # --- Introspection: let clients discover installed LoRAs without paying
    # for a generation. ---
    if job_input.get("action") == "list_loras":
        return {"loras": loras.available()}

    job_input.setdefault("resolution", DEFAULT_RESOLUTION)

    try:
        params = wan_core.validate_params(job_input)
    except ValueError as exc:
        return {"error": str(exc)}

    # --- Resolve LoRAs ---
    try:
        applied_loras = loras.apply(pipe, loras.parse_request(job_input.get("loras")))
    except loras.LoraError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"Failed to apply LoRAs: {exc}"}

    # --- Load image ---
    runpod.serverless.progress_update(job, "Loading input image...")
    try:
        image = wan_core.load_image(job_input.get("image"), job_input.get("image_url"))
    except Exception as exc:
        return {"error": f"Failed to load image: {exc}"}

    # --- Compute dimensions ---
    width, height = wan_core.calculate_dimensions(pipe, image, params["resolution"])
    image = image.resize((width, height))
    params["width"], params["height"] = width, height

    # --- Generate ---
    runpod.serverless.progress_update(
        job,
        f"Generating {params['resolution']} video ({width}x{height}, "
        f"{params['num_frames']} frames, {params['num_inference_steps']} steps)...",
    )

    generator = torch.Generator(device="cuda")
    if params["seed"] is not None:
        generator.manual_seed(int(params["seed"]))

    try:
        frames = wan_core.generate(pipe, image, params, generator=generator)
    except Exception as exc:
        return {"error": f"Video generation failed: {exc}"}

    # --- Encode and deliver output ---
    runpod.serverless.progress_update(job, "Encoding output video...")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        wan_core.write_video(frames, tmp_path, fps=params["fps"])
        delivery = deliver_video(tmp_path)
    except Exception as exc:
        return {"error": f"Failed to deliver video: {exc}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return {
        **delivery,
        "width": width,
        "height": height,
        "num_frames": params["num_frames"],
        "fps": params["fps"],
        "resolution": params["resolution"],
        "loras": applied_loras,
        "seed": params["seed"],
    }


runpod.serverless.start({"handler": handler})
