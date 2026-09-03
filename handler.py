"""
RunPod Serverless Handler — Wan2.2 Image-to-Video (I2V)
Model: Wan-AI/Wan2.2-I2V-A14B-Diffusers

Queue-based worker (/run, /runsync). Generation takes minutes, so the queue
endpoint type is the right fit — submit with /run and poll /status.
"""

import base64
import os
import tempfile
from io import BytesIO
from urllib.request import urlopen

import numpy as np
import runpod
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

import loras
import storage

# ---------------------------------------------------------------------------
# Configuration (from environment variables)
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
ENABLE_CPU_OFFLOAD = os.environ.get("ENABLE_CPU_OFFLOAD", "false").lower() == "true"
DEFAULT_RESOLUTION = os.environ.get("RESOLUTION", "480p")

DTYPE = torch.bfloat16
DEVICE = "cuda"

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, jittery, distorted, static, overexposed, "
    "watermark, text, logo, artifacts, worst quality, bad anatomy"
)

# /run caps payloads at 10 MB and base64 adds ~33%. Stay under it with room
# for the rest of the JSON body.
MAX_INLINE_VIDEO_BYTES = 7 * 1024 * 1024
MAX_IMAGE_DOWNLOAD_BYTES = 32 * 1024 * 1024

# ---------------------------------------------------------------------------
# Model loading — runs once at worker startup
# ---------------------------------------------------------------------------
print(f"[wan2.2-i2v] Loading pipeline from: {MODEL_ID}")

pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    token=HF_TOKEN,
)

if ENABLE_CPU_OFFLOAD:
    print("[wan2.2-i2v] CPU offload enabled.")
    pipe.enable_model_cpu_offload()
else:
    pipe.to(DEVICE)

print(f"[wan2.2-i2v] Pipeline ready. LoRAs available: {loras.available() or 'none'}")
print(f"[wan2.2-i2v] Video delivery: {'S3 URL' if storage.is_configured() else 'inline base64'}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(image_b64: str | None, image_url: str | None) -> Image.Image:
    """Load the conditioning image from either a base64 blob or a URL."""
    if image_url:
        if not image_url.lower().startswith(("http://", "https://")):
            raise ValueError("'image_url' must be an http(s) URL.")
        with urlopen(image_url, timeout=60) as response:
            data = response.read(MAX_IMAGE_DOWNLOAD_BYTES + 1)
        if len(data) > MAX_IMAGE_DOWNLOAD_BYTES:
            raise ValueError("Image at 'image_url' exceeds 32 MB.")
    else:
        if "," in image_b64:
            # Strip data URI header: data:image/png;base64,<data>
            image_b64 = image_b64.split(",", 1)[1]
        data = base64.b64decode(image_b64)

    return Image.open(BytesIO(data)).convert("RGB")


def calculate_dimensions(image: Image.Image, resolution: str) -> tuple[int, int]:
    """
    Compute output (width, height) that preserves aspect ratio and is
    compatible with the VAE / patch-size requirements.
    """
    max_area = 720 * 1280 if resolution == "720p" else 480 * 832
    mod_value = (
        pipe.vae_scale_factor_spatial
        * pipe.transformer.config.patch_size[1]
    )
    aspect_ratio = image.height / image.width
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    return int(width), int(height)


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

    # --- Required ---
    image_b64 = job_input.get("image")
    image_url = job_input.get("image_url")
    if not image_b64 and not image_url:
        return {"error": "Provide either 'image' (base64) or 'image_url'."}

    # --- Optional with defaults ---
    prompt = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    resolution = job_input.get("resolution", DEFAULT_RESOLUTION)
    num_frames = int(job_input.get("num_frames", 81))
    guidance_scale = float(job_input.get("guidance_scale", 3.5))
    guidance_scale_2 = job_input.get("guidance_scale_2")
    num_inference_steps = int(job_input.get("num_inference_steps", 40))
    fps = int(job_input.get("fps", 16))
    seed = job_input.get("seed", None)

    if resolution not in ("480p", "720p"):
        return {"error": f"Invalid resolution '{resolution}'. Must be '480p' or '720p'."}

    if not (1 <= num_frames <= 200):
        return {"error": "num_frames must be between 1 and 200."}

    # --- Resolve LoRAs ---
    try:
        requested_loras = loras.parse_request(job_input.get("loras"))
        applied_loras = loras.apply(pipe, requested_loras)
    except loras.LoraError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"Failed to apply LoRAs: {exc}"}

    # --- Load image ---
    runpod.serverless.progress_update(job, "Loading input image...")
    try:
        image = load_image(image_b64, image_url)
    except Exception as exc:
        return {"error": f"Failed to load image: {exc}"}

    # --- Compute dimensions ---
    width, height = calculate_dimensions(image, resolution)
    image = image.resize((width, height))

    # --- Generate ---
    runpod.serverless.progress_update(
        job,
        f"Generating {resolution} video ({width}x{height}, "
        f"{num_frames} frames, {num_inference_steps} steps)...",
    )

    generator = torch.Generator(device=DEVICE)
    if seed is not None:
        generator.manual_seed(int(seed))

    try:
        result = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            # Wan 2.2 runs a second guidance scale for the low-noise expert;
            # None makes diffusers reuse `guidance_scale`.
            guidance_scale_2=float(guidance_scale_2) if guidance_scale_2 is not None else None,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        frames = result.frames[0]
    except Exception as exc:
        return {"error": f"Video generation failed: {exc}"}

    # --- Encode and deliver output ---
    runpod.serverless.progress_update(job, "Encoding output video...")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        export_to_video(frames, tmp_path, fps=fps)
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
        "num_frames": num_frames,
        "fps": fps,
        "resolution": resolution,
        "loras": applied_loras,
        "seed": seed,
    }


runpod.serverless.start({"handler": handler})
