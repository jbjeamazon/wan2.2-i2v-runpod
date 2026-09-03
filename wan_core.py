"""
Shared Wan 2.2 pipeline setup and generation.

Used by both deployment targets: the RunPod serverless handler and the local
server. Keeping the dimension maths and VRAM tiering in one place means the two
produce identical output for identical inputs.
"""

import base64
from io import BytesIO
from urllib.request import urlopen

import numpy as np
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

DTYPE = torch.bfloat16

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, jittery, distorted, static, overexposed, "
    "watermark, text, logo, artifacts, worst quality, bad anatomy"
)

# Wan 2.2 A14B in bfloat16 is ~54 GB of transformer weights plus a umT5-XXL
# text encoder, so only the largest cards hold it resident.
VRAM_MODES = ("high", "balanced", "low")

MAX_IMAGE_DOWNLOAD_BYTES = 32 * 1024 * 1024


def detect_vram_gb() -> float:
    """Total VRAM on device 0, or 0.0 when no CUDA device is present."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def resolve_vram_mode(explicit: str | None = None) -> str:
    """
    Pick a memory strategy. `explicit` wins unless it is None or "auto".

    high     — weights resident on the GPU. Needs ~80 GB.
    balanced — whole modules moved to the GPU as needed. Fits 24-32 GB.
    low      — per-layer offload. Fits 16 GB, markedly slower.
    """
    if explicit and explicit.lower() != "auto":
        mode = explicit.lower()
        if mode not in VRAM_MODES:
            raise ValueError(f"VRAM_MODE must be one of {VRAM_MODES + ('auto',)}, got {explicit!r}")
        return mode

    gb = detect_vram_gb()
    if gb >= 48:
        return "high"
    if gb >= 20:
        return "balanced"
    return "low"


def load_pipeline(
    model_id: str,
    hf_token: str | None = None,
    vram_mode: str | None = None,
) -> WanImageToVideoPipeline:
    """Load Wan 2.2 I2V and apply the memory strategy for this machine."""
    mode = resolve_vram_mode(vram_mode)
    gb = detect_vram_gb()
    print(f"[wan] Loading {model_id} (detected {gb:.0f} GB VRAM, mode: {mode})")

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        token=hf_token,
    )

    if mode == "high":
        pipe.to("cuda")
    elif mode == "balanced":
        pipe.enable_model_cpu_offload()
    else:
        # Per-layer offload: the slowest option, but it fits the model onto
        # cards that cannot hold a whole module at once.
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()

    if mode in ("balanced", "low"):
        # Decoding a whole 81-frame latent at once is the single largest
        # allocation in the pipeline. Tiling trades a little speed to avoid it.
        # AutoencoderKLWan exposes enable_tiling() but has no enable_slicing().
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
            print("[wan] VAE tiling enabled.")

    print(f"[wan] Pipeline ready ({mode}).")
    return pipe


def calculate_dimensions(pipe, image, resolution: str) -> tuple[int, int]:
    """
    Output (width, height) preserving aspect ratio and satisfying the
    VAE / patch-size stride.
    """
    max_area = 720 * 1280 if resolution == "720p" else 480 * 832
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    aspect_ratio = image.height / image.width
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    return int(width), int(height)


def generate(pipe, image, params: dict, generator=None):
    """Run the pipeline and return the frame list for the single output video."""
    guidance_scale_2 = params.get("guidance_scale_2")
    result = pipe(
        image=image,
        prompt=params.get("prompt", ""),
        negative_prompt=params.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
        height=params["height"],
        width=params["width"],
        num_frames=params["num_frames"],
        guidance_scale=params.get("guidance_scale", 3.5),
        # Wan 2.2 applies a second guidance scale to the low-noise expert.
        # None makes diffusers reuse `guidance_scale`.
        guidance_scale_2=float(guidance_scale_2) if guidance_scale_2 is not None else None,
        num_inference_steps=params.get("num_inference_steps", 40),
        generator=generator,
    )
    return result.frames[0]


def write_video(frames, path: str, fps: int = 16) -> str:
    export_to_video(frames, path, fps=fps)
    return path


def load_image(image_b64: str | None, image_url: str | None) -> Image.Image:
    """Load the conditioning image from either a base64 blob or an http(s) URL."""
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


def validate_params(job_input: dict) -> dict:
    """
    Normalise and range-check request fields shared by both deployments.

    Raises ValueError with a caller-facing message on bad input.
    """
    if not job_input.get("image") and not job_input.get("image_url"):
        raise ValueError("Provide either 'image' (base64) or 'image_url'.")

    resolution = job_input.get("resolution", "480p")
    if resolution not in ("480p", "720p"):
        raise ValueError(f"Invalid resolution '{resolution}'. Must be '480p' or '720p'.")

    num_frames = int(job_input.get("num_frames", 81))
    if not (1 <= num_frames <= 200):
        raise ValueError("num_frames must be between 1 and 200.")

    return {
        "prompt": job_input.get("prompt", ""),
        "negative_prompt": job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
        "resolution": resolution,
        "num_frames": num_frames,
        "guidance_scale": float(job_input.get("guidance_scale", 3.5)),
        "guidance_scale_2": job_input.get("guidance_scale_2"),
        "num_inference_steps": int(job_input.get("num_inference_steps", 40)),
        "fps": int(job_input.get("fps", 16)),
        "seed": job_input.get("seed"),
    }
