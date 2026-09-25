"""
Keyframe generation: one still per scene, which the I2V stage then animates.

Talks to a local A1111-compatible HTTP endpoint (/sdapi/v1/txt2img), which
AUTOMATIC1111, Forge and several ComfyUI wrappers all expose. That keeps the
choice of checkpoint entirely yours — this stage is where an uncensored
checkpoint goes, since the video model only continues whatever it is handed.
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

DEFAULT_NEGATIVE = "blurry, low quality, watermark, text, logo, deformed, extra limbs"


class StillsError(RuntimeError):
    pass


async def render_still(cfg, prompt: str, out: Path, width: int = 832,
                       height: int = 1472, negative: str = DEFAULT_NEGATIVE,
                       steps: int = 28, cfg_scale: float = 6.0,
                       seed: int | None = None) -> Path:
    """
    Generate one keyframe.

    Dimensions default to a 9:16-ish aspect that survives the later crop to
    1080x1920 without losing the subject.
    """
    payload = {
        "prompt": prompt,
        "negative_prompt": negative,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sampler_name": "DPM++ 2M",
    }
    if seed is not None:
        payload["seed"] = seed
    if cfg.t2i_model:
        payload["override_settings"] = {"sd_model_checkpoint": cfg.t2i_model}

    async with httpx.AsyncClient(timeout=600) as client:
        r = await client.post(f"{cfg.t2i_base.rstrip('/')}/sdapi/v1/txt2img", json=payload)
    if r.status_code >= 400:
        raise StillsError(f"txt2img returned {r.status_code}: {r.text[:200]}")

    images = r.json().get("images") or []
    if not images:
        raise StillsError("txt2img returned no images")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(images[0].split(",", 1)[-1]))
    return out
