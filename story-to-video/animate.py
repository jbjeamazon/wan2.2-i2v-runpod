"""
Image-to-video via the local server already in this repo (local/server.py).

Submit and poll rather than blocking: a scene is several chained passes and
takes minutes.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import httpx

from config import SEGMENT_SECONDS


class AnimateError(RuntimeError):
    pass


def segments_for(seconds: float) -> int:
    """
    Passes needed to cover `seconds` of narration.

    Rounded up and then trimmed to the exact narration length downstream,
    because running short leaves a gap where the voice keeps talking over
    nothing.
    """
    return max(1, math.ceil(seconds / SEGMENT_SECONDS))


async def animate(cfg, image: Path, motion_prompt: str, seconds: float,
                  out: Path, on_status=None) -> Path:
    headers = {"Content-Type": "application/json"}
    if cfg.i2v_key:
        headers["Authorization"] = f"Bearer {cfg.i2v_key}"
    base = cfg.i2v_base.rstrip("/")

    import base64
    payload = {
        "image": base64.b64encode(image.read_bytes()).decode(),
        "prompt": motion_prompt,
        "segments": segments_for(seconds),
        "num_inference_steps": cfg.i2v_steps,
        "resolution": "480p",
    }
    if cfg.i2v_loras:
        payload["loras"] = [n.strip() for n in cfg.i2v_loras.split(",") if n.strip()]

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{base}/run", headers=headers, json={"input": payload})
        if r.status_code >= 400:
            raise AnimateError(f"I2V submit returned {r.status_code}: {r.text[:200]}")
        job_id = r.json().get("id")
        if not job_id:
            raise AnimateError(f"I2V returned no job id: {r.text[:200]}")

        while True:
            await asyncio.sleep(10)
            s = await client.get(f"{base}/status/{job_id}", headers=headers)
            body = s.json()
            status = body.get("status")
            if status == "COMPLETED":
                output = body.get("output") or {}
                url = output.get("video_url")
                if not url:
                    raise AnimateError("I2V completed without a video_url")
                video = await client.get(url, headers=headers, timeout=600)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(video.content)
                return out
            if status == "FAILED":
                raise AnimateError((body.get("output") or {}).get("error", "I2V failed"))
            if on_status and body.get("progress"):
                await on_status(body["progress"])
