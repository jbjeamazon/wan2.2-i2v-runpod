#!/usr/bin/env python3
"""
MCP server exposing a Wan 2.2 image-to-video endpoint to OpenClaw
(or any other MCP client).

Works against either backend, because local/server.py mirrors RunPod's API
shape — only the base URL and auth header differ.

Generation takes minutes, so the work is deliberately split across two tools:
`generate_video` submits and returns immediately with a job id, and
`check_video_job` polls it. A single blocking tool would stall the agent's
turn and, on most clients, time out before the video is ready.

Environment:
    WAN_BACKEND         "local" (default) or "runpod"

  local backend:
    WAN_LOCAL_URL       Base URL of local/server.py  (default http://127.0.0.1:8080)
    WAN_API_KEY         Bearer token, if the server sets one     (optional)

  runpod backend:
    RUNPOD_API_KEY      RunPod API key                           (required)
    RUNPOD_ENDPOINT_ID  Serverless endpoint id                   (required)

    WAN_TIMEOUT         HTTP timeout in seconds                  (default 120)
"""

import os
import base64
import mimetypes
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

BACKEND = os.environ.get("WAN_BACKEND", "local").lower()
TIMEOUT = float(os.environ.get("WAN_TIMEOUT", os.environ.get("RUNPOD_TIMEOUT", "120")))

HEADERS = {"Content-Type": "application/json"}

if BACKEND == "runpod":
    API_KEY = os.environ.get("RUNPOD_API_KEY")
    ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
    if not API_KEY or not ENDPOINT_ID:
        raise SystemExit(
            "WAN_BACKEND=runpod requires RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID."
        )
    BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
    HEADERS["Authorization"] = f"Bearer {API_KEY}"
elif BACKEND == "local":
    BASE_URL = os.environ.get("WAN_LOCAL_URL", "http://127.0.0.1:8080").rstrip("/")
    if os.environ.get("WAN_API_KEY"):
        HEADERS["Authorization"] = f"Bearer {os.environ['WAN_API_KEY']}"
else:
    raise SystemExit(f"WAN_BACKEND must be 'local' or 'runpod', got {BACKEND!r}.")

mcp = FastMCP("wan22-i2v")


async def _post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(f"{BASE_URL}/{path}", headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json()


async def _get(path: str) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(f"{BASE_URL}/{path}", headers=HEADERS)
        response.raise_for_status()
        return response.json()


def _image_field(image: str) -> dict[str, str]:
    """Send a remote image by URL and a local one as base64."""
    if image.startswith(("http://", "https://")):
        return {"image_url": image}

    path = os.path.expanduser(image)
    if not os.path.isfile(path):
        raise ValueError(f"No such image file: {image}")

    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return {"image": f"data:{mime};base64,{encoded}"}


@mcp.tool()
async def generate_video(
    image: str,
    prompt: str = "",
    negative_prompt: str | None = None,
    resolution: str = "480p",
    num_frames: int = 81,
    num_inference_steps: int = 40,
    guidance_scale: float = 3.5,
    fps: int = 16,
    seed: int | None = None,
    loras: list[str] | None = None,
) -> dict[str, Any]:
    """
    Start an image-to-video generation on the RunPod endpoint.

    Returns a job_id immediately — generation takes several minutes. Poll it
    with check_video_job; do not wait in a loop inside a single turn.

    Args:
        image: URL of the source image, or a path to a local image file.
        prompt: Description of the motion and scene to animate.
        negative_prompt: Concepts to steer away from. Omit for the default.
        resolution: "480p" or "720p".
        num_frames: Frame count; 81 is about 5 seconds at 16 fps.
        num_inference_steps: Denoising steps. Lower is faster and rougher.
        guidance_scale: Prompt adherence strength.
        fps: Frames per second of the exported MP4.
        seed: Fixed seed for reproducible output.
        loras: Names of installed LoRAs to apply, from list_loras.
    """
    payload: dict[str, Any] = {
        "prompt": prompt,
        "resolution": resolution,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "fps": fps,
        **_image_field(image),
    }
    if negative_prompt is not None:
        payload["negative_prompt"] = negative_prompt
    if seed is not None:
        payload["seed"] = seed
    if loras:
        payload["loras"] = loras

    result = await _post("run", {"input": payload})
    return {
        "job_id": result.get("id"),
        "status": result.get("status", "IN_QUEUE"),
        "note": "Generation takes several minutes. Poll check_video_job with this job_id.",
    }


@mcp.tool()
async def check_video_job(job_id: str) -> dict[str, Any]:
    """
    Check a generation job and return the finished video URL when ready.

    Status is one of IN_QUEUE, IN_PROGRESS, COMPLETED, FAILED or CANCELLED.
    While it is IN_QUEUE or IN_PROGRESS, wait before checking again rather
    than polling tightly.

    Args:
        job_id: The id returned by generate_video.
    """
    result = await _get(f"status/{job_id}")
    status = result.get("status")

    if status != "COMPLETED":
        return {
            "job_id": job_id,
            "status": status,
            "detail": result.get("error") or result.get("output"),
        }

    output = result.get("output") or {}
    if "error" in output:
        return {"job_id": job_id, "status": "FAILED", "error": output["error"]}

    response = {
        "job_id": job_id,
        "status": "COMPLETED",
        "resolution": output.get("resolution"),
        "num_frames": output.get("num_frames"),
        "loras": output.get("loras"),
    }
    if "video_url" in output:
        response["video_url"] = output["video_url"]
    else:
        # The endpoint has no object storage configured and returned the MP4
        # inline. Don't hand a multi-megabyte base64 blob back to the model.
        response["note"] = (
            "Endpoint returned the video inline as base64 rather than a URL. "
            "Configure S3 on the RunPod endpoint to get a link instead."
        )
    return response


@mcp.tool()
async def list_loras() -> dict[str, Any]:
    """List the LoRAs installed on the endpoint, usable in generate_video."""
    result = await _post("runsync", {"input": {"action": "list_loras"}})
    output = result.get("output") or {}
    return {"loras": output.get("loras", []), "status": result.get("status")}


if __name__ == "__main__":
    mcp.run()
