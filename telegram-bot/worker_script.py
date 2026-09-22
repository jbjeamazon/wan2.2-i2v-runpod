#!/usr/bin/env python3
"""
Runs on the rented Vast.ai GPU. Self-contained — no imports from the bot repo.

Reads job.json, generates the requested number of chained segments, and writes
output.mp4. Progress is appended to status.json so the bot can poll it over SSH
and relay live updates to the chat.

Chaining: a diffusion video model produces a fixed window (81 frames ~ 5s for
Wan 2.2). Longer clips are made by feeding the last frame of one segment in as
the conditioning image for the next. Quality drifts as segments accumulate —
that is inherent to the technique, not a bug here.

    python3 worker_script.py /workspace/job

Content filtering: none of the video pipelines used here ship a safety checker
(Wan, CogVideoX and I2VGen-XL have no such module). For SD-derived pipelines
that do accept one, it is explicitly disabled below.
"""

import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from PIL import Image

JOB_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/job")
STATUS = JOB_DIR / "status.json"
OUTPUT = JOB_DIR / "output.mp4"

FRAMES_PER_SEGMENT = 81
FPS = 16


def report(stage: str, detail: str = "", **extra) -> None:
    """Append a status line the bot polls over SSH."""
    payload = {"ts": time.time(), "stage": stage, "detail": detail, **extra}
    try:
        STATUS.write_text(json.dumps(payload))
    except OSError:
        pass
    print(f"[worker] {stage}: {detail}", flush=True)


def load_pipeline(model_id: str, dtype=torch.bfloat16):
    """
    Load an image-to-video pipeline and disable any content filter it exposes.

    AutoPipelineForImage2Video does not exist in diffusers, so the pipeline
    class is chosen from the model id.
    """
    report("loading", f"Loading {model_id}")
    lowered = model_id.lower()

    if "wan" in lowered:
        from diffusers import WanImageToVideoPipeline as Pipe
    elif "cogvideo" in lowered:
        from diffusers import CogVideoXImageToVideoPipeline as Pipe
    elif "i2vgen" in lowered:
        from diffusers import I2VGenXLPipeline as Pipe
    else:
        from diffusers import DiffusionPipeline as Pipe

    kwargs = {"torch_dtype": dtype}
    # Only pass safety_checker to pipelines whose __init__ accepts it —
    # passing it blindly raises on the video pipelines that lack the module.
    try:
        params = inspect.signature(Pipe.__init__).parameters
        if "safety_checker" in params:
            kwargs["safety_checker"] = None
        if "requires_safety_checker" in params:
            kwargs["requires_safety_checker"] = False
    except (TypeError, ValueError):
        pass

    pipe = Pipe.from_pretrained(model_id, **kwargs)

    # Belt and braces for any pipeline that attaches one post-construction.
    for attr in ("safety_checker", "watermarker", "nsfw_detector"):
        if hasattr(pipe, attr):
            setattr(pipe, attr, None)
    if hasattr(pipe, "register_to_config"):
        try:
            pipe.register_to_config(requires_safety_checker=False)
        except Exception:
            pass

    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total_vram >= 48:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
    report("loaded", f"{total_vram:.0f} GB VRAM")
    return pipe


def fit_dimensions(pipe, image: Image.Image, resolution: str) -> tuple[int, int]:
    import numpy as np
    max_area = 720 * 1280 if resolution == "720p" else 480 * 832
    try:
        mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    except AttributeError:
        mod = 16
    ar = image.height / image.width
    h = round(np.sqrt(max_area * ar)) // mod * mod
    w = round(np.sqrt(max_area / ar)) // mod * mod
    return int(w), int(h)


def generate_segment(pipe, image, prompt, negative, w, h, steps, guidance, seed, index):
    kwargs = dict(
        image=image, prompt=prompt, negative_prompt=negative,
        height=h, width=w, num_frames=FRAMES_PER_SEGMENT,
        num_inference_steps=steps, guidance_scale=guidance,
    )
    gen = torch.Generator(device="cuda")
    if seed is not None:
        # Offset per segment so chained parts are not identical re-rolls.
        gen.manual_seed(int(seed) + index)
        kwargs["generator"] = gen

    accepted = inspect.signature(pipe.__call__).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return pipe(**kwargs).frames[0]


def concat(segment_paths: list[Path], out: Path) -> None:
    """Join segment MP4s. ffmpeg stream-copy when available, else re-encode."""
    if len(segment_paths) == 1:
        segment_paths[0].replace(out)
        return
    listing = out.parent / "segments.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in segment_paths))
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
           "-c", "copy", str(out)]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
            capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr.decode()[:400]}")


def main() -> int:
    job = json.loads((JOB_DIR / "job.json").read_text())
    segments = int(job["segments"])
    resolution = job.get("resolution", "480p")
    steps = int(job.get("num_inference_steps", 30))
    guidance = float(job.get("guidance_scale", 3.5))
    seed = job.get("seed")
    negative = job.get("negative_prompt", "")
    model_id = job.get("model_id") or os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")

    # prompts may be one string (same throughout) or one per segment, which is
    # how "extend this video, and here is what happens next" is expressed.
    prompts = job["prompt"]
    if isinstance(prompts, str):
        prompts = [prompts] * segments
    while len(prompts) < segments:
        prompts.append(prompts[-1])

    pipe = load_pipeline(model_id)

    image = Image.open(JOB_DIR / "input.png").convert("RGB")
    w, h = fit_dimensions(pipe, image, resolution)
    image = image.resize((w, h))

    paths = []
    for i in range(segments):
        report("rendering", f"Segment {i + 1} of {segments}", segment=i + 1, total=segments)
        frames = generate_segment(pipe, image, prompts[i], negative, w, h,
                                  steps, guidance, seed, i)
        seg = JOB_DIR / f"segment_{i:03d}.mp4"
        export_to_video(frames, str(seg), fps=FPS)
        paths.append(seg)
        # Condition the next segment on where this one ended.
        last = frames[-1]
        image = last if isinstance(last, Image.Image) else Image.fromarray(
            (last * 255).astype("uint8") if getattr(last, "dtype", None) != "uint8" else last)

    report("encoding", f"Joining {len(paths)} segment(s)")
    concat(paths, OUTPUT)

    size_mb = OUTPUT.stat().st_size / 1e6
    report("done", f"{size_mb:.1f} MB", bytes=OUTPUT.stat().st_size)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        report("error", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
