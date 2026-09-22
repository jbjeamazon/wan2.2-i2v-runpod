"""
End-to-end job orchestration: rent a GPU, run the worker, retrieve the video,
destroy the GPU.

Cost note: with a stock PyTorch image, every job pays to `pip install` and then
download the model weights (~60 GB for Wan 2.2) before any rendering starts —
often more than the render itself. Point VAST_IMAGE at an image with the
dependencies and weights already baked in and that cost disappears. The
Dockerfile at the root of this repo builds exactly such an image.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from vast_manager import Instance, VastError, VastManager

log = logging.getLogger(__name__)

REMOTE_DIR = "/workspace/job"

DEFAULT_SETUP = (
    "pip install --no-cache-dir -q "
    "'diffusers>=0.32.0' 'transformers>=4.46.0' 'accelerate>=1.0.0' "
    "'huggingface_hub>=0.24.0' safetensors sentencepiece ftfy "
    "opencv-python-headless pillow numpy imageio imageio-ffmpeg"
)

DEFAULT_NEGATIVE = (
    "low quality, blurry, jittery, distorted, static, overexposed, "
    "watermark, text, logo, artifacts, worst quality"
)


@dataclass
class JobSpec:
    image_path: Path
    prompts: list[str]
    segments: int
    resolution: str = "480p"
    num_inference_steps: int = 30
    guidance_scale: float = 3.5
    seed: int | None = None
    negative_prompt: str = DEFAULT_NEGATIVE
    model_id: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_payload(self) -> dict:
        return {
            "prompt": self.prompts,
            "segments": self.segments,
            "resolution": self.resolution,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "model_id": self.model_id,
        }


@dataclass
class JobResult:
    video_path: Path
    seconds: float
    dph: float
    gpu: str

    @property
    def cost(self) -> float:
        return self.dph * self.seconds / 3600


async def _poll_status(vast: VastManager, inst: Instance, on_status, timeout: int) -> None:
    """Relay the worker's status file to the chat until it reports done."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            raw = await vast.exec(inst, f"cat {REMOTE_DIR}/status.json 2>/dev/null || true", timeout=60)
        except VastError:
            await asyncio.sleep(15)
            continue

        raw = raw.strip()
        if raw:
            try:
                status = json.loads(raw)
            except ValueError:
                status = None
            if status:
                stage, detail = status.get("stage"), status.get("detail", "")
                if stage == "error":
                    raise VastError(f"Worker failed: {detail}")
                if stage == "done":
                    await on_status(f"Render complete ({detail}). Downloading...")
                    return
                line = f"{stage.capitalize()}: {detail}" if detail else stage.capitalize()
                if line != last:
                    last = line
                    await on_status(line)
        await asyncio.sleep(15)
    raise VastError(f"Job exceeded {timeout}s.")


async def run_job(cfg, vast: VastManager, spec: JobSpec, on_status) -> JobResult:
    """Rent, render, retrieve, destroy. The instance dies on every exit path."""
    local_dir = cfg.work_dir / spec.job_id
    local_dir.mkdir(parents=True, exist_ok=True)
    out_path = local_dir / "output.mp4"

    public_key = cfg.ssh_pub_key_path.read_text().strip()
    worker = Path(__file__).parent / "worker_script.py"
    started = time.monotonic()

    async with vast.rent(
        query=cfg.gpu_query,
        gpu_names=cfg.vast_gpu_names,
        image=cfg.vast_image,
        disk_gb=cfg.vast_disk_gb,
        public_key=public_key,
        boot_timeout=cfg.instance_boot_timeout,
        on_status=on_status,
    ) as (inst, offer):
        await on_status("Uploading assets...")
        await vast.exec(inst, f"mkdir -p {REMOTE_DIR}", timeout=120)
        await vast.upload(inst, worker, f"{REMOTE_DIR}/worker_script.py")
        await vast.upload(inst, spec.image_path, f"{REMOTE_DIR}/input.png")

        payload = local_dir / "job.json"
        payload.write_text(json.dumps(spec.to_payload()))
        await vast.upload(inst, payload, f"{REMOTE_DIR}/job.json")

        await on_status("Installing dependencies on the GPU...")
        await vast.exec(inst, DEFAULT_SETUP, timeout=1800)

        await on_status(f"Rendering {spec.segments} segment(s)...")
        launch = (
            f"cd {REMOTE_DIR} && "
            f"nohup python3 worker_script.py {REMOTE_DIR} > worker.log 2>&1 & echo started"
        )
        await vast.exec(inst, launch, timeout=120)

        await _poll_status(vast, inst, on_status, cfg.job_timeout)
        await vast.download(inst, f"{REMOTE_DIR}/output.mp4", out_path)

    elapsed = time.monotonic() - started
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise VastError("Job finished but no video was retrieved.")

    return JobResult(video_path=out_path, seconds=elapsed, dph=offer.dph, gpu=offer.gpu_name)


def extract_last_frame(video: Path, out_image: Path) -> Path:
    """Pull the final frame so an extension continues from where it stopped."""
    import subprocess
    cmd = ["ffmpeg", "-y", "-sseof", "-1", "-i", str(video),
           "-update", "1", "-q:v", "2", str(out_image)]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not out_image.exists():
        raise RuntimeError(f"Could not extract final frame: {result.stderr.decode()[:300]}")
    return out_image
