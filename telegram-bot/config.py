"""
Configuration loaded from .env, with validation that fails loudly at startup
rather than halfway through a paid GPU rental.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional if the env is exported another way
    pass


class ConfigError(RuntimeError):
    """Raised when required settings are missing or malformed."""


def _int(name: str, default: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        if default is None:
            raise ConfigError(f"{name} is not set. See .env.example.")
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}.")


def _str(name: str, default: str | None = None) -> str:
    raw = os.environ.get(name) or default
    if raw is None:
        raise ConfigError(f"{name} is not set. See .env.example.")
    return raw


# How many frames one generation pass produces, and at what rate. Wan 2.2's
# native window is 81 frames; at 16 fps that is ~5 seconds of video. Longer
# durations are produced by chaining passes, not by one long generation.
FRAMES_PER_SEGMENT = 81
SEGMENT_FPS = 16
SEGMENT_SECONDS = FRAMES_PER_SEGMENT / SEGMENT_FPS  # ~5.06s

DURATION_CHOICES = (5, 10, 15, 20, 25, 30, 60)
EXTENSION_CHOICES = (5, 10, 15, 30)


def segments_for(seconds: int) -> int:
    """
    How many chained generation passes a requested duration needs.

    Each pass is conditioned on the final frame of the previous one, so a
    60-second clip is 12 passes — 12x the GPU time and 12x the opportunity for
    visual drift, not a single long render.
    """
    if seconds <= 0:
        raise ValueError("Duration must be positive.")
    return max(1, round(seconds / SEGMENT_SECONDS))


@dataclass(frozen=True)
class Config:
    telegram_token: str
    authorized_user_id: int

    vast_api_key: str
    vast_api_base: str
    vast_image: str
    vast_min_vram_gb: int
    vast_max_dph: float
    vast_disk_gb: int
    vast_gpu_names: tuple[str, ...]
    vast_label: str
    vast_verified_only: bool
    vast_datacenter_only: bool

    ssh_key_path: Path
    ssh_pub_key_path: Path

    model_id: str
    work_dir: Path
    log_rejected_ids: bool
    instance_boot_timeout: int
    job_timeout: int

    # Telegram bots may upload at most 50 MB through the public Bot API.
    max_upload_bytes: int = 50 * 1024 * 1024

    @property
    def gpu_query(self) -> dict:
        q = {
            "gpu_ram": {"gte": self.vast_min_vram_gb * 1024},
            "dph_total": {"lte": self.vast_max_dph},
            "rentable": {"eq": True},
            "reliability2": {"gt": 0.95},
            "num_gpus": {"eq": 1},
            "inet_down": {"gt": 200},
        }
        # A Vast.ai host has root on the machine your container runs on, so who
        # the host is, is a real consideration. `verified` restricts to hosts
        # Vast has vetted (their own CLI defaults to this); `datacenter`
        # narrows further to datacenter operators rather than individuals with
        # machines at home. Neither removes host root — only confidential
        # computing does that — but they change who holds it.
        if self.vast_verified_only:
            q["verified"] = {"eq": True}
        if self.vast_datacenter_only:
            q["datacenter"] = {"eq": True}
        return q


def load_config() -> Config:
    ssh_key = Path(_str("SSH_KEY_PATH", "./secrets/vast_ed25519")).expanduser()
    gpu_names = tuple(
        n.strip() for n in _str("VAST_GPU_NAMES", "RTX_3090,RTX_4090,A100_PCIE,A100_SXM4").split(",")
        if n.strip()
    )

    cfg = Config(
        telegram_token=_str("TELEGRAM_BOT_TOKEN"),
        authorized_user_id=_int("AUTHORIZED_USER_ID"),
        vast_api_key=_str("VAST_API_KEY"),
        vast_api_base=_str("VAST_API_BASE", "https://console.vast.ai/api/v0"),
        vast_image=_str("VAST_IMAGE", "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"),
        vast_min_vram_gb=_int("VAST_MIN_VRAM_GB", 24),
        vast_max_dph=float(_str("VAST_MAX_DPH", "1.50")),
        vast_disk_gb=_int("VAST_DISK_GB", 100),
        vast_gpu_names=gpu_names,
        vast_label=_str("VAST_LABEL", "i2v-bot"),
        vast_verified_only=_str("VAST_VERIFIED_ONLY", "true").lower() == "true",
        vast_datacenter_only=_str("VAST_DATACENTER_ONLY", "false").lower() == "true",
        ssh_key_path=ssh_key,
        ssh_pub_key_path=Path(_str("SSH_PUBLIC_KEY_PATH", str(ssh_key) + ".pub")).expanduser(),
        model_id=_str("MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-Diffusers"),
        work_dir=Path(_str("WORK_DIR", "./jobs")).expanduser(),
        log_rejected_ids=_str("LOG_REJECTED_IDS", "false").lower() == "true",
        instance_boot_timeout=_int("INSTANCE_BOOT_TIMEOUT", 900),
        job_timeout=_int("JOB_TIMEOUT", 7200),
    )

    if cfg.authorized_user_id <= 0:
        raise ConfigError("AUTHORIZED_USER_ID must be your numeric Telegram user id.")
    return cfg
