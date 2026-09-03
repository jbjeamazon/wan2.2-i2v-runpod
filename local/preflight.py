#!/usr/bin/env python3
"""
Check whether this machine can run the local Wan 2.2 server.

Run this after building the box and installing drivers, but BEFORE downloading
~60 GB of weights. Every check degrades gracefully, so it is safe to run before
the Python dependencies are installed.

    python3 local/preflight.py
"""

import os
import shutil
import subprocess
import sys

# Wan 2.2 A14B: ~54 GB of bfloat16 transformer weights plus a umT5-XXL text
# encoder, downloaded once into the HuggingFace cache.
WEIGHTS_GB = 65
MIN_PYTHON = (3, 10)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str) -> None:
    results.append((name, status, detail))


def human_gb(n_bytes: float) -> str:
    return f"{n_bytes / (1024 ** 3):.0f} GB"


def check_python() -> None:
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PYTHON
    check("Python", PASS if ok else FAIL,
          f"{v.major}.{v.minor}.{v.micro}" + ("" if ok else f" (need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})"))


def check_driver() -> None:
    if not shutil.which("nvidia-smi"):
        check("NVIDIA driver", FAIL,
              "nvidia-smi not found — install the proprietary NVIDIA driver first")
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    except Exception as exc:
        check("NVIDIA driver", FAIL, f"nvidia-smi failed: {exc}")
        return
    for line in out.splitlines():
        check("GPU", PASS, line.strip())


def check_torch() -> float:
    """Returns detected VRAM in GB, or 0.0."""
    try:
        import torch
    except ImportError:
        check("PyTorch", WARN, "not installed yet — pip install torch (with CUDA wheels)")
        return 0.0

    if not torch.cuda.is_available():
        check("PyTorch CUDA", FAIL,
              f"torch {torch.__version__} sees no CUDA device — "
              "you likely installed the CPU-only wheel")
        return 0.0

    gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    check("PyTorch CUDA", PASS,
          f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, {gb:.0f} GB VRAM")
    return gb


def check_vram_mode(gb: float) -> None:
    if gb <= 0:
        check("VRAM mode", WARN, "cannot determine until PyTorch sees the GPU")
        return
    if gb >= 48:
        check("VRAM mode", PASS, f"{gb:.0f} GB -> 'high' (weights resident, fastest)")
    elif gb >= 20:
        check("VRAM mode", PASS, f"{gb:.0f} GB -> 'balanced' (module offload + VAE tiling)")
    else:
        check("VRAM mode", WARN,
              f"{gb:.0f} GB -> 'low' (per-layer offload). It will run, but slowly. "
              "24 GB is the comfortable minimum for this model.")


def check_system_ram() -> None:
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        check("System RAM", WARN, "could not determine")
        return
    gb = total / (1024 ** 3)
    if gb >= 60:
        check("System RAM", PASS, f"{gb:.0f} GB")
    elif gb >= 30:
        check("System RAM", WARN,
              f"{gb:.0f} GB — offload modes push weights into system RAM; "
              "64 GB is strongly recommended on a 24 GB card")
    else:
        check("System RAM", FAIL,
              f"{gb:.0f} GB — too little for CPU offload; expect out-of-memory kills")


def check_disk() -> None:
    cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    probe = cache
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        free = shutil.disk_usage(probe or "/").free
    except OSError as exc:
        check("Disk space", WARN, f"could not stat {probe}: {exc}")
        return
    need = WEIGHTS_GB * (1024 ** 3)
    status = PASS if free >= need else FAIL
    check("Disk space", status,
          f"{human_gb(free)} free at {cache} (need ~{WEIGHTS_GB} GB for weights)")


def check_deps() -> None:
    missing = []
    for mod, label in [("diffusers", "diffusers"), ("transformers", "transformers"),
                       ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
                       ("PIL", "pillow"), ("cv2", "opencv-python-headless")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(label)
    if missing:
        check("Python deps", WARN,
              f"missing: {', '.join(missing)} — pip install -r local/requirements.txt")
    else:
        check("Python deps", PASS, "all present")


def check_port() -> None:
    import socket
    host = os.environ.get("WAN_HOST", "127.0.0.1")
    port = int(os.environ.get("WAN_PORT", "8080"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        in_use = s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0
    check("Port", WARN if in_use else PASS,
          f"{host}:{port}" + (" already in use" if in_use else " free"))


def check_loras() -> None:
    d = os.environ.get("LORA_DIR", "/runpod-volume/loras")
    if not os.path.isdir(d):
        check("LoRA dir", WARN,
              f"{d} does not exist — set LORA_DIR to a local path (e.g. ./loras) "
              "if you want per-request LoRAs")
        return
    n = len([f for f in os.listdir(d) if f.endswith(".safetensors")])
    check("LoRA dir", PASS, f"{d} ({n} LoRA{'s' if n != 1 else ''})")


def main() -> int:
    print("Wan 2.2 local server — preflight\n")
    check_python()
    check_driver()
    gb = check_torch()
    check_vram_mode(gb)
    check_system_ram()
    check_disk()
    check_deps()
    check_port()
    check_loras()

    width = max(len(n) for n, _, _ in results)
    icons = {PASS: "  ok  ", WARN: " warn ", FAIL: " FAIL "}
    for name, status, detail in results:
        print(f"[{icons[status]}] {name.ljust(width)}  {detail}")

    fails = [r for r in results if r[1] == FAIL]
    warns = [r for r in results if r[1] == WARN]
    print()
    if fails:
        print(f"{len(fails)} blocking problem(s). Fix these before starting the server.")
        return 1
    if warns:
        print(f"Ready, with {len(warns)} warning(s) above.")
        return 0
    print("Ready. Start with: python local/server.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
