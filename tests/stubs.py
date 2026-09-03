"""Shared fakes for torch / diffusers / PIL so the repo's real logic can run CPU-only."""
import sys, types, math

import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


class FakeImg:
    def __init__(self, w=1024, h=768):
        self.width, self.height = w, h
    def convert(self, m): return self
    def resize(self, wh):
        self.width, self.height = wh
        return self


class FakePipe:
    """Stands in for WanImageToVideoPipeline; records what it was asked to do."""
    vae_scale_factor_spatial = 8
    transformer = types.SimpleNamespace(config=types.SimpleNamespace(patch_size=[1, 2, 2]))
    last_call = None

    def __init__(self):
        self.moved_to = None
        self.offload = None
        self.vae = types.SimpleNamespace(
            enable_tiling=lambda *a, **k: setattr(self, "tiling", True))
        self.tiling = False

    def to(self, d): self.moved_to = d
    def enable_model_cpu_offload(self): self.offload = "model"
    def enable_sequential_cpu_offload(self): self.offload = "sequential"
    def load_lora_weights(self, p, adapter_name=None, load_into_transformer_2=False): pass
    def set_adapters(self, n, adapter_weights=None): pass
    def enable_lora(self): pass
    def disable_lora(self): pass
    def __call__(self, **kw):
        FakePipe.last_call = kw
        return types.SimpleNamespace(frames=[["f"] * kw["num_frames"]])


def export_to_video(frames, path, fps=16):
    with open(path, "wb") as f:
        f.write(b"\0" * export_to_video.size


                )
export_to_video.size = 1000


def install(vram_gb=80.0, cuda=True):
    """Install all stubs. vram_gb drives wan_core's VRAM-mode auto-detection."""
    mod("numpy", sqrt=math.sqrt)

    class Gen:
        def manual_seed(self, s): self.seed = s

    cuda_ns = types.SimpleNamespace(
        is_available=lambda: cuda,
        get_device_properties=lambda i: types.SimpleNamespace(
            total_memory=int(vram_gb * 1024 ** 3)),
    )
    mod("torch", bfloat16="bf16", Generator=lambda device=None: Gen(), cuda=cuda_ns)

    pil_image = types.SimpleNamespace(open=lambda b: FakeImg(), Image=FakeImg)
    mod("PIL", Image=pil_image)
    sys.modules["PIL.Image"] = pil_image

    mod("diffusers", WanImageToVideoPipeline=types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakePipe()))
    mod("diffusers.utils", export_to_video=export_to_video)

    if REPO not in sys.path:
        sys.path.insert(0, REPO)
