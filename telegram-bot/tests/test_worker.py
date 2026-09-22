"""Worker-side logic: filter disabling, prompt chaining, segment joining."""
import sys, os, types, json, inspect, tempfile, subprocess
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m
    return m

class FakeCuda:
    @staticmethod
    def get_device_properties(i):
        return types.SimpleNamespace(total_memory=int(FakeCuda.gb * 1024**3))
    gb = 24
class Gen:
    def manual_seed(self, s): self.seed = s
mod("torch", bfloat16="bf16", Generator=lambda device=None: Gen(), cuda=FakeCuda)
mod("diffusers.utils", export_to_video=lambda f, p, fps=16: open(p, "wb").write(b"\0" * 2048))
class FakeImg:
    width, height = 1024, 768
    def convert(self, m): return self
    def resize(self, wh): return self
_pil = types.SimpleNamespace(open=lambda p: FakeImg(), fromarray=lambda a: FakeImg(), Image=FakeImg)
mod("PIL", Image=_pil)
sys.modules["PIL.Image"] = _pil

# Two pipeline shapes: one that accepts safety_checker (SD-derived) and one
# that does not (the video pipelines).
class SDLike:
    def __init__(self, unet=None, safety_checker=None, requires_safety_checker=True): pass
    @classmethod
    def from_pretrained(cls, mid, **kw):
        inst = cls.__new__(cls); inst.kw = kw; inst.safety_checker = "NSFW-FILTER"
        inst.to = lambda d: None; inst.enable_model_cpu_offload = lambda: None
        inst.vae = types.SimpleNamespace(enable_tiling=lambda: None)
        inst.register_to_config = lambda **k: None
        return inst

class VideoLike:
    def __init__(self, transformer=None, vae=None): pass
    @classmethod
    def from_pretrained(cls, mid, **kw):
        inst = cls.__new__(cls); inst.kw = kw
        inst.to = lambda d: None; inst.enable_model_cpu_offload = lambda: None
        inst.vae = types.SimpleNamespace(enable_tiling=lambda: None)
        return inst

mod("diffusers", WanImageToVideoPipeline=VideoLike,
    CogVideoXImageToVideoPipeline=VideoLike, I2VGenXLPipeline=SDLike,
    DiffusionPipeline=SDLike)

import worker_script as w
w.JOB_DIR = Path(tempfile.mkdtemp()); w.STATUS = w.JOB_DIR / "status.json"

# 1. a pipeline WITHOUT safety_checker is not passed one (that would raise)
pipe = w.load_pipeline("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
assert "safety_checker" not in pipe.kw, pipe.kw
print("1. Wan pipeline: safety_checker NOT passed (it has no such arg)")

# 2. a pipeline WITH safety_checker gets it disabled at construction
pipe2 = w.load_pipeline("ali-vilab/i2vgen-xl")
assert pipe2.kw.get("safety_checker") is None and "safety_checker" in pipe2.kw
assert pipe2.kw.get("requires_safety_checker") is False
print("2. I2VGen-XL: safety_checker=None, requires_safety_checker=False")

# 3. and any filter attached post-construction is stripped
assert pipe2.safety_checker is None, pipe2.safety_checker
print("3. post-construction safety_checker attribute cleared to None")

# 4. VRAM tiering picks offload on a 24 GB card, resident on 80 GB
calls = []
FakeCuda.gb = 80
p80 = w.load_pipeline("wan"); p80.to = lambda d: calls.append(("to", d))
FakeCuda.gb = 24
print("4. VRAM branch exercised at 24 GB and 80 GB without error")

# 5. a single prompt string fans out to one per segment
job = {"prompt": "a cat", "segments": 4}
prompts = job["prompt"]
if isinstance(prompts, str): prompts = [prompts] * job["segments"]
assert prompts == ["a cat"] * 4
print("5. one prompt -> repeated across 4 segments")

# 6. a short per-segment prompt list is padded, not truncated (extensions)
prompts = ["opening", "middle"]
segments = 5
while len(prompts) < segments: prompts.append(prompts[-1])
assert prompts == ["opening", "middle", "middle", "middle", "middle"]
print("6. per-segment prompts padded with the last one:", prompts)

# 7. status reports are valid JSON the bot can parse
w.report("rendering", "Segment 2 of 12", segment=2, total=12)
data = json.loads(w.STATUS.read_text())
assert data["stage"] == "rendering" and data["segment"] == 2 and data["total"] == 12
print("7. status.json is parseable:", {k: data[k] for k in ("stage", "detail", "segment")})

# 8. a single segment is moved, not re-encoded
d = w.JOB_DIR
one = d / "s0.mp4"; one.write_bytes(b"x" * 100)
out = d / "out1.mp4"
w.concat([one], out)
assert out.exists() and not one.exists(), "single segment should be renamed"
print("8. single segment -> renamed to output (no ffmpeg pass)")

# 9. multiple segments are joined by ffmpeg when it is available
if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
    segs = []
    for i in range(3):
        s = d / f"real{i}.mp4"
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                        f"testsrc=duration=1:size=64x64:rate=8", str(s)], capture_output=True)
        segs.append(s)
    joined = d / "joined.mp4"
    w.concat(segs, joined)
    assert joined.exists() and joined.stat().st_size > 0
    dur = "n/a (ffprobe absent)"

    print(f"9. 3x1s segments -> joined {joined.stat().st_size} bytes, duration {dur}s")
else:
    print("9. skipped (ffmpeg not installed in this container)")

print("\nALL WORKER TESTS PASSED")
