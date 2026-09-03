"""VRAM tiering and shared param handling."""
import sys, importlib, stubs

def fresh(vram_gb, cuda=True):
    for m in ("wan_core",):
        sys.modules.pop(m, None)
    stubs.install(vram_gb=vram_gb, cuda=cuda)
    return importlib.import_module("wan_core")

# 1. auto-detection maps card size to strategy
for gb, expect in [(80, "high"), (48, "high"), (32, "balanced"), (24, "balanced"), (16, "low"), (12, "low")]:
    wc = fresh(gb)
    got = wc.resolve_vram_mode("auto")
    assert got == expect, f"{gb}GB -> {got}, expected {expect}"
    print(f"1. {gb:>2} GB -> {got}")

# no CUDA at all falls to the most conservative mode
assert fresh(0, cuda=False).resolve_vram_mode("auto") == "low"
print("1. no CUDA -> low")

# 2. explicit mode overrides detection; bad mode is rejected
wc = fresh(80)
assert wc.resolve_vram_mode("low") == "low"
try:
    wc.resolve_vram_mode("turbo")
except ValueError as e:
    print("2. explicit override honoured; bad mode rejected:", str(e)[:52])
else:
    sys.exit("FAIL: bad mode accepted")

# 3. each mode configures the pipeline the way it claims
for gb, expect_move, expect_offload, expect_tiling in [
    (80, "cuda", None, False),
    (24, None, "model", True),
    (16, None, "sequential", True),
]:
    wc = fresh(gb)
    pipe = wc.load_pipeline("fake/model", None, "auto")
    assert pipe.moved_to == expect_move, (gb, pipe.moved_to)
    assert pipe.offload == expect_offload, (gb, pipe.offload)
    assert pipe.tiling == expect_tiling, (gb, pipe.tiling)
    print(f"3. {gb:>2} GB -> to={pipe.moved_to} offload={pipe.offload} vae_tiling={pipe.tiling}")

# 4. dimensions honour aspect ratio and the VAE stride (8 * patch 2 = 16)
wc = fresh(80)
pipe = wc.load_pipeline("f", None, "high")
for w, h, res in [(1024, 768, "480p"), (768, 1024, "480p"), (1920, 1080, "720p"), (1000, 1000, "480p")]:
    img = stubs.FakeImg(w, h)
    ow, oh = wc.calculate_dimensions(pipe, img, res)
    cap = 720 * 1280 if res == "720p" else 480 * 832
    assert ow % 16 == 0 and oh % 16 == 0, (ow, oh)
    assert ow * oh <= cap, (ow * oh, cap)
    print(f"4. {w}x{h} {res} -> {ow}x{oh} (stride-16, {ow*oh} <= {cap})")

# 5. validate_params rejects bad input with caller-facing messages
for bad, frag in [
    ({}, "image"),
    ({"image": "x", "resolution": "1080p"}, "480p"),
    ({"image": "x", "num_frames": 0}, "between 1 and 200"),
    ({"image": "x", "num_frames": 500}, "between 1 and 200"),
]:
    try:
        wc.validate_params(bad)
    except ValueError as e:
        assert frag in str(e), (bad, str(e))
        print(f"5. rejected {bad} -> {str(e)[:48]}")
    else:
        sys.exit(f"FAIL: accepted {bad}")

# 6. defaults survive round-trip
p = wc.validate_params({"image": "x"})
assert p["num_frames"] == 81 and p["resolution"] == "480p" and p["fps"] == 16
assert p["negative_prompt"] == wc.DEFAULT_NEGATIVE_PROMPT
assert p["seed"] is None and p["guidance_scale_2"] is None
print("6. defaults:", {k: p[k] for k in ("resolution", "num_frames", "fps", "guidance_scale")})

# 7. generate() forwards guidance_scale_2 correctly
p.update(width=832, height=480)
wc.generate(pipe, stubs.FakeImg(), p)
assert stubs.FakePipe.last_call["guidance_scale_2"] is None
p["guidance_scale_2"] = 4.0
wc.generate(pipe, stubs.FakeImg(), p)
assert stubs.FakePipe.last_call["guidance_scale_2"] == 4.0
print("7. guidance_scale_2: omitted -> None, set -> 4.0")

print("\nALL WAN_CORE TESTS PASSED")
