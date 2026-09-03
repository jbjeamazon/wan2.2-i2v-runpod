"""RunPod handler paths, still working after the wan_core refactor."""
import sys, os, types, base64, tempfile, warnings
warnings.filterwarnings("ignore")

import stubs
TMP = tempfile.mkdtemp()
os.environ["LORA_DIR"] = os.path.join(TMP, "loras")
os.makedirs(os.environ["LORA_DIR"])
open(os.path.join(os.environ["LORA_DIR"], "remix.safetensors"), "wb").write(b"x")
stubs.install(vram_gb=80)

# runpod SDK stub (handler-only dependency)
stubs.mod("runpod", serverless=types.SimpleNamespace(
    progress_update=lambda job, msg: None, start=lambda cfg: None))

import handler
import storage

PNG = base64.b64encode(b"fakepng").decode()
def ok(r):
    assert "error" not in r, r["error"]
    return r

# 1. introspection needs no image
assert handler.handler({"input": {"action": "list_loras"}}) == {"loras": ["remix"]}
print("1. list_loras ->", handler.handler({"input": {"action": "list_loras"}}))

# 2. missing image rejected via the shared validator
r = handler.handler({"input": {"prompt": "hi"}})
assert "error" in r and "image_url" in r["error"]
print("2. missing image rejected:", r["error"])

# 3. happy path, base64 fallback when no S3
r = ok(handler.handler({"input": {"image": PNG, "prompt": "waves", "num_frames": 17}}))
assert "video_base64" in r and "video_url" not in r
assert r["width"] % 16 == 0 and r["height"] % 16 == 0
print(f"3. base64 fallback -> {r['width']}x{r['height']}, keys={sorted(r)}")

# 4. guidance_scale_2 passthrough
handler.handler({"input": {"image": PNG, "guidance_scale_2": 4.0}})
assert stubs.FakePipe.last_call["guidance_scale_2"] == 4.0
handler.handler({"input": {"image": PNG}})
assert stubs.FakePipe.last_call["guidance_scale_2"] is None
print("4. guidance_scale_2: set -> 4.0, omitted -> None")

# 5. lora applied and echoed
r = ok(handler.handler({"input": {"image": PNG, "loras": [{"name": "remix", "scale": 0.8}]}}))
assert r["loras"] == ["remix"]
print("5. lora echoed ->", r["loras"])

# 6. bad lora name is a clean error
r = handler.handler({"input": {"image": PNG, "loras": ["../../secret"]}})
assert "error" in r and "Invalid LoRA name" in r["error"]
print("6. bad lora rejected:", r["error"][:52])

# 7. oversized inline video fails loudly
stubs.export_to_video.size = 9 * 1024 * 1024
r = handler.handler({"input": {"image": PNG}})
assert "error" in r and "10 MB" in r["error"]
print("7. oversized inline rejected:", r["error"][:74], "...")

# 8. with S3 the same video returns a URL
storage.is_configured = lambda: True
storage.upload_video = lambda p, key_prefix="videos": "https://cdn.example.com/v/a.mp4"
r = ok(handler.handler({"input": {"image": PNG}}))
assert r["video_url"] == "https://cdn.example.com/v/a.mp4" and "video_base64" not in r
print("8. S3 configured -> video_url")
stubs.export_to_video.size = 1000

# 9. non-http image_url rejected
r = handler.handler({"input": {"image_url": "file:///etc/passwd"}})
assert "error" in r and "http(s)" in r["error"]
print("9. non-http image_url rejected:", r["error"])

# 10. bad resolution / frame count via shared validator
assert "480p" in handler.handler({"input": {"image": PNG, "resolution": "4k"}})["error"]
assert "between 1 and 200" in handler.handler({"input": {"image": PNG, "num_frames": 999}})["error"]
print("10. shared validator rejects bad resolution and num_frames")

# 11. no leaked temp files
assert not [f for f in os.listdir(tempfile.gettempdir()) if f.endswith(".mp4")]
print("11. no leaked .mp4 temp files")

print("\nALL HANDLER TESTS PASSED")
