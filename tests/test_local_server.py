"""Exercise local/server.py's real FastAPI routes with the GPU stack stubbed."""
import os, sys, time, tempfile, threading, warnings
warnings.filterwarnings("ignore")

import stubs
stubs.install(vram_gb=24)

TMP = tempfile.mkdtemp()
os.makedirs(os.path.join(TMP, "loras"), exist_ok=True)
open(os.path.join(TMP, "loras", "remix.safetensors"), "wb").write(b"x")
os.environ.update(
    LORA_DIR=os.path.join(TMP, "loras"),
    OUTPUT_DIR=os.path.join(TMP, "out"),
    WAN_API_KEY="s3cret",
    WAN_PUBLIC_BASE="http://127.0.0.1:8080",
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "local"))
import server
from fastapi.testclient import TestClient

client = TestClient(server.app)
AUTH = {"Authorization": "Bearer s3cret"}

# 1. /health is reachable without a key and reports the detected tier
r = client.get("/health")
assert r.status_code == 200, r.text
h = r.json()
assert h["model_loaded"] is False and h["vram_mode"] == "balanced", h
print("1. /health (no auth needed):", {k: h[k] for k in ("status", "model_loaded", "vram_gb", "vram_mode")})

# 2. every other route requires the key
assert client.get("/status/none").status_code == 401
assert client.post("/run", json={"input": {}}).status_code == 401
print("2. unauthenticated /run and /status -> 401")

# 3. list_loras works before the model is loaded, and needs no GPU
r = client.post("/run", json={"input": {"action": "list_loras"}}, headers=AUTH)
assert r.status_code == 200 and r.json()["output"]["loras"] == ["remix"], r.text
print("3. list_loras before model load ->", r.json()["output"])

# 4. a real generation request is refused while the model is still loading
r = client.post("/run", json={"input": {"image": "eA=="}}, headers=AUTH)
assert r.status_code == 503, r.text
print("4. generation while model loading -> 503:", r.json()["detail"][:48])

# --- bring the model up, as _load_model would ---
server._pipe = stubs.FakePipe()
threading.Thread(target=server._worker, daemon=True).start()

def wait(job_id, timeout=10):
    for _ in range(timeout * 20):
        s = client.get(f"/status/{job_id}", headers=AUTH).json()
        if s["status"] in ("COMPLETED", "FAILED"):
            return s
        time.sleep(0.05)
    raise AssertionError("job never finished")

# 5. submit returns immediately, then the job completes with a local URL
r = client.post("/run", json={"input": {
    "image": "eA==", "prompt": "waves", "num_frames": 17, "loras": ["remix"]}}, headers=AUTH)
assert r.status_code == 200, r.text
job_id = r.json()["id"]
assert r.json()["status"] == "IN_QUEUE"
print("5. submitted, got job id immediately:", job_id[:12])

s = wait(job_id)
assert s["status"] == "COMPLETED", s
out = s["output"]
assert out["video_url"].startswith("http://127.0.0.1:8080/videos/"), out
assert out["loras"] == ["remix"] and out["num_frames"] == 17
assert "video_base64" not in out, "local backend must never inline base64"
print("5. completed ->", out["video_url"])
print("   ", {k: out[k] for k in ("width", "height", "num_frames", "resolution", "loras")})

# 6. the video is actually served, and only from OUTPUT_DIR
name = out["video_url"].rsplit("/", 1)[-1]
r = client.get(f"/videos/{name}", headers=AUTH)
assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
print(f"6. GET /videos/{name[:12]}... -> 200, {len(r.content)} bytes, video/mp4")

for bad in ["../../../etc/passwd", "..%2Fsecret.mp4", "sub/dir.mp4", "notavideo.txt"]:
    code = client.get(f"/videos/{bad}", headers=AUTH).status_code
    assert code in (400, 404), f"{bad} -> {code}"
print("6. traversal / non-mp4 names rejected (400/404)")

# 7. bad input fails the job cleanly rather than killing the worker
r = client.post("/run", json={"input": {"image": "eA==", "resolution": "4k"}}, headers=AUTH)
s = wait(r.json()["id"])
assert s["status"] == "FAILED" and "480p" in s["output"]["error"], s
print("7. bad resolution -> FAILED:", s["output"]["error"][:52])

r = client.post("/run", json={"input": {"image": "eA==", "loras": ["../../etc/x"]}}, headers=AUTH)
s = wait(r.json()["id"])
assert s["status"] == "FAILED" and "Invalid LoRA name" in s["output"]["error"]
print("7. lora traversal -> FAILED:", s["output"]["error"][:52])

# 8. worker survived every failure and still serves new work
r = client.post("/run", json={"input": {"image": "eA==", "num_frames": 9}}, headers=AUTH)
assert wait(r.json()["id"])["status"] == "COMPLETED"
print("8. worker still healthy after failed jobs")

# 9. unknown job id
assert client.get("/status/deadbeef", headers=AUTH).status_code == 404
print("9. unknown job id -> 404")

# 10. retention sweep deletes only old files
server.RETAIN_HOURS = 1
old = os.path.join(server.OUTPUT_DIR, "old.mp4")
open(old, "wb").write(b"x")
os.utime(old, (time.time() - 7200, time.time() - 7200))
fresh_file = os.path.join(server.OUTPUT_DIR, "fresh.mp4")
open(fresh_file, "wb").write(b"x")
server._sweep_old_videos()
assert not os.path.exists(old) and os.path.exists(fresh_file)
print("10. retention sweep removed the 2h-old file, kept the new one")

print("\nALL LOCAL SERVER TESTS PASSED")
