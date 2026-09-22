"""Vast.ai request shaping and the guarantee that instances always get destroyed."""
import asyncio, json, sys, os, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import vast_manager
from vast_manager import VastManager, VastError, Instance

run = asyncio.run
CALLS = []

OFFERS = {"offers": [
    {"id": 101, "gpu_name": "RTX 4090", "gpu_ram": 24576, "dph_total": 0.44, "cuda_max_good": "12.4"},
    {"id": 102, "gpu_name": "RTX 3090", "gpu_ram": 24576, "dph_total": 0.21, "cuda_max_good": "12.2"},
    {"id": 103, "gpu_name": "A100 SXM4", "gpu_ram": 81920, "dph_total": 1.30, "cuda_max_good": "12.4"},
    {"id": 104, "gpu_name": "RTX 5000", "gpu_ram": 16384, "dph_total": 0.09, "cuda_max_good": "12.1"},
]}
STATE = {"instances": [], "destroyed": [], "create_fails": False}


def handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content) if request.content else None
    CALLS.append((request.method, request.url.path, body, dict(request.headers)))
    p, m = request.url.path, request.method

    if p == "/api/v0/search/asks/" and m == "PUT":
        return httpx.Response(200, json=OFFERS)
    if p.startswith("/api/v0/asks/") and m == "PUT":
        if STATE["create_fails"]:
            return httpx.Response(500, text="capacity gone")
        return httpx.Response(200, json={"success": True, "new_contract": 5150})
    if p == "/api/v0/instances/" and m == "GET":
        return httpx.Response(200, json={"instances": STATE["instances"]})
    if p.startswith("/api/v0/instances/") and m == "DELETE":
        STATE["destroyed"].append(int(p.rstrip("/").split("/")[-1]))
        return httpx.Response(200, json={"success": True})
    return httpx.Response(404, text=f"unmocked {m} {p}")


_real = httpx.AsyncClient
def patched(*a, **kw):
    kw["transport"] = httpx.MockTransport(handler)
    return _real(*a, **kw)
httpx.AsyncClient = patched
vast_manager.httpx = httpx

tmp = Path(tempfile.mkdtemp())
(tmp / "k").write_text("KEY"); (tmp / "k.pub").write_text("ssh-ed25519 AAAA test")
vast = VastManager("sk-test", "https://console.vast.ai/api/v0", tmp / "k", label="i2v-bot")

# 1. auth header + endpoint + the documented search body shape
offer = run(vast.find_cheapest({"gpu_ram": {"gte": 24576}}, ("RTX_3090", "RTX_4090")))
m, path, body, headers = CALLS[0]
assert (m, path) == ("PUT", "/api/v0/search/asks/"), (m, path)
assert headers["authorization"] == "Bearer sk-test"
assert body["select_cols"] == ["*"] and "q" in body
assert body["q"]["type"] == "on-demand"
print(f"1. search -> {m} {path}, Bearer auth, select_cols/q body")

# 2. cheapest MATCHING offer wins — not cheapest overall
assert offer.offer_id == 102 and offer.gpu_name == "RTX 3090", offer
print(f"2. picked {offer} (skipped the $0.09 RTX 5000 — not in VAST_GPU_NAMES)")

# 3. an empty name list accepts anything, so the true cheapest wins
any_gpu = run(vast.find_cheapest({}, ()))
assert any_gpu.offer_id == 104, any_gpu
print(f"3. no name filter -> cheapest overall: {any_gpu}")

# 4. no offers is a clear, actionable error
def empty(request): return httpx.Response(200, json={"offers": []})
httpx.AsyncClient = lambda *a, **kw: _real(*a, transport=httpx.MockTransport(empty), **kw)
try:
    run(vast.find_cheapest({}, ()))
except VastError as e:
    assert "VAST_MAX_DPH" in str(e)
    print(f"4. no offers -> actionable error: {str(e)[:58]}...")
httpx.AsyncClient = patched

# 5. create uses PUT /asks/{offer_id}/ and injects the public key
CALLS.clear()
iid = run(vast.create(offer, "pytorch/pytorch:2.4.0", 100, "ssh-ed25519 AAAA test"))
m, path, body, _ = CALLS[0]
assert (m, path) == ("PUT", "/api/v0/asks/102/"), (m, path)
assert iid == 5150
assert body["client_id"] == "me" and body["runtype"] == "ssh" and body["disk"] == 100
assert body["env"]["PUBLIC_KEY"] == "ssh-ed25519 AAAA test"
assert "authorized_keys" in body["onstart"]
assert body["label"] == "i2v-bot"
print(f"5. create -> {m} {path} -> instance {iid}; key injected via env + onstart")

# 6. destroy uses DELETE and never raises, even on API failure
STATE["destroyed"].clear()
assert run(vast.destroy(5150)) is True and STATE["destroyed"] == [5150]
def boom(request): return httpx.Response(500, text="nope")
httpx.AsyncClient = lambda *a, **kw: _real(*a, transport=httpx.MockTransport(boom), **kw)
assert run(vast.destroy(999)) is False      # returns False, does not raise
print("6. destroy -> DELETE; returns False on API error instead of raising")
httpx.AsyncClient = patched

# 7. rent() destroys the instance on the SUCCESS path
STATE["destroyed"].clear()
STATE["instances"] = [{"id": 5150, "ssh_host": "1.2.3.4", "ssh_port": 22, "actual_status": "running"}]
async def noop_ssh(self, inst, timeout=420): return None
VastManager.wait_for_ssh = noop_ssh

async def happy():
    async with vast.rent({}, (), "img", 100, "pub", 60) as (inst, off):
        assert inst.ssh_host == "1.2.3.4"
        return "worked"
assert run(happy()) == "worked"
assert STATE["destroyed"] == [5150], STATE["destroyed"]
print("7. rent() success path -> instance destroyed:", STATE["destroyed"])

# 8. rent() destroys the instance when the body RAISES
STATE["destroyed"].clear()
async def explodes():
    async with vast.rent({}, (), "img", 100, "pub", 60) as (inst, off):
        raise RuntimeError("render blew up")
try:
    run(explodes())
except RuntimeError:
    pass
assert STATE["destroyed"] == [5150], "LEAKED A PAID INSTANCE ON EXCEPTION"
print("8. rent() exception path -> instance still destroyed:", STATE["destroyed"])

# 9. rent() destroys the instance on CANCELLATION too
STATE["destroyed"].clear()
async def cancelled():
    async with vast.rent({}, (), "img", 100, "pub", 60) as (inst, off):
        await asyncio.sleep(3600)
async def driver():
    task = asyncio.create_task(cancelled())
    await asyncio.sleep(0.2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
run(driver())
assert STATE["destroyed"] == [5150], "LEAKED A PAID INSTANCE ON CANCEL"
print("9. rent() cancellation path -> instance still destroyed:", STATE["destroyed"])

# 10. reap_orphans destroys only OUR label, never someone else's instance
STATE["destroyed"].clear()
STATE["instances"] = [
    {"id": 1, "label": "i2v-bot", "ssh_host": "h", "ssh_port": 1, "actual_status": "running"},
    {"id": 2, "label": "someone-elses-training-run", "ssh_host": "h", "ssh_port": 1},
    {"id": 3, "label": "i2v-bot", "ssh_host": "h", "ssh_port": 1},
]
reaped = run(vast.reap_orphans())
assert reaped == [1, 3], reaped
assert 2 not in STATE["destroyed"], "DESTROYED AN UNRELATED INSTANCE"
print(f"10. reap_orphans destroyed {reaped}, left the unlabelled instance alone")

print("\nALL VAST TESTS PASSED")
