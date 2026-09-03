"""Exercise the MCP server's request shaping with mcp/httpx stubbed."""
import sys, os, types, asyncio, tempfile, base64

def mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m
    return m

CALLS = []
class FakeResp:
    def __init__(self, data): self._d = data
    def raise_for_status(self): pass
    def json(self): return self._d

RESPONSES = {}
class FakeClient:
    def __init__(self, timeout=None): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def post(self, url, headers=None, json=None):
        CALLS.append(("POST", url, json))
        return FakeResp(RESPONSES.get(url, {"id": "job-123", "status": "IN_QUEUE"}))
    async def get(self, url, headers=None):
        CALLS.append(("GET", url, None))
        return FakeResp(RESPONSES.get(url, {"status": "IN_PROGRESS"}))

mod("httpx", AsyncClient=FakeClient)

class FakeMCP:
    def __init__(self, name): self.name = name; self.tools = {}
    def tool(self):
        def deco(fn): self.tools[fn.__name__] = fn; return fn
        return deco
    def run(self): pass
mod("mcp")
mod("mcp.server")
mod("mcp.server.fastmcp", FastMCP=FakeMCP)

os.environ["WAN_BACKEND"] = "runpod"
os.environ["RUNPOD_API_KEY"] = "test-key"
os.environ["RUNPOD_ENDPOINT_ID"] = "ep123"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "openclaw-mcp"))
import server

run = asyncio.run

# 1. remote image -> image_url, no base64 blowup
CALLS.clear()
r = run(server.generate_video("https://example.com/cat.jpg", prompt="tail swish"))
payload = CALLS[0][2]["input"]
assert payload["image_url"] == "https://example.com/cat.jpg" and "image" not in payload
assert r["job_id"] == "job-123"
print("1. remote image -> image_url; returned job_id immediately:", r["job_id"])

# 2. local file -> base64 data URI
tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
tmp.write(b"pngbytes"); tmp.close()
CALLS.clear()
run(server.generate_video(tmp.name, prompt="x"))
payload = CALLS[0][2]["input"]
assert payload["image"].startswith("data:image/png;base64,")
assert base64.b64decode(payload["image"].split(",")[1]) == b"pngbytes"
print("2. local file -> base64 data URI, round-trips correctly")

# 3. missing local file is a clear error
try:
    run(server.generate_video("/nope/missing.png"))
except ValueError as e:
    print("3. missing file rejected:", e)
else:
    sys.exit("FAIL: missing file accepted")

# 4. optional args omitted stay out of the payload
CALLS.clear()
run(server.generate_video("https://e.com/a.jpg"))
payload = CALLS[0][2]["input"]
assert "seed" not in payload and "negative_prompt" not in payload and "loras" not in payload
print("4. omitted optionals absent from payload; sent keys:", sorted(payload))

# 5. loras + seed passed through
CALLS.clear()
run(server.generate_video("https://e.com/a.jpg", seed=7, loras=["remix"]))
payload = CALLS[0][2]["input"]
assert payload["seed"] == 7 and payload["loras"] == ["remix"]
print("5. seed + loras passed through")

# 6. in-progress poll
RESPONSES["https://api.runpod.ai/v2/ep123/status/job-123"] = {"status": "IN_PROGRESS"}
r = run(server.check_video_job("job-123"))
assert r["status"] == "IN_PROGRESS" and "video_url" not in r
print("6. in-progress poll ->", r["status"])

# 7. completed poll returns the URL, never a base64 blob
RESPONSES["https://api.runpod.ai/v2/ep123/status/job-123"] = {
    "status": "COMPLETED",
    "output": {"video_url": "https://cdn/x.mp4", "resolution": "480p",
               "num_frames": 81, "loras": ["remix"], "video_base64": "AAAA" * 9999},
}
r = run(server.check_video_job("job-123"))
assert r["video_url"] == "https://cdn/x.mp4"
assert "video_base64" not in str(r), "base64 must never reach the model context"
print("7. completed ->", r["video_url"], "| base64 stripped from context")

# 8. handler-level error surfaces as FAILED
RESPONSES["https://api.runpod.ai/v2/ep123/status/job-123"] = {
    "status": "COMPLETED", "output": {"error": "LoRA 'x' not found."}}
r = run(server.check_video_job("job-123"))
assert r["status"] == "FAILED" and "not found" in r["error"]
print("8. handler error surfaced as:", r["status"], "-", r["error"])

# 9. list_loras uses runsync
RESPONSES["https://api.runpod.ai/v2/ep123/runsync"] = {
    "status": "COMPLETED", "output": {"loras": ["remix", "lightning"]}}
CALLS.clear()
r = run(server.list_loras())
assert r["loras"] == ["remix", "lightning"]
assert CALLS[0][2] == {"input": {"action": "list_loras"}}
print("9. list_loras ->", r["loras"])

# 10. the same module drives a local backend with only env changes
for m in list(sys.modules):
    if m == "server":
        del sys.modules[m]
os.environ["WAN_BACKEND"] = "local"
os.environ["WAN_LOCAL_URL"] = "http://127.0.0.1:8080"
os.environ["WAN_API_KEY"] = "s3cret"
import server as local_server
assert local_server.BASE_URL == "http://127.0.0.1:8080"
assert local_server.HEADERS["Authorization"] == "Bearer s3cret"
print("10. WAN_BACKEND=local -> base", local_server.BASE_URL, "| auth header set from WAN_API_KEY")

CALLS.clear()
run(local_server.generate_video("https://e.com/a.jpg", prompt="x"))
assert CALLS[0][1] == "http://127.0.0.1:8080/run", CALLS[0][1]
print("11. local backend posts to", CALLS[0][1], "- same code path, no RunPod")

# 12. a local server with no key sends no Authorization header
for m in list(sys.modules):
    if m == "server":
        del sys.modules[m]
del os.environ["WAN_API_KEY"]
import server as open_local
assert "Authorization" not in open_local.HEADERS
print("12. no WAN_API_KEY -> no Authorization header sent")

# 13. an unknown backend fails loudly at startup
for m in list(sys.modules):
    if m == "server":
        del sys.modules[m]
os.environ["WAN_BACKEND"] = "azure"
try:
    import server as bad
except SystemExit as e:
    print("13. unknown backend rejected:", str(e)[:56])
else:
    sys.exit("FAIL: unknown backend accepted")

print("\nALL MCP TESTS PASSED")
