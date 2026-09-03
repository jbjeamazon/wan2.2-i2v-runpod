# Tests

These run CPU-only: `stubs.py` fakes torch, diffusers and PIL so the repo's real
control flow (validation, VRAM tiering, LoRA handling, routing, auth, delivery)
can be exercised without a GPU or the ~60 GB of weights.

```bash
pip install fastapi 'httpx<1'
for t in tests/test_*.py; do python "$t" || break; done
```

`test_local_server.py` uses FastAPI's real TestClient against the real routes.
The others drive the modules directly. What they do *not* cover is anything
requiring actual inference — image quality, VRAM headroom, generation speed.
Those need real hardware.
