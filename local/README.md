# Local Wan 2.2 image-to-video server

Runs the same model as the RunPod worker, on your own hardware. Nothing leaves
the machine: no managed queue, no object storage, no third-party job records.

The API deliberately mirrors RunPod's (`/run`, `/runsync`, `/status/{id}`), so
`openclaw-mcp/server.py` drives either backend with only environment changes.

## Setup

Buying hardware? **[HARDWARE.md](HARDWARE.md)** covers what to get and why.
Already have a machine? **[SETUP.md](SETUP.md)** walks the whole path: what hardware you
need around the GPU, drivers, install, and running it as an always-on service.
Run `python3 local/preflight.py` to check a machine before you start.

The short version:

```bash
# 1. PyTorch for your CUDA version, first and separately
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. Everything else
pip install -r local/requirements.txt

# 3. Run it
python local/server.py
```

First start downloads ~60 GB of weights from HuggingFace into `~/.cache/huggingface`.
That is the only outbound traffic; generation itself is entirely local.

## GPU tiers

`VRAM_MODE` defaults to `auto`, which picks a strategy from the detected card:

| Detected VRAM | Mode       | How it runs                                    |
|---------------|------------|------------------------------------------------|
| ≥ 48 GB       | `high`     | Weights resident on the GPU. Fastest.          |
| 20–48 GB      | `balanced` | Whole modules moved to the GPU on demand, VAE tiling. |
| < 20 GB       | `low`      | Per-layer offload, VAE tiling. Slowest, but it fits. |

Override with `VRAM_MODE=balanced` if auto-detection guesses wrong. On a 24 GB
card, expect roughly 8–12 minutes for an 81-frame 480p clip; a 32 GB card is
substantially quicker and can hold 720p more comfortably.

## Configuration

| Variable          | Default                 | Description                                        |
|-------------------|-------------------------|----------------------------------------------------|
| `WAN_HOST`        | `127.0.0.1`             | Bind address. **Leave as-is** unless you mean to expose it. |
| `WAN_PORT`        | `8080`                  | Port.                                              |
| `WAN_API_KEY`     | _(empty)_               | If set, all routes except `/health` require `Authorization: Bearer <key>`. |
| `WAN_PUBLIC_BASE` | `http://WAN_HOST:PORT`  | Base URL used to build returned `video_url`s.      |
| `OUTPUT_DIR`      | `./outputs`             | Where MP4s are written.                            |
| `RETAIN_HOURS`    | `0`                     | Delete generated videos older than N hours. `0` keeps them forever. |
| `VRAM_MODE`       | `auto`                  | `auto`, `high`, `balanced` or `low`.               |
| `LORA_DIR`        | `/runpod-volume/loras`  | Set this to a local path, e.g. `./loras`.          |
| `MODEL_ID`        | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | Model to load.                  |

The server binds to loopback by default, so it is unreachable from your network
as shipped. If you change `WAN_HOST` it will warn you when no `WAN_API_KEY` is
set — at that point anyone who can reach the port can generate video.

## Use

```bash
curl -X POST localhost:8080/run -H 'Content-Type: application/json' -d '{
  "input": {
    "image_url": "https://example.com/cat.jpg",
    "prompt": "A cat gently swishing its tail",
    "num_frames": 81,
    "loras": [{"name": "my_lora", "scale": 0.8}]
  }
}'
# -> {"id": "...", "status": "IN_QUEUE"}

curl localhost:8080/status/<id>
# -> {"status": "COMPLETED", "output": {"video_url": "http://127.0.0.1:8080/videos/<id>.mp4", ...}}
```

`GET /health` reports detected VRAM, the chosen mode, whether weights are
loaded, and the queue depth — useful while the model is still starting up.

## Point OpenClaw at it

```bash
export WAN_BACKEND=local
export WAN_LOCAL_URL=http://127.0.0.1:8080
export WAN_API_KEY=...        # only if the server sets one
python openclaw-mcp/server.py
```

That is the whole migration from RunPod. The tool definitions, submit/poll
split and LoRA handling are unchanged.

## Notes

One GPU means one job at a time, so requests are queued and served FIFO by a
single worker. Job records live in memory only — restarting the server forgets
its history, which is what you want on a private box. Generated files persist
on disk until you delete them or set `RETAIN_HOURS`.
