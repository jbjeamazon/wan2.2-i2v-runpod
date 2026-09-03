# Wan2.2 Image-to-Video — RunPod Serverless

Generate cinematic videos from a single image using **[Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)** — a 27B Mixture-of-Experts model (14B active parameters) that supports 480P and 720P output.

---

## Hardware Requirements

| Resolution | Recommended GPU       | Min VRAM |
|------------|-----------------------|----------|
| 480P       | A100 80GB / H100 80GB | ~40 GB   |
| 720P       | A100 80GB / H100 80GB | ~80 GB   |
| 480P (CPU offload) | RTX 4090 (24 GB) | ~24 GB |

---

## API Input

Send a POST request to your endpoint with a JSON body:

```json
{
  "input": {
    "image_url":           "https://example.com/cat.jpg",
    "prompt":              "A cat gently swishing its tail on a wooden floor.",
    "negative_prompt":     "low quality, blurry, watermark",
    "resolution":          "480p",
    "num_frames":          81,
    "guidance_scale":      3.5,
    "num_inference_steps": 40,
    "fps":                 16,
    "seed":                42,
    "loras":               [{"name": "my_lora", "scale": 0.8}]
  }
}
```

Generation takes several minutes, so submit with `/run` and poll `/status/{id}`
rather than holding a `/runsync` connection open.

### Input Fields

| Field                  | Type    | Default      | Description                                                  |
|------------------------|---------|--------------|--------------------------------------------------------------|
| `image`                | string  | *see note*   | Base64-encoded input image (JPEG or PNG). Data-URI prefix optional. |
| `image_url`            | string  | *see note*   | HTTP(S) URL of the input image. Use this instead of `image` to stay under the request payload cap. |
| `prompt`               | string  | `""`         | Text description of the desired motion/scene.                |
| `negative_prompt`      | string  | (see below)  | Concepts to avoid in the generated video.                    |
| `resolution`           | string  | `"480p"`     | Output resolution: `"480p"` or `"720p"`.                     |
| `num_frames`           | integer | `81`         | Number of video frames (~5 s at 16 fps).                     |
| `guidance_scale`       | float   | `3.5`        | Classifier-free guidance strength (high-noise expert).       |
| `guidance_scale_2`     | float   | = `guidance_scale` | Guidance for Wan 2.2's low-noise expert. See *Two experts* below. |
| `num_inference_steps`  | integer | `40`         | Denoising steps (more = higher quality, slower).             |
| `fps`                  | integer | `16`         | Frames per second of the exported MP4.                       |
| `seed`                 | integer | random       | RNG seed for reproducible results.                           |
| `loras`                | array   | `[]`         | LoRAs to apply, as `["name"]` or `[{"name": "x", "scale": 0.8}]`. See *LoRAs*. |
| `action`               | string  | —            | Set to `"list_loras"` to return the installed LoRA names without generating. |

One of `image` or `image_url` is required. If both are given, `image_url` wins.

**Default negative prompt:**
```
low quality, blurry, jittery, distorted, static, overexposed,
watermark, text, logo, artifacts, worst quality, bad anatomy
```

---

## API Output

With object storage configured (recommended):

```json
{
  "video_url":  "https://your-bucket.example.com/videos/8f3a....mp4",
  "width":      832,
  "height":     480,
  "num_frames": 81,
  "fps":        16,
  "resolution": "480p",
  "loras":      ["my_lora"],
  "seed":       42
}
```

Without it, the MP4 comes back inline as `video_base64` instead:

```python
import base64

data = response["output"]["video_base64"]
with open("output.mp4", "wb") as f:
    f.write(base64.b64decode(data))
```

> **Configure object storage for anything beyond short test clips.** RunPod caps
> job payloads at 10 MB on `/run` (20 MB on `/runsync`) and base64 inflates bytes
> by about a third, so a full-length 720p clip cannot be returned inline. The
> handler rejects an oversized inline video with an explicit error rather than
> returning a truncated one. Set `S3_BUCKET`, `S3_ACCESS_KEY_ID` and
> `S3_SECRET_ACCESS_KEY` and it returns a presigned URL instead.

---

## Quick-Start — Python Client

```python
import runpod, base64, time

runpod.api_key = "YOUR_RUNPOD_API_KEY"

with open("my_image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

run = endpoint.run({
    "image":               image_b64,
    "prompt":              "Ocean waves gently rolling onto a sandy shore at sunset.",
    "resolution":          "480p",
    "num_frames":          81,
    "num_inference_steps": 40,
    "guidance_scale":      3.5,
    "fps":                 16,
    "seed":                42,
})

output = run.output(timeout=1800)

if "video_url" in output:
    print("Video:", output["video_url"])
else:
    with open("output.mp4", "wb") as f:
        f.write(base64.b64decode(output["video_base64"]))
    print("Saved output.mp4")
```

---

## Environment Variables

| Variable                | Default                             | Description                                      |
|-------------------------|-------------------------------------|--------------------------------------------------|
| `MODEL_ID`              | `Wan-AI/Wan2.2-I2V-A14B-Diffusers`  | HuggingFace model ID (locked; do not change).    |
| `RESOLUTION`            | `480p`                              | Default resolution when not specified per-request.|
| `ENABLE_CPU_OFFLOAD`    | `false`                             | Set `true` to enable CPU offload for lower VRAM. |
| `HF_TOKEN`              | _(empty)_                           | HuggingFace token for private/gated models.      |
| `LORA_DIR`              | `/runpod-volume/loras`              | Directory of `.safetensors` LoRA files.          |
| `MAX_LORAS`             | `4`                                 | Maximum LoRAs applied per request.               |
| `S3_BUCKET`             | _(empty)_                           | Bucket for generated videos. Enables URL output. |
| `S3_ENDPOINT_URL`       | _(empty)_                           | S3-compatible endpoint. Empty for AWS S3; set for R2/B2/MinIO. |
| `S3_REGION`             | `auto`                              | Bucket region (`auto` for Cloudflare R2).        |
| `S3_ACCESS_KEY_ID`      | _(empty)_                           | Bucket access key.                               |
| `S3_SECRET_ACCESS_KEY`  | _(empty)_                           | Bucket secret key.                               |
| `S3_PUBLIC_URL_BASE`    | _(empty)_                           | Public CDN base. When set, returns permanent public URLs instead of expiring presigned ones. |
| `S3_URL_EXPIRY`         | `86400`                             | Presigned URL lifetime in seconds.               |

---

## Repository Structure

```
.
├── .runpod/
│   ├── hub.json        # RunPod Hub metadata & deployment config
│   └── tests.json      # Automated test cases
├── openclaw-mcp/
│   ├── server.py       # MCP server exposing the endpoint to OpenClaw
│   └── requirements.txt
├── handler.py          # RunPod serverless handler
├── loras.py            # Per-request LoRA loading (both Wan 2.2 experts)
├── storage.py          # S3-compatible upload + presigned URLs
├── Dockerfile          # Container build definition
├── icon.png            # Hub listing icon
└── README.md           # This file
```

---

## LoRAs

Drop `.safetensors` LoRA files into `LORA_DIR` (a mounted network volume is the
practical choice — it keeps model changes out of the Docker image) and reference
them by filename, without the extension:

```json
{"input": {"image_url": "...", "loras": [{"name": "my_lora", "scale": 0.8}]}}
```

`{"input": {"action": "list_loras"}}` returns what is installed. Weights stay
resident on the worker between jobs, so alternating between LoRAs does not pay
the load cost each time.

### Two experts, one LoRA

Wan 2.2 A14B is a mixture-of-experts model with two denoisers: `transformer`
handles the high-noise timesteps and `transformer_2` the low-noise ones, split
at the pipeline's `boundary_ratio`. Diffusers loads a LoRA into the **first
denoiser only** unless `load_into_transformer_2=True` is passed, which leaves it
inactive for roughly half the denoising schedule — the usual cause of "my LoRA
barely does anything" on Wan 2.2. `loras.py` always loads into both.

The same split is why `guidance_scale_2` exists: it sets guidance for the
low-noise expert, and defaults to whatever `guidance_scale` is.

---

## OpenClaw Integration

`openclaw-mcp/server.py` is an MCP server that exposes this endpoint as three
tools: `generate_video`, `check_video_job` and `list_loras`.

```bash
pip install -r openclaw-mcp/requirements.txt
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=...
python openclaw-mcp/server.py
```

Register it with OpenClaw as an MCP server (see the
[OpenClaw MCP docs](https://docs.openclaw.ai/cli/mcp) for the current CLI
syntax), passing `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` through the server's
environment rather than storing them in a skill file.

**Submission and polling are deliberately separate tools.** Generation takes
minutes; a single blocking tool would stall the agent's turn and time out on
most clients. `generate_video` returns a `job_id` immediately and
`check_video_job` reports progress, so the agent can reply "started it" and
follow up when the video lands.

`generate_video` accepts either a URL or a local file path for `image` — a URL
is passed straight through, and a local file is base64-encoded client-side. When
the endpoint has no object storage configured, `check_video_job` reports that
rather than returning the base64 blob, which would otherwise flood the agent's
context with several megabytes of useless text.

---

## Model License

The Wan2.2 model is released under the [Wan Community License](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers/blob/main/LICENSE). Please review the license terms before commercial use.

---

[![RunPod Hub](https://api.runpod.io/badge/YOUR_USERNAME/YOUR_REPO)](https://runpod.io/hub)
