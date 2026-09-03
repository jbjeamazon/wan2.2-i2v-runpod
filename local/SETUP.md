# From bought GPU to always-on server

## First: a GPU is not a computer

A graphics card is a component. It has no CPU, no RAM, no storage, no operating
system — it plugs into a PCIe slot inside a desktop PC and does nothing on its
own. So there are two paths:

**You already own a desktop PC.** Check three things before buying anything:
a free full-length PCIe x16 slot, a power supply with enough headroom (850 W+
for a 4090, 1000 W+ for a 5090) and the right connectors, and physical clearance
— these cards are ~13 inches long and three slots thick. If all three hold, the
card drops in and you're done.

**You don't.** You're building or buying a whole machine. Budget roughly
$700–1,200 on top of the card:

| Part | What matters | Rough cost |
|------|--------------|-----------|
| CPU | Anything modern; inference is GPU-bound | $150–300 |
| Motherboard | One PCIe x16 slot, 4 RAM slots | $120–200 |
| **RAM** | **64 GB.** Offload modes push weights into system RAM | $150–200 |
| SSD | 1 TB NVMe minimum — weights alone are ~60 GB | $70–120 |
| PSU | 850–1000 W, 80+ Gold, correct connectors | $120–200 |
| Case | Must physically fit the card | $80–150 |

RAM is the one people under-buy. On a 24 GB card the server offloads to system
memory constantly; 32 GB will thrash and 16 GB will be killed by the OOM reaper.

A prebuilt gaming PC with the GPU already installed is a legitimate shortcut if
you'd rather not assemble one, though you usually pay a premium.

## 1. Operating system

**Ubuntu 24.04 LTS.** Driver support is best, and everything below assumes it.

Windows works via WSL2 but adds a virtualisation layer, complicates GPU
passthrough, and makes the always-on service step harder. If this box is
dedicated to generation, install Linux on it.

## 2. NVIDIA driver

```bash
sudo ubuntu-drivers install
sudo reboot
nvidia-smi          # should print your card and driver version
```

You do **not** need a separate CUDA toolkit install — the PyTorch wheels bundle
what they need. You only need the driver.

## 3. The code

```bash
sudo apt update && sudo apt install -y python3-venv git ffmpeg
git clone <your-private-repo> ~/wan22-i2v
cd ~/wan22-i2v
python3 -m venv .venv && source .venv/bin/activate

# PyTorch first, matched to your driver's CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r local/requirements.txt
```

## 4. Preflight

```bash
python3 local/preflight.py
```

Checks the driver, whether PyTorch actually sees the GPU, VRAM and which mode
it selects, system RAM, free disk, dependencies and the port. **Run this before
the first start** — it catches the common mistakes (CPU-only torch wheel, too
little RAM, not enough disk) in seconds rather than after a 60 GB download.

## 5. First run

```bash
export LORA_DIR=~/wan22-i2v/loras
python local/server.py
```

The first start downloads ~60 GB of weights into `~/.cache/huggingface`. Expect
10–40 minutes depending on your connection. This is a one-time cost and the only
outbound traffic the server ever makes — generation itself is entirely local.

Watch it come up:

```bash
curl localhost:8080/health
# {"status":"ok","model_loaded":true,"vram_gb":24.0,"vram_mode":"balanced",...}
```

Then generate something short to confirm the whole path works:

```bash
curl -X POST localhost:8080/run -H 'Content-Type: application/json' \
  -d '{"input":{"image_url":"https://example.com/photo.jpg","prompt":"gentle motion","num_frames":17,"num_inference_steps":10}}'
curl localhost:8080/status/<id>
```

## 6. Make it standalone

Right now it dies when you close the terminal. `local/wan-i2v.service` turns it
into a system service that starts on boot and restarts on failure:

```bash
sudo cp local/wan-i2v.service /etc/systemd/system/
sudo nano /etc/systemd/system/wan-i2v.service    # set User and the two paths
sudo nano /etc/wan-i2v.env                       # WAN_API_KEY=..., LORA_DIR=...
sudo systemctl daemon-reload
sudo systemctl enable --now wan-i2v
```

```bash
systemctl status wan-i2v      # is it running
journalctl -u wan-i2v -f      # live logs
sudo systemctl restart wan-i2v
```

From here the machine boots, the server comes up, and it stays up. That's the
"standalone thing" — a box on your desk you don't interact with directly.

## 7. Point OpenClaw at it

**OpenClaw on the same machine** — nothing else to do:

```bash
export WAN_BACKEND=local
export WAN_LOCAL_URL=http://127.0.0.1:8080
python openclaw-mcp/server.py
```

**OpenClaw on a different machine** (laptop, home server): the GPU box binds to
loopback by default, so it is not reachable across your network yet. Two options,
in order of preference:

1. **Tailscale** (recommended). Install on both machines, and they get private
   addresses reachable from anywhere without opening a single port to the
   internet. Set `WAN_HOST=0.0.0.0` and `WAN_API_KEY=<something long>`, then
   point `WAN_LOCAL_URL` at the GPU box's Tailscale address.
2. **LAN only.** Set `WAN_HOST=0.0.0.0` and a `WAN_API_KEY`, and reach it at the
   box's local IP. Works at home, not away from it.

**Never port-forward this to the open internet.** The server will warn you if
you bind it publicly without a key set, but the warning is not a substitute for
not doing it.

## Troubleshooting

| Symptom | Cause |
|---------|-------|
| `preflight` says torch sees no CUDA device | You installed the CPU-only wheel. Reinstall with the `--index-url` above. |
| CUDA out of memory | Force `VRAM_MODE=low`, or lower `num_frames` / use 480p. |
| Process killed during generation | System RAM, not VRAM. Offload needs 64 GB. |
| Very slow (20+ min per clip) | Probably in `low` mode — check `/health`. |
| `nvidia-smi` works, service won't start | Wrong `User=` or paths in the unit file. Check `journalctl -u wan-i2v`. |
