# Buying hardware for this project

The workload is Wan 2.2 A14B image-to-video: ~54 GB of bfloat16 transformer
weights, 40 denoising steps over 81 frames. Two properties drive every buying
decision.

**VRAM is a hard floor, not a preference.** `local/preflight.py` picks a memory
strategy from the card it finds:

| VRAM | Mode | Reality |
|------|------|---------|
| ≥ 48 GB | `high` | Weights resident. Fastest. |
| 24–32 GB | `balanced` | Module offload + VAE tiling. **The target.** |
| 12–16 GB | `low` | Per-layer offload. Works, ~20+ min per clip. |
| < 12 GB | `low` | Painful. Avoid. |

**Video diffusion is compute-bound, unlike LLM inference which is
memory-bandwidth-bound.** This is why "AI PCs" marketed on huge unified memory
(Strix Halo, Apple Silicon) disappoint here: they can hold the model but not
push the FLOPs. It is also why CUDA matters — Wan on Metal is experimental and
frequently broken.

## The ranking that actually matters

For this workload VRAM outranks GPU tier. A 16 GB RTX 5060 Ti beats a 12 GB
RTX 5070 despite the 5070 being the faster card, because a model that does not
fit does not run.

| GPU | VRAM | Verdict |
|-----|------|---------|
| RTX 3090 (used) | 24 GB | Best value. `balanced` mode, CUDA, ~$850. |
| RTX 4090 (used) | 24 GB | Same VRAM, meaningfully faster, ~$1,200–1,800. |
| RTX 5090 | 32 GB | Ideal, but street price runs well over MSRP. |
| RTX 5060 Ti **16 GB** | 16 GB | Entry point that works. `low` mode. |
| RTX 5070 Ti / 4070 Ti Super | 16 GB | Faster than the 5060 Ti, same ceiling. |
| RTX 5070 | 12 GB | Marginal. Skip for this. |
| RTX 5060 Ti **8 GB** | 8 GB | **Trap.** Same model name, half the VRAM. |

The 5060 Ti ships in 8 GB and 16 GB versions under nearly identical listings.
Check the number before buying.

## Whole-system rules

- **64 GB system RAM.** Offload modes push weights into system memory. 32 GB
  thrashes; 16 GB gets killed by the OOM reaper. This is the most commonly
  under-bought part.
- **PSU headroom.** 850 W+ for a 3090 or 4090, 1000 W+ for a 5090. A prebuilt
  shipped with a mid-range card often has a 650 W supply — too small if you
  later swap in a 3090.
- **1 TB NVMe minimum.** Weights alone are ~60 GB.
- **Physical clearance.** A 3090/4090 is ~13 inches long and three slots thick.
- **Linux.** Ubuntu 24.04 LTS. You need the NVIDIA driver, not a separate CUDA
  toolkit — the PyTorch wheels bundle that.

## Two ways to spend ~$1,500

**A. Cheap prebuilt + used 3090.** A $700–900 prebuilt, then swap the GPU for a
used 24 GB RTX 3090. Gets you into `balanced` mode, which is the difference
between minutes and tens of minutes per clip. Verify the PSU can take it, and
budget for an upgrade if not. Sell the original card to offset.

**B. Prebuilt with a 16 GB card.** Simpler, no assembly, works out of the box —
but you are in `low` mode permanently, at roughly 20+ minutes per clip.

A is the better machine; B is the lower-effort one.

## Verify before you commit

Run this on any candidate machine before downloading 60 GB of weights:

```bash
python3 local/preflight.py
```

It reports the driver, whether PyTorch actually sees the GPU, detected VRAM and
the resulting mode, system RAM, free disk, and dependencies. On a Mac it fails
immediately with "torch sees no CUDA device" — which is the answer, delivered in
two seconds instead of after an afternoon.

## Buying used

Cards coming off AI workloads show wear that ordinary testing misses, and the
3090's GDDR6X runs hot enough that degraded thermal pads are common. Before
accepting one:

- `memtest_vulkan` for 15–30 minutes — require **zero** errors
- 20 minutes of sustained load while watching the memory junction temperature
- Listen for fan bearing grind
- Buy somewhere with a returns window
