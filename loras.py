"""
Per-request LoRA loading for Wan 2.2.

Wan 2.2 A14B is a two-expert MoE: `transformer` denoises the high-noise
timesteps and `transformer_2` the low-noise ones, switched at the pipeline's
`boundary_ratio`. Diffusers loads a LoRA into the *first* denoiser only unless
`load_into_transformer_2=True` is passed, so a single load_lora_weights() call
leaves the LoRA inactive for roughly half the denoising schedule — the usual
cause of "my LoRA barely does anything" on Wan 2.2. We always load into both.
"""

import os
import re

LORA_DIR = os.environ.get("LORA_DIR", "/runpod-volume/loras")
MAX_LORAS = int(os.environ.get("MAX_LORAS", "4"))

# Adapter names are derived from caller-supplied filenames, so keep them to a
# conservative character set and reject anything with path separators.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

# Adapter names currently registered on the pipeline, so repeat requests for
# the same LoRA re-use the loaded weights instead of re-reading from disk.
_loaded: set[str] = set()


class LoraError(ValueError):
    """Raised for a malformed or missing LoRA request."""


def available() -> list[str]:
    """List LoRA files present on the volume, without extensions."""
    if not os.path.isdir(LORA_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(LORA_DIR)
        if f.endswith(".safetensors")
    )


def _resolve(name: str) -> str:
    """Map a request-supplied LoRA name to a path inside LORA_DIR."""
    if not isinstance(name, str) or not _SAFE_NAME.match(name):
        raise LoraError(
            f"Invalid LoRA name {name!r}: use only letters, digits, '.', '_' and '-'."
        )

    path = os.path.join(LORA_DIR, f"{name}.safetensors")
    # Belt and braces against traversal via '..' surviving the regex.
    if os.path.dirname(os.path.abspath(path)) != os.path.abspath(LORA_DIR):
        raise LoraError(f"Invalid LoRA name {name!r}.")
    if not os.path.isfile(path):
        raise LoraError(
            f"LoRA {name!r} not found. Available: {available() or 'none installed'}"
        )
    return path


def parse_request(raw) -> list[tuple[str, float]]:
    """
    Normalise the `loras` input field into (name, scale) pairs.

    Accepts ["name"] or [{"name": "x", "scale": 0.8}] or a bare "name".
    """
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise LoraError("'loras' must be a list of names or {name, scale} objects.")
    if len(raw) > MAX_LORAS:
        raise LoraError(f"At most {MAX_LORAS} LoRAs may be applied per request.")

    parsed = []
    for entry in raw:
        if isinstance(entry, str):
            parsed.append((entry, 1.0))
        elif isinstance(entry, dict):
            name = entry.get("name")
            if not name:
                raise LoraError("Each LoRA object needs a 'name' field.")
            try:
                scale = float(entry.get("scale", 1.0))
            except (TypeError, ValueError):
                raise LoraError(f"LoRA {name!r} has a non-numeric 'scale'.")
            parsed.append((name, scale))
        else:
            raise LoraError("'loras' entries must be strings or objects.")
    return parsed


def apply(pipe, requested: list[tuple[str, float]]) -> list[str]:
    """
    Activate exactly `requested` on the pipeline and return the applied names.

    Weights stay resident across jobs; only the active set changes, so a worker
    serving alternating LoRAs does not pay the load cost every time.
    """
    if not requested:
        if _loaded:
            pipe.disable_lora()
        return []

    names = [name for name, _ in requested]
    scales = [scale for _, scale in requested]

    for name in names:
        if name in _loaded:
            continue
        path = _resolve(name)
        # Both calls use the same adapter_name; set_adapters() then activates it
        # across both denoisers (it handles Wan 2.2's two-transformer layout).
        pipe.load_lora_weights(path, adapter_name=name)
        pipe.load_lora_weights(path, adapter_name=name, load_into_transformer_2=True)
        _loaded.add(name)

    pipe.set_adapters(names, adapter_weights=scales)
    pipe.enable_lora()
    return names
