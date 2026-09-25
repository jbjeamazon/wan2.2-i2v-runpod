"""
Narration. Two local engines, chosen by TTS_ENGINE.

  kokoro      82M params, 54 built-in voices, runs on CPU, Apache-2.0.
              Cannot clone a voice.
  chatterbox  Clones from ~10s of reference audio, MIT, wants a GPU.

Licensing is worth checking before you build on one: Kokoro and Chatterbox are
permissive, while XTTS-v2 (CPML) and F5-TTS weights (CC-BY-NC) are
non-commercial, which matters if the output is ever monetised.
"""

from __future__ import annotations

from pathlib import Path

SAMPLE_RATE = 24000


class TTSError(RuntimeError):
    pass


def _write_wav(samples, path: Path, rate: int = SAMPLE_RATE) -> Path:
    import soundfile as sf
    sf.write(str(path), samples, rate)
    return path


def _kokoro(text: str, out: Path, voice: str, speed: float) -> Path:
    try:
        from kokoro import KPipeline
    except ImportError:
        raise TTSError("Kokoro not installed: pip install kokoro soundfile")

    import numpy as np
    # Lang code is the voice prefix: 'a' American, 'b' British, and so on.
    pipeline = KPipeline(lang_code=voice[0] if voice else "a")
    chunks = [audio for _, _, audio in pipeline(text, voice=voice, speed=speed)]
    if not chunks:
        raise TTSError("Kokoro produced no audio")
    return _write_wav(np.concatenate(chunks), out)


def _chatterbox(text: str, out: Path, ref_audio: str) -> Path:
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError:
        raise TTSError("Chatterbox not installed: pip install chatterbox-tts")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=device)
    wav = model.generate(text, audio_prompt_path=ref_audio) if ref_audio else model.generate(text)
    import soundfile as sf
    sf.write(str(out), wav.squeeze(0).cpu().numpy(), model.sr)
    return out


def speak(cfg, text: str, out: Path) -> Path:
    """Render `text` to a wav at `out`."""
    if not text.strip():
        raise TTSError("Nothing to speak")
    out.parent.mkdir(parents=True, exist_ok=True)

    engine = cfg.tts_engine.lower()
    if engine == "kokoro":
        return _kokoro(text, out, cfg.tts_voice, cfg.tts_speed)
    if engine == "chatterbox":
        return _chatterbox(text, out, cfg.tts_ref_audio)
    raise TTSError(f"Unknown TTS_ENGINE {cfg.tts_engine!r}; use kokoro or chatterbox")
