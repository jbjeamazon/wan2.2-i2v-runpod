"""Settings for the story-to-video pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Vertical short-form. 1080x1920 is what TikTok/Reels/Shorts expect.
WIDTH, HEIGHT = 1080, 1920

# Wan 2.2's native window, shared with the rest of the repo.
SEGMENT_SECONDS = 81 / 16


def _s(name, default=None):
    v = os.environ.get(name) or default
    if v is None:
        raise RuntimeError(f"{name} is not set")
    return v


def _i(name, default):
    return int(os.environ.get(name) or default)


def _f(name, default):
    return float(os.environ.get(name) or default)


@dataclass(frozen=True)
class Config:
    # Script generation — any OpenAI-compatible server. Ollama, llama.cpp and
    # vLLM all expose this, which keeps the whole pipeline local and unmetered.
    llm_base: str = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    llm_model: str = os.environ.get("LLM_MODEL", "llama3.1:8b")
    llm_key: str = os.environ.get("LLM_API_KEY", "not-needed")

    # Image-to-video: the local server already in this repo.
    i2v_base: str = os.environ.get("WAN_LOCAL_URL", "http://127.0.0.1:8080")
    i2v_key: str = os.environ.get("WAN_API_KEY", "")
    i2v_steps: int = _i("I2V_STEPS", 4)
    i2v_loras: str = os.environ.get("I2V_LORAS", "")

    # Text-to-image keyframes: a local ComfyUI or diffusers server.
    t2i_base: str = os.environ.get("T2I_BASE_URL", "http://127.0.0.1:7860")
    t2i_model: str = os.environ.get("T2I_MODEL", "")

    # Voice.
    tts_engine: str = os.environ.get("TTS_ENGINE", "kokoro")
    tts_voice: str = os.environ.get("TTS_VOICE", "af_heart")
    tts_speed: float = _f("TTS_SPEED", 1.0)
    tts_ref_audio: str = os.environ.get("TTS_REF_AUDIO", "")

    # Captions.
    whisper_model: str = os.environ.get("WHISPER_MODEL", "base")
    caption_words: int = _i("CAPTION_WORDS_ON_SCREEN", 4)
    caption_font: str = os.environ.get("CAPTION_FONT", "DejaVu Sans")
    caption_size: int = _i("CAPTION_SIZE", 96)
    caption_fill: str = os.environ.get("CAPTION_FILL", "&H00FFFFFF")
    caption_active: str = os.environ.get("CAPTION_ACTIVE", "&H0000E5FF")
    caption_outline: str = os.environ.get("CAPTION_OUTLINE", "&H00000000")
    caption_margin_v: int = _i("CAPTION_MARGIN_V", 420)

    # Output.
    work_dir: Path = Path(os.environ.get("STORY_WORK_DIR", "./stories"))
    music_path: str = os.environ.get("MUSIC_PATH", "")
    music_gain_db: float = _f("MUSIC_GAIN_DB", -22.0)
    max_scenes: int = _i("MAX_SCENES", 20)
