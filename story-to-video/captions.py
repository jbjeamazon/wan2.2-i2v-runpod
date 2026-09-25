"""
Word-level karaoke captions as an ASS subtitle file.

The look is the one every short-form video uses: a few words on screen at a
time, the word currently being spoken tinted and slightly enlarged.

Implemented with per-word {\\c}{\\fscx} override tags rather than ASS \\k
karaoke timing. \\k is the "correct" mechanism but renders inconsistently
across libass versions and gives no control over scale; emitting one Dialogue
line per spoken word, with the active word overridden inside the visible
group, is predictable and styles freely.

One ffmpeg pass burns the result in — no browser renderer, no per-frame
compositing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Word:
    text: str
    start: float
    end: float


def ass_time(seconds: float) -> str:
    """ASS timestamps are H:MM:SS.cc — centiseconds, single-digit hours."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs == 100:  # rounding pushed us to the next second
        cs = 0
        s += 1
        if s == 60:
            s = 0
            m += 1
            if m == 60:
                m = 0
                h += 1
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape(text: str) -> str:
    """Neutralise ASS override syntax in spoken text."""
    return (text.replace("\\", "\\\\")
                .replace("{", "\\{")
                .replace("}", "\\}")
                .replace("\n", " ")
                .strip())


def group_words(words: list[Word], per_group: int) -> list[list[Word]]:
    if per_group < 1:
        raise ValueError("per_group must be at least 1")
    return [words[i:i + per_group] for i in range(0, len(words), per_group)]


def header(cfg) -> str:
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {cfg_width(cfg)}
PlayResY: {cfg_height(cfg)}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{cfg.caption_font},{cfg.caption_size},{cfg.caption_fill},{cfg.caption_active},{cfg.caption_outline},&H64000000,1,0,0,0,100,100,0,0,1,6,3,2,60,60,{cfg.caption_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, Effect, Text
"""


def cfg_width(cfg) -> int:
    return getattr(cfg, "width", 1080)


def cfg_height(cfg) -> int:
    return getattr(cfg, "height", 1920)


def build_ass(words: list[Word], cfg) -> str:
    """Render word timings into a complete ASS file."""
    out = [header(cfg)]
    if not words:
        return "".join(out)

    for group in group_words(words, cfg.caption_words):
        for active in range(len(group)):
            parts = []
            for i, w in enumerate(group):
                token = escape(w.text)
                if not token:
                    continue
                if i == active:
                    # Tint and enlarge the spoken word, then reset for the rest.
                    parts.append(
                        f"{{\\c{cfg.caption_active}\\fscx112\\fscy112}}{token}{{\\r}}")
                else:
                    parts.append(token)
            text = " ".join(parts)

            start = group[active].start
            # Hold each word until the next begins so there is never a gap.
            end = group[active + 1].start if active + 1 < len(group) else group[active].end
            if end <= start:
                end = start + 0.08

            # Format declares 9 fields: Layer, Start, End, Style, Name,
            # MarginL, MarginR, Effect, Text. Everything after the 8th comma is
            # Text, so an extra field here leaks a comma into the caption.
            out.append(
                f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Karaoke,,0,0,,{text}\n")

    return "".join(out)


def transcribe_words(audio_path: str, model_name: str = "base") -> list[Word]:
    """
    Word-level timings from the narration audio.

    Timing comes from the rendered audio rather than the script text, because
    TTS pacing never matches a naive estimate and drift is what makes captions
    look wrong.
    """
    import whisper  # imported lazily; it pulls in torch

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, word_timestamps=True)

    words: list[Word] = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            text = str(w.get("word", "")).strip()
            if text:
                words.append(Word(text=text, start=float(w["start"]), end=float(w["end"])))
    return words
