"""
Synthetic stand-ins for the four stages that need models.

`pipeline.py --dry-run` swaps the LLM, TTS, text-to-image and image-to-video
stages for ffmpeg-generated placeholders and then runs the *real* assembly:
reframing, fitting visuals to narration, joining, mixing and burning captions.

The point is to prove the orchestration and the whole back half work — timings,
ordering, caption sync, the final mux — before you spend money on a GPU or
spend an afternoon installing model weights. The output is a genuine mp4 you
can play; only the pixels and the voice are fake.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from captions import Word
from scenes import Scene

# Distinct background per scene so a mis-ordered or dropped scene is obvious
# on playback rather than silently plausible.
PALETTE = ["0x1a2740", "0x3d2246", "0x14323a", "0x402a1c", "0x2b1f3d", "0x1c3a2e"]

CANNED = [
    ("She found the door behind the waterfall.",
     "a mossy stone door behind falling water, shafts of light",
     "water falling, slow push in"),
    ("It had been waiting for her the whole time.",
     "close on a carved handprint in wet stone, cold light",
     "dust drifting, handheld sway"),
    ("She pressed her palm against the cold stone.",
     "a hand on carved stone, warm glow spreading from the contact",
     "glow spreading outward, slow zoom"),
    ("The door remembered her name.",
     "wide shot of a door opening into golden light",
     "door swinging open, light spilling forward"),
    ("And it opened.",
     "silhouette stepping into blinding golden light",
     "figure walking forward, light blooming"),
]


def script(n_scenes: int) -> list[Scene]:
    return [Scene(narration=n, image_prompt=i, motion_prompt=m, index=k)
            for k, (n, i, m) in enumerate(CANNED[:n_scenes])]


def narrate(text: str, out: Path) -> Path:
    """
    A near-silent tone lasting roughly as long as the line would take to read.

    ~2.8 words per second is a normal narration pace; this only has to be
    plausible, because the pipeline measures whatever it actually produces
    rather than trusting an estimate.
    """
    seconds = max(1.2, len(text.split()) / 2.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=190:duration={seconds:.2f}",
         "-af", "volume=-30dB", "-c:a", "pcm_s16le", "-ar", "44100", str(out)],
        capture_output=True, check=True)
    return out


def still(index: int, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    colour = PALETTE[index % len(PALETTE)]
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={colour}:size=832x1472",
         "-vf", "noise=alls=10:allf=t", "-frames:v", "1", str(out)],
        capture_output=True, check=True)
    return out


def animate(index: int, seconds: float, out: Path) -> Path:
    """A slowly drifting field, long enough to be trimmed to the narration."""
    out.parent.mkdir(parents=True, exist_ok=True)
    colour = PALETTE[index % len(PALETTE)]
    # Deliberately generated a little long, so fit_to_duration exercises its
    # trim path rather than always padding.
    duration = seconds + 1.5
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"color=c={colour}:size=832x1472:duration={duration:.2f}:rate=16",
         "-vf", "noise=alls=12:allf=t+u,hue=s=0.7",
         "-pix_fmt", "yuv420p", str(out)],
        capture_output=True, check=True)
    return out


def time_words(scenes: list[Scene], durations: list[float]) -> list[Word]:
    """
    Caption timings spread evenly across each scene's narration.

    Whisper would derive these from the audio; here the audio is a tone, so the
    words are distributed across the measured duration instead. Even spacing is
    enough to verify that captions stay inside their scene and never run past
    the end of the video.
    """
    words: list[Word] = []
    clock = 0.0
    for scene, duration in zip(scenes, durations):
        tokens = scene.narration.split()
        if not tokens:
            clock += duration
            continue
        step = duration / len(tokens)
        for i, token in enumerate(tokens):
            start = clock + i * step
            words.append(Word(text=token, start=start, end=start + step * 0.92))
        clock += duration
    return words
