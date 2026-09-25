"""
ffmpeg operations: reframe to vertical, fit video to narration, join, mix,
burn captions.

Durations are probed from the rendered files rather than estimated. Narration
length is the source of truth — the visuals are cut to it, never the other way
round, which is what stops the audio drifting out of sync with the captions.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def _run(argv: list[str], what: str) -> subprocess.CompletedProcess:
    result = subprocess.run(argv, capture_output=True)
    if result.returncode != 0:
        raise FFmpegError(f"{what} failed: {result.stderr.decode(errors='replace')[-600:]}")
    return result


def probe_duration(path: str | Path) -> float:
    """
    Seconds of media in `path`.

    Prefers ffprobe, but falls back to parsing ffmpeg's own banner, because
    plenty of minimal installs (including pip's imageio-ffmpeg) ship ffmpeg
    without ffprobe.
    """
    path = str(path)
    if shutil.which("ffprobe"):
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path], capture_output=True)
        if r.returncode == 0:
            try:
                return float(json.loads(r.stdout)["format"]["duration"])
            except (ValueError, KeyError, TypeError):
                pass

    r = subprocess.run(["ffmpeg", "-i", path], capture_output=True)
    m = re.search(r"Duration:\s*(\d+):(\d\d):(\d\d)\.(\d+)", r.stderr.decode(errors="replace"))
    if not m:
        raise FFmpegError(f"Could not determine duration of {path}")
    h, mnt, s, frac = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + int(s) + float(f"0.{frac}")


def to_vertical(src: Path, dst: Path, width: int = 1080, height: int = 1920) -> Path:
    """
    Reframe to 9:16 by scaling to cover and centre-cropping.

    `increase` then `crop` fills the frame without pillarboxing; a landscape
    source loses its edges, which is the right trade for short-form.
    """
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=increase,"
          f"crop={width}:{height},setsar=1")
    _run(["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-c:v", "libx264",
          "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
          "-an", str(dst)], f"reframe {src.name}")
    return dst


def fit_to_duration(src: Path, dst: Path, seconds: float) -> Path:
    """
    Make `src` exactly `seconds` long.

    Too long is trimmed. Too short is extended by holding the final frame,
    which reads as a deliberate beat, where looping reads as a glitch.
    """
    have = probe_duration(src)
    if have >= seconds:
        _run(["ffmpeg", "-y", "-i", str(src), "-t", f"{seconds:.3f}",
              "-c:v", "libx264", "-preset", "medium", "-crf", "20",
              "-pix_fmt", "yuv420p", "-an", str(dst)], f"trim {src.name}")
    else:
        pad = seconds - have
        _run(["ffmpeg", "-y", "-i", str(src),
              "-vf", f"tpad=stop_mode=clone:stop_duration={pad:.3f}",
              "-t", f"{seconds:.3f}", "-c:v", "libx264", "-preset", "medium",
              "-crf", "20", "-pix_fmt", "yuv420p", "-an", str(dst)],
             f"extend {src.name}")
    return dst


def concat(parts: list[Path], dst: Path) -> Path:
    """Join clips. Re-encodes, since segments may differ in encoder settings."""
    if not parts:
        raise FFmpegError("Nothing to concatenate")
    if len(parts) == 1:
        shutil.copy(parts[0], dst)
        return dst

    listing = dst.parent / f"{dst.stem}_concat.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    try:
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
              "-c", "copy", str(dst)], "concat (copy)")
    except FFmpegError:
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
              "-c:v", "libx264", "-preset", "medium", "-crf", "20",
              "-pix_fmt", "yuv420p", str(dst)], "concat (re-encode)")
    return dst


def concat_audio(parts: list[Path], dst: Path) -> Path:
    if len(parts) == 1:
        shutil.copy(parts[0], dst)
        return dst
    listing = dst.parent / f"{dst.stem}_concat.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
          "-c:a", "pcm_s16le", "-ar", "44100", str(dst)], "concat audio")
    return dst


def mix_music(narration: Path, music: Path, dst: Path, gain_db: float = -22.0) -> Path:
    """Lay music under narration at `gain_db`, trimmed to the narration."""
    _run(["ffmpeg", "-y", "-i", str(narration), "-stream_loop", "-1", "-i", str(music),
          "-filter_complex",
          f"[1:a]volume={gain_db}dB[bed];[0:a][bed]amix=inputs=2:duration=first:dropout_transition=0[a]",
          "-map", "[a]", "-c:a", "pcm_s16le", "-ar", "44100", str(dst)], "mix music")
    return dst


def mux(video: Path, audio: Path, dst: Path) -> Path:
    """Attach the audio track and stop at whichever stream ends first."""
    _run(["ffmpeg", "-y", "-i", str(video), "-i", str(audio),
          "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
          "-c:a", "aac", "-b:a", "192k", "-shortest",
          "-movflags", "+faststart", str(dst)], "mux audio")
    return dst


def burn_captions(video: Path, ass: Path, dst: Path) -> Path:
    """Burn the ASS file in with libass, in a single pass."""
    # The subtitles filter takes a filter-graph argument, so ':' and '\' in the
    # path have to be escaped or the graph parser mangles them.
    escaped = str(ass.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    _run(["ffmpeg", "-y", "-i", str(video), "-vf", f"subtitles='{escaped}'",
          "-c:v", "libx264", "-preset", "medium", "-crf", "20",
          "-pix_fmt", "yuv420p", "-c:a", "copy",
          "-movflags", "+faststart", str(dst)], "burn captions")
    return dst


def has_libass() -> bool:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True)
    return b"subtitles" in r.stdout
