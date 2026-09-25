#!/usr/bin/env python3
"""
Story premise -> finished vertical video.

    python3 pipeline.py "a diver finds a sunken city" --scenes 5

Stages:
    1. LLM writes a shot list (narration + image prompt + motion prompt)
    2. TTS renders each narration line; its real duration drives everything
    3. T2I renders one keyframe per scene
    4. I2V animates each keyframe for exactly its narration's length
    5. Whisper reads the rendered audio back for word-level caption timing
    6. ffmpeg reframes to 9:16, joins, mixes music, burns captions

Narration duration is the clock. Visuals are cut to the voice, never the
reverse — that is what keeps audio, picture and captions locked together.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import animate
import assemble
import captions
import scenes as scenes_mod
import stills
import voice
from config import HEIGHT, WIDTH, Config


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def build(premise: str, n_scenes: int, style: str, cfg: Config,
                outfile: Path | None = None, dry_run: bool = False) -> Path:
    job = cfg.work_dir / f"story_{int(time.time())}"
    for sub in ("audio", "stills", "clips", "vertical"):
        (job / sub).mkdir(parents=True, exist_ok=True)

    if dry_run:
        import dryrun
        log("DRY RUN: the LLM, TTS, image and video stages are synthetic.")
        log("         Assembly, timing and captions are the real code.")

    # 1. shot list -----------------------------------------------------------
    if dry_run:
        shots = dryrun.script(n_scenes)
        log(f"Canned shot list: {len(shots)} scenes")
    else:
        log(f"Writing a {n_scenes}-scene shot list with {cfg.llm_model}...")
        shots = await scenes_mod.write_script(cfg, premise, min_scenes=max(2, n_scenes - 1),
                                              max_scenes=n_scenes, style=style)
        log(f"  {len(shots)} scenes")
    (job / "script.json").write_text(json.dumps([s.to_dict() for s in shots], indent=2))

    # 2. narration first: its duration is the clock everything else runs on ---
    narration_paths, durations = [], []
    for s in shots:
        wav = job / "audio" / f"scene_{s.index:02d}.wav"
        log(f"Voicing scene {s.index + 1}/{len(shots)}...")
        if dry_run:
            dryrun.narrate(s.narration, wav)
        else:
            voice.speak(cfg, s.narration, wav)
        d = assemble.probe_duration(wav)
        narration_paths.append(wav)
        durations.append(d)
        log(f"  {d:.1f}s")

    total = sum(durations)
    log(f"Narration total: {total:.1f}s")

    # 3. keyframes -----------------------------------------------------------
    still_paths = []
    for s in shots:
        png = job / "stills" / f"scene_{s.index:02d}.png"
        log(f"Keyframe {s.index + 1}/{len(shots)}...")
        if dry_run:
            dryrun.still(s.index, png)
        else:
            await stills.render_still(cfg, s.image_prompt, png)
        still_paths.append(png)

    # 4. animate each keyframe for its scene's exact length -------------------
    clip_paths = []
    for s, png, seconds in zip(shots, still_paths, durations):
        clip = job / "clips" / f"scene_{s.index:02d}.mp4"
        passes = animate.segments_for(seconds)
        log(f"Animating scene {s.index + 1}/{len(shots)} "
            f"({seconds:.1f}s = {passes} pass{'es' if passes > 1 else ''})...")
        if dry_run:
            dryrun.animate(s.index, seconds, clip)
        else:
            await animate.animate(cfg, png, s.motion_prompt, seconds, clip)
        clip_paths.append(clip)

    # 5. reframe to 9:16 and cut to the narration ----------------------------
    fitted = []
    for s, clip, seconds in zip(shots, clip_paths, durations):
        tall = job / "vertical" / f"scene_{s.index:02d}_tall.mp4"
        exact = job / "vertical" / f"scene_{s.index:02d}.mp4"
        assemble.to_vertical(clip, tall, WIDTH, HEIGHT)
        assemble.fit_to_duration(tall, exact, seconds)
        fitted.append(exact)

    log("Joining scenes...")
    silent = assemble.concat(fitted, job / "silent.mp4")
    full_narration = assemble.concat_audio(narration_paths, job / "narration.wav")

    # 6. music, captions, final mux ------------------------------------------
    track = full_narration
    if cfg.music_path and Path(cfg.music_path).exists():
        log("Mixing music bed...")
        track = assemble.mix_music(full_narration, Path(cfg.music_path),
                                   job / "mixed.wav", cfg.music_gain_db)

    with_audio = assemble.mux(silent, track, job / "with_audio.mp4")

    if dry_run:
        log("Timing captions from scene durations (Whisper skipped)...")
        words = dryrun.time_words(shots, durations)
    else:
        log(f"Transcribing narration for caption timing ({cfg.whisper_model})...")
        words = captions.transcribe_words(str(full_narration), cfg.whisper_model)
    ass = job / "captions.ass"
    ass.write_text(captions.build_ass(words, cfg))
    log(f"  {len(words)} words timed")

    final = outfile or (job / "final.mp4")
    log("Burning captions...")
    assemble.burn_captions(with_audio, ass, final)

    log(f"Done: {final}  ({assemble.probe_duration(final):.1f}s)")
    return final


def main() -> int:
    p = argparse.ArgumentParser(description="Story premise to vertical video")
    p.add_argument("premise", help="what the story is about")
    p.add_argument("--scenes", type=int, default=6, help="maximum scenes (default 6)")
    p.add_argument("--style", default="", help="visual style applied to every keyframe")
    p.add_argument("--out", type=Path, default=None, help="output path")
    p.add_argument("--dry-run", action="store_true",
                   help="use synthetic stages; needs only ffmpeg. Proves the "
                        "orchestration and assembly work before you install any models.")
    args = p.parse_args()

    cfg = Config()
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(build(args.premise, min(args.scenes, cfg.max_scenes),
                          args.style, cfg, args.out, dry_run=args.dry_run))
    except Exception as exc:
        print(f"\nFailed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
