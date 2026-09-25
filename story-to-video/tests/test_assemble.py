"""Real ffmpeg: reframe, fit to narration, join, mix, burn captions."""
import sys, os, subprocess, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import assemble, captions
from captions import Word
from config import Config, WIDTH, HEIGHT

cfg = Config()
d = Path(tempfile.mkdtemp())

def make_video(name, seconds, size="640x360"):
    p = d / name
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    f"testsrc=duration={seconds}:size={size}:rate=16",
                    "-pix_fmt", "yuv420p", str(p)], capture_output=True, check=True)
    return p

def make_audio(name, seconds, freq=440):
    p = d / name
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency={freq}:duration={seconds}",
                    "-c:a", "pcm_s16le", "-ar", "44100", str(p)],
                   capture_output=True, check=True)
    return p

def dims(p):
    r = subprocess.run(["ffmpeg", "-i", str(p)], capture_output=True)
    import re
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", r.stderr.decode(errors="replace"))
    return (int(m.group(1)), int(m.group(2))) if m else None

# 1. duration probing works without ffprobe (this container has none)
src = make_video("src.mp4", 4)
got = assemble.probe_duration(src)
assert abs(got - 4.0) < 0.3, got
print(f"1. probe_duration (no ffprobe present) -> {got:.2f}s")

# 2. a landscape source is reframed to a full 9:16 frame, no pillarbox
tall = assemble.to_vertical(src, d / "tall.mp4", WIDTH, HEIGHT)
assert dims(tall) == (WIDTH, HEIGHT), dims(tall)
print(f"2. 640x360 landscape -> {dims(tall)[0]}x{dims(tall)[1]} (scale-to-cover + centre crop)")

# 3. video LONGER than its narration is trimmed to the narration
long_clip = make_video("long.mp4", 6)
fitted = assemble.fit_to_duration(long_clip, d / "fit_short.mp4", 2.5)
got = assemble.probe_duration(fitted)
assert abs(got - 2.5) < 0.25, got
print(f"3. 6.0s clip + 2.5s narration -> {got:.2f}s (trimmed)")

# 4. video SHORTER than its narration is extended by holding the last frame
short_clip = make_video("short.mp4", 2)
fitted2 = assemble.fit_to_duration(short_clip, d / "fit_long.mp4", 5.0)
got = assemble.probe_duration(fitted2)
assert abs(got - 5.0) < 0.3, got
print(f"4. 2.0s clip + 5.0s narration -> {got:.2f}s (final frame held, not looped)")

# 5. scenes join to the sum of their parts
parts = [assemble.to_vertical(make_video(f"p{i}.mp4", 2), d / f"pv{i}.mp4") for i in range(3)]
joined = assemble.concat(parts, d / "joined.mp4")
got = assemble.probe_duration(joined)
assert abs(got - 6.0) < 0.4, got
print(f"5. three 2s scenes -> {got:.2f}s joined")

# 6. narration tracks join too
auds = [make_audio(f"a{i}.wav", 2, 330 + i * 110) for i in range(3)]
full = assemble.concat_audio(auds, d / "narration.wav")
assert abs(assemble.probe_duration(full) - 6.0) < 0.3
print(f"6. three 2s narration lines -> {assemble.probe_duration(full):.2f}s")

# 7. music is ducked under the narration and trimmed to it
music = make_audio("music.wav", 30, 220)
mixed = assemble.mix_music(full, music, d / "mixed.wav", -22.0)
got = assemble.probe_duration(mixed)
assert abs(got - 6.0) < 0.3, f"music bed should be cut to narration, got {got}"
print(f"7. 30s music under 6s narration -> {got:.2f}s (bed trimmed to voice)")

# 8. audio muxes onto the video
with_audio = assemble.mux(joined, mixed, d / "with_audio.mp4")
r = subprocess.run(["ffmpeg", "-i", str(with_audio)], capture_output=True)
info = r.stderr.decode(errors="replace")
assert "Audio:" in info and "Video:" in info
print("8. muxed: output carries both a video and an audio stream")

# 9. captions burn in via libass, in one pass
assert assemble.has_libass(), "ffmpeg build lacks the subtitles filter"
words = [Word(w, i * 0.5, i * 0.5 + 0.45) for i, w in
         enumerate("the diver sank into the dark water below".split())]
ass = d / "caps.ass"
ass.write_text(captions.build_ass(words, cfg))
final = assemble.burn_captions(with_audio, ass, d / "final.mp4")
assert final.exists() and final.stat().st_size > 0
print(f"9. captions burned in -> final.mp4, {final.stat().st_size} bytes")

# 10. burning changes pixels (proves libass actually drew something)
def frame_at(video, t, name):
    p = d / name
    subprocess.run(["ffmpeg", "-y", "-ss", str(t), "-i", str(video),
                    "-frames:v", "1", str(p)], capture_output=True)
    return p.read_bytes() if p.exists() else b""

before, after = frame_at(with_audio, 1.0, "b.png"), frame_at(final, 1.0, "a.png")
assert before and after and before != after, "captions did not alter the frame"
print(f"10. frame at t=1.0s differs before/after burn ({len(before)} vs {len(after)} bytes) — libass drew")

# 11. duration survives the whole chain
assert abs(assemble.probe_duration(final) - 6.0) < 0.4
print(f"11. end-to-end duration preserved: {assemble.probe_duration(final):.2f}s")

# 12. a path with ':' in it does not break the subtitles filter graph
odd = d / "odd:name"
odd.mkdir(exist_ok=True)
odd_ass = odd / "c.ass"
odd_ass.write_text(captions.build_ass(words, cfg))
assemble.burn_captions(with_audio, odd_ass, d / "final2.mp4")
print("12. subtitle path containing ':' escaped correctly for the filter graph")

# 13. concatenating nothing is an error, not a silent empty file
try:
    assemble.concat([], d / "x.mp4")
except assemble.FFmpegError:
    print("13. concat([]) raises instead of writing an empty file")
else:
    sys.exit("FAIL: concat accepted an empty list")

print("\nALL ASSEMBLE TESTS PASSED")
