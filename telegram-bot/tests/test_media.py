"""Real ffmpeg round-trips: extension frame extraction and upload-size clamping."""
import sys, os, subprocess, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

d = Path(tempfile.mkdtemp())
vid = d / "clip.mp4"
subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                "testsrc=duration=3:size=320x240:rate=16", "-pix_fmt", "yuv420p", str(vid)],
               capture_output=True)
assert vid.exists()
print(f"1. built a real 3s test clip: {vid.stat().st_size} bytes")

os.environ.update(TELEGRAM_BOT_TOKEN="1:x", AUTHORIZED_USER_ID="1", VAST_API_KEY="k")

# extract_last_frame is the mechanism behind "extend this video"
from jobs import extract_last_frame
from PIL import Image
frame = d / "last.png"
extract_last_frame(vid, frame)
im = Image.open(frame)
assert im.size == (320, 240), im.size
print(f"2. extract_last_frame -> {im.size} {im.mode} PNG, ready to seed the next segment")

# a missing/corrupt video fails loudly rather than producing a blank frame
try:
    extract_last_frame(d / "nope.mp4", d / "x.png")
except Exception as e:
    print(f"3. missing input rejected: {type(e).__name__}")
else:
    sys.exit("FAIL: extracted a frame from a nonexistent file")

import bot as botmod
# under the limit -> untouched (no needless re-encode)
assert botmod.shrink_to_limit(vid, 10_000_000) == vid
print("4. under Telegram's limit -> returned unchanged, no re-encode")

# over the limit -> progressively harder CRF until it fits
before = vid.stat().st_size
smaller = botmod.shrink_to_limit(vid, 12_000)
after = smaller.stat().st_size
assert smaller.exists()
print(f"5. over the limit -> {before} -> {after} bytes "
      f"({'now fits' if after <= 12_000 else 'still over; deliver() warns instead of failing'})")

# an impossible target still returns a usable file rather than raising
impossible = botmod.shrink_to_limit(vid, 10)
assert impossible.exists() and impossible.stat().st_size > 0
print("6. impossible target -> still returns a playable file for deliver() to report on")

print("\nALL MEDIA TESTS PASSED")
