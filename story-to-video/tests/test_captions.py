"""ASS karaoke generation: timing format, grouping, highlighting, escaping."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from captions import Word, ass_time, escape, group_words, build_ass
from config import Config

cfg = Config()

# 1. ASS timestamps are H:MM:SS.cc
cases = [(0, "0:00:00.00"), (1.5, "0:00:01.50"), (61.23, "0:01:01.23"),
         (3661.07, "1:01:01.07"), (-5, "0:00:00.00")]
for secs, want in cases:
    got = ass_time(secs)
    assert got == want, f"{secs} -> {got}, want {want}"
print("1. ass_time:", ", ".join(f"{s}->{ass_time(s)}" for s, _ in cases[:4]))

# 2. rounding at the .995 boundary must not produce ".100"
for secs in (0.999, 59.999, 3599.999):
    t = ass_time(secs)
    assert re.match(r"^\d:\d\d:\d\d\.\d\d$", t), t
    assert not t.endswith(".100"), t
print("2. centisecond rounding carries correctly:", ass_time(59.999))

# 3. override syntax in spoken text is neutralised
assert escape("{drop}") == "\\{drop\\}"
assert escape("a\\b") == "a\\\\b"
assert escape("two\nlines") == "two lines"
print("3. braces/backslashes/newlines escaped in narration text")

# 4. grouping
ws = [Word(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(10)]
groups = group_words(ws, 4)
assert [len(g) for g in groups] == [4, 4, 2]
try:
    group_words(ws, 0)
except ValueError:
    print("4. grouped 10 words into 4/4/2; per_group=0 rejected")

# 5. one Dialogue line per word — the highlight advances word by word
ass = build_ass(ws, cfg)
lines = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
assert len(lines) == len(ws), f"{len(lines)} dialogue lines for {len(ws)} words"
print(f"5. {len(ws)} words -> {len(lines)} Dialogue events (one per word)")

# 6. in each event exactly one word carries the active override
for line in lines:
    assert line.count("\\fscx112") == 1, line
print("6. exactly one highlighted word per event")

# 7. the highlight walks through the group in order
first4 = lines[:4]
highlighted = [re.search(r"\\fscx112\\fscy112\}(\w+)", l).group(1) for l in first4]
assert highlighted == ["w0", "w1", "w2", "w3"], highlighted
print("7. highlight advances:", " -> ".join(highlighted))

# 8. all group members stay on screen while one is highlighted
assert all(f"w{i}" in first4[0] for i in range(4)), first4[0]
print("8. whole group visible throughout; only the tint moves")

# 9. no gaps — each word is held until the next starts
starts_ends = []
for l in lines[:4]:
    m = re.match(r"Dialogue: 0,([\d:.]+),([\d:.]+),", l)
    starts_ends.append((m.group(1), m.group(2)))
for i in range(3):
    assert starts_ends[i][1] == starts_ends[i + 1][0], starts_ends
print("9. consecutive events abut exactly — no caption flicker")

# 10. a zero-length word still gets a visible minimum duration
z = build_ass([Word("blip", 1.0, 1.0)], cfg)
m = re.search(r"Dialogue: 0,([\d:.]+),([\d:.]+),", z)
assert m.group(1) != m.group(2), "zero-duration event would never render"
print(f"10. zero-length word padded: {m.group(1)} -> {m.group(2)}")

# 11. the header declares the vertical canvas and a valid style row
assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
style = [l for l in ass.splitlines() if l.startswith("Style:")][0]
assert style.count(",") == 22, f"V4+ style needs 23 fields, got {style.count(',') + 1}"
print(f"11. header: 1080x1920, style row has {style.count(',') + 1} fields (V4+ requires 23)")

# 12. empty input yields a valid, empty file rather than crashing
empty = build_ass([], cfg)
assert "[Events]" in empty and "Dialogue:" not in empty
print("12. no words -> valid empty ASS file")

# 13. Dialogue field count must match the Format row, or Text picks up a
#     stray leading comma (caught by rendering a frame and looking at it)
fmt = [l for l in ass.splitlines() if l.startswith("Format:") and "Layer" in l][0]
n_fields = len(fmt.split(":", 1)[1].split(","))
body = lines[0].split(":", 1)[1]
head, sep, text = body.partition(",,")
prefix_fields = len(lines[0].split(",")[:n_fields - 1])
assert n_fields == 9, n_fields
# Take the first 8 commas as fields; everything after is Text.
text_part = lines[0].split(",", 8)[8]
assert not text_part.startswith(","), f"leading comma leaked into caption text: {text_part[:30]!r}"
assert text_part.lstrip().startswith(("w", "{")), text_part[:30]
print(f"13. Dialogue has exactly {n_fields} fields; caption text starts clean: {text_part[:24]!r}")

# 14. and the rendered text contains no stray separator before the first word
for l in lines[:3]:
    t = l.split(",", 8)[8]
    assert not t.startswith(","), t[:20]
print("14. no stray comma on any event")

print("\nALL CAPTION TESTS PASSED")
