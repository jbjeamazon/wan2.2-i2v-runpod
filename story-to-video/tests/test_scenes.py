"""Shot-list parsing, including the messy output small local models produce."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenes import parse_scenes, _extract_json, ScriptError

GOOD = [{"narration": "The diver sank into the dark.",
         "image_prompt": "a diver descending, shafts of light, deep blue",
         "motion_prompt": "slow descent, bubbles rising"}]

# 1. clean JSON
s = parse_scenes(json.dumps(GOOD))
assert len(s) == 1 and s[0].index == 0
print("1. clean JSON array parsed")

# 2. wrapped in a markdown fence (very common)
s = parse_scenes("```json\n" + json.dumps(GOOD) + "\n```")
assert s[0].narration.startswith("The diver")
print("2. ```json fence stripped")

# 3. prose before and after the array (also very common)
messy = "Sure! Here is your shot list:\n\n" + json.dumps(GOOD) + "\n\nLet me know if you want changes!"
s = parse_scenes(messy)
assert len(s) == 1
print("3. surrounding prose ignored, array extracted")

# 4. nested arrays inside the objects don't break bracket matching
nested = json.dumps([dict(GOOD[0], tags=["a", "b"])])
assert len(parse_scenes("blah " + nested + " thanks")) == 1
print("4. nested arrays handled by depth counting")

# 5. max_scenes truncates rather than overrunning the budget
many = json.dumps(GOOD * 50)
assert len(parse_scenes(many, max_scenes=6)) == 6
print("5. max_scenes caps the list at 6")

# 6. indices are assigned in order
s = parse_scenes(json.dumps(GOOD * 4))
assert [x.index for x in s] == [0, 1, 2, 3]
print("6. scene indices assigned 0..n")

# 7. a scene missing a required field is rejected by name
for field in ("narration", "image_prompt", "motion_prompt"):
    bad = dict(GOOD[0]); bad.pop(field)
    try:
        parse_scenes(json.dumps([bad]))
    except ScriptError as e:
        assert field in str(e), e
    else:
        sys.exit(f"FAIL: accepted a scene missing {field}")
print("7. missing narration/image_prompt/motion_prompt each rejected by name")

# 8. blank values count as missing
blank = dict(GOOD[0], narration="   ")
try:
    parse_scenes(json.dumps([blank]))
except ScriptError:
    print("8. whitespace-only field treated as missing")
else:
    sys.exit("FAIL: accepted a blank narration")

# 9. no array at all -> clear error
for junk in ("I cannot help with that.", "", "{}"):
    try:
        parse_scenes(junk)
    except ScriptError:
        pass
    else:
        sys.exit(f"FAIL: accepted {junk!r}")
print("9. non-array responses rejected")

print("\nALL SCENE TESTS PASSED")
