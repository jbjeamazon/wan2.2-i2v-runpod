"""Duration maths, keyboard contents, and handler wiring."""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from config import segments_for, DURATION_CHOICES, SEGMENT_SECONDS

# 1. every offered duration maps to a sane segment count
print(f"   (one pass = {config.FRAMES_PER_SEGMENT} frames @ {config.SEGMENT_FPS}fps = {SEGMENT_SECONDS:.2f}s)")
expected = {5: 1, 10: 2, 15: 3, 20: 4, 25: 5, 30: 6, 60: 12}
for secs in DURATION_CHOICES:
    n = segments_for(secs)
    assert n == expected[secs], f"{secs}s -> {n}, expected {expected[secs]}"
    print(f"1. {secs:>2}s -> {n:>2} pass(es) (~{n * SEGMENT_SECONDS:.0f}s actual)")

# 2. the button set matches the spec exactly
assert DURATION_CHOICES == (5, 10, 15, 20, 25, 30, 60), DURATION_CHOICES
print("2. duration options are exactly 5/10/15/20/25/30/60")

# 3. never zero segments, and bad input is rejected
assert segments_for(1) == 1, "sub-segment durations must still render one pass"
for bad in (0, -5):
    try:
        segments_for(bad)
    except ValueError:
        pass
    else:
        sys.exit(f"FAIL: segments_for({bad}) accepted")
print("3. minimum one pass; zero/negative rejected")

# --- wiring -----------------------------------------------------------------
os.environ.update(
    TELEGRAM_BOT_TOKEN="123456789:AAFakeTokenForTestingOnly_NotReal12345678",
    AUTHORIZED_USER_ID="424242", VAST_API_KEY="sk-test",
    SSH_KEY_PATH="/tmp/nope", WORK_DIR="/tmp/jobs-test",
)
import bot as botmod
from telegram.ext import TypeHandler, ConversationHandler

cfg = config.load_config()
app = botmod.build_application(cfg)

# 4. the auth guard sits in group -1, ahead of every other handler
groups = sorted(app.handlers.keys())
assert groups[0] == -1, groups
guard_handler = app.handlers[-1][0]
assert isinstance(guard_handler, TypeHandler)
assert guard_handler.callback.authorized_user_id == 424242
print(f"4. AuthGuard installed in group {groups[0]} (before groups {groups[1:]}), locked to 424242")

# 5. the conversation covers the full specified flow
conv = next(h for h in app.handlers[0] if isinstance(h, ConversationHandler))
states = set(conv.states)
assert states == {botmod.IMAGE, botmod.PROMPT, botmod.DURATION,
                  botmod.EXTEND_CHOICE, botmod.EXTEND_PROMPT}, states
print("5. states: IMAGE -> PROMPT -> DURATION -> (render) -> EXTEND_CHOICE -> EXTEND_PROMPT")

# 6. an image sent cold starts the conversation (no /start required)
assert len(conv.entry_points) == 2
print("6. entry points: /start and a bare image upload")

# 7. keyboards carry the right callback data
dur_kb = botmod.duration_keyboard()
data = [b.callback_data for row in dur_kb.inline_keyboard for b in row]
assert data == [f"dur:{s}" for s in DURATION_CHOICES] + ["dur:cancel"], data
ext_kb = botmod.extend_keyboard()
ext = [b.callback_data for row in ext_kb.inline_keyboard for b in row]
assert ext == [f"ext:{s}" for s in config.EXTENSION_CHOICES] + ["ext:no"], ext
print(f"7. duration buttons {data[:3]}... / extend buttons {ext}")

# 8. one job at a time — the busy flag exists and starts clear
assert app.bot_data["busy"] is False
assert app.bot_data["cfg"].authorized_user_id == 424242
assert app.bot_data["vast"].label == "i2v-bot"
print("8. bot_data seeded: busy=False, vast label 'i2v-bot'")

# 9. config refuses to start without the lockdown id
os.environ["AUTHORIZED_USER_ID"] = ""
try:
    config.load_config()
except config.ConfigError as e:
    print(f"9. missing AUTHORIZED_USER_ID refuses startup: {str(e)[:52]}")
else:
    sys.exit("FAIL: started without an authorized user id")
os.environ["AUTHORIZED_USER_ID"] = "424242"

# 10. a non-numeric id is rejected rather than silently coerced
os.environ["AUTHORIZED_USER_ID"] = "@jesse"
try:
    config.load_config()
except config.ConfigError as e:
    print(f"10. @username rejected: {str(e)[:56]}")
else:
    sys.exit("FAIL: accepted a non-numeric user id")

os.environ["AUTHORIZED_USER_ID"] = "424242"

# 11. host-vetting filters reach the Vast query, and default to verified-only
cfg2 = config.load_config()
assert cfg2.vast_verified_only is True, "verified hosts should be the default"
assert cfg2.gpu_query["verified"] == {"eq": True}
assert "datacenter" not in cfg2.gpu_query
print("11. default query restricts to verified hosts:", cfg2.gpu_query["verified"])

os.environ["VAST_DATACENTER_ONLY"] = "true"
assert config.load_config().gpu_query["datacenter"] == {"eq": True}
print("12. VAST_DATACENTER_ONLY=true narrows further to datacenter operators")

os.environ["VAST_VERIFIED_ONLY"] = "false"
os.environ["VAST_DATACENTER_ONLY"] = "false"
q = config.load_config().gpu_query
assert "verified" not in q and "datacenter" not in q
print("13. both off -> unrestricted host pool (cheapest, least vetted)")

os.environ["VAST_VERIFIED_ONLY"] = "true"

# 14. render knobs are configurable, since a 60s video pays each step 12 times
os.environ.update(DEFAULT_STEPS="4", DEFAULT_LORAS="Wan2.2-Lightning, extra-lora",
                  DEFAULT_RESOLUTION="720p")
c = config.load_config()
assert c.default_steps == 4 and c.default_resolution == "720p"
assert c.default_loras == ("Wan2.2-Lightning", "extra-lora"), c.default_loras
print(f"14. steps={c.default_steps}, res={c.default_resolution}, loras={c.default_loras}")

# 15. those reach the JobSpec the bot sends to the GPU
from jobs import JobSpec
from pathlib import Path
spec = JobSpec(image_path=Path("/tmp/x.png"), prompts=["p"] * 12, segments=12,
               model_id=c.model_id, num_inference_steps=c.default_steps,
               resolution=c.default_resolution, loras=c.default_loras)
payload = spec.to_payload()
assert payload["num_inference_steps"] == 4 and payload["loras"] == ["Wan2.2-Lightning", "extra-lora"]
assert payload["segments"] == 12
print(f"15. payload -> steps={payload['num_inference_steps']}, loras={payload['loras']}")

# 16. the cost lever, stated plainly: steps x segments is what you pay for
for steps, label in ((30, "standard"), (4, "Lightning")):
    print(f"16. 60s video @ {steps} steps = {12 * steps} denoising passes total ({label})")

print("\nALL BOT TESTS PASSED")
