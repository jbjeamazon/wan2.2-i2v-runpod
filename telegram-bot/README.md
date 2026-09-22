# Private image-to-video Telegram bot

A Telegram bot that only you can talk to. Send it an image and a prompt, pick a
duration, and it rents the cheapest suitable GPU on Vast.ai, renders the video,
sends it back, and destroys the instance so billing stops.

```
image → prompt → duration → [rent GPU] → render → deliver → "extend?" ⟲
```

## Files

| File | Role |
|------|------|
| `bot.py` | Conversation state machine, status relay, video delivery |
| `auth.py` | Single-user lockdown, runs before every handler |
| `vast_manager.py` | Vast.ai search / create / SSH / destroy |
| `jobs.py` | End-to-end orchestration and frame extraction for extensions |
| `worker_script.py` | Uploaded to the GPU; runs the chained inference |
| `config.py` | Env loading, validation, duration→segment maths |
| `verify.py` | Connectivity check — run before the bot |
| `setup.sh` | One-shot local setup |

## Setup

```bash
cd telegram-bot
./setup.sh
```

It checks prerequisites, generates an ed25519 keypair (private key `chmod 600`),
creates `.env` with the key paths filled in, installs dependencies into `.venv`,
prompts for your three secrets with the input hidden, prints the public key to
paste into Vast.ai, and runs `verify.py`.

You need three things before you start:

| Value | Where |
|-------|-------|
| `TELEGRAM_BOT_TOKEN` | [@BotFather](https://t.me/botfather) → `/newbot` |
| `AUTHORIZED_USER_ID` | [@userinfobot](https://t.me/userinfobot) → your numeric id |
| `VAST_API_KEY` | https://cloud.vast.ai/account/ → API keys |

Then paste the printed public key into Vast.ai under **Account → SSH Keys**, and:

```bash
source .venv/bin/activate
python3 bot.py
```

## Privacy model

`auth.py` registers a `TypeHandler` in handler group `-1`, so it sees every
update — messages, commands, callback queries, edits — before any other handler.
Anything not from `AUTHORIZED_USER_ID` raises `ApplicationHandlerStop`, which
prevents every later handler from running.

Nothing is sent back. No "access denied", no typing indicator, no error. To an
outsider the bot is indistinguishable from one that does not exist.

Message content is never logged, in any configuration. `LOG_REJECTED_IDS=true`
adds the numeric sender id of rejected updates and nothing else; the default is
to log nothing at all. `tests/test_auth.py` asserts both, including that a
secret prompt string never reaches the log.

Every handler additionally carries a `filters.User` restriction, so the lockdown
survives someone removing the guard by accident.

## Duration and chaining

Diffusion video models produce a fixed window — 81 frames at 16 fps, about 5
seconds for Wan 2.2. Longer clips are made by **chaining**: the last frame of one
segment becomes the conditioning image for the next.

| Button | Passes | Meaning |
|--------|--------|---------|
| 5s | 1 | one generation |
| 30s | 6 | six chained generations |
| 60s | 12 | twelve chained generations |

A 60-second clip therefore costs roughly twelve times a 5-second one, and drifts
visually as it goes — colour, identity and lighting wander as each segment
re-conditions on the last frame of the previous one. That is inherent to the
technique. In practice 5–15 seconds holds together well; 60 seconds is a slideshow
of related moments more than one continuous shot.

"Extend" uses the same mechanism with a fresh prompt, which is the more useful
way to reach long durations — you steer each step instead of committing upfront.

## Cost control

- The cheapest offer meeting `VAST_MIN_VRAM_GB` and `VAST_MAX_DPH` wins.
- `rent()` destroys the instance in a `finally` block, so teardown runs on
  success, on exception, and on cancellation alike. `tests/test_vast.py` asserts
  all three.
- `destroy()` never raises — a teardown failure is logged loudly rather than
  masking the original error.
- Every instance carries `VAST_LABEL`. On startup the bot destroys any instance
  with that label, catching orphans left by a crash. It never touches instances
  without the label.
- Only one job runs at a time, so a second request cannot rent a second GPU.

**The largest cost is not the render.** With a stock PyTorch image, every job
pays to `pip install` and pull ~60 GB of weights before rendering starts — often
several dollars, repeated per job. Build the `Dockerfile` at the root of this
repo, push it to a registry, and set `VAST_IMAGE` to it. Then the instance boots
ready to work.

## Content filtering

None of the video pipelines used here ship a safety checker — Wan, CogVideoX and
I2VGen-XL have no such module, so there is nothing to disable. For SD-derived
pipelines that *do* accept one, `worker_script.py` passes `safety_checker=None`
and `requires_safety_checker=False`, and strips the attribute after
construction. It inspects the constructor signature first, because passing
`safety_checker` to a pipeline that does not accept it raises.

## Tests

```bash
./tests/run.sh
```

Five suites, no GPU and no credentials required:

| Suite | Covers |
|-------|--------|
| `test_auth.py` | Lockdown across update types, silence, content never logged |
| `test_vast.py` | API shapes against a mock transport; teardown on all three exit paths; orphan reaping |
| `test_bot.py` | Duration maths, keyboards, handler wiring, config rejection |
| `test_worker.py` | Filter disabling per pipeline type, prompt chaining, segment joining |
| `test_media.py` | Real ffmpeg: last-frame extraction, upload-size clamping |

What they do **not** cover: actual inference, a real Vast.ai rental, or a live
Telegram connection. Those need credentials and a GPU.

## Limits worth knowing

- **Telegram caps bot uploads at 50 MB.** `deliver()` re-encodes progressively
  to fit; if it still cannot, it reports the path on the host rather than failing.
- **Boot latency.** Renting, booting, installing and loading weights takes
  several minutes before rendering begins. This is per job, by design — the
  instance is destroyed afterwards.
- **Vast.ai hosts have root on their machines.** The marketplace is cheap and
  permissive, not private. See `../PRIVACY.md`.
