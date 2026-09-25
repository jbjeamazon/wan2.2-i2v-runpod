# Story → vertical video

A local pipeline that takes a premise and produces a captioned 9:16 video with
narration. This is the open-source equivalent of the hosted "AI story to video"
tools, running on your own hardware with no per-video credits.

```
premise → LLM shot list → TTS narration → T2I keyframes
        → I2V animation → Whisper timing → ffmpeg assembly → final.mp4
```

## The design decision that matters

**Narration duration is the clock.** The voice is rendered first, its real
length measured, and the visuals cut to fit — never the reverse. Estimating
speech length from word count drifts by seconds over a minute, and that drift is
why amateur output has captions sliding out of sync with the audio.

Each scene is animated for `ceil(narration / 5.06s)` passes and then trimmed to
the exact narration length. A clip that comes up short holds its final frame
rather than looping, because a held frame reads as a deliberate beat and a loop
reads as a glitch.

## Stages

| Stage | Module | What it uses |
|---|---|---|
| Shot list | `scenes.py` | Any OpenAI-compatible endpoint — Ollama, llama.cpp, vLLM |
| Narration | `voice.py` | Kokoro (CPU, Apache-2.0) or Chatterbox (cloning, MIT) |
| Keyframes | `stills.py` | A1111-compatible `/sdapi/v1/txt2img` — your checkpoint |
| Animation | `animate.py` | The `local/server.py` Wan 2.2 endpoint in this repo |
| Caption timing | `captions.py` | Whisper word-level timestamps |
| Assembly | `assemble.py` | ffmpeg + libass |

Every stage is a local HTTP call or a local model. Nothing is metered and
nothing is filtered — the checkpoint you point `stills.py` at decides what the
video can contain, since the video model only continues whatever still it is
handed.

## Captions

Groups of four words, the spoken one tinted and scaled 112%. Implemented with
per-word `{\c}{\fscx}` override tags rather than ASS `\k` karaoke timing —
`\k` is the nominally correct mechanism but renders inconsistently across libass
versions and gives no control over scale. One Dialogue event per spoken word,
held until the next begins so there is never a flicker between them.

Burned in with a single ffmpeg pass. No browser renderer, no per-frame
compositing.

## Setup

```bash
pip install -r requirements.txt

# an LLM — anything OpenAI-compatible
ollama serve && ollama pull llama3.1:8b

# the image-to-video server from this repo
python3 ../local/server.py

# your text-to-image server on :7860 (A1111, Forge, …)
```

Then:

```bash
python3 pipeline.py "a diver finds a sunken city" --scenes 5 --style "cinematic, 35mm"
```

Configuration lives in `config.py`, all overridable by environment variable:
`LLM_BASE_URL`, `LLM_MODEL`, `WAN_LOCAL_URL`, `T2I_BASE_URL`, `TTS_ENGINE`,
`TTS_VOICE`, `WHISPER_MODEL`, `CAPTION_*`, `MUSIC_PATH`.

## Choosing a TTS engine

| | Kokoro | Chatterbox |
|---|---|---|
| Size | 82M | larger |
| Hardware | CPU is fine | wants a GPU |
| Voice cloning | no, 54 built-in voices | yes, from ~10s |
| Licence | Apache-2.0 | MIT |

Both are safe to build on commercially. **XTTS-v2 (CPML) and F5-TTS weights
(CC-BY-NC) are not** — worth knowing before the output gets monetised.

## Tests

```bash
./tests/run.sh
```

| Suite | Covers |
|---|---|
| `test_scenes.py` | Shot-list parsing, including fenced and prose-wrapped LLM output |
| `test_captions.py` | ASS timing format, grouping, highlight advance, escaping, field count |
| `test_assemble.py` | Real ffmpeg: reframe, fit-to-narration, join, music mix, caption burn |

`test_assemble.py` runs actual ffmpeg end to end and asserts that burning
captions changes the pixels, so a silently-failing libass cannot pass.

What is **not** covered: the LLM, TTS, T2I and I2V stages, which need models and
servers. Those are adapters over HTTP; their shapes are exercised, their outputs
are not.

## Cost

Electricity. The tradeoff is latency and setup: a hosted tool returns in a
minute, this takes as long as your GPU needs, and you assemble the stack
yourself.
