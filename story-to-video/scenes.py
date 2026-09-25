"""
Story → structured shot list.

The LLM is asked for strict JSON: one entry per scene carrying the narration
line, a still-image prompt for the keyframe, and a motion prompt describing what
moves. Splitting image and motion matters — a text-to-image model wants a
composition, an image-to-video model wants a verb.

Any OpenAI-compatible endpoint works, so this runs against a local Ollama or
llama.cpp server with no metering and no content filter.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

import httpx

SYSTEM = """You are a shot planner for short vertical videos.

Return ONLY a JSON array. No prose, no markdown fence. Each element:
{
  "narration":    "one or two sentences to be read aloud",
  "image_prompt": "a still frame: subject, setting, lighting, camera, style",
  "motion_prompt": "what MOVES in this shot, as a short verb phrase"
}

Rules:
- Narration must be speakable in under 12 seconds.
- image_prompt describes a photograph, never an action over time.
- motion_prompt describes motion only: "hair drifting, slow push in".
- Keep the subject's appearance identical across scenes; repeat the
  descriptive words verbatim so the character stays consistent.
- Between {min_scenes} and {max_scenes} scenes."""


@dataclass
class Scene:
    narration: str
    image_prompt: str
    motion_prompt: str
    index: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class ScriptError(RuntimeError):
    pass


def _extract_json(text: str) -> list:
    """
    Pull the JSON array out of a model response.

    Small local models wrap output in prose or a markdown fence however firmly
    you instruct them not to, so take the first bracketed array rather than
    trusting the whole response to parse.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, depth = None, 0
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    raise ScriptError(f"No JSON array found in model output: {text[:200]}")


def parse_scenes(raw: str, max_scenes: int = 20) -> list[Scene]:
    """Validate and normalise the model's array into Scene objects."""
    data = _extract_json(raw)
    if not isinstance(data, list) or not data:
        raise ScriptError("Model returned no scenes.")

    scenes = []
    for i, item in enumerate(data[:max_scenes]):
        if not isinstance(item, dict):
            raise ScriptError(f"Scene {i} is not an object: {item!r}")
        missing = [k for k in ("narration", "image_prompt", "motion_prompt")
                   if not str(item.get(k, "")).strip()]
        if missing:
            raise ScriptError(f"Scene {i} missing {', '.join(missing)}")
        scenes.append(Scene(
            narration=str(item["narration"]).strip(),
            image_prompt=str(item["image_prompt"]).strip(),
            motion_prompt=str(item["motion_prompt"]).strip(),
            index=i,
        ))
    return scenes


async def write_script(cfg, premise: str, min_scenes: int = 4,
                       max_scenes: int = 8, style: str = "") -> list[Scene]:
    system = SYSTEM.format(min_scenes=min_scenes, max_scenes=max_scenes)
    user = premise if not style else f"{premise}\n\nVisual style: {style}"

    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(
            f"{cfg.llm_base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {cfg.llm_key}"},
            json={
                "model": cfg.llm_model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "temperature": 0.9,
            },
        )
    if r.status_code >= 400:
        raise ScriptError(f"LLM returned {r.status_code}: {r.text[:200]}")

    body = r.json()
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise ScriptError(f"Unexpected LLM response shape: {str(body)[:200]}")
    return parse_scenes(content, max_scenes)
