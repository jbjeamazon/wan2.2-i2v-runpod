#!/usr/bin/env python3
"""
Connectivity and configuration check. Run after setup.sh, before starting the
bot. Touches no GPU and rents nothing.

    python3 verify.py
"""

import asyncio
import os
import sys
from pathlib import Path

PASS, FAIL, WARN = "  ok  ", " FAIL ", " warn "
results: list[tuple[str, str, str]] = []


def add(name: str, status: str, detail: str) -> None:
    results.append((name, status, detail))


async def main() -> int:
    try:
        import config
        cfg = config.load_config()
        add("Config", PASS, f"loaded; locked to user id {cfg.authorized_user_id}")
    except Exception as exc:
        add("Config", FAIL, str(exc))
        render()
        return 1

    # SSH keys
    if not cfg.ssh_key_path.exists():
        add("SSH key", FAIL, f"{cfg.ssh_key_path} missing — run ./setup.sh")
    else:
        mode = oct(cfg.ssh_key_path.stat().st_mode)[-3:]
        if mode != "600":
            add("SSH key", FAIL, f"{cfg.ssh_key_path} has mode {mode}; ssh requires 600")
        else:
            add("SSH key", PASS, f"{cfg.ssh_key_path} (mode 600)")
    if cfg.ssh_pub_key_path.exists():
        add("SSH public key", PASS, cfg.ssh_pub_key_path.read_text().split()[0])
    else:
        add("SSH public key", FAIL, f"{cfg.ssh_pub_key_path} missing")

    # Local tools
    import shutil
    for tool in ("ssh", "scp", "ffmpeg"):
        where = shutil.which(tool)
        add(f"`{tool}`", PASS if where else FAIL, where or "not found on PATH")

    # Telegram
    try:
        from telegram import Bot
        bot = Bot(cfg.telegram_token)
        async with bot:
            me = await bot.get_me()
        add("Telegram", PASS, f"@{me.username} (id {me.id})")
    except Exception as exc:
        add("Telegram", FAIL, f"{type(exc).__name__}: {exc}")

    # Vast.ai
    try:
        from vast_manager import VastManager
        vast = VastManager(cfg.vast_api_key, cfg.vast_api_base,
                           cfg.ssh_key_path, cfg.vast_label)
        data = await vast.ping()
        running = len(data.get("instances", []))
        add("Vast.ai API", PASS, f"key accepted; {running} instance(s) currently on the account")

        try:
            offer = await vast.find_cheapest(cfg.gpu_query, cfg.vast_gpu_names)
            add("Vast.ai offers", PASS, f"cheapest match: {offer}")
        except Exception as exc:
            add("Vast.ai offers", WARN, str(exc))
    except Exception as exc:
        add("Vast.ai API", FAIL, f"{type(exc).__name__}: {exc}")

    return render()


def render() -> int:
    width = max(len(n) for n, _, _ in results)
    for name, status, detail in results:
        print(f"[{status}] {name.ljust(width)}  {detail}")
    fails = [r for r in results if r[1] == FAIL]
    print()
    if fails:
        print(f"{len(fails)} problem(s) to fix before running the bot.")
        return 1
    print("All checks passed. Start the bot with: python3 bot.py")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
