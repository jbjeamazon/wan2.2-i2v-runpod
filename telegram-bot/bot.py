#!/usr/bin/env python3
"""
Private image-to-video Telegram bot.

Flow: image -> prompt -> duration -> render on a rented GPU -> deliver -> offer
to extend, repeatedly.

Only AUTHORIZED_USER_ID can interact with it. See auth.py — the guard runs
before every handler and drops everything else in silence.
"""

import asyncio
import html
import logging
import subprocess
import time
import warnings
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler, ContextTypes,
    ConversationHandler, MessageHandler, filters,
)

import auth
import config
from jobs import JobSpec, extract_last_frame, run_job
from vast_manager import VastError, VastManager

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("bot")

IMAGE, PROMPT, DURATION, EXTEND_CHOICE, EXTEND_PROMPT = range(5)


# ---------------------------------------------------------------------------
# Status relay: edit one message rather than flooding the chat
# ---------------------------------------------------------------------------

class StatusRelay:
    def __init__(self, message, min_interval: float = 3.0):
        self.message = message
        self.min_interval = min_interval
        self._last_sent = 0.0
        self._last_text = ""
        self._lines: list[str] = []

    async def __call__(self, text: str) -> None:
        if text == self._last_text:
            return
        self._last_text = text
        self._lines.append(text)
        self._lines = self._lines[-6:]

        now = time.monotonic()
        if now - self._last_sent < self.min_interval:
            return
        self._last_sent = now
        body = "\n".join(f"• {html.escape(l)}" for l in self._lines)
        try:
            await self.message.edit_text(f"<b>Working…</b>\n{body}", parse_mode="HTML")
        except Exception:
            pass  # message unchanged or rate-limited; never fail a job over status


# ---------------------------------------------------------------------------
# Keyboards
# ---------------------------------------------------------------------------

def duration_keyboard() -> InlineKeyboardMarkup:
    opts = config.DURATION_CHOICES
    rows, row = [], []
    for s in opts:
        row.append(InlineKeyboardButton(f"{s}s", callback_data=f"dur:{s}"))
        if len(row) == 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("Cancel", callback_data="dur:cancel")])
    return InlineKeyboardMarkup(rows)


def extend_keyboard() -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton(f"+{s}s", callback_data=f"ext:{s}")
           for s in config.EXTENSION_CHOICES]
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("No, I'm done", callback_data="ext:no")]])


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------

def shrink_to_limit(video: Path, limit: int) -> Path:
    """Re-encode progressively harder until the file fits Telegram's limit."""
    if video.stat().st_size <= limit:
        return video
    for crf in (26, 30, 34):
        out = video.with_name(f"{video.stem}_crf{crf}.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video), "-c:v", "libx264", "-crf", str(crf),
             "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
            capture_output=True)
        if out.exists() and out.stat().st_size <= limit:
            return out
    return video


async def deliver(update: Update, cfg, result) -> None:
    chat = update.effective_chat
    video = shrink_to_limit(result.video_path, cfg.max_upload_bytes)
    size = video.stat().st_size

    caption = (
        f"{result.gpu} · {result.seconds / 60:.1f} min · "
        f"about ${result.cost:.2f}"
    )

    if size > cfg.max_upload_bytes:
        await chat.send_message(
            f"Rendered, but the file is {size / 1e6:.0f} MB and Telegram caps bot "
            f"uploads at {cfg.max_upload_bytes // (1024 * 1024)} MB even after "
            f"re-encoding.\n\nIt is on the bot host at:\n<code>{html.escape(str(video))}</code>\n\n"
            f"{caption}", parse_mode="HTML")
        return

    await chat.send_action(ChatAction.UPLOAD_VIDEO)
    with open(video, "rb") as fh:
        await chat.send_video(video=fh, caption=caption, supports_streaming=True,
                              read_timeout=300, write_timeout=300)


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text(
        "Send me an image to animate.\n\n/cancel at any point to start over.")
    return IMAGE


async def got_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    cfg = context.bot_data["cfg"]
    msg = update.message

    if msg.photo:
        tg_file = await msg.photo[-1].get_file()
    elif msg.document and (msg.document.mime_type or "").startswith("image/"):
        tg_file = await msg.document.get_file()
    else:
        await msg.reply_text("That is not an image. Send a photo or an image file.")
        return IMAGE

    session = cfg.work_dir / f"session_{int(time.time())}"
    session.mkdir(parents=True, exist_ok=True)
    path = session / "input.png"
    await tg_file.download_to_drive(custom_path=str(path))

    context.user_data["image_path"] = path
    context.user_data["session"] = session
    await msg.reply_text("Got it. Now describe what should happen in the video.")
    return PROMPT


async def got_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["prompt"] = update.message.text.strip()
    await update.message.reply_text(
        "How long should the video be?\n\n"
        "<i>Each ~5s is one generation pass, chained from the last frame of the "
        "previous one. Longer clips cost proportionally more and drift more.</i>",
        parse_mode="HTML", reply_markup=duration_keyboard())
    return DURATION


async def _render(update: Update, context: ContextTypes.DEFAULT_TYPE,
                  image_path: Path, prompts: list[str], segments: int) -> bool:
    cfg = context.bot_data["cfg"]
    vast: VastManager = context.bot_data["vast"]

    if context.bot_data.get("busy"):
        await update.effective_chat.send_message(
            "A render is already running. Wait for it to finish — two jobs would "
            "rent two GPUs.")
        return False

    context.bot_data["busy"] = True
    status_msg = await update.effective_chat.send_message("<b>Working…</b>", parse_mode="HTML")
    relay = StatusRelay(status_msg)

    spec = JobSpec(
        image_path=image_path, prompts=prompts, segments=segments,
        model_id=cfg.model_id,
    )
    try:
        result = await run_job(cfg, vast, spec, relay)
    except VastError as exc:
        await status_msg.edit_text(f"Job failed: {html.escape(str(exc))}\n\n"
                                   "The GPU instance was destroyed.", parse_mode="HTML")
        return False
    except Exception as exc:
        log.exception("Unexpected job failure")
        await status_msg.edit_text(
            f"Unexpected failure: {html.escape(type(exc).__name__)}: {html.escape(str(exc))}",
            parse_mode="HTML")
        return False
    finally:
        context.bot_data["busy"] = False

    context.user_data["last_video"] = result.video_path
    await deliver(update, cfg, result)
    return True


async def chose_duration(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    choice = query.data.split(":", 1)[1]

    if choice == "cancel":
        await query.edit_message_text("Cancelled.")
        return ConversationHandler.END

    seconds = int(choice)
    segments = config.segments_for(seconds)
    await query.edit_message_text(
        f"Rendering ~{seconds}s ({segments} segment{'s' if segments > 1 else ''}).")

    ok = await _render(update, context,
                       context.user_data["image_path"],
                       [context.user_data["prompt"]] * segments,
                       segments)
    if not ok:
        return ConversationHandler.END

    await update.effective_chat.send_message(
        "Would you like to extend this video?", reply_markup=extend_keyboard())
    return EXTEND_CHOICE


async def chose_extension(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    choice = query.data.split(":", 1)[1]

    if choice == "no":
        await query.edit_message_text("Done. Send another image whenever you like.")
        return ConversationHandler.END

    context.user_data["extend_seconds"] = int(choice)
    await query.edit_message_text(
        f"Extending by ~{choice}s. What happens next in the sequence?")
    return EXTEND_PROMPT


async def got_extension_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    cfg = context.bot_data["cfg"]
    prompt = update.message.text.strip()
    seconds = context.user_data["extend_seconds"]
    segments = config.segments_for(seconds)

    previous = context.user_data.get("last_video")
    if not previous or not Path(previous).exists():
        await update.message.reply_text("I no longer have the previous video. Send a new image.")
        return ConversationHandler.END

    session = context.user_data["session"]
    frame = session / f"extend_{int(time.time())}.png"
    try:
        extract_last_frame(Path(previous), frame)
    except Exception as exc:
        await update.message.reply_text(f"Could not read the final frame: {exc}")
        return ConversationHandler.END

    ok = await _render(update, context, frame, [prompt] * segments, segments)
    if not ok:
        return ConversationHandler.END

    await update.effective_chat.send_message(
        "Extend again?", reply_markup=extend_keyboard())
    return EXTEND_CHOICE


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("Cancelled. Send an image to begin again.")
    return ConversationHandler.END


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("Handler error", exc_info=context.error)


async def post_init(application: Application) -> None:
    """Destroy anything a previous crash left running before accepting work."""
    vast: VastManager = application.bot_data["vast"]
    reaped = await vast.reap_orphans()
    if reaped:
        log.warning("Startup reaped orphaned instances: %s", reaped)


def build_application(cfg) -> Application:
    app = Application.builder().token(cfg.telegram_token).post_init(post_init).build()

    app.bot_data["cfg"] = cfg
    app.bot_data["vast"] = VastManager(
        api_key=cfg.vast_api_key, api_base=cfg.vast_api_base,
        ssh_key=cfg.ssh_key_path, label=cfg.vast_label)
    app.bot_data["busy"] = False

    # Group -1: runs before everything, drops non-authorized updates silently.
    auth.install(app, auth.AuthGuard(cfg.authorized_user_id, cfg.log_rejected_ids))

    only_me = filters.User(user_id=cfg.authorized_user_id)

    # This conversation deliberately mixes message and callback handlers, so
    # per_message=False is correct here. Silence the advisory rather than print
    # it on every startup.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*per_message=False.*")
        conv = ConversationHandler(
            entry_points=[
                CommandHandler("start", start, filters=only_me),
                MessageHandler((filters.PHOTO | filters.Document.IMAGE) & only_me, got_image),
            ],
            states={
                IMAGE: [MessageHandler((filters.PHOTO | filters.Document.IMAGE) & only_me, got_image)],
                PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND & only_me, got_prompt)],
                DURATION: [CallbackQueryHandler(chose_duration, pattern=r"^dur:")],
                EXTEND_CHOICE: [CallbackQueryHandler(chose_extension, pattern=r"^ext:")],
                EXTEND_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND & only_me,
                                               got_extension_prompt)],
            },
            fallbacks=[CommandHandler("cancel", cancel, filters=only_me)],
            allow_reentry=True,
        )
    app.add_handler(conv)
    app.add_error_handler(on_error)
    return app


def main() -> None:
    cfg = config.load_config()
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    log.info("Locked to Telegram user id %s", cfg.authorized_user_id)
    build_application(cfg).run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
