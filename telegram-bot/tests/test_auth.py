"""The privacy lockdown: only one user gets through, and outsiders get silence."""
import asyncio, sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram import Update, User, Message, Chat, CallbackQuery
from telegram.ext import ApplicationHandlerStop
from auth import AuthGuard

ME, STRANGER = 111222333, 999888777
run = asyncio.run


def make_update(user_id, text="hello secret text", kind="message"):
    user = User(id=user_id, first_name="x", is_bot=False) if user_id else None
    chat = Chat(id=user_id or 0, type="private")
    if kind == "message":
        msg = Message(message_id=1, date=None, chat=chat, from_user=user, text=text)
        return Update(update_id=1, message=msg)
    if kind == "callback":
        msg = Message(message_id=1, date=None, chat=chat, from_user=user, text=text)
        cq = CallbackQuery(id="1", from_user=user, chat_instance="c", data="dur:60", message=msg)
        return Update(update_id=1, callback_query=cq)
    if kind == "edited":
        msg = Message(message_id=1, date=None, chat=chat, from_user=user, text=text)
        return Update(update_id=1, edited_message=msg)
    raise ValueError(kind)


guard = AuthGuard(ME)

# 1. the authorized user passes through every update type
for kind in ("message", "callback", "edited"):
    run(guard(make_update(ME, kind=kind), None))   # must not raise
    print(f"1. authorized user passes: {kind}")

# 2. a stranger is stopped dead, on every update type
for kind in ("message", "callback", "edited"):
    try:
        run(guard(make_update(STRANGER, kind=kind), None))
    except ApplicationHandlerStop:
        print(f"2. stranger stopped: {kind} -> ApplicationHandlerStop")
    else:
        sys.exit(f"FAIL: stranger's {kind} was not stopped")

# 3. an update with no identifiable user is also stopped
try:
    run(guard(Update(update_id=9), None))
except ApplicationHandlerStop:
    print("3. update with no user -> stopped")
else:
    sys.exit("FAIL: userless update passed")

# 4. nothing is ever sent back — the guard has no bot/reply calls at all
#    (a reply would require context, which we pass as None; any attempt to use
#     it would have raised AttributeError above rather than ApplicationHandlerStop)
print("4. guard never touches context/bot — no reply is possible")

# 5. message content is NEVER logged, even with id logging on
records = []
class Capture(logging.Handler):
    def emit(self, r): records.append(r.getMessage())

logger = logging.getLogger("auth")
logger.addHandler(Capture()); logger.setLevel(logging.DEBUG)

loud = AuthGuard(ME, log_rejected_ids=True)
SECRET = "extremely-private-prompt-text-do-not-log"
try:
    run(loud(make_update(STRANGER, text=SECRET), None))
except ApplicationHandlerStop:
    pass
joined = " ".join(records)
assert SECRET not in joined, f"CONTENT LEAKED INTO LOGS: {joined}"
assert str(STRANGER) in joined, "id logging was requested but id absent"
print(f"5. LOG_REJECTED_IDS=true logs the id only: {records[-1]!r}")

# 6. with the default setting, nothing at all is logged
records.clear()
quiet = AuthGuard(ME, log_rejected_ids=False)
try:
    run(quiet(make_update(STRANGER, text=SECRET), None))
except ApplicationHandlerStop:
    pass
assert records == [], f"expected silence, got {records}"
print("6. default (LOG_REJECTED_IDS=false) logs nothing at all")

# 7. rejections are counted so you can see scanning without seeing content
# guard saw 3 stranger updates plus 1 userless update
assert quiet.rejected_count == 1, quiet.rejected_count
assert guard.rejected_count == 4, guard.rejected_count
print(f"7. rejected_count tracked: {guard.rejected_count}")

print("\nALL AUTH TESTS PASSED")
