"""
Single-user lockdown.

Registered in handler group -1 so it runs before every other handler for every
update type — messages, commands, callback queries, edits, everything. An
update from anyone other than AUTHORIZED_USER_ID is dropped by raising
ApplicationHandlerStop, which prevents any later handler from seeing it.

Nothing is sent back. An outsider messaging the bot gets silence, which is
indistinguishable from the bot not existing: no "access denied", no typing
indicator, no error. Message content is never logged, regardless of settings.
"""

import logging

from telegram import Update
from telegram.ext import ApplicationHandlerStop, ContextTypes, TypeHandler

log = logging.getLogger(__name__)


class AuthGuard:
    """Drops every update that does not come from the one authorized user."""

    def __init__(self, authorized_user_id: int, log_rejected_ids: bool = False):
        self.authorized_user_id = authorized_user_id
        self.log_rejected_ids = log_rejected_ids
        self.rejected_count = 0

    def is_authorized(self, update: Update) -> bool:
        user = update.effective_user
        # An update with no identifiable user (channel posts, some service
        # messages) can never match the authorized id, so it is not authorized.
        return user is not None and user.id == self.authorized_user_id

    async def __call__(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if self.is_authorized(update):
            return

        self.rejected_count += 1
        if self.log_rejected_ids:
            user = update.effective_user
            # The id only — never the text, caption, file or chat title.
            log.info("Dropped update from unauthorized id=%s", user.id if user else "unknown")

        # Stops the update dead: no further handler runs, and nothing is sent.
        raise ApplicationHandlerStop


def install(application, guard: AuthGuard) -> None:
    """Attach the guard ahead of all other handlers."""
    application.add_handler(TypeHandler(Update, guard), group=-1)
