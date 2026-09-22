#!/usr/bin/env bash
#
# Hands-off setup. Run this on the machine that will host the bot.
#
#   chmod +x setup.sh && ./setup.sh
#
# It generates the SSH keypair, writes .env with the paths already filled in,
# prompts for your three secrets (never echoed, never leaving this machine),
# installs dependencies, and runs the connectivity check.

set -euo pipefail
cd "$(dirname "$0")"

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'
step() { echo; echo "${BOLD}==> $*${RESET}"; }
ok()   { echo "${GREEN}  ✓${RESET} $*"; }
warn() { echo "${YELLOW}  !${RESET} $*"; }

KEY_DIR="./secrets"
KEY_PATH="$KEY_DIR/vast_ed25519"

# ---------------------------------------------------------------------------
step "Checking prerequisites"
missing=()
for tool in python3 ssh-keygen ssh scp ffmpeg; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [ ${#missing[@]} -gt 0 ]; then
    warn "Missing: ${missing[*]}"
    echo "    Debian/Ubuntu: sudo apt install -y python3 python3-venv openssh-client ffmpeg"
    echo "    macOS:         brew install ffmpeg"
    exit 1
fi
ok "python3, ssh-keygen, ssh, scp, ffmpeg all present"

# ---------------------------------------------------------------------------
step "SSH keypair"
mkdir -p "$KEY_DIR"
chmod 700 "$KEY_DIR"
if [ -f "$KEY_PATH" ]; then
    ok "Reusing existing key at $KEY_PATH"
else
    ssh-keygen -t ed25519 -f "$KEY_PATH" -N "" -C "vast-i2v-bot" >/dev/null
    ok "Generated $KEY_PATH"
fi
chmod 600 "$KEY_PATH"
chmod 644 "$KEY_PATH.pub"
ok "Permissions set (private key 600)"

# ---------------------------------------------------------------------------
step "Python dependencies"
if [ ! -d .venv ]; then
    python3 -m venv .venv
    ok "Created virtualenv at .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
ok "Installed $(wc -l < requirements.txt) requirement groups into .venv"

# ---------------------------------------------------------------------------
step "Environment file"
if [ -f .env ]; then
    ok ".env already exists — leaving it alone"
else
    cp .env.example .env
    chmod 600 .env
    ok "Created .env from template (mode 600)"
fi

set_env() {  # set_env KEY VALUE — rewrites in place, no duplicate lines
    local key="$1" value="$2"
    if grep -q "^${key}=" .env; then
        python3 - "$key" "$value" <<'PY'
import sys, pathlib
key, value = sys.argv[1], sys.argv[2]
p = pathlib.Path(".env")
lines = p.read_text().splitlines()
out = [f"{key}={value}" if l.startswith(f"{key}=") else l for l in lines]
p.write_text("\n".join(out) + "\n")
PY
    else
        printf '%s=%s\n' "$key" "$value" >> .env
    fi
}

set_env SSH_KEY_PATH "$KEY_PATH"
set_env SSH_PUBLIC_KEY_PATH "$KEY_PATH.pub"
ok "Wrote SSH key paths into .env"

# ---------------------------------------------------------------------------
step "Credentials"
echo "${DIM}   Typed secrets are hidden and written only to ./.env on this machine.${RESET}"
echo

prompt_secret() {  # prompt_secret KEY "Human label" "hint"
    local key="$1" label="$2" hint="$3" current value
    current=$(grep "^${key}=" .env | cut -d= -f2- || true)
    if [ -n "$current" ]; then
        ok "$label already set — skipping"
        return
    fi
    echo "   ${BOLD}${label}${RESET}"
    echo "   ${DIM}${hint}${RESET}"
    read -rsp "   > " value
    echo
    if [ -z "$value" ]; then
        warn "Left blank — fill $key into .env before running the bot"
    else
        set_env "$key" "$value"
        ok "Saved $key"
    fi
    echo
}

prompt_secret TELEGRAM_BOT_TOKEN "Telegram bot token" \
    "Message @BotFather, send /newbot, paste the token it gives you."
prompt_secret AUTHORIZED_USER_ID "Your numeric Telegram user id" \
    "Message @userinfobot — it replies with your numeric id. Not your @username."
prompt_secret VAST_API_KEY "Vast.ai API key" \
    "https://cloud.vast.ai/account/ -> API keys -> copy."

# ---------------------------------------------------------------------------
step "Your Vast.ai public key"
echo "${DIM}   Paste this into https://cloud.vast.ai/account/ -> SSH Keys.${RESET}"
echo
cat "$KEY_PATH.pub"
echo

# ---------------------------------------------------------------------------
step "Verifying"
python3 verify.py || true

echo
echo "${BOLD}Next:${RESET} source .venv/bin/activate && python3 bot.py"
