#!/usr/bin/env bash
#
# One command to see what works on this machine.
#
#   ./check.sh
#
# Runs every test suite, then reports which components are ready to run for
# real and what each one is still missing. Rents nothing, needs no credentials,
# touches no GPU.

cd "$(dirname "$0")"
BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'
pass=0; fail=0

hdr()  { echo; echo "${BOLD}$*${RESET}"; }
ok()   { echo "  ${GREEN}✓${RESET} $*"; pass=$((pass+1)); }
bad()  { echo "  ${RED}✗${RESET} $*"; fail=$((fail+1)); }
note() { echo "  ${DIM}·${RESET} ${DIM}$*${RESET}"; }
warn() { echo "  ${YELLOW}!${RESET} $*"; }

have() { command -v "$1" >/dev/null 2>&1; }
pyhas() { python3 -c "import $1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
hdr "Tooling"
have python3 && ok "python3 $(python3 -V 2>&1 | cut -d' ' -f2)" || bad "python3 missing"
have ffmpeg  && ok "ffmpeg"  || bad "ffmpeg missing — needed by every video stage"
have ssh     && ok "ssh"     || warn "ssh missing — only needed for the Vast.ai bot"
have nvidia-smi && ok "nvidia-smi ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1))" \
                || note "no NVIDIA GPU here — local rendering will not run"

# ---------------------------------------------------------------------------
hdr "Test suites (no GPU, no credentials)"
run_suite() {
    local name="$1" path="$2"
    if [ ! -x "$path" ]; then note "$name: no suite"; return; fi
    if out=$("$path" 2>&1); then
        ok "$name — $(echo "$out" | grep -c 'PASSED') suites passed"
    else
        bad "$name failed:"; echo "$out" | tail -5 | sed 's/^/      /'
    fi
}
run_suite "core (wan_core, handler, loras, mcp)" ./tests/run.sh
run_suite "telegram-bot"                          ./telegram-bot/tests/run.sh
run_suite "story-to-video"                        ./story-to-video/tests/run.sh

# ---------------------------------------------------------------------------
hdr "Component readiness"

echo "  ${BOLD}local/ — Wan 2.2 image-to-video server${RESET}"
if have nvidia-smi && pyhas torch; then
    ok "  ready: python3 local/preflight.py"
else
    note "  needs an NVIDIA GPU and torch. Full check: python3 local/preflight.py"
fi

echo
echo "  ${BOLD}telegram-bot/ — private bot + rented Vast.ai GPUs${RESET}"
if [ -f telegram-bot/.env ]; then
    ok "  .env present: cd telegram-bot && python3 verify.py"
else
    note "  needs setup: cd telegram-bot && ./setup.sh"
fi
note "  this is the cheapest way to see a REAL render — no GPU purchase needed"

echo
echo "  ${BOLD}story-to-video/ — premise to captioned vertical video${RESET}"
if have ffmpeg; then
    ok "  dry run available now:"
    echo "        cd story-to-video && python3 pipeline.py \"any premise\" --scenes 3 --dry-run"
    note "  real run also needs: an LLM endpoint, a TTS engine, a txt2img server,"
    note "  and the local/ I2V server"
else
    bad "  needs ffmpeg"
fi

# ---------------------------------------------------------------------------
hdr "Summary"
echo "  ${GREEN}${pass} ok${RESET}   ${RED}${fail} problem(s)${RESET}"
echo
if [ "$fail" -eq 0 ]; then
    echo "  ${BOLD}Everything testable here passes.${RESET} Next, in increasing cost:"
    echo "    1. ${BOLD}Free${RESET}   story-to-video dry run (above) — proves assembly end to end"
    echo "    2. ${BOLD}~\$1${RESET}    telegram-bot against Vast.ai — proves real generation"
    echo "    3. ${BOLD}Hardware${RESET} local/ on your own GPU — the private configuration"
else
    echo "  Fix the ✗ items above before moving on."
fi
exit "$([ "$fail" -eq 0 ] && echo 0 || echo 1)"
