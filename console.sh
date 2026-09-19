#!/bin/zsh
# macbrow console: start / stop / status / log
# Usage: ./console.sh start|stop|status|log
cd "$(dirname "$0")"
LOG="${MACBROW_LOG:-/tmp/macbrow-console.log}"
case "${1:-start}" in
  start)
    if pgrep -f "agent.py console" >/dev/null; then echo "already running (pid $(pgrep -f 'agent.py console' | head -1))"; exit 0; fi
    set -a; [ -f .env.local ] && source .env.local; set +a
    # Keys exported only in the interactive shell profile (e.g. ~/.zshrc) aren't visible to a
    # detached start; pull them in when missing.
    for v in TYPESAFE_API_KEY GRADIUM_API_KEY; do
      if [ -z "${(P)v}" ]; then
        val=$(zsh -ic "print -r -- \${$v}" 2>/dev/null); [ -n "$val" ] && export "$v=$val"
      fi
    done
    missing=(); for v in TYPESAFE_API_KEY GRADIUM_API_KEY; do [ -z "${(P)v}" ] && missing+=("$v"); done
    if [ ${#missing[@]} -gt 0 ]; then echo "missing: ${missing[*]} (set in .env.local or your shell profile)"; exit 1; fi
    nohup uv run python agent.py console > "$LOG" 2>&1 &
    sleep 3; echo "started (pid $!), log: $LOG" ;;
  stop)
    pkill -INT -f "agent.py console" 2>/dev/null && sleep 2; pkill -9 -f "agent.py console" 2>/dev/null; echo "stopped" ;;
  status)
    pgrep -fl "agent.py console" || echo "not running" ;;
  log)
    sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$LOG" | grep -E "user_transcript|macbrow\.router +route|\"role\": \"assistant\"" | sed -E 's/^ *[0-9:.]* *(DEBUG|INFO) *//' ;;
  *) echo "usage: $0 start|stop|status|log"; exit 1 ;;
esac
