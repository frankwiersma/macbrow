"""Safety policy: what voice control is allowed to touch on this Mac.

Allowed by design: launching/switching/quitting ordinary apps, opening files, folders and
URLs, browser tabs and page actions, messaging (Slack, Messages, Mail), notes, reminders,
calendar, media playback, volume, notifications, screenshots, appearance (dark mode).

Blocked by design: anything that changes how the machine or a dev environment is set up.
System Settings and preference writes, shell commands beyond a tiny allowlist, package
managers and Python/Node/Homebrew tooling, dotfiles and system paths, terminals, power
and session control, deleting or moving files, admin privileges, keychain/passwords.

The policy is enforced three times: when a tool is loaded (violating tools are never
offered to the router), when a new tool is generated (rejected instead of learned), and
right before execution on the fully rendered script (spoken arguments filled in).
Set MACBROW_POLICY=off to disable for debugging; the agent will announce that at start.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass

ENABLED = os.environ.get("MACBROW_POLICY", "strict").lower() != "off"

# --- Applications voice control may never script or open -------------------------------
BLOCKED_APPS = {
    "system settings",
    "system preferences",
    "terminal",
    "iterm",
    "iterm2",
    "ghostty",
    "warp",
    "alacritty",
    "kitty",
    "hyper",
    "xcode",
    "activity monitor",
    "disk utility",
    "keychain access",
    "console",
    "script editor",
    "automator",
    "installer",
    "migration assistant",
    "directory utility",
    "system information",
    "boot camp assistant",
    "screen sharing",
    "docker",
    "docker desktop",
    "orbstack",
    "utm",
    "parallels desktop",
    "vmware fusion",
    "1password",
    "bitwarden",
    "little snitch",
    "lulu",
}

# --- Shell: `do shell script` is allowed only for these command heads ------------------
ALLOWED_SHELL_HEADS = {
    "open",
    "screencapture",
    "date",
    "pmset",
    "pbcopy",
    "pbpaste",
    "say",
    "afplay",
    "osascript_denied_placeholder",
}
ALLOWED_SHELL_HEADS.discard("osascript_denied_placeholder")
# pmset is read-only only: `pmset -g ...`
READONLY_ONLY = {"pmset": re.compile(r"^pmset\s+-g\b")}

# Anything in this list anywhere in a shell line is blocked regardless of the head.
BLOCKED_SHELL_TOKENS = re.compile(
    r"(?<![\w-])("
    r"sudo|su|defaults|killall|kill|pkill|launchctl|networksetup|systemsetup|scutil|dscl|diskutil|"
    r"tmutil|softwareupdate|csrutil|spctl|tccutil|nvram|security|plutil|crontab|chmod|chown|chflags|"
    r"rm|rmdir|mv|cp|ln|mkdir|touch|tee|dd|shred|srm|"
    r"pip|pip3|python|python3|uv|uvx|pyenv|conda|poetry|pipx|brew|npm|npx|pnpm|yarn|node|bun|deno|"
    r"cargo|rustup|gem|bundle|go|docker|kubectl|git|ssh|scp|rsync|curl|wget|nc|ncat|telnet|"
    r"osascript|osacompile|automator|shutdown|reboot|halt|logout|"
    r"xattr|codesign|installer|hdiutil|bless|fdesetup|sysctl|purge|caffeinate|"
    r"export|source|eval|exec|sh|bash|zsh|fish|perl|ruby|php|env|nohup|screen|tmux"
    r")(?![\w-])",
    re.IGNORECASE,
)
SHELL_INJECTION = re.compile(r"[;&|`$><]")  # no chaining, pipes, subshells or redirects

# --- Paths voice control may never reference -------------------------------------------
BLOCKED_PATHS = re.compile(
    r"(~?/(Library|System|usr|etc|private|var|bin|sbin|opt|cores|Volumes/[^/\s\"']+/(System|Library))\b|"
    r"~/Library|/Applications/Utilities|/Applications/Python|"
    # dotfiles/dirs: the dot must start a path segment (not a hostname like docs.livekit.io)
    r"(?<![\w.-])\.(zshrc|zprofile|zshenv|bashrc|bash_profile|profile|gitconfig|npmrc|pypirc|netrc|"
    r"ssh|aws|gnupg|config|claude|venv|pyenv|local|cargo|rustup|docker|kube|lmstudio|livekit)\b(?![\w-])|"
    r"site-packages|pyproject\.toml|requirements\.txt|package\.json|uv\.lock|poetry\.lock|"
    r"Homebrew|/opt/homebrew|node_modules|\.plist\b|LaunchAgents|LaunchDaemons|Keychains)",
    re.IGNORECASE,
)

# --- AppleScript constructs that are never allowed ---------------------------------------
BLOCKED_APPLESCRIPT = [
    (re.compile(r"with administrator privileges", re.I), "administrator privileges"),
    (re.compile(r"\bpassword\b", re.I), "password handling"),
    (re.compile(r"\bshut\s*down\b|\brestart\b|\blog\s*out\b", re.I), "power/session control"),
    (re.compile(r"\bto sleep\b|\bsleep\b(?!\s*\()", re.I), "putting the Mac to sleep"),
    (re.compile(r"\bempty\s+(the\s+)?trash\b", re.I), "emptying the trash"),
    (re.compile(r"\bdelete\b|\bmove\s+(to\s+)?trash\b", re.I), "deleting items"),
    (re.compile(r"\bmove\b(?!\s+(to\s+)?(the\s+)?(front|back|top|bottom))", re.I), "moving files"),
    (re.compile(r"\bduplicate\b", re.I), "duplicating files"),
    (re.compile(r"\beject\b|\bmount volume\b|\bunmount\b", re.I), "mounting or ejecting volumes"),
    (
        re.compile(r"\b(dock|security|network|CD and DVD|expose|screen saver|universal access)\s+preferences\b", re.I),
        "system preference writes",
    ),
    (re.compile(r"\blogin\s+items?\b", re.I), "login items"),
    (re.compile(r"\bmake\s+new\s+(login item|user|volume)\b", re.I), "creating system objects"),
    (re.compile(r"\bkeychain\b", re.I), "keychain access"),
    (re.compile(r"\bset\s+(the\s+)?clipboard\s+to\b", re.I), "overwriting the clipboard"),
    (re.compile(r"\bdo shell script\s+[^\n]*\bwith\s+administrator", re.I), "administrator privileges"),
]

# Only appearance preferences may be changed through System Events.
_APP_TELL = re.compile(r"\b(?:tell\s+|activate\s+)?application\s+\"([^\"]+)\"", re.I)
_OPEN_APP = re.compile(r"\bopen\s+(?:-a|-b|--app|--bundle)\s+(\"[^\"]+\"|'[^']+'|\S+)", re.I)
_SHELL_LINE = re.compile(r"do shell script\s+(.+)", re.I)
_STRING_LIT = re.compile(r"\"((?:[^\"\\]|\\.)*)\"")


@dataclass(frozen=True)
class Violation:
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.detail}"


def check(script: str, scope: str | None = None) -> list[Violation]:
    """Return every policy violation in an AppleScript (empty list = allowed)."""
    if not ENABLED:
        return []
    v: list[Violation] = []
    if scope and scope.strip().lower() in BLOCKED_APPS:
        v.append(Violation("blocked app", scope))
    for m in _APP_TELL.finditer(script):
        if m.group(1).strip().lower() in BLOCKED_APPS:
            v.append(Violation("blocked app", m.group(1)))
    if BLOCKED_PATHS.search(script):
        v.append(Violation("protected path", BLOCKED_PATHS.search(script).group(0)))  # type: ignore[union-attr]
    for rx, label in BLOCKED_APPLESCRIPT:
        m = rx.search(script)
        if m:
            v.append(Violation(label, m.group(0)))
    for m in _SHELL_LINE.finditer(script):
        v.extend(_check_shell_line(m.group(1)))
    # de-duplicate, keep order
    seen: set[str] = set()
    out: list[Violation] = []
    for x in v:
        if str(x) not in seen:
            seen.add(str(x))
            out.append(x)
    return out


def _check_shell_line(rest: str) -> list[Violation]:
    """`rest` is everything after `do shell script` on that line (may concatenate literals)."""
    v: list[Violation] = []
    lits = _STRING_LIT.findall(rest)
    text = " ".join(lits) if lits else rest
    if "&" in re.sub(_STRING_LIT, "", rest) and not lits:
        v.append(Violation("shell", "command built from variables only; must start with a literal"))
    if SHELL_INJECTION.search(text):
        v.append(Violation("shell", f"chaining/pipes/redirects not allowed: {text[:60]!r}"))
    hit = BLOCKED_SHELL_TOKENS.search(text)
    if hit:
        v.append(Violation("shell", f"command not allowed: {hit.group(1)}"))
    head = ""
    if lits:
        try:
            parts = shlex.split(lits[0].replace('\\"', '"'))
        except ValueError:
            parts = lits[0].split()
        head = parts[0].lower() if parts else ""
    if head not in ALLOWED_SHELL_HEADS:
        v.append(Violation("shell", f"only {sorted(ALLOWED_SHELL_HEADS)} may be run; got {head or rest[:40]!r}"))
    elif head in READONLY_ONLY and not READONLY_ONLY[head].match(text.strip()):
        v.append(Violation("shell", f"{head} is read-only here (-g)"))
    if head == "open":
        m = _OPEN_APP.search(text)
        if m and m.group(1).strip("\"'").lower() in BLOCKED_APPS:
            v.append(Violation("blocked app", m.group(1)))
        if re.search(r"x-apple\.systempreferences|prefPane|System%20Settings", text, re.I):
            v.append(Violation("system preferences", "opening System Settings panes"))
    return v


def describe_for_llm() -> str:
    """Policy summary embedded in the code-generation prompt."""
    return (
        "POLICY (hard rules; a script that breaks one is rejected, so set feasible=false instead):\n"
        "- Allowed: launch/switch/quit ordinary apps; open files, folders and URLs; browser tabs and "
        "in-page JavaScript; messaging in Slack, Messages, Mail; Notes, Reminders, Calendar; media playback; "
        "volume; notifications; screenshots; dark mode.\n"
        "- Never script or open: System Settings/Preferences, Terminal or any terminal emulator, Xcode, "
        "Activity Monitor, Disk Utility, Keychain Access, Console, Script Editor, Automator, Docker, "
        "password managers.\n"
        f"- `do shell script` only with these commands: {', '.join(sorted(ALLOWED_SHELL_HEADS))}. "
        "No pipes, `;`, `&&`, redirects or `$()`. Never defaults/killall/launchctl/networksetup/sudo/rm/mv/cp/"
        "mkdir/chmod, and never pip/python/uv/brew/npm/node/git/ssh/curl.\n"
        "- Never reference ~/Library, /Library, /System, /usr, /etc, /private, dotfiles (.zshrc, .ssh, .config, "
        ".venv), site-packages, pyproject.toml, package.json, node_modules, Homebrew, .plist files.\n"
        "- Never delete, move, duplicate, eject, empty the trash, shut down, restart, log out, sleep, change "
        "dock/security/network/login-item preferences, touch the keychain, or use administrator privileges."
    )


# --- Browser tasks (jev-ultrafast driving the user's Chrome) -----------------------------
# Never: money, credentials, account lifecycle. Confirm first: anything that changes state on a site.
# Hard terms that never belong in a voice-driven browser task. Whether a goal *requires* buying,
# paying, signing in or changing an account is a semantic question ("look up handbags I can buy" is
# a search) and is judged by Jev in the router, not by word matching.
BROWSER_BLOCKED_GOAL = re.compile(
    r"\b(card number|credit card|debit card|cvv|iban|password|passcode|two[- ]factor|2fa|"
    r"verification code|otp|ssn|social security|passport number|"
    r"delete (my |the )?account|close (my |the )?account)\b",
    re.IGNORECASE,
)
BROWSER_CONFIRM_GOAL = re.compile(
    r"\b(add(ing)? to (my |the )?(cart|basket|bag|list|wishlist)|cart|basket|send|post|tweet|reply|submit|"
    r"book|reserve|order|subscribe|follow|unfollow|like|comment|apply|upload|delete|remove|rsvp|accept|"
    r"decline|invite|share|message|dm|email|draft|save|create|schedule|update|edit|change|rate|review)\b",
    re.IGNORECASE,
)


def check_browser_goal(goal: str) -> list[Violation]:
    if not ENABLED:
        return []
    m = BROWSER_BLOCKED_GOAL.search(goal)
    return [Violation("browser goal", m.group(0))] if m else []


def browser_goal_needs_confirm(goal: str) -> bool:
    return bool(BROWSER_CONFIRM_GOAL.search(goal))
