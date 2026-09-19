"""Chrome profile resolution: which --profile-directory matches the configured account.

Chrome opens new windows in the last-used profile, which on a multi-profile Mac is often
the wrong account. Every Chrome-launching script uses

    open -na "Google Chrome" --args --profile-directory=<dir> [--new-window] [URL]

with <dir> filled from the `{{chrome_profile}}` built-in placeholder, resolved here by
matching MACBROW_CHROME_PROFILE_EMAIL against Chrome's Local State file.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger("macbrow.chrome")

LOCAL_STATE = Path.home() / "Library/Application Support/Google/Chrome/Local State"
PROFILE_EMAIL = os.environ.get("MACBROW_CHROME_PROFILE_EMAIL", "")  # empty: use Chrome's last-used profile
PROFILE_DIR_OVERRIDE = os.environ.get("MACBROW_CHROME_PROFILE_DIR")  # e.g. "Profile 4"
HOME_URL = os.environ.get("MACBROW_CHROME_HOME", "")  # empty = Chrome's new-tab page

_cache: tuple[float, str] | None = None
_TTL = 60.0


def profiles() -> dict[str, dict[str, str]]:
    """{profile_dir: {"name":..., "email":...}} from Local State; empty if unreadable."""
    try:
        info = json.loads(LOCAL_STATE.read_text())["profile"]["info_cache"]
    except (OSError, KeyError, ValueError):
        return {}
    return {k: {"name": v.get("name", ""), "email": v.get("user_name", "")} for k, v in info.items()}


def profile_dir() -> str:
    """Directory name for the configured account. Falls back to Chrome's last-used profile."""
    global _cache
    if PROFILE_DIR_OVERRIDE:
        return PROFILE_DIR_OVERRIDE
    now = time.monotonic()
    if _cache and now - _cache[0] < _TTL:
        return _cache[1]
    found = ""
    if PROFILE_EMAIL:
        for d, meta in profiles().items():
            if meta["email"].lower() == PROFILE_EMAIL.lower():
                found = d
                break
    if not found:
        try:
            found = json.loads(LOCAL_STATE.read_text())["profile"].get("last_used", "Default")
        except (OSError, KeyError, ValueError):
            found = "Default"
        if PROFILE_EMAIL:
            log.warning("no Chrome profile signed in as %s; using %s", PROFILE_EMAIL, found)
    _cache = (now, found)
    return found


def system_vars() -> dict[str, str]:
    """Built-in placeholders every tool script may use."""
    return {"chrome_profile": profile_dir(), "chrome_home": HOME_URL}
