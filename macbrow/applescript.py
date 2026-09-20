"""Thin async wrappers around osascript and macOS environment probing."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

OSASCRIPT_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class ScriptResult:
    ok: bool
    output: str
    error: str = ""

    @property
    def text(self) -> str:
        return self.output if self.ok else self.error


async def run_applescript(script: str, timeout: float = OSASCRIPT_TIMEOUT_S) -> ScriptResult:
    """Run an AppleScript source string with osascript and capture its result."""
    proc = await asyncio.create_subprocess_exec(
        "osascript",
        "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(script.encode("utf-8")), timeout)
    except TimeoutError:
        proc.kill()
        return ScriptResult(False, "", f"AppleScript timed out after {timeout:.0f}s")
    output = out.decode("utf-8", "replace").strip()
    error = err.decode("utf-8", "replace").strip()
    if proc.returncode != 0:
        # osascript prefixes errors like "123:145: execution error: ... (-1728)"
        error = re.sub(r"^\d+:\d+:\s*(execution|syntax) error:\s*", "", error)
        return ScriptResult(False, output, error or f"osascript exited {proc.returncode}")
    return ScriptResult(True, output, error)


def escape_applescript_string(value: str) -> str:
    """Escape a Python string for interpolation inside an AppleScript "..." literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


# A process name ("MSTeams") is not always the app's bundle name ("Microsoft Teams"), and only the
# bundle name is what `tell application "..."` resolves and what get_installed_apps() reports. Ask
# System Events for the process's bundle file so both lists speak one vocabulary; fall back to the
# process name for the rare process that has no file.
_BUNDLE = "name of file of"
_PROC = "name of"
_FRONTMOST = 'tell application "System Events" to get {} first application process whose frontmost is true'
_RUNNING = (
    'tell application "System Events" to get {} every application process whose background only is false'
)


def _app_name(raw: str) -> str:
    """'Microsoft Teams.app' -> 'Microsoft Teams'."""
    name = raw.strip()
    return name[: -len(".app")] if name.endswith(".app") else name


# Asking for a whole list of bundle files errors outright if any process starts or exits
# mid-enumeration -- which is exactly what open_app causes. Retry before dropping to process
# names, or a launch would silently put the old vocabulary back in front of the router.
_BUNDLE_ATTEMPTS = 3


async def _app_names(template: str) -> str | None:
    for prop in (*(_BUNDLE,) * _BUNDLE_ATTEMPTS, _PROC):
        res = await run_applescript(template.format(prop), timeout=5)
        if res.ok and res.output:
            return res.output
    return None


async def get_active_app() -> str:
    out = await _app_names(_FRONTMOST)
    return _app_name(out) if out else "Finder"


async def get_running_apps() -> list[str]:
    out = await _app_names(_RUNNING)
    if not out:
        return ["Finder"]
    return sorted({_app_name(a) for a in out.split(",") if a.strip()}, key=str.lower)


_APP_DIRS = (Path("/Applications"), Path("/System/Applications"), Path.home() / "Applications")


def get_installed_apps(limit: int = 200) -> list[str]:
    """Names of installed .app bundles (top level only). Cheap filesystem scan."""
    names: set[str] = set()
    for d in _APP_DIRS:
        if not d.is_dir():
            continue
        for p in d.glob("*.app"):
            names.add(p.stem)
    return sorted(names, key=str.lower)[:limit]


@dataclass(frozen=True)
class MacContext:
    active_app: str
    running_apps: list[str]
    installed_apps: list[str]


async def snapshot_context() -> MacContext:
    active, running = await asyncio.gather(get_active_app(), get_running_apps())
    return MacContext(active_app=active, running_apps=running, installed_apps=get_installed_apps())


class ContextPoller:
    """Keeps a fresh MacContext in the background so a voice turn never waits on osascript.

    Frontmost/running apps refresh every ``interval`` seconds; the installed-app
    scan (filesystem) refreshes every ``installed_interval`` seconds.
    """

    def __init__(self, interval: float = 1.0, installed_interval: float = 60.0):
        self.interval = interval
        self.installed_interval = installed_interval
        self._ctx: MacContext | None = None
        self._task: asyncio.Task | None = None
        self._installed: list[str] = []
        self._installed_at = 0.0

    async def start(self) -> MacContext:
        if self._task is None:
            self._ctx = await self._refresh()
            self._task = asyncio.create_task(self._loop(), name="macbrow-context-poller")
        assert self._ctx is not None
        return self._ctx

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def latest(self) -> MacContext:
        """Cached snapshot; starts the poller on first use."""
        if self._ctx is None:
            return await self.start()
        return self._ctx

    async def _refresh(self) -> MacContext:
        import time

        now = time.monotonic()
        if now - self._installed_at > self.installed_interval:
            self._installed = get_installed_apps()
            self._installed_at = now
        active, running = await asyncio.gather(get_active_app(), get_running_apps())
        return MacContext(active_app=active, running_apps=running, installed_apps=self._installed)

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            try:
                self._ctx = await self._refresh()
            except Exception:  # keep polling on transient osascript failures
                pass
