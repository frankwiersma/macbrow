import json
from pathlib import Path

from macbrow.applescript import _app_name

SEED = json.loads(Path("tools/seed.json").read_text())


def _tool(name: str) -> dict:
    tools = SEED if isinstance(SEED, list) else SEED.get("tools", SEED)
    if isinstance(tools, dict):
        tools = list(tools.values())
    return next(t for t in tools if t["name"] == name)


def test_app_name_strips_bundle_suffix():
    assert _app_name("Microsoft Teams.app") == "Microsoft Teams"
    assert _app_name("  Finder.app  ") == "Finder"
    assert _app_name("Finder") == "Finder"
    assert _app_name("Scrapp") == "Scrapp"  # only a trailing ".app" is a suffix


def test_open_app_matches_on_bundle_name_not_process_name():
    """A process name ("MSTeams") is not the app name ("Microsoft Teams"): matching the
    rendered {{app}} against a process name made "open Teams" fail with -1728."""
    script = _tool("open_app")["script"]
    assert 'process "{{app}}"' not in script
    assert "name of file of first application process" in script
