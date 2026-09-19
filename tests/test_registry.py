import json

from macbrow import chrome, registry
from macbrow.applescript import MacContext
from macbrow.registry import ArgSpec, PolicyError, Tool, ToolRegistry

CTX = MacContext(active_app="Finder", running_apps=["Finder", "Safari"], installed_apps=["Finder", "Safari", "Slack"])


def test_render_escapes_arguments_and_fills_builtins(monkeypatch):
    monkeypatch.setattr(chrome, "system_vars", lambda: {"chrome_profile": "Profile 9", "chrome_home": ""})
    tool = Tool(
        name="t",
        description="d",
        script='say "{{msg}}" -- {{chrome_profile}}',
        args=[ArgSpec(name="msg", kind="text", instructions="x")],
    )
    rendered = tool.render({"msg": 'he said "hi" \\ bye'})
    assert rendered == 'say "he said \\"hi\\" \\\\ bye" -- Profile 9'


def test_scoped_tool_only_available_when_app_running():
    tool = Tool(name="safari_x", description="d", script='tell application "Safari" to activate', scope="Safari")
    assert tool.is_available(CTX)
    assert not tool.is_available(MacContext("Finder", ["Finder"], []))


def test_blocked_tool_is_never_available():
    tool = Tool(name="bad", description="d", script='tell application "Terminal" to activate')
    assert tool.blocked and "Terminal" in tool.policy_report()
    assert not tool.is_available(CTX)


def test_dynamic_enum_criteria_follow_context():
    spec = ArgSpec(name="app", kind="enum", instructions="which", dynamic="apps")
    crit = spec.resolve_criteria(CTX)
    assert list(crit)[:2] == ["Finder", "Safari"]  # running apps first
    assert crit["Finder"] == "running" and crit["Slack"] is None


def test_learned_tools_persist_and_policy_gate(tmp_path):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps([{"name": "seeded", "description": "d", "script": 'display notification "x"'}]))
    reg = ToolRegistry(seed_path=seed, learned_path=tmp_path / "learned.json")
    ok = reg.add_learned(Tool(name="new_tool", description="d", script='display notification "y"'))
    assert ok.source == "learned"
    assert json.loads((tmp_path / "learned.json").read_text())[0]["name"] == "new_tool"
    try:
        reg.add_learned(Tool(name="evil", description="d", script='do shell script "rm -rf ~"'))
        raise AssertionError("policy should reject")
    except PolicyError as e:
        assert "rm" in str(e)
    # a learned tool may not shadow a seed tool
    shadow = reg.add_learned(Tool(name="seeded", description="d", script='display notification "z"'))
    assert shadow.name == "seeded_v2"
    assert reg.remove_learned("new_tool") and not reg.remove_learned("seeded")


def test_risky_detection_from_script():
    t = Tool.from_dict({"name": "x", "description": "d", "script": 'tell application "Finder" to move f to g'}, "seed")
    assert t.risky
    assert registry.RISKY_PATTERNS.search("keystroke")
