"""Tool registry: AppleScript tools with typed argument slots.

Tools live in two JSON files: ``tools/seed.json`` (checked in) and
``tools/learned.json`` (written at runtime when the LLM generates a new tool).
Each tool declares:

- ``scope``: ``null`` for global/system tools, or an application name. App-scoped
  tools are only offered to Jev while that app is running, which keeps the Choice
  small and relevant to the current desktop.
- ``args``: argument slots. ``enum`` slots become Jev Choice questions with fixed
  criteria (or ``"dynamic": "running_apps" | "installed_apps"`` to fill criteria
  from the live environment). ``text`` slots are filled by span selection.
- ``script``: AppleScript with ``{{arg_name}}`` placeholders.
- ``speak``: ``"done"`` (confirm), ``"result"`` (read the script output aloud), or a
  template string containing ``{result}``.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from . import chrome, policy
from .applescript import MacContext, escape_applescript_string

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
SEED_PATH = TOOLS_DIR / "seed.json"
LEARNED_PATH = TOOLS_DIR / "learned.json"

MAX_CHOICE_OPTIONS = 255  # Jev Choice limit
BUILTIN_PLACEHOLDERS = {"chrome_profile", "chrome_home"}  # filled by chrome.system_vars(), never by Jev

ArgKind = Literal["enum", "text"]
DynamicSource = Literal["running_apps", "installed_apps", "apps"]

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,48}$")

# Patterns that mark a script as needing spoken confirmation before it runs.
RISKY_PATTERNS = re.compile(
    r"(delete|trash|erase|empty\s+trash|rm\s+-r|rm\s+-f|\bmove\b|\bmv\s|shutdown|shut\s+down|restart|"
    r"log\s*out|sudo|keystroke|key\s+code|diskutil|kill\s|killall|format|"
    r"System Preferences|System Settings|security|password|osascript)",
    re.IGNORECASE,
)


@dataclass
class ArgSpec:
    name: str
    kind: ArgKind
    instructions: str
    criteria: dict[str, str | None] = field(default_factory=dict)
    dynamic: DynamicSource | None = None
    default: str | None = None

    def resolve_criteria(self, ctx: MacContext) -> dict[str, str | None]:
        if self.dynamic == "running_apps":
            return {a: None for a in ctx.running_apps[:MAX_CHOICE_OPTIONS]}
        if self.dynamic == "installed_apps":
            return {a: None for a in ctx.installed_apps[:MAX_CHOICE_OPTIONS]}
        if self.dynamic == "apps":
            merged = list(ctx.running_apps) + [a for a in ctx.installed_apps if a not in ctx.running_apps]
            return {a: ("running" if a in ctx.running_apps else None) for a in merged[:MAX_CHOICE_OPTIONS]}
        return dict(self.criteria)


@dataclass
class Tool:
    name: str
    description: str
    script: str
    scope: str | None = None
    args: list[ArgSpec] = field(default_factory=list)
    speak: str = "done"
    risky: bool = False
    source: Literal["seed", "learned"] = "seed"
    examples: list[str] = field(default_factory=list)
    not_for: str | None = None
    verified: float | None = None  # Jev's p(script works), set for learned tools
    computed: dict[str, dict[str, str]] = field(default_factory=dict)  # {{name}} <- resolvers.run(fn, args[from])
    runner: str = "applescript"  # "applescript" | "browser" (jev-ultrafast web task; script unused)

    @property
    def blocked_by(self) -> list[str]:
        """Policy violations for this tool's template (arguments filled with placeholders)."""
        rendered = self.render({a.name: (a.default or "x") for a in self.args})
        return [str(v) for v in policy.check(rendered, self.scope)]

    @property
    def blocked(self) -> bool:
        return bool(self.blocked_by)

    def is_available(self, ctx: MacContext) -> bool:
        if self.blocked:
            return False
        if self.scope is None:
            return True
        return self.scope.lower() in {a.lower() for a in ctx.running_apps}

    def render(self, args: dict[str, str]) -> str:
        script = self.script
        for spec in self.args:
            value = args.get(spec.name, spec.default or "")
            script = script.replace("{{" + spec.name + "}}", escape_applescript_string(value))
        for name, value in chrome.system_vars().items():  # built-ins: {{chrome_profile}}, {{chrome_home}}
            script = script.replace("{{" + name + "}}", escape_applescript_string(value))
        for name in self.computed:  # filled by the agent via resolvers before execution
            if name in args:
                script = script.replace("{{" + name + "}}", escape_applescript_string(args[name]))
        return script

    def choice_description(self) -> dict[str, Any]:
        desc: dict[str, Any] = {"what": self.description}
        if self.scope:
            desc["app"] = self.scope
        if self.examples:
            desc["examples"] = self.examples[:4]
        if self.not_for:
            desc["not_for"] = self.not_for
        return desc

    @classmethod
    def from_dict(cls, d: dict[str, Any], source: str) -> Tool:
        args = [ArgSpec(**a) for a in d.get("args", [])]
        return cls(
            name=d["name"],
            description=d["description"],
            script=d["script"],
            scope=d.get("scope"),
            args=args,
            speak=d.get("speak", "done"),
            risky=bool(d.get("risky", False)) or bool(RISKY_PATTERNS.search(d["script"])),
            source=source,  # type: ignore[arg-type]
            examples=list(d.get("examples", [])),
            not_for=d.get("not_for"),
            verified=d.get("verified"),
            computed=dict(d.get("computed") or {}),
            runner=d.get("runner", "applescript"),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("source", None)
        return d

    def policy_report(self) -> str:
        return "blocked: " + "; ".join(self.blocked_by) if self.blocked else "allowed"


class PolicyError(ValueError):
    def __init__(self, violations: list[str]):
        super().__init__("; ".join(violations))
        self.violations = violations


class ToolRegistry:
    """In-memory registry backed by seed + learned JSON files. Thread-safe writes."""

    def __init__(self, seed_path: Path = SEED_PATH, learned_path: Path = LEARNED_PATH):
        self._seed_path = seed_path
        self._learned_path = learned_path
        self._lock = threading.Lock()
        self.tools: dict[str, Tool] = {}
        self.reload()

    def reload(self) -> None:
        tools: dict[str, Tool] = {}
        for path, source in ((self._seed_path, "seed"), (self._learned_path, "learned")):
            if not path.exists():
                continue
            for raw in json.loads(path.read_text() or "[]"):
                tool = Tool.from_dict(raw, source)
                tools[tool.name] = tool
        self.tools = tools

    def available(self, ctx: MacContext) -> list[Tool]:
        return [t for t in self.tools.values() if t.is_available(ctx)]

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def add_learned(self, tool: Tool) -> Tool:
        """Insert or replace a learned tool and persist it. Raises PolicyError if it violates policy."""
        if tool.blocked:
            raise PolicyError(tool.blocked_by)
        if not _NAME_RE.match(tool.name):
            tool.name = re.sub(r"[^a-z0-9_]+", "_", tool.name.lower()).strip("_")[:48] or "learned_tool"
        if tool.name in self.tools and self.tools[tool.name].source == "seed":
            tool.name = f"{tool.name}_v2"
        tool.source = "learned"
        with self._lock:
            self.tools[tool.name] = tool
            learned = [t.to_dict() for t in self.tools.values() if t.source == "learned"]
            self._learned_path.parent.mkdir(parents=True, exist_ok=True)
            self._learned_path.write_text(json.dumps(learned, indent=2, ensure_ascii=False) + "\n")
        return tool

    def remove_learned(self, name: str) -> bool:
        tool = self.tools.get(name)
        if not tool or tool.source != "learned":
            return False
        with self._lock:
            del self.tools[name]
            learned = [t.to_dict() for t in self.tools.values() if t.source == "learned"]
            self._learned_path.write_text(json.dumps(learned, indent=2, ensure_ascii=False) + "\n")
        return True
