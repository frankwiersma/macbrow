"""Fallback tier: an LLM writes a new AppleScript tool for an unseen request.

Default backend is LiveKit Inference (hosted, billed to your LiveKit Cloud project; needs
LIVEKIT_URL/API_KEY/API_SECRET). Set MACBROW_LLM_PROVIDER=openai to use any OpenAI-compatible endpoint instead.

Runs once per novel intent (a couple of seconds); the result is persisted to
``tools/learned.json`` so Jev routes to it in ~150ms next time. Generated tools
are validated for syntax with ``osacompile`` before they are registered.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import re
import tempfile
from typing import Any, Literal

import openai
from livekit.agents import inference, llm
from livekit.agents.llm import utils as llm_utils
from pydantic import BaseModel, Field
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul

from . import policy
from .applescript import MacContext
from .registry import BUILTIN_PLACEHOLDERS, RISKY_PATTERNS, ArgSpec, PolicyError, Tool, ToolRegistry

log = logging.getLogger("macbrow.generator")

PROVIDER = os.environ.get("MACBROW_LLM_PROVIDER", "livekit")  # "livekit" | "openai"
# "openai" is any OpenAI-compatible endpoint: LM Studio, Ollama, api.openai.com, a local gateway.
OPENAI_BASE_URL = os.environ.get("MACBROW_OPENAI_BASE_URL", "http://localhost:1234/v1")
OPENAI_API_KEY = os.environ.get("MACBROW_OPENAI_API_KEY", "local")
MODEL = os.environ.get("MACBROW_CODEGEN_MODEL", "openai/gpt-5-mini" if PROVIDER == "livekit" else "qwen/qwen3.5-9b")
# LiveKit: "low" is plenty for short scripts. Local Qwen 3.5: "none", otherwise the model spends
# the whole budget in the reasoning channel and returns empty text.
REASONING_EFFORT = os.environ.get("MACBROW_REASONING_EFFORT", "low" if PROVIDER == "livekit" else "none")
MAX_ATTEMPTS = int(os.environ.get("MACBROW_CODEGEN_ATTEMPTS", "3"))  # first draft + compiler-guided repairs
MAX_OUTPUT_TOKENS = int(os.environ.get("MACBROW_CODEGEN_MAX_TOKENS", "1500"))  # bounds a runaway generation
# Jev reviews each compiled script: p(script really performs the request). Below this it is sent
# back for repair. Probed values: fake/invented scripts 0.05-0.37, working ones 0.49-0.83.
VERIFY_THRESHOLD = float(os.environ.get("MACBROW_VERIFY_THRESHOLD", "0.4"))

# A script must contain at least one line that plausibly *does* something. Small models
# happily emit `tell application "Slack" / set x to y / return "done"`, which compiles and lies.
EFFECT_PATTERNS = re.compile(
    r"(do shell script|keystroke|key code|\bclick\b|open location|\bopen\b|activate|\bquit\b|make new|"
    r"\bdelete\b|set volume|display (notification|dialog|alert)|reload|\bclose\b|execute .*javascript|"
    r"do javascript|playpause|\bplay\b|\bpause\b|next track|previous track|empty trash|"
    r"set [\w\s']+ of [\w\s']+ to|set (autohide|dark mode|visible|frontmost|value|url|zoom|volume|"
    r"minimized|bounds|position|size|selected|current tab|index) to|\bsave\b|\bprint\b|\bsend\b|"
    r"\bmove\b|\bduplicate\b|\breveal\b|\bselect\b|\bsleep\b|\brestart\b|\bshut down\b|\blog out\b|"
    r"\breturn\b.*\b(of|'s)\b|\bget\b.*\bof\b)",
    re.IGNORECASE,
)


def _has_effect(script: str) -> bool:
    return any(EFFECT_PATTERNS.search(line) for line in script.splitlines() if not line.strip().startswith("--"))


class GeneratedCriterion(BaseModel):
    value: str = Field(description="The enum value as it will appear in the script placeholder.")
    description: str = Field(description="Short meaning of this value, or empty string.")


class GeneratedArg(BaseModel):
    name: str = Field(description="Placeholder name used in the script as {{name}}.")
    kind: Literal["enum", "text"] = Field(description="enum for a closed set of values, text for free-form content.")
    instructions: str = Field(description="Question the router asks to fill this slot from the user's words.")
    criteria: list[GeneratedCriterion] = Field(description="For enum args: 2-30 allowed values. Empty for text args.")
    default: str | None = Field(description="Value used when the user does not specify one, or null.")


class GeneratedTool(BaseModel):
    feasible: bool = Field(
        description="False only if the request truly cannot be done with AppleScript/shell on macOS."
    )
    reason: str = Field(description="One spoken sentence explaining why, if not feasible; else empty string.")
    tool_name: str = Field(description="snake_case, prefixed with the app name when app-specific, e.g. chrome_zoom.")
    description: str = Field(
        description="One sentence: what the tool does. Used by the router to match future requests."
    )
    scope: str | None = Field(
        description="Exact macOS application name the tool controls (e.g. 'Google Chrome'), or null for system-wide tools."
    )
    args: list[GeneratedArg] = Field(description="Argument slots referenced in the script as {{name}}.")
    script: str = Field(
        description="Complete AppleScript with {{arg}} placeholders (substituted as escaped string literals). Ends with `return` of a short result string."
    )
    speak: str = Field(
        description="'done' to just confirm, 'result' to read the returned string aloud, or a template containing {result}."
    )
    examples: list[str] = Field(description="2-4 alternative phrasings of the request.")


# OpenAI-style response_format dict, used for the LM Studio path (LiveKit takes the class directly).
TOOL_RESPONSE_FORMAT: dict[str, Any] = llm_utils.to_openai_response_format(GeneratedTool)


SYSTEM = """You write a NEW AppleScript tool for a voice-controlled macOS assistant.

You receive the user's spoken request, the app in front, the running apps, and the names of
tools that ALREADY exist. Your job is to create one tool that does NOT exist yet and that
fulfils the request. The existing list is only there so you pick a different, non-clashing
tool_name; it is never a reason to say the request is infeasible.

Rules:
- Set feasible=true whenever AppleScript (optionally with `do shell script`) can plausibly do it. Only set feasible=false for things macOS truly cannot do this way (e.g. sending money, reading another user's files) and explain in one spoken sentence.
- Prefer the app's scripting dictionary (`tell application "X"`). If the app has no dictionary, fall back to `do shell script` (e.g. `defaults write`, `open`, `osascript`-free CLI tools) or System Events GUI scripting as a last resort.
- Implement exactly the requested action and nothing more: no extra modes, no unrelated toggles, no GUI clicking when a direct command exists. Short scripts (under 15 lines) are best.
- Handy direct commands: System Settings panes open with `do shell script "open 'x-apple.systempreferences:com.apple.<pane>'"` (e.g. com.apple.wifi-settings-extension, com.apple.Bluetooth-Settings.extension, com.apple.Sound-Settings.extension, com.apple.Displays-Settings.extension); Wi-Fi power is `networksetup -setairportpower Wi-Fi on|off`; Dock/Finder settings via `defaults write` then `killall Dock` / `killall Finder`; the frontmost app via `tell application "System Events" to get name of first application process whose frontmost is true`.
- Google Chrome: never use `tell application "Google Chrome"` to open URLs or windows (it lands in the wrong profile). Always launch through the shell with the built-in placeholder {{chrome_profile}} (not an arg):
    do shell script "open -na 'Google Chrome' --args " & quoted form of ("--profile-directory=" & "{{chrome_profile}}") & " " & quoted form of theURL
  Reading or acting on the current tab (`URL of active tab of front window`, `execute ... javascript`, `reload`) via the Chrome dictionary is fine.
- scope is the exact macOS application name the tool controls ("Google Chrome", "Slack") or null for system-wide tools.
- Parameterise values the user is likely to vary as {{arg}} placeholders. Enum args need 2-30 concrete values with short descriptions; use kind "text" only for free-form content. Placeholders are substituted as escaped string literals, so write them inside double quotes.
- AppleScript has NO inline if / ternary expression.
  WRONG:  return "Dock " & (if hidden then "hidden" else "shown")
  RIGHT:  if hidden then
            set r to "hidden"
          else
            set r to "shown"
          end if
          return "Dock " & r
  Keep the final return simple; a fixed string like return "dock hidden" is fine.
- String concatenation is &. Do not name variables text, type, status, name, result, or other reserved words. Coerce numbers with `as text`.
- Only use commands and properties you are certain exist in that app's scripting dictionary. Chrome/Safari expose tabs, windows, URL, reload, and `execute javascript`/`do JavaScript`; use JavaScript for page-level actions like zoom. Electron apps such as Slack, Discord, and VS Code have NO dictionary: use their URL schemes via `open location`, `do shell script`, or System Events keystrokes.
- Never return a script that only pretends: assigning variables and returning a success string is not an action. If the app has no scripting dictionary and no URL scheme or shell command reaches the feature, either drive it with System Events keystrokes/menu clicks, or set feasible=false and say so.
- Never embed secrets, never ask for passwords, never call osascript from the script.
- End the script with `return` of a short result string that can be read aloud.
- speak: "done" to just confirm, "result" to read the returned string aloud, or a template containing {result}.

Example
request: "toggle the dock hiding"  frontmost: Finder
answer: {"feasible": true, "reason": "", "tool_name": "dock_toggle_autohide", "description": "Turn Dock auto-hiding on or off.", "scope": null,
 "args": [{"name": "mode", "kind": "enum", "instructions": "Should the Dock auto-hide be turned on, off, or toggled?", "criteria": [{"value": "on", "description": "hide the Dock"}, {"value": "off", "description": "keep the Dock visible"}, {"value": "toggle", "description": "flip the current setting"}], "default": "toggle"}],
 "script": "set m to \"{{mode}}\"\ntell application \"System Events\" to tell dock preferences\n  if m is \"on\" then\n    set autohide to true\n  else if m is \"off\" then\n    set autohide to false\n  else\n    set autohide to not autohide\n  end if\n  return \"dock autohide \" & autohide\nend tell",
 "speak": "result", "examples": ["hide the dock", "show the dock", "toggle dock hiding"]}
(Note: that example predates the policy below and would now be rejected as a dock preference write;
it is kept only to show the JSON shape.)

""" + policy.describe_for_llm()


class ToolGenerator:
    def __init__(
        self,
        registry: ToolRegistry,
        client: openai.AsyncOpenAI | None = None,
        jev: AsyncTypeSafeClient | None = None,
    ):
        self.registry = registry
        self.jev = jev
        self.provider = PROVIDER if client is None else "openai"
        self.client: openai.AsyncOpenAI | None = None
        self.lk_llm: inference.LLM | None = None
        if self.provider == "livekit":
            self.lk_llm = inference.LLM(model=MODEL, extra_kwargs={"reasoning_effort": REASONING_EFFORT})
        else:
            self.client = client or openai.AsyncOpenAI(
                base_url=OPENAI_BASE_URL,
                api_key=OPENAI_API_KEY,
                timeout=90.0,
                max_retries=1,
            )

    async def generate(self, utterance: str, ctx: MacContext) -> tuple[Tool | None, str]:
        """Returns (tool, spoken_message). tool is None when generation failed.

        If the generated tool duplicates an existing available one, that existing tool is
        returned instead with message == DUPLICATE and nothing new is registered.
        """
        user = "Create a NEW tool for this request.\n" + json.dumps(
            {
                "request": utterance,
                "frontmost_app": ctx.active_app,
                "running_apps": ctx.running_apps,
                "existing_tool_names_to_avoid": [t.name for t in self.registry.tools.values()],
            },
            ensure_ascii=False,
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        last_error = ""
        policy_strikes = 0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                text, finish = await self._complete(messages)
            except _Unavailable as e:
                log.error("codegen backend unavailable: %s", e)
                return None, str(e)
            try:
                data = GeneratedTool.model_validate_json(text).model_dump() if text.strip() else None
            except ValueError:  # JSONDecodeError / pydantic ValidationError
                data = None
            if data is None:
                # Truncated or degenerate output: retry from scratch rather than feeding it back.
                last_error = f"unparseable output (finish_reason={finish}, {len(text)} chars)"
                log.warning("codegen attempt %d/%d: %s", attempt, MAX_ATTEMPTS, last_error)
                continue
            if not data.get("feasible", False):
                return None, data.get("reason") or "I can't do that with AppleScript."

            tool = self._to_tool(data)
            violations = policy.check(tool.render({a.name: (a.default or "x") for a in tool.args}), tool.scope)
            if violations:
                policy_strikes += 1
                log.warning(
                    "codegen attempt %d/%d violates policy: %s", attempt, MAX_ATTEMPTS, "; ".join(map(str, violations))
                )
                if policy_strikes >= 2:
                    return None, "That would change how this Mac is set up, which voice control isn't allowed to do."
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That script violates the POLICY: "
                            + "; ".join(map(str, violations))
                            + ". Either achieve the request within the allowed actions (apps, files, browser, messaging, notes) "
                            "or set feasible=false with a one-sentence spoken reason."
                        ),
                    }
                )
                continue
            ok, err = await _compiles(tool.render({a.name: (a.default or "x") for a in tool.args}))
            if ok and not _has_effect(tool.script):
                ok, err = (
                    False,
                    (
                        "The script compiles but has no effect: it only assigns variables and returns a string. "
                        "It must actually perform the action (a scripting-dictionary command, `do shell script`, "
                        "`open location`, or System Events keystrokes/clicks). If that isn't possible, set feasible=false."
                    ),
                )
            if ok:
                tool.verified = await self._verify(utterance, tool)
                if tool.verified is not None and tool.verified < VERIFY_THRESHOLD:
                    ok, err = (
                        False,
                        (
                            f"A reviewer judged this script unlikely to actually perform the request "
                            f"(confidence {tool.verified:.0%}). Re-check that every command, application, shell tool, "
                            "and flag really exists and does what you assume; simplify, or set feasible=false."
                        ),
                    )
            if ok:
                break
            last_error = err
            log.warning("codegen attempt %d/%d failed to compile: %s", attempt, MAX_ATTEMPTS, err)
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"osacompile rejected that script:\n{err}\n\n"
                        "Return the corrected tool as JSON. Fix only what is needed. If the app has no "
                        "scripting command for this, switch to JavaScript (browsers), `do shell script`, "
                        "or System Events instead of guessing dictionary terms."
                    ),
                }
            )
        else:
            log.warning("codegen gave up after %d attempts: %s", MAX_ATTEMPTS, last_error)
            if "unparseable" in last_error:
                return None, "The local model kept returning something I couldn't parse."
            return None, "I tried to write a script for that but couldn't get a working one, so I didn't run it."

        dup = await self.find_duplicate(tool, self.registry.available(ctx))
        if dup is not None:
            return dup, DUPLICATE
        try:
            tool = self.registry.add_learned(tool)  # final gate; should not trigger after the check above
        except PolicyError as e:
            log.error("policy rejected tool at registration: %s", e)
            return None, "That would change how this Mac is set up, which voice control isn't allowed to do."
        log.info("learned tool %s (scope=%s, args=%s)", tool.name, tool.scope, [a.name for a in tool.args])
        return tool, ""

    async def _complete(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        """One structured-output completion on the configured backend -> (text, finish_reason)."""
        if self.lk_llm is not None:
            ctx = llm.ChatContext()
            for m in messages:
                ctx.add_message(role=m["role"], content=m["content"])  # type: ignore[arg-type]
            try:
                stream = self.lk_llm.chat(chat_ctx=ctx, response_format=GeneratedTool)
                async with stream:
                    parts = [c async for c in stream.to_str_iterable()]
            except Exception as e:  # APIConnectionError, APIStatusError, auth errors
                raise _Unavailable(f"LiveKit Inference request failed: {_short(e)}") from e
            return "".join(parts), "stop"
        assert self.client is not None
        try:
            resp = await self.client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.2,
                messages=messages,  # type: ignore[arg-type]
                response_format=TOOL_RESPONSE_FORMAT,  # type: ignore[arg-type]
                extra_body={"reasoning_effort": REASONING_EFFORT},
            )
        except openai.APIConnectionError as e:
            raise _Unavailable("The local model server isn't running, so I can't learn that action.") from e
        except openai.APIStatusError as e:
            raise _Unavailable(f"The local model couldn't write that action ({e.status_code}).") from e
        return resp.choices[0].message.content or "", resp.choices[0].finish_reason or ""

    async def aclose(self) -> None:
        if self.lk_llm is not None:
            await self.lk_llm.aclose()
        if self.client is not None:
            await self.client.close()

    async def find_duplicate(self, new_tool: Tool, available: list[Tool]) -> Tool | None:
        """If an existing available tool already does what the new one does, return it.

        Guards against the router sending a re-phrased request to codegen and the registry
        filling up with near-identical tools (three Slack DM tools in one session).
        """
        if self.jev is None or not available:
            return None
        criteria: dict[str, Any] = {t.name: t.choice_description() for t in available[:250]}
        criteria["__none__"] = "No existing tool does the same thing as the new tool."
        try:
            resp = await self.jev.system_one(
                state={
                    "new_tool": {
                        "description": new_tool.description,
                        "app": new_tool.scope,
                        "arguments": [a.name for a in new_tool.args],
                    }
                },
                questions={
                    "same_as": Choice(
                        instructions=(
                            "`new_tool` was just written for a voice assistant. Which existing tool performs the same "
                            "action with the same kind of arguments (differences in wording do not matter)? "
                            "Choose __none__ if the new tool does something no listed tool does."
                        ),
                        criteria=criteria,
                    )
                },
            )
        except Exception:
            log.exception("duplicate check failed; skipping")
            return None
        ans = resp.choices["same_as"]
        if ans.choice != "__none__" and ans.confidence >= 0.6:
            log.info("new tool %s duplicates existing %s (conf %.2f)", new_tool.name, ans.choice, ans.confidence)
            return self.registry.get(ans.choice)
        return None

    async def _verify(self, request: str, tool: Tool) -> float | None:
        """Ask Jev whether the script would really perform the request. None if Jev is unavailable."""
        if self.jev is None:
            return None
        try:
            resp = await self.jev.system_one(
                state={"request": request, "script": tool.script, "arguments": [a.name for a in tool.args]},
                questions={
                    "works": Noul(
                        instructions=(
                            "`script` is an AppleScript a voice assistant wrote to carry out `request` on macOS "
                            "(`arguments` are placeholders filled in before running). Would running this script "
                            "actually accomplish the request? Judge whether every command, application, shell tool, "
                            "and flag used really exists and does what the script assumes; a script that only assigns "
                            "variables, invents commands or flags, or drives the wrong UI does not accomplish it."
                        ),
                        criteria={
                            "true": "The script would really perform the request as written.",
                            "false": "It would fail, do nothing, do something else, or relies on commands/flags that do not exist.",
                        },
                    )
                },
            )
            return float(resp.nouls["works"].noul)
        except Exception:
            log.exception("Jev verification failed; skipping")
            return None

    def _to_tool(self, data: dict[str, Any]) -> Tool:
        tool = Tool(
            name=data["tool_name"],
            description=data["description"],
            script=data["script"],
            scope=data.get("scope") or None,
            args=[
                ArgSpec(
                    name=a["name"],
                    kind=a["kind"],
                    instructions=a["instructions"],
                    criteria={c["value"]: (c["description"] or None) for c in (a.get("criteria") or [])},
                    default=a.get("default"),
                )
                for a in data.get("args", [])
            ],
            speak=data.get("speak") or "done",
            risky=bool(RISKY_PATTERNS.search(data["script"])),
            source="learned",
            examples=list(data.get("examples", [])),
        )
        # Built-in placeholders are filled by the system, never by Jev: drop them if the model declared them.
        tool.args = [a for a in tool.args if a.name not in BUILTIN_PLACEHOLDERS]
        # Every other {{placeholder}} in the script must be a declared arg.
        declared = {a.name for a in tool.args}
        for ph in set(re.findall(r"\{\{(\w+)\}\}", tool.script)) - BUILTIN_PLACEHOLDERS:
            if ph not in declared:
                tool.args.append(
                    ArgSpec(name=ph, kind="text", instructions=f"The value for {ph} in the user's request.")
                )
        return tool


DUPLICATE = "__duplicate__"  # generate() message meaning: returned tool is an existing one


class _Unavailable(RuntimeError):
    """The codegen backend could not serve the request; message is user-speakable."""


def _short(e: BaseException, n: int = 120) -> str:
    msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return msg if len(msg) <= n else msg[: n - 1] + "…"


async def _compiles(script: str) -> tuple[bool, str]:
    """Syntax/dictionary check with osacompile. Returns (ok, error_with_line_context)."""
    with tempfile.TemporaryDirectory() as d:
        src = f"{d}/tool.applescript"
        pathlib.Path(src).write_text(script)
        proc = await asyncio.create_subprocess_exec(
            "osacompile",
            "-o",
            f"{d}/tool.scpt",
            src,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await asyncio.wait_for(proc.communicate(), 20)
    if proc.returncode == 0:
        return True, ""
    msg = err.decode("utf-8", "replace").strip()
    # "…/tool.applescript:10: error: Expected expression but found “if”. (-2741)"
    m = re.search(r":(\d+): error: (.*)$", msg, re.MULTILINE)
    if not m:
        return False, msg
    line_no, detail = int(m.group(1)), m.group(2).strip()
    lines = script.splitlines()
    line_text = lines[line_no - 1].strip() if 0 < line_no <= len(lines) else ""
    hint = ""
    if re.search(r"\(\s*if\b|\bif\b.*\bthen\b.*\belse\b", line_text):
        hint = " AppleScript has no inline if/ternary expression: compute the value with a multi-line if/else block into a variable, then return the variable."
    return False, f"line {line_no}: {detail} Offending line: `{line_text}`.{hint}"
