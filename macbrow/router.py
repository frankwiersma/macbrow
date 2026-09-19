"""Jev routing: utterance + live Mac state -> tool, arguments, confidence.

One request selects the tool (Choice over the tools currently available) and,
speculatively in the same request, every enum argument for every candidate
tool. Only the selected tool's answers are consumed. Free-text arguments need
the selected tool first, so they are filled in a second, tiny request that
asks Jev to *select* the right span of the utterance (no generation).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul

from .applescript import MacContext
from .registry import MAX_CHOICE_OPTIONS, Tool, ToolRegistry

log = logging.getLogger("macbrow.router")

CHAT = "chat"
NEW_ACTION = "new_action"
STOP = "stop_listening"

MIN_TOOL_CONFIDENCE = 0.45  # below this we treat the pick as uncertain
NEW_ACTION_MIN_CONFIDENCE = 0.6  # hesitant new_action -> ask about the best existing tool instead
UNCERTAIN_TOOL_MIN_PROB = 0.2
MAX_TEXT_CANDIDATES = 200  # Jev Choice allows 255 options


@dataclass
class Route:
    kind: str  # "tool" | "chat" | "new_action" | "uncertain" | "stop"
    tool: Tool | None = None
    args: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    is_confirmation: float = 0.0
    is_denial: float = 0.0
    browser_followup: float = 0.0  # p(utterance continues/corrects the recent browser task)
    web_goal_complete: float = 1.0  # p(request has the concrete details a website form needs)
    web_goal_forbidden: float = 0.0  # p(request requires buying/paying/signing in/changing an account)
    web_goal_missing: str = "nothing"  # what a clarifying question should ask for
    latency_ms: float = 0.0
    weakest_arg: tuple[str, float] | None = None

    @property
    def summary(self) -> str:
        if self.tool:
            return f"{self.tool.name}({', '.join(f'{k}={v!r}' for k, v in self.args.items())})"
        return self.kind


def _state(utterance: str, ctx: MacContext, recent_browser: dict[str, str] | None = None) -> dict[str, Any]:
    import datetime as _dt

    st: dict[str, Any] = {
        "utterance": utterance,
        "today": _dt.date.today().strftime("%A %d %B %Y"),
        "frontmost_app": ctx.active_app,
        "running_apps": ctx.running_apps,
    }
    if recent_browser:
        st["recent_browser_task"] = recent_browser
    return st


def _arg_qid(tool: Tool, arg_name: str) -> str:
    return f"arg::{tool.name}::{arg_name}"


class JevRouter:
    def __init__(self, registry: ToolRegistry, client: AsyncTypeSafeClient | None = None):
        self.registry = registry
        self.client = client or AsyncTypeSafeClient()

    async def aclose(self) -> None:
        await self.client.aclose()

    # ------------------------------------------------------------------ routing
    async def route(
        self,
        utterance: str,
        ctx: MacContext,
        *,
        awaiting_confirmation: bool = False,
        recent_browser: dict[str, str] | None = None,
    ) -> Route:
        t0 = time.perf_counter()
        tools = self.registry.available(ctx)
        criteria: dict[str, Any] = {}
        for t in tools[: MAX_CHOICE_OPTIONS - 2]:
            criteria[t.name] = t.choice_description()
        criteria[CHAT] = {
            "what": "The user is chatting, asking a general question, or thinking aloud; "
            "they are not asking the assistant to do something on the Mac.",
            "examples": ["how are you", "what's the capital of France", "thanks"],
        }
        criteria[STOP] = {
            "what": "The user wants the voice assistant itself to stop, quit, go to sleep, or stop listening.",
            "examples": [
                "stop listening",
                "quit macbrow",
                "quit your application wherever you are running",
                "shut yourself down",
                "stop running",
                "goodbye assistant, you can stop now",
            ],
            "not_for": "Quitting a named application such as Slack or Chrome, or cancelling a pending action.",
        }
        criteria[NEW_ACTION] = {
            "what": "The user wants an action performed on the Mac (in an app or the system) "
            "that none of the listed tools can do.",
            "not_for": "Requests a listed tool already covers, even if worded differently.",
        }

        questions: dict[str, Any] = {
            "intent": Choice(
                instructions=(
                    "The user spoke `utterance` to a voice assistant that controls this Mac. "
                    "`frontmost_app` is the app currently in focus and `running_apps` are open. "
                    "Which tool best fulfils the request? Prefer a tool scoped to `frontmost_app` "
                    "when the request is ambiguous between apps. A website, URL, or web search "
                    "goes to a browser tool, not to opening an application. Closing, switching or reloading "
                    "tabs and windows is a browser-control tool, never a web task."
                ),
                criteria=criteria,
            )
        }
        # Speculative enum-argument questions for every candidate tool.
        for t in tools:
            for spec in t.args:
                if spec.kind != "enum":
                    continue
                crit = spec.resolve_criteria(ctx)
                if not crit:
                    continue
                questions[_arg_qid(t, spec.name)] = Choice(
                    instructions=[
                        f"Assume the user wants to run the tool '{t.name}' ({t.description}).",
                        spec.instructions,
                        "Base the answer on `utterance`; use `frontmost_app` when the user says "
                        "'this app' or leaves the target implicit.",
                    ],
                    criteria=crit,
                )
        if any(t.runner == "browser" for t in tools):
            questions.update(_web_goal_questions())
        if recent_browser:
            questions["browser_followup"] = Noul(
                instructions=(
                    "`recent_browser_task` describes a web task the assistant just performed in Chrome, or a "
                    "clarifying question it asked about one (see `question`). Is `utterance` a follow-up, "
                    "correction, answer, or next step for that same task, rather than a new unrelated request?"
                ),
                criteria={
                    "true": "Continues, corrects, or refines the recent task: 'yes, now set the return date', "
                    "'no, the other one', 'the return should be November 4th', 'add it to the cart'",
                    "false": "A new request unrelated to that page, or a general Mac command",
                },
            )
        if awaiting_confirmation:
            questions["confirm"] = Noul(
                instructions="Does `utterance` confirm or approve going ahead with a pending action?",
                criteria={"true": "yes, go ahead, do it, confirmed, sure", "false": "anything else"},
            )
            questions["deny"] = Noul(
                instructions="Does `utterance` cancel, decline, or say no to a pending action?",
                criteria={"true": "no, cancel, stop, never mind, don't", "false": "anything else"},
            )

        resp = await self.client.system_one(state=_state(utterance, ctx, recent_browser), questions=questions)
        intent = resp.choices["intent"]
        route = Route(
            kind="tool",
            confidence=float(intent.confidence),
            probabilities={k: round(float(v), 3) for k, v in intent.probabilities.items()},
        )
        if "web_goal_complete" in resp.nouls:
            route.web_goal_complete = float(resp.nouls["web_goal_complete"].noul)
            route.web_goal_forbidden = float(resp.nouls["web_goal_forbidden"].noul)
            route.web_goal_missing = resp.choices["web_goal_missing"].choice
        if recent_browser:
            route.browser_followup = float(resp.nouls["browser_followup"].noul)
        if awaiting_confirmation:
            route.is_confirmation = float(resp.nouls["confirm"].noul)
            route.is_denial = float(resp.nouls["deny"].noul)

        picked = intent.choice
        if picked == CHAT:
            route.kind = "chat"
        elif picked == STOP:
            route.kind = "stop"
        elif picked == NEW_ACTION:
            route.kind = "new_action"
            # A hesitant new_action with a plausible existing tool is a question, not a codegen trigger.
            if intent.confidence < NEW_ACTION_MIN_CONFIDENCE:
                best = max(
                    ((k, v) for k, v in intent.probabilities.items() if k not in (CHAT, NEW_ACTION, STOP)),
                    key=lambda kv: kv[1],
                    default=None,
                )
                if best and best[1] >= UNCERTAIN_TOOL_MIN_PROB and (tool := self.registry.get(best[0])):
                    picked = tool.name
                    route.kind = "uncertain"
                    route.tool = tool
        if picked not in (CHAT, NEW_ACTION, STOP):
            tool = self.registry.get(picked)
            if tool is None:
                route.kind = "new_action"
                route.tool = None
            else:
                route.tool = tool
                if route.kind != "uncertain" and intent.confidence < MIN_TOOL_CONFIDENCE:
                    route.kind = "uncertain"
                # Consume only the selected tool's speculative answers.
                weakest: tuple[str, float] | None = None
                for spec in tool.args:
                    if spec.kind != "enum":
                        continue
                    ans = resp.choices.get(_arg_qid(tool, spec.name))
                    if ans is None:
                        continue
                    route.args[spec.name] = ans.choice
                    if weakest is None or ans.confidence < weakest[1]:
                        weakest = (spec.name, float(ans.confidence))
                route.weakest_arg = weakest
                # Second request only when a free-text slot exists.
                text_specs = [s for s in tool.args if s.kind == "text"]
                if text_specs:
                    await self._fill_text_args(utterance, ctx, tool, text_specs, route)

        route.latency_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "route %s conf=%.2f %.0fms probs=%s",
            route.summary,
            route.confidence,
            route.latency_ms,
            _top(route.probabilities),
        )
        return route

    async def web_goal_check(self, text: str, ctx: MacContext) -> tuple[float, str]:
        """Completeness of a (merged) web-task goal: (p_complete, most_missing)."""
        resp = await self.client.system_one(state=_state(text, ctx), questions=_web_goal_questions())
        return float(resp.nouls["web_goal_complete"].noul), resp.choices["web_goal_missing"].choice

    # ---------------------------------------------------------- text arguments
    async def _fill_text_args(
        self, utterance: str, ctx: MacContext, tool: Tool, specs: list[Any], route: Route
    ) -> None:
        """Select-not-generate: Jev picks which span of the utterance is the argument."""
        candidates = _span_candidates(utterance, MAX_TEXT_CANDIDATES)
        if not candidates:
            for spec in specs:
                route.args[spec.name] = spec.default or utterance
            return
        crit = {c: None for c in candidates}
        crit["__none__"] = "No part of the utterance is this argument"
        questions = {
            spec.name: Choice(
                instructions=[
                    f"The user is running the tool '{tool.name}' ({tool.description}).",
                    spec.instructions,
                    "Choose the option that is exactly and only that value, copied from `utterance`, "
                    "without leading command words like 'open', 'search for', 'say', or 'that says'.",
                ],
                criteria=crit,
            )
            for spec in specs
        }
        resp = await self.client.system_one(state=_state(utterance, ctx), questions=questions)
        for spec in specs:
            ans = resp.choices[spec.name]
            value = ans.choice if ans.choice != "__none__" else (spec.default or utterance)
            route.args[spec.name] = _clean_value(value)
            if route.weakest_arg is None or ans.confidence < route.weakest_arg[1]:
                route.weakest_arg = (spec.name, float(ans.confidence))


def _web_goal_questions() -> dict[str, Any]:
    return {
        "web_goal_complete": Noul(
            instructions=(
                "Suppose `utterance` is a task to carry out on a website (shopping, flights, email, calendar, forms). "
                "Could an assistant fill every field the site REQUIRES using only the words in `utterance` plus "
                "`today`, without inventing a value? A generic product ('a vacuum cleaner') is enough for a search "
                "box; a short message is enough content; emails need no dates; a calendar event needs only a "
                "title and a time, attendees are optional. A relative date that resolves "
                "unambiguously from `today` ('Monday', 'tomorrow', 'next Friday at 2pm') IS a usable date. "
                "A travel search needs both cities/airports (a country is not enough) and dates that resolve to "
                "specific days: 'in November' or 'for five days' do not."
            ),
            criteria={
                "true": "Every required field has a usable value in the utterance (given today's date)",
                "false": "A required field has no usable value: vague/missing dates, a country instead of a city, "
                "no recipient, no idea what to search for",
            },
        ),
        "web_goal_forbidden": Noul(
            instructions=(
                "Would carrying out `utterance` on a website require the assistant ITSELF to place an order or pay, "
                "sign in or enter credentials, or change account settings? Browsing, searching, comparing prices, "
                "reading, filtering, opening product pages and adding to a cart do NOT count, even if the user "
                "mentions wanting to buy something eventually."
            ),
            criteria={
                "true": "The task cannot be completed without a purchase/payment, a sign-in, or an account change",
                "false": "It is research, navigation, search, reading, or cart-building; no money or credentials involved",
            },
        ),
        "web_goal_missing": Choice(
            instructions="If `utterance` were a website task, which single required detail is most clearly missing or vague?",
            criteria={
                "exact_dates": "Dates are missing or don't resolve to specific days ('in November', 'for five days'); "
                "'Monday' or 'tomorrow' DO resolve and are not missing",
                "destination": "Where to is missing, or too broad (a country or region instead of a city/airport)",
                "origin": "Where from",
                "product": "What item to search for or which one to pick",
                "recipient": "Who the message/email is for",
                "content": "What the message/note/form should say",
                "nothing": "Nothing important is missing",
            },
        ),
    }


_WORD_RE = re.compile(r"\S+")


def _span_candidates(utterance: str, max_candidates: int = 200) -> list[str]:
    """Word-bounded substrings of the utterance, longest first, so Jev can pick exactly the
    argument ("send Constance a message saying hi" -> "Constance" and "hi") instead of a
    whole trailing clause. Suffixes are always included; inner spans fill the remaining budget.
    """
    text = utterance.strip().rstrip(".!?")
    words = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    n = len(words)
    spans: list[str] = []
    seen: set[str] = set()

    def add(i: int, j: int) -> None:
        span = text[words[i][0] : words[j - 1][1]].strip().rstrip(".,;:!?").strip(",;:")
        key = span.lower()
        if len(span) >= 2 and key not in seen:
            seen.add(key)
            spans.append(span)

    for i in range(n):  # suffixes first: the common case for spoken commands
        add(i, n)
    for length in range(n - 1, 0, -1):  # then every shorter inner span
        for i in range(n - length):
            if len(spans) >= max_candidates:
                return spans
            add(i, i + length)
    return spans


def _clean_value(v: str) -> str:
    v = v.strip().strip("\"'").rstrip(".,;:!?")
    # Spoken URLs: "github dot com" -> "github.com"
    v = re.sub(r"\s+dot\s+", ".", v, flags=re.IGNORECASE)
    v = re.sub(r"\s+slash\s+", "/", v, flags=re.IGNORECASE)
    return v


def _top(probs: dict[str, float], n: int = 3) -> dict[str, float]:
    return dict(sorted(probs.items(), key=lambda kv: -kv[1])[:n])
