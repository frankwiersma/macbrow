"""DynamicMacAgent: the state machine that ties routing, execution and learning together.

States
------
IDLE                -> every utterance is routed by Jev.
AWAITING_CONFIRM    -> a risky tool is staged; Jev judges whether the next
                       utterance confirms or cancels it (and, if neither, the
                       utterance is routed as a fresh command).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from . import browser_task, policy, resolvers
from .applescript import ContextPoller, run_applescript
from .generator import DUPLICATE, ToolGenerator
from .registry import Tool, ToolRegistry
from .router import JevRouter, Route

log = logging.getLogger("macbrow.agent")

CONFIRM_THRESHOLD = 0.6
FOLLOWUP_THRESHOLD = 0.5
GOAL_COMPLETE_THRESHOLD = 0.4  # below: ask for the missing detail instead of driving Chrome
FORBIDDEN_THRESHOLD = 0.6  # p(goal requires buying/paying/signing in) at or above which we refuse
ACHIEVED_THRESHOLD = 0.5  # p(final page shows the success criterion); below: one more round, then be honest
BROWSER_MEMORY_S = 15 * 60  # how long a finished browser task stays "recent" for follow-ups
ARG_CONFIDENCE_FLOOR = 0.35  # weakest argument below this -> ask instead of act
# Read the result of a finished action back out loud. 0: act silently. Questions, confirmations,
# refusals and failures are still spoken either way -- muting those would strand the conversation.
SPEAK_RESULTS = os.environ.get("MACBROW_SPEAK_RESULTS", "1") != "0"


PRONOUNS = {
    "her",
    "him",
    "them",
    "it",
    "that",
    "this",
    "that person",
    "the same person",
    "same",
    "same one",
    "that one",
}


class State(StrEnum):
    IDLE = "idle"
    AWAITING_CONFIRM = "awaiting_confirm"


@dataclass
class Outcome:
    """What the voice layer should do after an utterance."""

    speak: str | None = None  # text to say directly (fast path)
    handoff_to_llm: bool = False  # let the conversational LLM answer instead
    stop: bool = False  # user asked the assistant itself to stop
    route: Route | None = None
    executed: bool = False
    learned: Tool | None = None
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class Pending:
    tool: Tool
    args: dict[str, str]
    staged_at: float
    utterance: str = ""


class DynamicMacAgent:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        enable_learning: bool = True,
        on_learning: Callable[[str], Any] | None = None,
    ):
        """``on_learning(message)`` is called right before the slow codegen tier runs, so the
        voice layer can say a filler line instead of leaving silence."""
        self.on_learning = on_learning
        self.registry = registry or ToolRegistry()
        self.router = JevRouter(self.registry)
        self.generator = ToolGenerator(self.registry, jev=self.router.client) if enable_learning else None
        self.state = State.IDLE
        self.pending: Pending | None = None
        self.context = ContextPoller()
        self.last_browser: dict[str, Any] | None = None  # goal, url, title, target_id, finished_at
        self.last_args: dict[str, dict[str, str]] = {}  # tool name -> last used arguments (for "send her another")

    async def start(self) -> None:
        """Warm the context poller (optional; handle() does it lazily)."""
        await self.context.start()

    async def aclose(self) -> None:
        await self.context.stop()
        if self.generator is not None:
            await self.generator.aclose()
        await self.router.aclose()

    # ----------------------------------------------------------------- public
    async def handle(self, utterance: str) -> Outcome:
        utterance = utterance.strip()
        if not utterance:
            return Outcome(handoff_to_llm=False)
        t0 = time.perf_counter()
        ctx = await self.context.latest()
        t_ctx = time.perf_counter()

        recent = self._recent_browser()
        route = await self.router.route(
            utterance, ctx, awaiting_confirmation=self.state is State.AWAITING_CONFIRM, recent_browser=recent
        )
        # A follow-up to the browser task wins over whatever else Jev matched (often a mis-scoped web_task).
        if (
            recent
            and route.browser_followup >= FOLLOWUP_THRESHOLD
            and route.kind in ("tool", "uncertain", "new_action", "chat")
        ):
            wt = self.registry.get("web_task")
            if wt is not None:
                route.kind, route.tool, route.args = "tool", wt, {"site": "current_tab"}
                route.weakest_arg = None
            lb = self.last_browser or {}
            if lb.get("status") == "needs_info" and wt is not None:
                # Answer to a clarifying question: merge, then check whether anything else is still missing.
                merged = f"{lb['goal']} {utterance.strip()}"
                asked = list(lb.get("asked") or [])
                p_complete, missing = await self.router.web_goal_check(merged, ctx)
                log.info("merged goal completeness %.2f missing=%s asked=%s", p_complete, missing, asked)
                # Re-ask an item at most once (the user may have answered a different gap first); three asks total.
                if (
                    p_complete < GOAL_COMPLETE_THRESHOLD
                    and missing != "nothing"
                    and asked.count(missing) < 2
                    and len(asked) < 3
                ):
                    question = _ask_for(missing, lb.get("site"))
                    if missing in asked:
                        question = "Sorry, I still need this. " + question
                    self.last_browser = {
                        **lb,
                        "goal": merged,
                        "question": question,
                        "asked": asked + [missing],
                        "finished_at": time.time(),
                    }
                    return Outcome(
                        route=route,
                        speak=question,
                        timings={"context_ms": (t_ctx - t0) * 1e3, "route_ms": (time.perf_counter() - t_ctx) * 1e3},
                    )
                self.last_browser = {**lb, "goal": merged, "finished_at": time.time()}
                utterance = merged  # run the fully specified goal
                recent = self._recent_browser()
        t_route = time.perf_counter()
        outcome = Outcome(route=route, timings={"context_ms": (t_ctx - t0) * 1e3, "route_ms": (t_route - t_ctx) * 1e3})

        if self.state is State.AWAITING_CONFIRM and self.pending:
            if route.is_confirmation >= CONFIRM_THRESHOLD and route.is_confirmation > route.is_denial:
                pending, self.pending, self.state = self.pending, None, State.IDLE
                await self._execute(pending.tool, pending.args, outcome, utterance=pending.utterance)
                return outcome
            if route.is_denial >= CONFIRM_THRESHOLD:
                self.pending, self.state = None, State.IDLE
                outcome.speak = "Cancelled."
                return outcome
            # Neither: drop the pending action and treat this as a new command.
            self.pending, self.state = None, State.IDLE

        if route.kind == "chat":
            outcome.handoff_to_llm = True
            return outcome

        if route.kind == "stop":
            outcome.stop = True
            outcome.speak = "Okay, stopping. Goodbye."
            return outcome

        if route.tool:
            unresolved = self._resolve_pronouns(route.tool, route.args)
            if unresolved:
                outcome.speak = f"Who do you mean by {unresolved[1]!r} for {_say_tool(route.tool)}?"
                return outcome

        if route.kind == "uncertain" and route.tool:
            top = sorted(route.probabilities.items(), key=lambda kv: -kv[1])[:2]
            if len(top) == 2 and top[1][0] not in ("chat", "new_action") and top[1][1] > 0.2:
                outcome.speak = f"Did you want {_say_tool(route.tool)} or {_say_tool(self.registry.get(top[1][0]))}?"
            else:
                outcome.speak = f"Should I {_say_tool(route.tool)}?"
            self._stage(route.tool, route.args)
            return outcome

        if route.kind == "tool" and route.tool:
            if route.weakest_arg and route.weakest_arg[1] < ARG_CONFIDENCE_FLOOR:
                name, _ = route.weakest_arg
                outcome.speak = f"Which {name} did you mean? I heard {route.args.get(name, 'nothing')}."
                self._stage(route.tool, route.args)
                return outcome
            if route.tool.runner == "browser" and route.weakest_arg and route.weakest_arg[1] < ARG_CONFIDENCE_FLOOR:
                route.args["site"] = "current_tab" if recent else "google"  # never ask "which site"; just go
                route.weakest_arg = None
            if route.tool.runner == "browser":
                refusal = _browser_refusal(utterance, route.web_goal_forbidden)
                if refusal:
                    outcome.speak = refusal
                    return outcome
                is_followup = bool(recent) and route.browser_followup >= FOLLOWUP_THRESHOLD
                if (
                    not is_followup
                    and route.web_goal_complete < GOAL_COMPLETE_THRESHOLD
                    and route.web_goal_missing != "nothing"
                ):
                    question = _ask_for(route.web_goal_missing, route.args.get("site"))
                    self.last_browser = {  # a clarification is a "recent task" the answer follows up on
                        "goal": utterance,
                        "url": "",
                        "title": "",
                        "target_id": None,
                        "status": "needs_info",
                        "site": route.args.get("site", "google"),
                        "question": question,
                        "asked": [route.web_goal_missing],
                        "finished_at": time.time(),
                    }
                    outcome.speak = question
                    return outcome
                if policy.browser_goal_needs_confirm(utterance):
                    self._stage(route.tool, route.args, utterance)
                    outcome.speak = f"I'll do this in Chrome: {utterance.rstrip('.?!')}. Go ahead?"
                    return outcome
                await self._execute(route.tool, route.args, outcome, utterance=utterance)
                return outcome
            if route.tool.risky:
                self._stage(route.tool, route.args)
                outcome.speak = f"That will {_say_tool(route.tool, route.args)}. Should I go ahead?"
                return outcome
            await self._execute(route.tool, route.args, outcome, utterance=utterance)
            return outcome

        # new_action -> heavy tier
        if self.generator is None:
            outcome.speak = "I don't have a tool for that yet."
            return outcome
        if self.on_learning:
            try:
                self.on_learning("I don't know that one yet. Give me a moment.")
            except Exception:  # never let UX sugar break the pipeline
                log.exception("on_learning callback failed")
        tool, msg = await self.generator.generate(utterance, ctx)
        outcome.timings["codegen_ms"] = (time.perf_counter() - t_route) * 1e3
        if tool is None:
            outcome.speak = msg
            return outcome

        # Fill the tool's arguments with Jev now that it is in the registry / identified.
        route2 = await self.router.route(utterance, ctx)
        args = route2.args if route2.tool and route2.tool.name == tool.name else {}

        if msg == DUPLICATE:
            # The router missed an existing tool; use it through the normal gates.
            log.info("codegen resolved to existing tool %s", tool.name)
            if tool.risky:
                self._stage(tool, args)
                outcome.speak = f"That will {_say_tool(tool, args)}. Should I go ahead?"
                return outcome
            await self._execute(tool, args, outcome)
            return outcome

        outcome.learned = tool
        # A freshly written script always gets a spoken confirmation before its first run;
        # generated code can compile and pass review yet still do the wrong thing.
        self._stage(tool, args)
        hedge = ", though I'm not certain it works" if tool.verified is not None and tool.verified < 0.6 else ""
        outcome.speak = f"I wrote a new action to {_say_tool(tool, args)}{hedge}. Should I run it?"
        return outcome

    # ---------------------------------------------------------------- helpers
    async def _execute_browser(self, tool: Tool, args: dict[str, str], outcome: Outcome, utterance: str) -> None:
        refusal = _browser_refusal(utterance, 0.0)
        if refusal:
            outcome.speak = refusal
            return
        site = args.get("site", "other")
        recent = self._recent_browser()
        lb = self.last_browser or {}
        reuse = None
        if site == "current_tab" and recent and browser_task.tab_exists(lb.get("target_id")):
            reuse = lb.get("target_id")
        if site == "current_tab":
            site = lb.get("site") or "google"
        if self.on_learning:
            try:
                self.on_learning("On it. Driving Chrome now." if not reuse else "Continuing in the same tab.")
            except Exception:
                log.exception("on_learning callback failed")

        # Rewrite the raw sentence into a self-contained objective with a visible success criterion.
        t0 = time.perf_counter()
        context: dict[str, Any] = {"today": time.strftime("%A %d %B %Y")}
        if recent:
            context["conversation_so_far"] = recent["goal"]
            if recent.get("page_title"):
                context["current_page"] = {"title": recent["page_title"], "url": recent["page_url"][:200]}
        try:
            composed = await browser_task.compose_goal(utterance, context)
            objective, success, search_query = composed.objective, composed.success, composed.search_query
        except Exception:
            log.exception("goal composition failed; using the raw utterance")
            objective, success, search_query = utterance.strip(), "", None
        outcome.timings["compose_ms"] = (time.perf_counter() - t0) * 1e3
        log.info("browser objective: %s | success: %s | search: %s", objective, success, search_query)
        goal = objective + (f"\nDone when: {success}" if success else "")
        if reuse:
            goal += "\nContinue from the current page state; do not redo completed steps."
        if search_query and site in browser_task.SEARCH_URLS:
            # A refined search: land straight on the new results, in the same tab when we have one.
            start_url = browser_task.start_url_for(site, objective, search_query)
        else:
            start_url = "" if reuse else browser_task.start_url_for(site, objective, None)

        t0 = time.perf_counter()
        result = await browser_task.run_task(start_url, goal, on_progress=self.on_learning, reuse_target=reuse)
        achieved: float | None = None
        if result.status == "blocked" and success and result.page_text:
            # The agent may have stopped because the results were already the answer.
            achieved = await self._verify_outcome(objective, success, result)
            if achieved is not None and achieved >= ACHIEVED_THRESHOLD:
                log.info("blocked run judged achieved (%.2f); reporting done", achieved)
                result.status = "done"
        if result.status == "done" and success:
            achieved = await self._verify_outcome(objective, success, result)
            if achieved is not None and achieved < ACHIEVED_THRESHOLD and browser_task.tab_exists(result.target_id):
                log.info("outcome not confirmed (%.2f); one more round", achieved)
                nudge = goal + f"\nNOT yet achieved: the page does not show {success}. Keep going from here."
                result = await browser_task.run_task(
                    "", nudge, on_progress=self.on_learning, reuse_target=result.target_id
                )
                if result.status == "done":
                    achieved = await self._verify_outcome(objective, success, result)
        outcome.timings["browser_ms"] = (time.perf_counter() - t0) * 1e3
        outcome.executed = result.status == "done" and (achieved is None or achieved >= ACHIEVED_THRESHOLD)
        log.info(
            "browser task %s: %s steps=%d %.0fms achieved=%s url=%s reuse=%s",
            result.status,
            objective[:70],
            result.steps,
            result.elapsed_ms,
            f"{achieved:.2f}" if achieved is not None else "n/a",
            result.url,
            bool(reuse),
        )
        if result.status != "error":
            self.last_browser = {
                "goal": (recent["goal"] + " | " + objective) if recent else objective,
                "url": result.url,
                "title": result.title,
                "target_id": result.target_id or reuse,
                "status": result.status,
                "site": site,
                "finished_at": time.time(),
            }
        if result.status == "done" and achieved is not None and achieved < ACHIEVED_THRESHOLD:
            where = f" Chrome is on {browser_task.spoken_title(result.title)}." if result.title else ""
            outcome.speak = (
                f"I got as far as I could, but I can't confirm the page shows {success}.{where} Take a look."
            )
            return
        if result.status in ("blocked", "timeout"):
            outcome.speak = result.spoken
            reason = await self._explain_stall(goal, result)
            if reason:
                outcome.speak = f"{result.spoken} {reason}"
        elif SPEAK_RESULTS:
            outcome.speak = result.spoken

    async def _verify_outcome(self, objective: str, success: str, result: browser_task.BrowserResult) -> float | None:
        """Independent check that the final page actually shows what the user asked for."""
        from typesafe_sdk import Noul

        try:
            resp = await self.router.client.system_one(
                state={
                    "objective": objective,
                    "success_criterion": success,
                    "final_page": {
                        "title": result.title,
                        "url": result.url[:300],
                        "visible_text": result.page_text[:3000],
                    },
                },
                questions={
                    "achieved": Noul(
                        instructions=(
                            "A browser agent claims it completed `objective`. Judging only from `final_page`, does the "
                            "page visibly satisfy `success_criterion`? A search box containing the right words, or a "
                            "results list for a different or broader query, does not count."
                        ),
                        criteria={
                            "true": "The page shows what the success criterion describes",
                            "false": "It does not, or shows something else (wrong item, unsubmitted search, generic results)",
                        },
                    )
                },
            )
            return float(resp.nouls["achieved"].noul)
        except Exception:
            log.exception("outcome verification failed; skipping")
            return None

    async def _explain_stall(self, goal: str, result: browser_task.BrowserResult) -> str:
        """One Jev judgment: why did the browser agent stop short? Returns a spoken hint or an empty string."""
        from typesafe_sdk import Choice

        try:
            resp = await self.router.client.system_one(
                state={
                    "goal": goal[:1500],
                    "final_page": {"title": result.title, "url": result.url[:300]},
                    "steps_taken": [f"{h['operation']} {h['action'][:60]}" for h in result.history[-8:]],
                    "stopped_because": result.error or result.status,
                },
                questions={
                    "why": Choice(
                        instructions="A browser agent stopped before finishing `goal`. From the final page and the steps taken, what is the most likely reason?",
                        criteria={
                            "missing_details": "The request lacked a concrete value a form needs (exact dates, a specific item, a name)",
                            "login_required": "The site wants sign-in, a password, or a verification step",
                            "unsupported_page": "The page uses controls the agent cannot operate (canvas, custom widgets, popups)",
                            "already_satisfied": "The goal looks achieved on the final page; the agent just did not recognise it",
                            "wandered": "The agent drifted into unrelated features or pages",
                            "unclear": "Cannot tell",
                        },
                    )
                },
            )
            why = resp.choices["why"]
            if float(why.probabilities.get(why.choice, 0)) < 0.55:
                return ""
            return _STALL_HINTS.get(why.choice, "")
        except Exception:
            log.exception("stall explanation failed")
            return ""

    def _recent_browser(self) -> dict[str, str] | None:
        lb = self.last_browser
        if not lb or time.time() - lb["finished_at"] > BROWSER_MEMORY_S:
            return None
        rec = {"goal": lb["goal"], "page_title": lb["title"], "page_url": lb["url"], "status": lb["status"]}
        if lb.get("question"):
            rec["question"] = lb["question"]
        return rec

    def _stage(self, tool: Tool, args: dict[str, str], utterance: str = "") -> None:
        self.pending = Pending(tool=tool, args=args, staged_at=time.time(), utterance=utterance)
        self.state = State.AWAITING_CONFIRM

    def _resolve_pronouns(self, tool: Tool, args: dict[str, str]) -> tuple[str, str] | None:
        """Replace pronoun text arguments ("her", "him", "them", "that") with the value used the
        last time this tool ran. Returns (arg, pronoun) if one can't be resolved."""
        previous = self.last_args.get(tool.name, {})
        for spec in tool.args:
            if spec.kind != "text":
                continue
            value = args.get(spec.name, "")
            if value.strip().lower().strip(".,!?") in PRONOUNS:
                if spec.name in previous:
                    args[spec.name] = previous[spec.name]
                else:
                    return spec.name, value
        return None

    async def _execute(
        self, tool: Tool, args: dict[str, str], outcome: Outcome, prefix: str = "", utterance: str = ""
    ) -> None:
        if tool.runner == "browser":
            await self._execute_browser(tool, args, outcome, utterance)
            return
        if tool.computed:
            args = dict(args)
            t_r = time.perf_counter()
            for name, spec in tool.computed.items():
                try:
                    args[name] = await resolvers.run(spec["fn"], args.get(spec.get("from", ""), ""))
                except resolvers.ResolveError as e:
                    outcome.speak = f"{prefix}{e}"
                    return
            outcome.timings["resolve_ms"] = (time.perf_counter() - t_r) * 1e3
        script = tool.render(args)
        violations = policy.check(script, tool.scope)
        if violations:
            log.warning("policy blocked %s(%s): %s", tool.name, args, "; ".join(map(str, violations)))
            outcome.speak = f"{prefix}I can't do that. Voice control isn't allowed to {_say_violation(violations[0])}."
            return
        t0 = time.perf_counter()
        result = await run_applescript(script)
        outcome.timings["exec_ms"] = (time.perf_counter() - t0) * 1e3
        outcome.executed = result.ok
        if result.ok and args:
            self.last_args[tool.name] = dict(args)
        if not result.ok:
            log.warning("tool %s failed: %s", tool.name, result.error)
            outcome.speak = f"{prefix}That didn't work: {_short(result.error)}"
            return
        out = result.output
        if tool.name in BROWSER_OPENERS:
            # A plain open/search is context for the next web task ("show me a black one").
            self.last_browser = {
                "goal": f"{tool.description.rstrip('.')}: {', '.join(f'{k}={v}' for k, v in args.items())}",
                "url": out if out.startswith("http") else "",
                "title": "",
                "target_id": None,
                "status": "done",
                "site": BROWSER_OPENERS[tool.name](args),
                "finished_at": time.time(),
            }
        if not SPEAK_RESULTS:
            return
        if tool.speak == "done":
            outcome.speak = f"{prefix}Done." if prefix else "Done."
        elif tool.speak == "result":
            outcome.speak = f"{prefix}{out or 'Done.'}"
        else:
            outcome.speak = prefix + tool.speak.replace("{result}", out or "nothing")


_ASK_FOR = {
    "exact_dates": "Which exact date should I use?",
    "destination": "Which city or airport? A country alone isn't enough for the search.",
    "origin": "Where are you flying from?",
    "product": "Which product exactly should I look for?",
    "recipient": "Who is it for?",
    "content": "What should it say?",
}
_STALL_HINTS = {
    "missing_details": "It looks like it needed more specifics from you, such as exact dates.",
    "login_required": "The site is asking to sign in, which I don't do.",
    "unsupported_page": "That page has controls I can't operate.",
    "already_satisfied": "Though the page may already show what you asked for.",
    "wandered": "It drifted off the main form; try a shorter, more specific request.",
}


# Tools that open a page without jev-ultrafast; they seed the "recent browser" context for follow-ups.
BROWSER_OPENERS: dict[str, Callable[[dict[str, str]], str]] = {
    "chrome_open": lambda a: "google" if a.get("site") in (None, "other") else a.get("site", "google"),
    "safari_open": lambda a: "google",
    "youtube_play": lambda a: "youtube",
}


def _browser_refusal(utterance: str, forbidden_p: float) -> str | None:
    """Spoken refusal for a browser goal, or None. Hard terms come from the policy regex; whether the task
    needs buying/paying/signing in is Jev's judgment (probability passed in)."""
    bad = policy.check_browser_goal(utterance)
    if bad:
        return f"I can't do that in the browser. Voice control doesn't handle {bad[0].detail}."
    if forbidden_p >= FORBIDDEN_THRESHOLD:
        return "That would need me to buy, pay, sign in, or change an account, which voice control doesn't do. I can search and compare for you."
    return None


def _ask_for(missing: str, site: str | None) -> str:
    if missing == "exact_dates" and site == "google_flights":
        return "Which exact dates? I need the departure and the return."
    return _ASK_FOR.get(missing, "Can you give me the missing details?")


def _say_tool(tool: Tool | None, args: dict[str, str] | None = None) -> str:
    if tool is None:
        return "that"
    text = tool.description.rstrip(".")
    text = text[0].lower() + text[1:]
    if args:
        text += " with " + ", ".join(f"{k} {v}" for k, v in args.items())
    return text


def _say_violation(v: policy.Violation) -> str:
    spoken = {
        "blocked app": f"control {v.detail}",
        "protected path": "touch system or configuration files",
        "shell": "run system commands",
        "system preferences": "open System Settings",
        "system preference writes": "change system preferences",
    }
    return spoken.get(v.rule, v.rule)


def _short(s: str, n: int = 140) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"
