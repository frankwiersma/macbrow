"""Multi-step web tasks in the user's real Chrome, driven by jev-ultrafast.

jev-ultrafast (Browser Use × TypeSafe) observes the page as an indexed element table and
lets Jev choose one operation and one target per step; a small LLM writes text only for
TYPE_TEXT. We add three things on top:

1. Profile pinning: the working tab is created inside the Chrome profile macbrow is
   configured for (see chrome.py), found by opening a probe URL in that profile through
   `open -na` and reading the probe tab's browserContextId over CDP.
2. Text helper on our own LLM backend (LiveKit Inference or LM Studio) instead of the
   upstream OpenRouter/DeepSeek helper, with a strict JSON schema.
3. A wall-clock budget, progress callbacks for the voice layer, and a spoken summary.

Chrome must have "Allow remote debugging for this browser instance" enabled at
chrome://inspect/#remote-debugging; browser-harness (installed with jev-ultrafast) does the
CDP attach. Run `uv run browser-harness --doctor` if the connection is unclear.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from . import chrome

log = logging.getLogger("macbrow.browser")

WALL_CLOCK_S = float(os.environ.get("MACBROW_BROWSER_BUDGET_S", "90"))
CONNECT_TIMEOUT_S = 25.0
PROGRESS_EVERY_S = 12.0  # spoken progress cadence during a browser task
VIEWPORT = (1120, 780)  # emulated layout size while a task runs; cleared afterwards
DATE_FIELD = re.compile(r"\b(date|departure|depart|return|check[- ]?in|check[- ]?out|arrival|arrive)\b", re.I)
# Appended to every spoken goal. Keeps the agent off ads and out of money/credential flows.
STANDING_RULES = (
    "Prefer organic results over items labelled Sponsored or Ad unless nothing else matches. "
    "Use the site's standard search form; do not open AI assistants, 'explore' or promotional features. "
    "If the goal lacks a value a field requires (an exact date, a name), do not invent one: choose BLOCKED. "
    "On travel sites, fields labelled Departure and Return are DATE fields, and Where from / Where to are place "
    "fields; never type a city into a date field. Set a date by TYPE_TEXT into its field in a short form such as "
    "'Nov 7', then CLICK Done; use calendar day cells only if typing is impossible. "
    "For any search or lookup, DONE only once the actual results (prices, listings, items) are visible on the "
    "page; a filled-in form is not a result, so submit it first (Done, Search, Enter). "
    "Never proceed to checkout, payment, sign-in, or password fields; if the task would need them, "
    "stop and report BLOCKED."
)

SITE_URLS: dict[str, str] = {
    "amazon": "https://www.amazon.com/",
    "gmail": "https://mail.google.com/",
    "google": "https://www.google.com/",
    "google_flights": "https://www.google.com/travel/flights?hl=en",
    "google_shopping": "https://www.google.com/shopping",
    "google_maps": "https://www.google.com/maps",
    "youtube": "https://www.youtube.com/",
    "x_twitter": "https://x.com/",
    "linkedin": "https://www.linkedin.com/",
    "github": "https://github.com/",
    "google_calendar": "https://calendar.google.com/",
    "google_drive": "https://drive.google.com/",
    "notion": "https://www.notion.so/",
    "current_tab": "",  # keep working in whatever tab is in front
    "other": "https://www.google.com/",
}


SEARCH_URLS: dict[str, str] = {
    "google": "https://www.google.com/search?q={q}",
    "google_shopping": "https://www.google.com/search?tbm=shop&q={q}",
    "other": "https://www.google.com/search?q={q}",
    "amazon": "https://www.amazon.com/s?k={q}",
    "youtube": "https://www.youtube.com/results?search_query={q}",
    "github": "https://github.com/search?q={q}",
    "linkedin": "https://www.linkedin.com/search/results/all/?keywords={q}",
    "x_twitter": "https://x.com/search?q={q}",
    "google_maps": "https://www.google.com/maps/search/{q}",
}


def start_url_for(site: str, goal: str, search_query: str | None = None) -> str:
    """Start URL for a task. Landing directly on a results page skips the fragile 'find the search box'
    steps: Google Flights parses a natural-language q=, and most sites accept a search URL."""
    import urllib.parse

    if search_query and site in SEARCH_URLS:
        return SEARCH_URLS[site].format(q=urllib.parse.quote_plus(search_query.strip()[:200]))
    if site == "google_flights":
        user_text = " ".join(
            line.split(":", 1)[-1].strip() if line.lower().startswith("additional details") else line
            for line in goal.splitlines()
            if line.strip()
            and not line.startswith(
                ("Prefer organic", "Use the site", "If the goal", "On travel", "For any", "Never proceed")
            )
        )
        query = _compact_flight_query(user_text) or user_text
        return "https://www.google.com/travel/flights?hl=en&q=" + urllib.parse.quote_plus(query[:300])
    return SITE_URLS.get(site, SITE_URLS["other"])


def _compact_flight_query(user_text: str) -> str:
    """One small LLM call: spoken request -> 'flights from X to Y on <date> returning <date>'."""
    try:
        value, _ = _field_text_sync(
            {
                "goal": user_text,
                "field": {"label": "Google Flights search query", "role": "searchbox", "value": ""},
                "page": {
                    "title": "Google Flights",
                    "text": "Write one line like: flights from Paris to Tokyo on November 2 2026 returning November 7 2026. "
                    "Use the city/airport names and exact dates from the goal; resolve relative dates against today; "
                    "omit anything else (no 'cheapest', no trip length).",
                },
                "recent_actions": [],
            }
        )
        return value.strip().splitlines()[0][:200]
    except Exception:
        log.exception("flight query compaction failed; using raw text")
        return ""


def spoken_title(title: str, limit: int = 60) -> str:
    """Page titles can be a whole search query; keep what a listener can absorb."""
    t = re.sub(r"\s*[-|–]\s*(Google Search|Google Shopping|YouTube|Amazon\.com)\s*$", "", title or "").strip()
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


@dataclass
class BrowserResult:
    status: str  # done | blocked | timeout | error
    steps: int
    elapsed_ms: int
    url: str = ""
    title: str = ""
    error: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    target_id: str | None = None  # the Chrome tab we worked in, for follow-ups
    page_text: str = ""  # visible text of the final page, for outcome verification

    @property
    def spoken(self) -> str:
        where = f" Chrome is on {spoken_title(self.title)}." if self.title else ""
        if self.status == "done":
            return f"Done.{where}" if self.steps == 0 else f"Done in {self.steps} steps.{where}"
        if self.status == "blocked":
            return f"I got stuck after {self.steps} steps.{where} Take a look."
        if self.status == "timeout":
            return f"I ran out of time after {self.steps} steps.{where}"
        return f"I couldn't drive Chrome: {self.error}"


# ----------------------------------------------------------------------------- text helper
class FieldText(BaseModel):
    text: str | None = Field(
        description="Exact string to type into the field, or null if the goal doesn't provide one."
    )


class ComposedGoal(BaseModel):
    objective: str = Field(
        description="One or two sentences a stranger could execute in a browser, self-contained: resolve pronouns and "
        "references ('it', 'that one', 'a black one') using the context, keep every constraint the user stated "
        "(brand, colour, price cap, dates, quantity), add nothing they did not ask for."
    )
    success: str = Field(
        description="What must be visible on screen when the task is complete, concretely (e.g. 'a listing or product "
        "page for a black Coach Women's Lola bag under 300 euros')."
    )
    search_query: str | None = Field(
        description="If the fastest route is a site search, the exact query to search for (all constraints included, "
        "e.g. 'black coach women's lola bag'); otherwise null."
    )


COMPOSE_SYSTEM = (
    "You rewrite a spoken request into a self-contained objective for a browser agent that only sees the current "
    "page and has no memory of the conversation. Use the conversation context to resolve what the user refers to "
    "('it', 'that one', 'a black one'). Keep every constraint the user actually stated (brand, colour, price cap, "
    "dates); add nothing else. Where the user is ('I'm in Paris') is context, not a requirement: never demand "
    "evidence of location, delivery, stock or 'available to buy now' unless they asked for it in those words. "
    "Phrases like 'that I can buy', 'to buy', 'I want to buy' just mean shopping results; never turn them into a "
    "requirement to show purchase options, stock, or a product page. "
    "objective: at most two plain sentences. success: only what the user asked to see. For find / search / look up / "
    "show requests the success is the results list for the refined query (e.g. 'search results showing black Coach "
    "Lola bags'), not an opened product page, unless the user said to open one. "
    "search_query: plain keywords only, at most 8 words, no quotes, no site: or OR operators, no punctuation, "
    "e.g. 'black coach lola bag'. Set it when the request is to find, search, look up or show items; set null when "
    "the request acts on what is already on screen (add to cart, open the first result, pick a date, click). "
    "Return JSON matching the schema."
)


def compose_goal_sync(utterance: str, context: dict[str, Any]) -> ComposedGoal:
    raw = _structured_sync(
        COMPOSE_SYSTEM, json.dumps({"utterance": utterance, **context}, ensure_ascii=False), ComposedGoal
    )
    return ComposedGoal.model_validate_json(raw)


async def compose_goal(utterance: str, context: dict[str, Any]) -> ComposedGoal:
    return await asyncio.to_thread(compose_goal_sync, utterance, context)


def _structured_sync(system: str, user: str, cls: type[BaseModel]) -> str:
    """One structured-output completion on the configured LLM backend; returns the raw JSON text."""
    from .generator import LMSTUDIO_BASE_URL, PROVIDER  # late import: generator pulls livekit

    if PROVIDER == "livekit":
        from livekit.agents import inference, llm

        async def go() -> str:
            m = inference.LLM(
                model=os.environ.get("MACBROW_CHAT_MODEL", "openai/gpt-5-mini"),
                extra_kwargs={"reasoning_effort": "minimal"},
            )
            try:
                ctx = llm.ChatContext()
                ctx.add_message(role="system", content=system)
                ctx.add_message(role="user", content=user)
                async with m.chat(chat_ctx=ctx, response_format=cls) as stream:
                    return "".join([c async for c in stream.to_str_iterable()])
            finally:
                await m.aclose()

        return _run_coro_blocking(go())
    import openai
    from livekit.agents.llm import utils as llm_utils

    client = openai.OpenAI(
        base_url=LMSTUDIO_BASE_URL, api_key=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"), timeout=30
    )
    resp = client.chat.completions.create(
        model=os.environ.get("MACBROW_CHAT_MODEL", "qwen/qwen3.5-9b"),
        max_tokens=400,
        temperature=0.2,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=llm_utils.to_openai_response_format(cls),  # type: ignore[arg-type]
        extra_body={"reasoning_effort": "none"},
    )
    return resp.choices[0].message.content or ""


def _field_text_sync(context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Replacement for jev_ultrafast.model.field_text: same contract, our LLM backend."""
    from jev_ultrafast.questions import TEXT_VALUE

    from .generator import LMSTUDIO_BASE_URL, PROVIDER  # late import: generator pulls livekit

    started = time.perf_counter()
    if PROVIDER == "livekit":
        from livekit.agents import inference, llm

        async def go() -> str:
            m = inference.LLM(
                model=os.environ.get("MACBROW_CHAT_MODEL", "openai/gpt-5-mini"),
                extra_kwargs={"reasoning_effort": "minimal"},
            )
            try:
                ctx = llm.ChatContext()
                ctx.add_message(role="system", content=TEXT_VALUE)
                ctx.add_message(role="user", content=json.dumps(context))
                async with m.chat(chat_ctx=ctx, response_format=FieldText) as stream:
                    return "".join([c async for c in stream.to_str_iterable()])
            finally:
                await m.aclose()

        raw = _run_coro_blocking(go())
        model_name = os.environ.get("MACBROW_CHAT_MODEL", "openai/gpt-5-mini")
    else:
        import openai
        from livekit.agents.llm import utils as llm_utils

        client = openai.OpenAI(
            base_url=LMSTUDIO_BASE_URL, api_key=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"), timeout=30
        )
        model_name = os.environ.get("MACBROW_CHAT_MODEL", "qwen/qwen3.5-9b")
        resp = client.chat.completions.create(
            model=model_name,
            max_tokens=300,
            temperature=0.2,
            messages=[{"role": "system", "content": TEXT_VALUE}, {"role": "user", "content": json.dumps(context)}],
            response_format=llm_utils.to_openai_response_format(FieldText),  # type: ignore[arg-type]
            extra_body={"reasoning_effort": "none"},
        )
        raw = resp.choices[0].message.content or ""
    value = FieldText.model_validate_json(raw).text
    if not value or not value.strip() or len(value) > 2000:
        raise ValueError("Text helper returned no valid field value; nothing typed.")
    return value, {"model": model_name, "latency_ms": round((time.perf_counter() - started) * 1000), "usage": {}}


# ------------------------------------------------------------------------ profile-pinned tab
_context_cache: dict[str, str] = {}
_reuse: dict[str, str | None] = {"target_id": None}  # consumed by ProfileBrowser.__init__


def tab_exists(target_id: str | None) -> bool:
    if not target_id:
        return False
    try:
        return any(t.get("targetId") == target_id for t in _cdp()("Target.getTargets").get("targetInfos", []))
    except Exception:
        return False


def _cdp():
    from browser_harness.helpers import cdp

    return cdp


def _context_id_for_profile(profile_dir: str) -> str | None:
    """browserContextId of the Chrome profile, via a probe tab opened with --profile-directory."""
    cdp = _cdp()
    cached = _context_cache.get(profile_dir)
    if cached:
        infos = cdp("Target.getTargets").get("targetInfos", [])
        if any(t.get("browserContextId") == cached for t in infos):
            return cached
    token = uuid.uuid4().hex
    probe = f"https://example.com/?macbrow_probe={token}"
    subprocess.run(["open", "-na", "Google Chrome", "--args", f"--profile-directory={profile_dir}", probe], check=False)
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        for t in cdp("Target.getTargets").get("targetInfos", []):
            if token in t.get("url", "") and t.get("type") == "page":
                ctx_id = t.get("browserContextId")
                try:
                    cdp("Target.closeTarget", targetId=t["targetId"])
                except Exception:
                    pass
                if ctx_id:
                    _context_cache[profile_dir] = ctx_id
                return ctx_id
        time.sleep(0.15)
    log.warning("could not locate Chrome profile %s over CDP; using default context", profile_dir)
    return None


def _install_patches() -> None:
    """Route jev-ultrafast's browser creation through the pinned profile and our text helper."""
    import jev_ultrafast.agent as ja
    from jev_ultrafast.browser import Browser

    if getattr(ja, "_macbrow_patched", False):
        return

    class ProfileBrowser(Browser):
        def __init__(self, url: str):
            from browser_harness.admin import ensure_daemon

            cdp = _cdp()
            ensure_daemon()
            reuse = _reuse.pop("target_id", None)
            _reuse["target_id"] = None
            if reuse and tab_exists(reuse):
                self.target = reuse  # continue in the previous task's tab; navigate only if a url was given
            else:
                ctx_id = _context_id_for_profile(chrome.profile_dir())
                params: dict[str, Any] = {"url": "about:blank", "background": False}
                if ctx_id:
                    params["browserContextId"] = ctx_id
                self.target = cdp("Target.createTarget", **params)["targetId"]
            self.session = cdp("Target.attachToTarget", targetId=self.target, flatten=True)["sessionId"]
            # Fixed layout viewport (same as upstream). Pop-ups wider than the user's window otherwise push
            # controls offscreen, and offscreen controls never enter the element table Jev chooses from.
            self.call(
                "Emulation.setDeviceMetricsOverride",
                width=VIEWPORT[0],
                height=VIEWPORT[1],
                deviceScaleFactor=1,
                mobile=False,
            )
            self.call("Emulation.setFocusEmulationEnabled", enabled=True)
            if url:
                self.call("Page.navigate", url=url)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if self.evaluate("document.readyState") == "complete":
                    break
                time.sleep(0.02)
            try:
                cdp("Target.activateTarget", targetId=self.target)
            except Exception:
                pass

        def close(self) -> None:
            # Leave the tab open so the user sees the result; remember it for follow-ups.
            self.last_target = self.target
            cdp = _cdp()
            for method in ("Emulation.clearDeviceMetricsOverride",):
                try:
                    cdp(method, session_id=self.session)  # give the tab its normal size back
                except Exception:
                    pass
            try:
                cdp("Target.detachFromTarget", sessionId=self.session)
            except Exception:
                pass
            self.target = None

    ja.Browser = ProfileBrowser
    ja.field_text = _field_text_sync
    _patch_click_executor()
    ja._macbrow_patched = True


def _patch_click_executor() -> None:
    """Scroll a click/fill target into view before the occlusion check.

    Upstream rejects a target whose centre isn't hit by elementFromPoint. Pop-ups that overflow
    the viewport (Google Flights' two-month calendar at ~1300px) make every cell in the second
    month "covered" by its scroll container, so the loop re-decides forever. Scrolling the
    element into view first is what a user's eyes and wheel would do; the freshness guards and
    the observed-node rule are unchanged, and model output still never becomes a selector.
    """
    import json as _json

    import jev_ultrafast.browser as jb
    from jev_ultrafast.browser import StalePage

    original = jb.browser_operation

    def patched(request):
        if request["operation"] != "act" or request["action"]["kind"] not in {"click", "fill"}:
            return original(request)
        action, session = request["action"], request["session"]
        cdp = _cdp()
        if type(action["node"]) is not int:
            raise ValueError("Invalid observed node")
        js = (
            """(action => {
          const e=window.__jevFast?.nodes.get(action.node);
          if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
              !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return null;
          if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
          const centre=()=>{const r=e.getBoundingClientRect(); return {r, x:r.x+r.width/2, y:r.y+r.height/2};};
          const ok=({r,x,y})=> r.width && r.height && x>=0 && y>=0 && x<innerWidth && y<innerHeight &&
                                e.contains(document.elementFromPoint(x,y));
          let c=centre();
          if (!ok(c)) { e.scrollIntoView({block:'nearest', inline:'nearest'}); c=centre(); }
          if (!ok(c)) return null;
          return {x:c.x, y:c.y};
        })("""
            + _json.dumps(action)
            + ")"
        )
        result = cdp("Runtime.evaluate", session_id=session, expression=js, returnByValue=True)
        if result.get("exceptionDetails"):
            raise StalePage("Document changed during evaluation")
        target = result.get("result", {}).get("value")
        if target is None:
            raise StalePage("Target changed or is covered. Observe again.")
        x, y = target["x"], target["y"]
        for event in ("mousePressed", "mouseReleased"):
            cdp("Input.dispatchMouseEvent", session_id=session, type=event, x=x, y=y, button="left", clickCount=1)
        if action["kind"] == "fill":
            import sys as _sys

            mods = 4 if _sys.platform == "darwin" else 2
            cdp(
                "Input.dispatchKeyEvent",
                session_id=session,
                type="keyDown",
                key="a",
                code="KeyA",
                modifiers=mods,
                commands=["selectAll"],
            )
            cdp("Input.dispatchKeyEvent", session_id=session, type="keyUp", key="a", code="KeyA", modifiers=mods)
            cdp("Input.insertText", session_id=session, text=request["text"])
            if DATE_FIELD.search(action.get("label", "")):
                # Date textboxes (Google Flights, booking sites) discard a typed value unless it is committed.
                for t in ("keyDown", "keyUp"):
                    cdp(
                        "Input.dispatchKeyEvent",
                        session_id=session,
                        type=t,
                        key="Enter",
                        code="Enter",
                        windowsVirtualKeyCode=13,
                    )
        return {"executed": action["id"]}

    jb.browser_operation = patched


# ------------------------------------------------------------------------------- run a task
def run_task_sync(
    start_url: str, goal: str, on_progress: Callable[[str], None] | None = None, reuse_target: str | None = None
) -> BrowserResult:
    """Blocking. Drive Chrome toward `goal` starting at `start_url`, or continue in `reuse_target`."""
    _install_patches()
    from jev_ultrafast import Agent

    url = start_url if reuse_target else (start_url or SITE_URLS["other"])
    log.info(
        "browser task start: %s%s",
        "reuse tab" if reuse_target else "",
        (" -> " if reuse_target and url else "") + url[:140],
    )
    _reuse["target_id"] = reuse_target
    import datetime as _dt

    goal = (
        goal.strip()
        + f"\nToday is {_dt.date.today().strftime('%A %d %B %Y')}; resolve relative dates against it.\n"
        + STANDING_RULES
    )
    started = time.perf_counter()
    try:
        agent = _with_timeout(lambda: Agent(url, goal), CONNECT_TIMEOUT_S)
    except TimeoutError:
        return BrowserResult("error", 0, 0, error="Chrome didn't accept the remote-debugging connection in time.")
    except Exception as e:  # daemon / CDP failures
        log.exception("browser connect failed")
        return BrowserResult("error", 0, 0, error=_short(e))
    state: dict[str, Any] = agent.state
    status, error = "timeout", ""
    last_report = time.perf_counter()
    try:
        for state in agent.run():
            now = time.perf_counter()
            if on_progress and state["history"] and now - last_report > PROGRESS_EVERY_S:
                last_report = now
                on_progress(f"Still working, {len(state['history'])} steps in.")
            if now - started > WALL_CLOCK_S:
                break
        else:
            status = state["status"]
    except ValueError as e:  # jev-ultrafast stop conditions (step budget, no progress)
        status, error = "blocked", _short(e)
        log.warning("browser task stopped: %s", error)
        _log_decisions(state)
    except Exception as e:
        status, error = "error", _short(e)
        log.exception("browser task failed")
    finally:
        try:
            agent.close()
        except Exception:
            pass
    page = state.get("page", {}) or {}
    for h in state.get("history", []):
        log.info(
            "  step %d %s %r text=%r p=%.2f changed=%s",
            h["step"],
            h["operation"],
            h["action"][:60],
            h["text"],
            h["probability"],
            h["page_changed"],
        )
    if status != "done":
        last = (state.get("decisions") or [{}])[-1]
        ops = {
            k: round(v, 2)
            for k, v in sorted((last.get("operation_probabilities") or {}).items(), key=lambda kv: -kv[1])[:5]
        }
        tgt = {
            k: round(v, 2)
            for k, v in sorted((last.get("target_probabilities") or {}).items(), key=lambda kv: -kv[1])[:4]
        }
        log.info(
            "  final decision: %s conf=%.2f ops=%s targets=%s",
            last.get("operation"),
            last.get("confidence", 0) or 0,
            ops,
            tgt,
        )
        page_actions = (state.get("page") or {}).get("actions") or []
        log.info(
            "  page offered %d controls: %s",
            len(page_actions),
            [a["label"].strip()[:22] for a in page_actions if a["kind"] == "click"][:24],
        )
    return BrowserResult(
        status,
        len(state.get("history", [])),
        round((time.perf_counter() - started) * 1000),
        url=page.get("url", ""),
        title=page.get("title", ""),
        page_text=(page.get("text") or "")[:4000],
        error=error,
        history=state.get("history", []),
        target_id=getattr(agent.browser, "last_target", None),
    )


async def run_task(
    start_url: str, goal: str, on_progress: Callable[[str], None] | None = None, reuse_target: str | None = None
) -> BrowserResult:
    return await asyncio.to_thread(run_task_sync, start_url, goal, on_progress, reuse_target)


def _log_decisions(state: dict[str, Any], n: int = 8) -> None:
    """Why did the loop stall? Summarise the trailing decisions (operation, target, probability)."""
    decisions = state.get("decisions") or []
    from collections import Counter

    ops = Counter(d.get("operation") for d in decisions)
    log.warning("decisions=%d ops=%s executed=%d", len(decisions), dict(ops), len(state.get("history") or []))
    for d in decisions[-n:]:
        tgt = d.get("target")
        label = ""
        try:
            label = next(a["label"] for a in state["page"]["actions"] if a["id"] == d.get("choice"))
        except (StopIteration, KeyError, TypeError):
            pass
        log.warning(
            "  %s target=%s p=%.2f conf=%.2f %s",
            d.get("operation"),
            tgt,
            max(d.get("probabilities", {}).values() or [0]),
            d.get("confidence", 0),
            label[:70],
        )


def _run_coro_blocking(coro: Any) -> Any:
    """Run a coroutine to completion from sync code, whether or not this thread has a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result(timeout=60)


def _with_timeout(fn: Callable[[], Any], seconds: float) -> Any:
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["v"] = fn()
        except BaseException as e:  # propagate to caller
            box["e"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError()
    if "e" in box:
        raise box["e"]
    return box["v"]


def _short(e: BaseException, n: int = 140) -> str:
    s = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return s if len(s) <= n else s[: n - 1] + "…"
