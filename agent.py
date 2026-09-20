"""LiveKit Agents entrypoint for macbrow.

    uv run python agent.py console      # local mic/speaker, no LiveKit server needed
    uv run python agent.py dev          # connect to LIVEKIT_URL as a worker
    uv run python agent.py download-files

Pipeline: Deepgram STT -> (Jev router -> AppleScript) | LLM for brief replies -> Deepgram TTS.
LLM defaults to LiveKit Inference (openai/gpt-5-mini); MACBROW_LLM_PROVIDER=openai uses any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
import time

from dotenv import load_dotenv

# Before any macbrow import: those modules read their configuration at import time, so loading
# the env afterwards leaves them on defaults unless the caller exported everything first (which
# is what console.sh does, and why running agent.py directly used to behave differently).
load_dotenv(".env.local")
load_dotenv()

from livekit import agents  # noqa: E402
from livekit.agents import (  # noqa: E402
    Agent,
    AgentServer,
    AgentSession,
    StopResponse,
    get_job_context,
    inference,
    llm,
)
from livekit.plugins import deepgram, silero  # noqa: E402
from livekit.plugins import openai as lk_openai  # noqa: E402

from macbrow import generator, policy  # noqa: E402
from macbrow.agent import DynamicMacAgent  # noqa: E402

log = logging.getLogger("macbrow.voice")

INSTRUCTIONS = """You are macbrow, a terse voice assistant that controls this Mac.
Mac actions are handled by a fast tool router before you see the message, so anything
that reaches you is small talk or a quick question. Answer in one short spoken sentence.
No markdown, no lists, no emoji, no follow-up questions."""

LLM_PROVIDER = os.environ.get("MACBROW_LLM_PROVIDER", "livekit")  # "livekit" | "openai"
STT_MODEL = os.environ.get("DEEPGRAM_STT_MODEL", "nova-3")
# Deepgram bakes the voice into the TTS model name (aura-2-<voice>-en), so there is no separate voice id.
TTS_MODEL = os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-andromeda-en")
# Deepgram bills for as long as the socket is open, silence included -- VAD only forwards a copy
# of the audio for turn detection, it never gates the stream. So an agent left running costs
# wall-clock time having heard nothing. End the session after this long without speech; 0 disables.
IDLE_TIMEOUT_S = float(os.environ.get("MACBROW_IDLE_TIMEOUT_S", "900"))


def build_chat_llm() -> llm.LLM:
    if LLM_PROVIDER == "livekit":
        return inference.LLM(
            model=os.environ.get("MACBROW_CHAT_MODEL", "openai/gpt-5-mini"),
            extra_kwargs={
                "reasoning_effort": os.environ.get("MACBROW_CHAT_REASONING", "minimal"),
                "max_completion_tokens": 80,
            },
        )
    return lk_openai.LLM(
        model=os.environ.get("MACBROW_CHAT_MODEL", "qwen/qwen3.5-9b"),
        base_url=generator.OPENAI_BASE_URL,
        api_key=generator.OPENAI_API_KEY,
        temperature=0.3,
        max_completion_tokens=60,
        # Qwen 3.5 thinks by default; LM Studio turns it off with reasoning_effort "none".
        extra_body={"reasoning_effort": os.environ.get("MACBROW_REASONING_EFFORT", "none")},
    )


class MacBrowAgent(Agent):
    def __init__(self, mac: DynamicMacAgent) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self.mac = mac
        self.last_speech = time.monotonic()

    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        text = new_message.text_content or ""
        if not text.strip():
            raise StopResponse()
        self.last_speech = time.monotonic()

        outcome = await self.mac.handle(text)
        r = outcome.route
        log.info(
            "turn %r -> %s timings=%s",
            text,
            "llm" if outcome.handoff_to_llm else (r.summary if r else "?"),
            {k: round(v) for k, v in outcome.timings.items()},
        )
        if outcome.handoff_to_llm:
            return  # normal LLM reply

        if outcome.stop:
            try:
                await self.session.say(outcome.speak or "Goodbye.", add_to_chat_ctx=False)
            except RuntimeError:
                pass
            get_job_context().shutdown(reason="user asked macbrow to stop")
            raise StopResponse()

        if outcome.speak:
            # Speak the deterministic result and keep it in history so the LLM has context later.
            try:
                self.session.say(outcome.speak, add_to_chat_ctx=True)
            except RuntimeError as e:  # session closing mid-turn (ctrl-c during a route)
                log.warning("could not speak result: %s", e)
        raise StopResponse()


server = AgentServer()


@server.rtc_session(agent_name=os.environ.get("MACBROW_AGENT_NAME", "macbrow"))
async def entrypoint(ctx: agents.JobContext) -> None:
    session: AgentSession | None = None

    def _filler(text: str) -> None:
        if session is None:
            return
        try:
            session.say(text, add_to_chat_ctx=False)
        except RuntimeError:
            pass

    mac = DynamicMacAgent(
        enable_learning=os.environ.get("MACBROW_LEARN", "1") != "0",
        on_learning=_filler,
    )
    await mac.start()

    session = AgentSession(
        stt=deepgram.STT(model=STT_MODEL, language=os.environ.get("MACBROW_LANG", "en")),
        llm=build_chat_llm(),
        tts=deepgram.TTS(model=TTS_MODEL),
        vad=silero.VAD.load(),
        preemptive_generation=False,  # we decide per-turn whether the LLM runs at all
    )

    watchdog: asyncio.Task[None] | None = None

    async def _close() -> None:
        if watchdog is not None:
            watchdog.cancel()
        await mac.aclose()

    ctx.add_shutdown_callback(_close)

    async def _idle_shutdown(agent: MacBrowAgent) -> None:
        while True:
            await asyncio.sleep(min(30.0, IDLE_TIMEOUT_S))
            idle = time.monotonic() - agent.last_speech
            if idle >= IDLE_TIMEOUT_S:
                log.info("no speech for %.0fs; closing the session so the STT stream stops billing", idle)
                # shutdown() ends the job and closes the Deepgram sockets, but leaves the worker
                # process up waiting for the next job, which reads as "still running" to anyone
                # checking. Ask for the clean exit Ctrl-C gives, once the session has closed.
                # A plain timer thread, because shutdown() cancels this task and stops the job's
                # event loop -- neither an await nor call_later survives it. Process-directed,
                # because the job runs off the main thread, where a raised signal is never handled.
                threading.Timer(3.0, os.kill, (os.getpid(), signal.SIGINT)).start()
                get_job_context().shutdown(reason="idle timeout")
                return

    log.info(
        "pipeline: stt=deepgram/%s tts=deepgram/%s llm=%s idle_timeout=%s",
        STT_MODEL,
        TTS_MODEL,
        LLM_PROVIDER,
        f"{IDLE_TIMEOUT_S:.0f}s" if IDLE_TIMEOUT_S > 0 else "off",
    )
    agent = MacBrowAgent(mac)
    await session.start(agent=agent, room=ctx.room)
    if IDLE_TIMEOUT_S > 0:
        watchdog = asyncio.create_task(_idle_shutdown(agent), name="macbrow-idle")
    greeting = "macbrow ready." if policy.ENABLED else "macbrow ready. Warning: the safety policy is off."
    session.say(greeting, add_to_chat_ctx=False)


if __name__ == "__main__":
    agents.cli.run_app(server)
