"""LiveKit Agents entrypoint for macbrow.

    uv run python agent.py console      # local mic/speaker, no LiveKit server needed
    uv run python agent.py dev          # connect to LIVEKIT_URL as a worker
    uv run python agent.py download-files

Pipeline: Deepgram STT -> (Jev router -> AppleScript) | LLM for brief replies -> Deepgram TTS.
LLM defaults to LiveKit Inference (openai/gpt-5-mini); MACBROW_LLM_PROVIDER=openai uses any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, StopResponse, get_job_context, inference, llm
from livekit.plugins import deepgram, silero
from livekit.plugins import openai as lk_openai

from macbrow import generator, policy
from macbrow.agent import DynamicMacAgent

load_dotenv(".env.local")
load_dotenv()

log = logging.getLogger("macbrow.voice")

INSTRUCTIONS = """You are macbrow, a terse voice assistant that controls this Mac.
Mac actions are handled by a fast tool router before you see the message, so anything
that reaches you is small talk or a quick question. Answer in one short spoken sentence.
No markdown, no lists, no emoji, no follow-up questions."""

LLM_PROVIDER = os.environ.get("MACBROW_LLM_PROVIDER", "livekit")  # "livekit" | "openai"
STT_MODEL = os.environ.get("DEEPGRAM_STT_MODEL", "nova-3")
# Deepgram bakes the voice into the TTS model name (aura-2-<voice>-en), so there is no separate voice id.
TTS_MODEL = os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-andromeda-en")


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

    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        text = new_message.text_content or ""
        if not text.strip():
            raise StopResponse()

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

    async def _close() -> None:
        await mac.aclose()

    ctx.add_shutdown_callback(_close)

    log.info("pipeline: stt=deepgram/%s tts=deepgram/%s llm=%s", STT_MODEL, TTS_MODEL, LLM_PROVIDER)
    await session.start(agent=MacBrowAgent(mac), room=ctx.room)
    greeting = "macbrow ready." if policy.ENABLED else "macbrow ready. Warning: the safety policy is off."
    session.say(greeting, add_to_chat_ctx=False)


if __name__ == "__main__":
    agents.cli.run_app(server)
