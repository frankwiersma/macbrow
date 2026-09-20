"""macbrow: voice-controlled macOS agent.

LiveKit Agents (Deepgram STT/TTS) -> Jev (TypeSafe System One) routes each
utterance to an AppleScript tool in ~150ms -> osascript executes it. Unknown
requests fall back to an LLM, which writes a new AppleScript tool that is
cached in the registry so the next request is fast. Multi-step website work
is delegated to jev-ultrafast (Browser Use x TypeSafe) inside the user's Chrome.
"""

__all__ = ["DynamicMacAgent"]

from .agent import DynamicMacAgent
