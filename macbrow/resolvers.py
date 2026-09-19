"""Computed arguments: Python-side lookups a tool can request before its script runs.

A tool declares  "computed": {"video_url": {"fn": "youtube_first_result", "from": "query"}}
and the agent fills {{video_url}} by calling the named function with the value of the
`query` argument. Keeps AppleScript free of JavaScript, curl, and API keys.
"""

from __future__ import annotations

import asyncio
import re
import urllib.parse
import urllib.request

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)


class ResolveError(RuntimeError):
    """Spoken-friendly message in str(e)."""


def _fetch(url: str, timeout: float = 8.0) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _UA,
            "Accept-Language": "en-US,en;q=0.9",
            # Skip the EU consent interstitial that would otherwise replace the results page.
            "Cookie": "SOCS=CAI; CONSENT=YES+cb",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def youtube_first_result(query: str) -> str:
    """Watch URL of the first video for a YouTube search."""
    q = query.strip()
    if not q:
        raise ResolveError("I didn't catch what to play.")
    html = _fetch("https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(q))
    m = re.search(r'"videoRenderer":\{"videoId":"([A-Za-z0-9_-]{11})"', html) or re.search(
        r'"videoId":"([A-Za-z0-9_-]{11})"', html
    )
    if not m:
        raise ResolveError(f"I couldn't find a YouTube video for {q}.")
    return f"https://www.youtube.com/watch?v={m.group(1)}"


RESOLVERS = {"youtube_first_result": youtube_first_result}


async def run(fn: str, value: str) -> str:
    f = RESOLVERS.get(fn)
    if f is None:
        raise ResolveError(f"unknown resolver {fn}")
    try:
        return await asyncio.wait_for(asyncio.to_thread(f, value), 12)
    except ResolveError:
        raise
    except TimeoutError as e:
        raise ResolveError("The lookup took too long.") from e
    except Exception as e:  # network errors etc.
        raise ResolveError("The lookup failed.") from e
