import asyncio

import pytest

from macbrow import resolvers


def test_unknown_resolver_is_a_spoken_error():
    with pytest.raises(resolvers.ResolveError):
        asyncio.run(resolvers.run("nope", "x"))


def test_youtube_parses_first_video_id(monkeypatch):
    html = '... "videoRenderer":{"videoId":"abcdefghijk","thumb":1} ... "videoId":"zzzzzzzzzzz"'
    monkeypatch.setattr(resolvers, "_fetch", lambda url, timeout=8.0: html)
    assert resolvers.youtube_first_result("anything") == "https://www.youtube.com/watch?v=abcdefghijk"


def test_youtube_empty_query_and_no_results(monkeypatch):
    with pytest.raises(resolvers.ResolveError):
        resolvers.youtube_first_result("   ")
    monkeypatch.setattr(resolvers, "_fetch", lambda url, timeout=8.0: "<html>nothing</html>")
    with pytest.raises(resolvers.ResolveError):
        resolvers.youtube_first_result("obscure")
