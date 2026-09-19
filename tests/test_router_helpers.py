from macbrow.router import _clean_value, _span_candidates


def test_span_candidates_prefers_suffixes_then_inner_spans():
    spans = _span_candidates("send Constance a message saying hi there.")
    assert spans[0] == "send Constance a message saying hi there"
    assert "hi there" in spans  # suffix
    assert "Constance" in spans  # inner span
    assert len(spans) == len({s.lower() for s in spans})  # de-duplicated


def test_span_candidates_respects_budget():
    long = " ".join(f"w{i}" for i in range(40))
    # every suffix is always offered; inner spans fill the rest of the budget
    assert len(_span_candidates(long, max_candidates=60)) == 60
    assert len(_span_candidates(long, max_candidates=10)) == 40


def test_clean_value_normalises_spoken_urls():
    assert _clean_value(" github dot com ") == "github.com"
    assert _clean_value('"docs dot livekit dot io slash agents"') == "docs.livekit.io/agents"


def test_clean_value_strips_trailing_punctuation():
    assert _clean_value("trailer of love hypothesis.") == "trailer of love hypothesis"
    assert _span_candidates("play the trailer. Done.")[0] == "play the trailer. Done"
