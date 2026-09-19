import json

from macbrow import chrome

LOCAL_STATE = {
    "profile": {
        "last_used": "Profile 2",
        "info_cache": {
            "Default": {"name": "You", "user_name": ""},
            "Profile 1": {"name": "Work", "user_name": "me@work.example"},
            "Profile 2": {"name": "Home", "user_name": "me@home.example"},
        },
    }
}


def _prep(monkeypatch, tmp_path, email):
    path = tmp_path / "Local State"
    path.write_text(json.dumps(LOCAL_STATE))
    monkeypatch.setattr(chrome, "LOCAL_STATE", path)
    monkeypatch.setattr(chrome, "PROFILE_EMAIL", email)
    monkeypatch.setattr(chrome, "PROFILE_DIR_OVERRIDE", None)
    monkeypatch.setattr(chrome, "_cache", None)


def test_profile_by_email(monkeypatch, tmp_path):
    _prep(monkeypatch, tmp_path, "ME@work.example")
    assert chrome.profile_dir() == "Profile 1"
    assert chrome.system_vars()["chrome_profile"] == "Profile 1"


def test_unset_email_uses_last_used(monkeypatch, tmp_path):
    _prep(monkeypatch, tmp_path, "")
    assert chrome.profile_dir() == "Profile 2"


def test_unknown_email_falls_back_to_last_used(monkeypatch, tmp_path):
    _prep(monkeypatch, tmp_path, "nobody@example.com")
    assert chrome.profile_dir() == "Profile 2"


def test_directory_override_wins(monkeypatch, tmp_path):
    _prep(monkeypatch, tmp_path, "me@work.example")
    monkeypatch.setattr(chrome, "PROFILE_DIR_OVERRIDE", "Profile 7")
    assert chrome.profile_dir() == "Profile 7"
