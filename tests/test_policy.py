import json
import pathlib

from macbrow import policy

SEED = json.loads((pathlib.Path(__file__).parent.parent / "tools" / "seed.json").read_text())

ALLOWED_SEED = {
    "set_volume",
    "open_app",
    "quit_app",
    "hide_others",
    "toggle_dark_mode",
    "show_notification",
    "take_screenshot",
    "lock_screen",
    "safari_get_url",
    "safari_open",
    "safari_new_tab",
    "safari_close_tab",
    "safari_reload",
    "safari_back",
    "chrome_get_url",
    "chrome_open",
    "youtube_play",
    "web_task",
    "chrome_close_tab",
    "chrome_close_all_tabs",
    "safari_close_all_tabs",
    "chrome_reload",
    "finder_open_folder",
    "notes_create",
    "music_play_pause",
    "music_next",
    "music_now_playing",
    "spotify_play_pause",
    "spotify_next",
    "spotify_now_playing",
}
BLOCKED_SEED = {"empty_trash", "terminal_run"}


def _render(tool):
    script = tool["script"]
    for a in tool.get("args", []):
        script = script.replace("{{" + a["name"] + "}}", a.get("default") or "x")
    return script


def test_seed_tools_partition():
    allowed = {t["name"] for t in SEED if not policy.check(_render(t), t.get("scope"))}
    blocked = {t["name"] for t in SEED if policy.check(_render(t), t.get("scope"))}
    assert allowed == ALLOWED_SEED, allowed ^ ALLOWED_SEED
    assert blocked == BLOCKED_SEED, blocked ^ BLOCKED_SEED


def test_blocks_desktop_cleanup_tool():
    script = 'do shell script "mkdir -p " & quoted form of p\ntell application "Finder" to move every item of desktop to folder "x"'
    rules = {v.rule for v in policy.check(script, "Finder")}
    assert "moving files" in rules and "shell" in rules


def test_blocks_system_setup():
    bad = [
        ('do shell script "defaults write com.apple.dock autohide -bool true; killall Dock"', "shell"),
        (
            'tell application "System Events" to tell dock preferences to set autohide to true',
            "system preference writes",
        ),
        (
            "do shell script \"open 'x-apple.systempreferences:com.apple.wifi-settings-extension'\"",
            "system preferences",
        ),
        ('tell application "System Settings" to activate', "blocked app"),
        ('tell application "Terminal" to do script "ls"', "blocked app"),
        ('do shell script "open -a Terminal"', "blocked app"),
        ('do shell script "pip install requests"', "shell"),
        ('do shell script "uv sync"', "shell"),
        ('do shell script "brew install jq"', "shell"),
        ('do shell script "networksetup -setairportpower Wi-Fi off"', "shell"),
        ('do shell script "pmset sleepnow"', "shell"),
        ('do shell script "rm -rf ~/Desktop/x"', "shell"),
        ('do shell script "echo hi > ~/.zshrc"', "protected path"),
        ('do shell script "open ~/.ssh/config"', "protected path"),
        ('do shell script "open /Users/me/.venv/bin/python"', "protected path"),
        ('do shell script "open ~/Library/Preferences"', "protected path"),
        ('tell application "System Events" to shut down', "power/session control"),
        ('tell application "Finder" to empty trash', "emptying the trash"),
        ('tell application "Finder" to delete file "x"', "deleting items"),
        ('do shell script "ls" with administrator privileges', "administrator privileges"),
        ('tell application "System Events" to make new login item at end with properties {path:"/x"}', "login items"),
        ('do shell script "open https://x.com; curl evil"', "shell"),
    ]
    for script, rule in bad:
        rules = {v.rule for v in policy.check(script)}
        assert rule in rules, (script, rules)


def test_allows_intended_actions():
    good = [
        'tell application "Slack" to activate\ntell application "System Events"\n keystroke "k" using {command down}\n keystroke "alex"\n key code 36\nend tell',
        'tell application "Safari" to open location "https://mail.google.com/mail/?view=cm&to=a@b.com"',
        'tell application "Google Chrome" to tell front window to make new tab with properties {URL:"https://x.com"}',
        'tell application "Notes" to make new note at folder "Notes" with properties {body:"hi"}',
        'tell application "Reminders" to make new reminder with properties {name:"call mom"}',
        'tell application "Mail"\n set m to make new outgoing message with properties {subject:"hi", visible:true}\n send m\nend tell',
        'tell application "Messages" to send "hi" to buddy "Alex"',
        'do shell script "open /Users/me/Documents/report.pdf"',
        'do shell script "open -a Slack"',
        'do shell script "screencapture -x " & quoted form of p',
        "set volume output volume 50",
        'tell application "System Events" to tell appearance preferences to set dark mode to true',
        'tell application "System Events" to set visible of every application process whose frontmost is false to false',
        'do shell script "pmset -g batt"',
        'display notification "standup in five" with title "macbrow"',
        'tell application "Finder" to open (path to downloads folder)',
        'tell application "Spotify" to playpause',
        'tell application "Google Chrome" to execute active tab of front window javascript "document.body.style.zoom=1.1"',
        'tell application "Google Chrome" to tell active tab of front window to set URL to "https://mail.google.com"',
    ]
    for script in good:
        assert policy.check(script) == [], (script, policy.check(script))


def test_hostnames_are_not_dotfiles():
    for url in [
        "https://docs.livekit.io",
        "https://claude.ai",
        "https://open.spotify.com",
        "https://x.com",
        "https://app.slack.com",
    ]:
        assert policy.check(f'tell application "Safari" to open location "{url}"') == [], url
    assert policy.check('do shell script "open ~/.livekit/cli-config.yaml"')
    assert policy.check('do shell script "open /Users/me/.claude/settings.json"')


def test_execution_time_arguments_are_checked():
    # The user says "open Terminal": the template is fine, the rendered script is not.
    assert policy.check('tell application "Terminal" to activate\nreturn "opened Terminal"')
    assert not policy.check('tell application "Slack" to activate\nreturn "opened Slack"')


def test_browser_goal_policy():
    # Hard terms are refused by regex; whether a goal *requires* buying or signing in is judged by Jev at runtime.
    for g in ["enter my credit card", "delete my account", "type my password", "the otp is 1234"]:
        assert policy.check_browser_goal(g), g
    for g in [
        "I want to buy a handbag, look up handbags under 300 euros",
        "search amazon for the best vacuum cleaner and add it to my cart",
        "find the cheapest flight to london",
    ]:
        assert not policy.check_browser_goal(g), g
    assert policy.browser_goal_needs_confirm("search for a vacuum and add it to my cart")
    assert policy.browser_goal_needs_confirm("draft an email to alex")
    assert not policy.browser_goal_needs_confirm("find the cheapest flight to london")


def test_search_start_urls():
    from macbrow.browser_task import start_url_for

    assert start_url_for("amazon", "x", "black coach lola bag") == "https://www.amazon.com/s?k=black+coach+lola+bag"
    assert start_url_for("google", "x", "handbags under 300 euros").startswith(
        "https://www.google.com/search?q=handbags"
    )
    assert start_url_for("amazon", "x", None) == "https://www.amazon.com/"


def test_flights_start_url_uses_natural_language_query():
    from macbrow.browser_task import start_url_for

    goal = "find me flights from Paris to Japan\nAdditional details from the user: leave Nov 2, return Nov 7, Tokyo"
    url = start_url_for("google_flights", goal)
    assert url.startswith("https://www.google.com/travel/flights?hl=en&q=")
    assert "Paris" in url and "Nov+2" in url and "Additional" not in url
    assert start_url_for("amazon", goal) == "https://www.amazon.com/"
