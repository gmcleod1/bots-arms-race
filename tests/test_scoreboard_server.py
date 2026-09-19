"""The scoreboard server serves the public feed and nothing that would leak the round."""
import json
import urllib.error
import urllib.request

import pytest

from scoreboard.feed import Feed
from scoreboard.server import serve


@pytest.fixture
def running(tmp_path):
    feed = Feed(tmp_path)
    feed.emit("round_started", round_id="r1", attempts_planned=2)
    for secret in ("agent-log.jsonl", "outcomes.json", "detector-timeline.json", "manifest.json"):
        (tmp_path / secret).write_text("SECRET")
    server, _ = serve(tmp_path, port=0)
    yield f"http://127.0.0.1:{server.server_port}", feed
    server.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, dict(r.headers), r.read()


def test_serves_the_dashboard_and_the_public_feed_with_cors_and_no_cache(running):
    base, feed = running
    status, headers, body = _get(base + "/")
    assert status == 200 and b"Bots Arms Race" in body and "text/html" in headers["Content-Type"]
    status, headers, body = _get(base + "/state.json?t=123")
    assert json.loads(body)["round"]["id"] == "r1"
    assert headers["Access-Control-Allow-Origin"] == "*" and headers["Cache-Control"] == "no-store"
    feed.emit("attempt_started", attempt=1)
    assert json.loads(_get(base + "/state.json")[2])["current"]["attempt"] == 1  # live
    assert b"round_started" in _get(base + "/events.jsonl")[2]


@pytest.mark.parametrize("path", ["/agent-log.jsonl", "/outcomes.json", "/detector-timeline.json", "/manifest.json",
                                  "/../secret", "/%2e%2e/secret", "/attempt-1.jsonl", "//etc/passwd"])
def test_nothing_else_in_the_round_directory_is_reachable(running, path):
    base, _ = running
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(base + path)
    assert exc.value.code == 404


def test_the_dashboard_avoids_the_obvious_script_injection_sinks():
    """A static guard only. Whether untrusted text is actually escaped is checked by rendering the
    page with a hostile detector label in a real browser (see the M5 verification notes)."""
    html = open("scoreboard/dashboard.html", encoding="utf-8").read()
    assert "eval(" not in html and "document.write" not in html and "new Function" not in html


def test_the_favicon_request_is_answered_with_no_content_not_an_error(running):
    base, _ = running
    status, _, body = _get(base + "/favicon.ico")
    assert status == 204 and body == b""
