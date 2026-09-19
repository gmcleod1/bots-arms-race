"""The defender's command line: parses commands and only reaches the console."""
import json

import pytest

from faceoff.console import DefenderConsole
from faceoff.host import DetectorHost
from faceoff.repl import handle, run_repl


@pytest.fixture(scope="module")
def host(duel_ref):
    return DetectorHost(duel_ref.events)


def _console(host):
    import threading

    return DefenderConsole(host, [], threading.Event(), threading.Event())


def test_status_and_help_and_unknown(host):
    c = _console(host)
    assert "detector v1" in handle(c, "status") and "budget 5%" in handle(c, "status")
    assert "patch" in handle(c, "help")
    assert "unknown command" in handle(c, "frobnicate")
    assert handle(c, "") == "" and handle(c, "quit") is None


def test_patch_parses_json_values_and_applies_or_rejects(duel_ref):
    c = _console(DetectorHost(duel_ref.events))
    out = handle(c, 'patch "accept more false positives" fpr_budget=0.10')
    assert out.startswith("applied as v2") and c.status()["fpr_budget"] == 0.10
    out = handle(c, "patch \"drop noisy typo feature\" disabled_features='[\"x_shared_oov\"]'")
    assert "v3" in out and "x_shared_oov" not in c.status()["features"]
    assert handle(c, 'patch "bad" fpr_budget=9').startswith("rejected")
    assert handle(c, 'patch "bad" warp=1').startswith("rejected")
    assert handle(c, "patch").startswith("usage") and "expected key=value" in handle(c, 'patch "x" oops')
    assert json.loads(json.dumps(c.status()))["version"] == "v3"  # bad patches changed nothing


def test_fair_play_patch_is_reported_as_queued(duel_ref):
    host = DetectorHost(duel_ref.events, pause_edits=True)
    c = _console(host)
    host.freeze()
    assert "queued at position 1" in handle(c, 'patch "later" fpr_budget=0.08')
    assert handle(c, "go").startswith("released")


def test_flags_before_any_attempt_and_the_loop_stops_on_quit_eof_or_round_end(host):
    c = _console(host)
    assert handle(c, "flags") == "no attempt has been scored yet"
    lines = iter(["status", "quit", "status"])
    out = []
    run_repl(c, lambda: False, read=lambda: next(lines), write=out.append)
    assert len(out) == 1  # stopped at quit

    def eof():
        raise EOFError

    run_repl(c, lambda: False, read=eof, write=out.append)  # returns cleanly on EOF
    run_repl(c, lambda: True, read=lambda: "status", write=out.append)  # round already over: no reads
    assert len(out) == 1
