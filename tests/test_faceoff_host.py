"""Hot-reload: versioned patches, atomic swaps, and the turn-based fair-play mode."""
import threading

import pytest

from detector.config import ConfigError, DetectorConfig
from faceoff.host import DetectorHost


@pytest.fixture(scope="module")
def reference(duel_ref):
    return duel_ref.events


def test_a_live_patch_can_turn_target_overlap_on(reference):
    host = DetectorHost(reference)
    assert host.current.alpha == pytest.approx(0.05 / 6)
    result = host.patch("catch slow lockstep", target_overlap=True)
    assert result.version.diff == {"target_overlap": [False, True]}
    assert result.version.alpha == pytest.approx(0.05 / 7)


def test_starts_at_v1_and_a_patch_creates_v2_with_a_diff(reference):
    seen = []
    host = DetectorHost(reference, on_event=lambda t, **d: seen.append((t, d)))
    assert host.current.id == "v1" and host.current.diff == {}
    result = host.patch("accept more false positives", fpr_budget=0.10)
    assert not result.queued and result.version.id == "v2" and host.current.id == "v2"
    assert result.version.diff == {"fpr_budget": [0.05, 0.10]}
    assert result.version.alpha == pytest.approx(0.10 / 6)
    assert [t for t, _ in seen] == ["detector_patched", "detector_patched"]


def test_an_invalid_patch_changes_nothing(reference):
    host = DetectorHost(reference)
    for bad in ({"warp": 1}, {"fpr_budget": 5}, {"disabled_features": ["nope"]}):
        with pytest.raises(ConfigError):
            host.patch("bad", **bad)
    with pytest.raises(ConfigError):
        host.patch("empty")
    assert [v.id for v in host.versions] == ["v1"]


def test_scoring_reports_the_version_it_used_and_records_it(reference, duel_eval):
    host = DetectorHost(reference)
    _, v1 = host.score(duel_eval.events)
    host.patch("looser", fpr_budget=0.10)
    _, v2 = host.score(duel_eval.events)
    assert (v1.id, v2.id) == ("v1", "v2")
    assert [(r.version_id, r.n_events) for r in host.score_log] == [("v1", len(duel_eval.events))] * 1 + [
        ("v2", len(duel_eval.events))]


def test_a_patch_during_scoring_never_changes_the_run_in_flight(reference, duel_eval):
    host = DetectorHost(reference)
    started, release = threading.Event(), threading.Event()
    original = host._detector.score

    def slow(events):
        started.set()
        release.wait(timeout=10)
        return original(events)

    host._detector.score = slow  # instance override: only v1's detector is slowed
    out = {}
    worker = threading.Thread(target=lambda: out.update(zip(("verdicts", "version"), host.score(duel_eval.events))))
    worker.start()
    assert started.wait(timeout=10)
    host.patch("patched mid-score", fpr_budget=0.10)  # swaps while v1 is still scoring
    release.set()
    worker.join(timeout=30)
    assert out["version"].id == "v1" and host.current.id == "v2"
    assert host.score(duel_eval.events)[1].id == "v2"


def test_pause_edits_queues_patches_until_thaw_and_applies_them_in_order(reference):
    host = DetectorHost(reference, pause_edits=True)
    host.freeze()
    a = host.patch("first", fpr_budget=0.08)
    b = host.patch("second", coordination_window=300)
    assert a.queued and b.queued and (a.queue_position, b.queue_position) == (1, 2)
    assert host.current.id == "v1" and len(host.pending) == 2  # nothing applied mid-attempt
    with pytest.raises(ConfigError):
        host.patch("bad while frozen", fpr_budget=9)  # rejected immediately, not at thaw
    applied = host.thaw()
    assert [v.id for v in applied] == ["v2", "v3"] and host.pending == []
    assert host.current.config.fpr_budget == 0.08 and host.current.config.coordination_window == 300


def test_freeze_is_a_no_op_unless_pause_edits_is_on(reference):
    host = DetectorHost(reference, pause_edits=False)
    host.freeze()
    assert not host.patch("live", fpr_budget=0.07).queued and host.current.id == "v2"


def test_history_is_json_serializable_and_complete(reference, duel_eval):
    import json

    host = DetectorHost(reference)
    host.score(duel_eval.events)
    host.patch("looser", fpr_budget=0.10)
    host.score(duel_eval.events)
    h = json.loads(json.dumps(host.history()))
    assert [v["id"] for v in h["versions"]] == ["v1", "v2"]
    assert DetectorConfig.from_json(h["versions"][1]["config"]).fpr_budget == 0.10
    assert [s["version"] for s in h["scores"]] == ["v1", "v2"]
