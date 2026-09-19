import numpy as np
import pytest

from detector.detector import Detector
from detector.signals import default_signals
from simulator.platform import Platform
from tests import botkit
from tests.helpers import cached_world


@pytest.fixture(scope="session")
def ref_world():
    """Clean historical data the detector calibrates on."""
    return cached_world(101, 800, 10)


@pytest.fixture(scope="session")
def eval_world():
    """A different seed, same scale: what the detector is evaluated on."""
    return cached_world(202, 800, 10)


@pytest.fixture(scope="session")
def duel_ref():
    """A small clean world for face-off tests: big enough to calibrate, fast to build."""
    return cached_world(301, 250, 6)


@pytest.fixture(scope="session")
def duel_eval():
    return cached_world(302, 250, 6)


@pytest.fixture(scope="session")
def fitted_signals(ref_world):
    signals = default_signals()
    for s in signals:
        s.fit(ref_world.events)
    return signals


@pytest.fixture(scope="session")
def detector(ref_world):
    """Calibrated once on clean history at a 5% false-positive budget."""
    d = Detector(fpr_budget=0.05)
    d.fit(ref_world.events)
    return d


@pytest.fixture(scope="session")
def human_features(eval_world, fitted_signals):
    """feature -> account -> raw value, humans only."""
    out = {}
    for s in fitted_signals:
        out.update(s.compute(eval_world.events))
    return out


def _long_posts(world, account):
    return [
        e for e in world.events
        if e.account_id == account and e.action == "post" and len(e.text.split()) >= 6
    ]


@pytest.fixture(scope="session")
def bot_world(eval_world):
    """Every bot pattern injected at once so features are computed once."""
    platform = Platform(eval_world)
    s = botkit.Script(platform)
    rng = np.random.default_rng(5)
    groups = {
        "fixed": botkit.fixed_interval_bots(s, rng),
        "jitter": botkit.jitter_bots(s, rng),
        "lockstep": botkit.lockstep_bots(s, rng, platform),
        "slow_lockstep": botkit.lockstep_bots(s, rng, platform, spread=7_200, prefix="sl"),
        "copy": botkit.copy_paste_bots(s, rng),
        "typo_copy": botkit.copy_paste_bots(s, rng, mutate=True, prefix="tc"),
        "novel_typo": botkit.novel_typo_bots(s, rng),
    }
    victim = next(a for a in eval_world.ground_truth if len(_long_posts(eval_world, a)) >= 8)
    groups["victim_copiers"] = botkit.copy_human_posts(s, rng, _long_posts(eval_world, victim)[:8])
    s.run()
    return {"platform": platform, "groups": groups, "victim": victim, "events": platform.events_until(10**9)}


@pytest.fixture(scope="session")
def bot_features(bot_world, fitted_signals):
    out = {}
    for s in fitted_signals:
        out.update(s.compute(bot_world["events"]))
    return out
