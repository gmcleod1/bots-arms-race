"""Small scripted bot patterns for exercising signals (NOT the M3 naive-bot baseline).

Each pattern drives the real Platform API, so it obeys the same rules a bot
would: time only moves forward and targets must exist. Actions are collected
first and replayed in time order.
"""
from simulator import content
from simulator.bots import PLAIN_STYLE as PLAIN
from simulator.bots import Script  # noqa: F401  (re-exported for the tests)

HORIZON = 10 * 86_400


def _text(rng):
    return content.compose(rng, "post", int(rng.integers(8)), PLAIN)


def fixed_interval_bots(s, rng, n=6, interval=1500, count=400, prefix="fx"):
    """24/7, exactly one post every `interval` seconds, unique text, no targets."""
    for i in range(n):
        acct = f"{prefix}-{i}"
        s.create(acct, 3_600)
        for k in range(count):
            s.post(acct, 3_600 + i * 7 + k * interval, _text(rng))
    return [f"{prefix}-{i}" for i in range(n)]


def jitter_bots(s, rng, n=6, lo=600, hi=2_400, prefix="jt"):
    """24/7 with uniform random gaps: the classic 'add jitter' evasion."""
    for i in range(n):
        acct = f"{prefix}-{i}"
        s.create(acct, 3_600)
        t = 3_600.0
        while t < HORIZON - hi:
            t += rng.uniform(lo, hi)
            s.post(acct, int(t), _text(rng))
    return [f"{prefix}-{i}" for i in range(n)]


def lockstep_bots(s, rng, platform, n=12, k=12, spread=10, prefix="ls"):
    """n bots like the same k human posts within `spread` seconds of each other."""
    posts = [e for e in platform.human_events if e.action == "post" and 86_400 <= e.sim_ts <= 8 * 86_400]
    picks = sorted(rng.choice(len(posts), size=k, replace=False))
    targets = [posts[int(i)] for i in picks]
    accts = [f"{prefix}-{i}" for i in range(n)]
    for a in accts:
        s.create(a, targets[0].sim_ts)
    for post in targets:
        for a in accts:
            s.like(a, post.sim_ts + 40 + int(rng.integers(0, spread + 1)), post.post_id)
    return accts


def copy_paste_bots(s, rng, n=10, k=8, mutate=False, prefix="cp"):
    """Every bot posts the same k texts at its own random times. `mutate` adds one dropped letter."""
    texts = [_text(rng) for _ in range(k)]
    accts = [f"{prefix}-{i}" for i in range(n)]
    for a in accts:
        s.create(a, 3_600)
        for text in texts:
            t = text
            if mutate:
                words = t.split(" ")
                j = int(rng.integers(len(words)))
                if len(words[j]) >= 5:
                    d = int(rng.integers(1, len(words[j]) - 1))
                    words[j] = words[j][:d] + words[j][d + 1 :]
                t = " ".join(words)
            s.post(a, int(rng.uniform(3_600, 9 * 86_400)), t)
    return accts


def novel_typo_bots(s, rng, n=10, k=10, token="recieved", prefix="nt"):
    """Otherwise unique posts that all share one misspelling nobody else uses."""
    accts = [f"{prefix}-{i}" for i in range(n)]
    for a in accts:
        s.create(a, 3_600)
        for _ in range(k):
            s.post(a, int(rng.uniform(3_600, 9 * 86_400)), f"{_text(rng)} I {token}")
    return accts


def copy_human_posts(s, rng, victim_posts, n=4, prefix="cv"):
    """Bots repost a human's exact long posts after the human wrote them."""
    accts = [f"{prefix}-{i}" for i in range(n)]
    start = max(p.sim_ts for p in victim_posts) + 60
    for a in accts:
        s.create(a, start)
        for j, p in enumerate(victim_posts):
            s.post(a, start + 100 + j * 50, p.text)
    return accts
