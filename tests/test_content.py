"""Increment 4 of human behavior: organic content with natural typos.

Posts and comments carry text. Every person writes in their own style, with
personal typo habits and a random typo rate. Two humans almost never write
the same thing, and one person's typos repeat across their own posts but not
across other people's. The content detector (M2) has to tell that apart from
a bot farm pasting one string everywhere.
"""
import random
import re
from collections import Counter

import pytest

import numpy as np

from simulator import content
from simulator.content import NOUNS, vocabulary
from simulator.events import Event
from tests.helpers import cached_world

N, DAYS = 200, 14


def _world():
    return cached_world(7, N, DAYS)


def _words(text):
    return re.findall(r"[a-z]+", text.lower())


def _texts(w, action):
    return [e for e in w.events if e.action == action]


def test_event_text_contract():
    for action in ("post", "comment"):
        with pytest.raises(ValueError):
            Event(0, 0, "a", action, None, None, "10.0.0.1")
        with pytest.raises(ValueError):
            Event(0, 0, "a", action, None, "", "10.0.0.1")
    for action in ("like", "follow"):
        with pytest.raises(ValueError):
            Event(0, 0, "a", action, "x", "hello", "10.0.0.1")


def test_generated_events_obey_the_text_contract():
    w = _world()
    assert all((e.text is not None) == (e.action in ("post", "comment")) for e in w.events)


def test_comments_are_shorter_than_posts():
    w = _world()
    mean = lambda es: sum(len(e.text) for e in es) / len(es)
    assert mean(_texts(w, "comment")) < 0.6 * mean(_texts(w, "post"))


def test_humans_almost_never_write_the_same_thing():
    w = _world()
    posts = [e.text for e in _texts(w, "post")]
    assert len(set(posts)) / len(posts) > 0.98

    grams = [
        {tuple(ws[i : i + 3]) for i in range(len(ws) - 2)}
        for ws in (_words(t) for t in posts)
    ]
    rng = random.Random(0)
    pairs = [(rng.randrange(len(grams)), rng.randrange(len(grams))) for _ in range(20_000)]
    near = sum(
        1 for i, j in pairs
        if i != j and len(grams[i] | grams[j]) and len(grams[i] & grams[j]) / len(grams[i] | grams[j]) >= 0.8
    )
    assert near / len(pairs) < 0.001  # a pasted bot post would score 1.0 against its copies


def test_topics_follow_community():
    w = _world()
    community_of = {n: c for c, nouns in enumerate(NOUNS) for n in nouns}
    hit = total = 0
    for e in _texts(w, "post"):
        for word in _words(e.text):
            if word in community_of:
                total += 1
                hit += community_of[word] == w.profiles[e.account_id].community
    assert total > 1000
    assert hit / total > 0.95


def test_typos_exist_but_are_modest():
    w = _world()
    vocab = vocabulary()
    words = [t for e in w.events if e.text for t in _words(e.text)]
    oov = sum(1 for t in words if t not in vocab) / len(words)
    assert 0.002 <= oov <= 0.06


def test_typo_habits_are_personal():
    """A person's misspellings concentrate on a few habits; strangers don't share them."""
    w = _world()
    vocab = vocabulary()
    per_person: dict[str, Counter] = {}
    for e in w.events:
        if e.text:
            per_person.setdefault(e.account_id, Counter()).update(
                t for t in _words(e.text) if t not in vocab
            )
    tops = [
        c.most_common(1)[0][1] / sum(c.values())
        for c in per_person.values()
        if sum(c.values()) >= 10
    ]
    assert len(tops) > 0.5 * N
    tops.sort()
    assert tops[len(tops) // 2] >= 0.3  # median top-misspelling share; random typos alone scatter far lower


def test_writing_styles_differ_across_people():
    w = _world()
    styles = [p.style for p in w.profiles.values()]
    rates = np.array([s.typo_rate for s in styles])
    assert rates.std() / rates.mean() > 0.4  # typo rates spread widely across people
    lower = sum(s.lowercase for s in styles) / len(styles)
    assert 0.1 <= lower <= 0.5
    assert len({s.habits for s in styles}) > 10


def test_community_nouns_are_single_distinct_words_not_shared_with_templates():
    """A noun that also appears in a template or another community would blur topic signal."""
    shared = set()
    for s in (content.POST_TEMPLATES + content.COMMENT_TEMPLATES + content.REACTS
              + content.ADJECTIVES + content.FEELINGS + content.TIMES + content.VERBS):
        shared.update(re.findall(r"[a-z]+", re.sub(r"\{[a-z0-9]+\}", " ", s.lower())))
    seen: Counter = Counter()
    for group in NOUNS:
        assert len(group) == 16
        for noun in group:
            assert re.fullmatch(r"[a-z]+", noun), noun
            assert noun not in shared, noun
            seen[noun] += 1
    assert all(k == 1 for k in seen.values()), [n for n, k in seen.items() if k > 1]
