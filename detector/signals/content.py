"""Content fingerprinting: the same words, the same slips, across accounts that should differ.

  x_dup_rate     fraction of an account's longer texts (6+ words) that repeat something
                 written EARLIER, by anyone, exactly or as a near-copy (char-shingle
                 Jaccard >= 0.75, so a dropped letter does not hide a copy). The first
                 occurrence is never a duplicate, so a person who gets copied is not
                 blamed for it.
  x_shared_oov   fraction of an account's texts containing a word the reference
                 corpus has never really seen (a likely misspelling) that several other
                 accounts also wrote. People's typos are personal and mostly unique; a
                 farm pastes the same slip everywhere.

`fit` learns the vocabulary from clean reference text. Words that many reference
people use, including popular misspellings, count as vocabulary, so ordinary
personal habits ("teh") never trip the shared-typo feature.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

from detector.minhash import near_duplicates
from detector.signals import Features
from simulator.events import Event

MIN_WORDS = 6
MIN_TEXTS = 5
JACCARD = 0.75
VOCAB_MIN_COUNT = 5  # a reference word seen at least this often is "known"
OOV_MIN_ACCOUNTS = 3  # an unknown word used by this many accounts is shared


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", text.lower())).strip()


class ContentSignal:
    name = "content"
    features = ("x_dup_rate", "x_shared_oov")

    def __init__(self) -> None:
        self.vocab: frozenset[str] = frozenset()

    def fit(self, events: Iterable[Event]) -> None:
        counts: Counter[str] = Counter()
        for e in events:
            if e.text:
                counts.update(normalize(e.text).split())
        self.vocab = frozenset(w for w, c in counts.items() if c >= VOCAB_MIN_COUNT)

    def compute(self, events: Iterable[Event]) -> Features:
        texts = [e for e in events if e.text]
        out: Features = {f: {} for f in self.features}
        self._dup_rate(texts, out["x_dup_rate"])
        self._shared_oov(texts, out["x_shared_oov"])
        return out

    def _dup_rate(self, events: list[Event], out: dict[str, float]) -> None:
        occurrences: dict[str, list[tuple[int, int, str]]] = defaultdict(list)  # text -> (ts, id, acct)
        for e in events:
            norm = normalize(e.text)
            if len(norm.split()) >= MIN_WORDS:
                occurrences[norm].append((e.sim_ts, e.event_id, e.account_id))
        unique = sorted(occurrences)  # sorted: index order must not depend on input order
        for occ in occurrences.values():
            occ.sort()
        neighbours = near_duplicates(unique, JACCARD)
        first = [occurrences[t][0][:2] for t in unique]

        seen: Counter[str] = Counter()
        dups: Counter[str] = Counter()
        for i, text in enumerate(unique):
            derivative = any(first[j] < first[i] for j in neighbours.get(i, ()))
            for k, (_, _, acct) in enumerate(occurrences[text]):
                seen[acct] += 1
                dups[acct] += 1 if (k > 0 or derivative) else 0
        for acct, n in seen.items():
            if n >= MIN_TEXTS:
                out[acct] = dups[acct] / n

    def _shared_oov(self, events: list[Event], out: dict[str, float]) -> None:
        accounts_using: dict[str, set[str]] = defaultdict(set)
        per_text: list[tuple[str, set[str]]] = []
        for e in events:
            words = {w for w in normalize(e.text).split() if len(w) >= 3 and w not in self.vocab}
            per_text.append((e.account_id, words))
            for w in words:
                accounts_using[w].add(e.account_id)
        shared = {w for w, accts in accounts_using.items() if len(accts) >= OOV_MIN_ACCOUNTS}
        seen: Counter[str] = Counter()
        hits: Counter[str] = Counter()
        for acct, words in per_text:
            seen[acct] += 1
            hits[acct] += 1 if words & shared else 0
        for acct, n in seen.items():
            if n >= MIN_TEXTS:
                out[acct] = hits[acct] / n
