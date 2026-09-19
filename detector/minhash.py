"""Near-duplicate text search: char shingles, MinHash, banded LSH, exact verification.

Written here instead of using a library because it must be byte-for-byte
deterministic across runs (stable crc32 shingle hashes, seeded permutations),
and because LSH candidates are verified with exact Jaccard anyway.
"""
from __future__ import annotations

import zlib
from collections import defaultdict

import numpy as np

_P = (1 << 31) - 1  # prime modulus; hashes and coefficients stay under 2**31 so products fit uint64
NUM_PERM = 60
ROWS = 6  # 10 bands x 6 rows: catches pairs at Jaccard 0.75 about 87% of the time, 0.83 about 98%
BANDS = NUM_PERM // ROWS


def shingles(text: str, k: int = 5) -> np.ndarray:
    """Sorted unique stable hashes of the text's character k-grams."""
    grams = {zlib.crc32(text[i : i + k].encode("utf-8")) & 0x7FFFFFFF for i in range(len(text) - k + 1)}
    return np.array(sorted(grams), dtype=np.uint64)


class MinHasher:
    def __init__(self, seed: int = 20260919) -> None:
        rng = np.random.default_rng(seed)
        self._a = rng.integers(1, _P, size=(NUM_PERM, 1), dtype=np.uint64)
        self._b = rng.integers(0, _P, size=(NUM_PERM, 1), dtype=np.uint64)

    def signature(self, hashes: np.ndarray) -> np.ndarray:
        return ((self._a * hashes[None, :] + self._b) % _P).min(axis=1)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - inter
    return inter / union if union else 0.0


def near_duplicates(texts: list[str], threshold: float = 0.75) -> dict[int, set[int]]:
    """For each text index, the other indexes whose char-shingle Jaccard is >= threshold."""
    sets = [shingles(t) for t in texts]
    hasher = MinHasher()
    sigs = [hasher.signature(s) if s.size else None for s in sets]
    neighbours: dict[int, set[int]] = defaultdict(set)
    for band in range(BANDS):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        for i, sig in enumerate(sigs):
            if sig is not None:
                buckets[sig[band * ROWS : (band + 1) * ROWS].tobytes()].append(i)
        for members in buckets.values():
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    i, j = members[x], members[y]
                    if j in neighbours[i]:
                        continue
                    if jaccard(sets[i], sets[j]) >= threshold:
                        neighbours[i].add(j)
                        neighbours[j].add(i)
    return neighbours
