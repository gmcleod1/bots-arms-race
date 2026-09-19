"""Seeded randomness with one independent stream per account.

Streams are keyed by a stable hash of the account id, not by creation order.
That way adding, removing or reordering accounts never shifts another
account's draws, and a recorded round replays identically.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _key(account_id: str) -> int:
    digest = hashlib.blake2b(account_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class SeedTree:
    def __init__(self, root_seed: int) -> None:
        self.root_seed = root_seed

    def for_account(self, account_id: str) -> np.random.Generator:
        """A fresh generator for this account, identical on every call."""
        seq = np.random.SeedSequence(entropy=self.root_seed, spawn_key=(_key(account_id),))
        return np.random.default_rng(seq)
