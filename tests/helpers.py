"""Shared test helpers."""
from functools import lru_cache

from simulator.world import generate_world


@lru_cache(maxsize=None)
def cached_world(seed: int, n_humans: int, days: int):
    """Generate once per (seed, size) and share. Read-only: tests must not mutate it.

    Determinism tests must call generate_world directly, never this, or comparing
    two "runs" would compare an object with itself.
    """
    return generate_world(seed=seed, n_humans=n_humans, days=days)
