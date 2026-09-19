"""The agent's action log, sealed until the round ends.

During a live round the defender must not read what the agent did or why, or they
could hard-fix exactly its moves instead of doing what a real defender does:
infer from their own telemetry. Reading raises until `unseal()`, which the round
runner calls when the round is over. The file itself is plain JSONL for replay and
the episode reveal; do not open it mid-round.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SealedError(RuntimeError):
    """The log cannot be read until the round has ended."""


class SealedLog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._records: list[dict[str, Any]] = []
        self._sealed = True
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def write(self, kind: str, **payload: Any) -> None:
        record = {"seq": len(self._records), "kind": kind, **payload}
        self._records.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True, default=str, ensure_ascii=True) + "\n")

    @property
    def sealed(self) -> bool:
        return self._sealed

    def unseal(self) -> None:
        self._sealed = False

    def read(self) -> list[dict[str, Any]]:
        if self._sealed:
            raise SealedError("the agent log is sealed until the round ends")
        return list(self._records)
