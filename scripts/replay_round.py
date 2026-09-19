"""Replay a recorded face-off without the model and check it reproduces.

    .venv/Scripts/python scripts/replay_round.py rounds/<id>

Rebuilds the worlds and every detector version from the round's own files, re-executes each
attempt's trace, and compares feedback, flagged accounts, confusion counts, engagement and the
audience state against what was recorded. Exit code 0 if identical, 1 if anything differs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faceoff.replay import replay_round  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    report = replay_round(Path(sys.argv[1]))
    if report.ok:
        print(f"REPRODUCED: {report.attempts_checked} attempt(s) replayed without the model and matched the recording.")
        return
    print(f"MISMATCH after replaying {report.attempts_checked} attempt(s):")
    for m in report.mismatches:
        print("  -", m)
    sys.exit(1)


if __name__ == "__main__":
    main()
