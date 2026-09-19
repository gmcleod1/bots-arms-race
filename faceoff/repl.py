"""The defender's command line during a round.

    status                          the live detector version, settings, and what is queued
    flags                           the latest attempt's flagged accounts (anonymised) and near misses
    versions                        every detector version so far
    patch "label" key=value ...     change the detector (JSON values), e.g.
                                      patch "accept more false positives" fpr_budget=0.10
                                      patch "drop the noisy typo feature" disabled_features='["x_shared_oov"]'
    go                              fair-play mode: release the gate so the next attempt can start
    help | quit

Reads and writes only through the DefenderConsole, so it cannot see labels, outcomes or the agent.
"""
from __future__ import annotations

import json
import shlex
from typing import Callable

from detector.config import ConfigError
from faceoff.console import DefenderConsole

HELP = __doc__


def _value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def handle(console: DefenderConsole, line: str) -> str | None:
    """Run one command; return its output. None means quit."""
    try:
        parts = shlex.split(line)
    except ValueError as exc:
        return f"could not parse that: {exc}"
    if not parts:
        return ""
    cmd, args = parts[0].lower(), parts[1:]
    if cmd in ("quit", "exit"):
        return None
    if cmd == "help":
        return HELP
    if cmd == "status":
        s = console.status()
        changed = {k: v for k, v in s["settings"].items() if v not in ((), [], None) and k != "fpr_budget"}
        return (f"detector {s['version']} ({s['label']}) | budget {s['fpr_budget']:.0%} | "
                f"per-feature alpha {s['alpha']:.4f} | features {len(s['features'])}\n"
                f"settings {json.dumps(s['settings'])}\n"
                f"queued patches {s['pending_patches'] or 'none'} | frozen {s['frozen']} | "
                f"waiting for you: {s['waiting_for_you']} | attempts scored {s['attempts_scored']}")
    if cmd == "flags":
        t = console.latest()
        if t is None:
            return "no attempt has been scored yet"
        rows = [f"  {r.account}  score {r.score:.2f}  fired {','.join(r.fired) or '-'}" for r in t.flagged[:15]]
        near = [f"  {r.account}  score {r.score:.2f}  top {r.top_feature}" for r in t.near_misses]
        more = f"\n  ... and {len(t.flagged) - 15} more" if len(t.flagged) > 15 else ""
        return (f"attempt {t.attempt}, detector {t.version}: {len(t.flagged)} flagged of {t.scored_accounts} scored\n"
                + "\n".join(rows) + more + "\nnear misses (highest-scoring, not flagged):\n" + "\n".join(near))
    if cmd == "versions":
        return "\n".join(f"  {v.id}  {v.label}  {json.dumps(v.diff)}" for v in console.versions())
    if cmd == "go":
        console.ready()
        return "released: the next attempt can start"
    if cmd == "patch":
        if not args:
            return 'usage: patch "label" key=value ...'
        label, changes = args[0], {}
        for token in args[1:]:
            if "=" not in token:
                return f"expected key=value, got {token!r}"
            k, v = token.split("=", 1)
            changes[k] = _value(v)
        try:
            result = console.patch(label, **changes)
        except ConfigError as exc:
            return f"rejected: {exc}"
        if result.queued:
            return f"queued at position {result.queue_position}; it applies when the current attempt is scored"
        v = result.version
        return f"applied as {v.id} in {v.build_seconds:.1f}s | alpha {v.alpha:.4f} | changed {json.dumps(v.diff)}"
    return f"unknown command {cmd!r}; type help"


def run_repl(console: DefenderConsole, done: Callable[[], bool], read: Callable[[], str] = lambda: input("defender> "),
             write: Callable[[str], None] = print) -> None:
    """Loop until quit, EOF, or the round is over (`done()`)."""
    while not done():
        try:
            line = read()
        except EOFError:
            return
        out = handle(console, line)
        if out is None:
            return
        if out:
            write(out)
