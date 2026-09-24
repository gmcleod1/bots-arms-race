# How to run and play Bots Arms Race

Everything below works from `bots-arms-race/` with the existing virtual environment. Nothing
in this file needs a paid API key unless the "Going live" section at the end says otherwise.

```powershell
cd C:\Users\netge\Projects\bots-arms-race
```

## 1. Run the recorded match (no setup, no API key)

This is the flagship 4-round escalation arc: a bot farm gets progressively stealthier
(naive → human-style → 15% stealth → 5% stealth) while the detector's defender live-patches
in response. Same worlds every time (fixed seeds), so it's reproducible.

```powershell
.\.venv\Scripts\python scripts\escalation_round.py
```

- Takes about 5 minutes (800 simulated humans over 10 days).
- Add `--small` for a ~1-minute run against a smaller 250-human world, if you just want to see
  it work end to end.
- While it runs, the console prints a line like:
  ```
  scoreboard: http://127.0.0.1:8430/   (round 20260924-122400-escalation)
  ```
  Open that URL in a browser. It updates once a second: live score, three charts, the
  detector's patch history, and a caption under the round title telling you which stealth
  level is currently running.
- When it finishes, the console prints a per-attempt summary (bots caught, precision/recall,
  engagement kept) and the artifact folder, e.g. `rounds\20260924-122400-escalation\`.

## 2. Watch it as an overlay (OBS or any browser)

Add a **Browser Source** pointed at `http://127.0.0.1:8430/`, sized `1920x1080`. Useful query
params, combine as needed:

| Param | Effect |
|---|---|
| `?compact=1` | Score, tiles, and the plan caption only — hides the charts/table/patch log. This is the one to use for a clean overlay. |
| `&bg=transparent` | Transparent page background, to key over other footage. |
| `&theme=dark` or `&theme=light` | Force a theme instead of following the OS. |

Example for filming: `http://127.0.0.1:8430/?compact=1&bg=transparent&theme=dark`

The scoreboard is read-only and only serves `dashboard.html`, `state.json`, and
`events.jsonl` — nothing that reveals which accounts are bots ahead of time.

## 3. Replay a finished round (no model, no re-simulation needed)

Every round writes everything needed to reproduce it exactly. To check a round is legitimate
or to re-watch its numbers without re-running anything random:

```powershell
.\.venv\Scripts\python scripts\replay_round.py rounds\<round-id>
```

Prints `REPRODUCED: N attempt(s) replayed without the model and matched the recording.` on
success, or a list of exactly what didn't match.

## 4. Actually play it yourself (interactive defender console)

`escalation_round.py` is scripted end-to-end — good for recording, but you don't get to make
any decisions. For a hands-on session where *you* patch the detector live while a farm attacks:

```powershell
.\.venv\Scripts\python scripts\faceoff.py --dry-run --small
```

This drops you into a `defender>` prompt once the round starts. Commands:

| Command | Does |
|---|---|
| `status` | Current detector version, config, and round phase |
| `flags` | Recent flagged/near-miss accounts (anonymised — no labels, by design) |
| `versions` | Every patch applied so far |
| `patch "label" key=value ...` | Apply a new detector setting live, e.g. `patch "catch slow lockstep" target_overlap=true` |
| `help` | Full command list |

The scripted `--dry-run` agent plays a fixed sequence of tactics against you, so you can
practice patching without spending anything. Drop `--dry-run` for a live round where an actual
Claude Opus 5 agent picks its own evasion tactics — see below, this costs money.

For a turn-based version where the round pauses after every attempt until you type `go`, add
`--pause-edits --gate`.

## Going live: a real autonomous AI opponent (costs money)

Everything above uses a scripted stand-in for the attacker, so it's free and repeatable.
Dropping `--dry-run` on `scripts\faceoff.py` (or `scripts\run_agent.py`) switches to a real
Claude Opus 5 agent that picks its own tactics — no canned plan, genuinely unpredictable.

```powershell
.\.venv\Scripts\python scripts\faceoff.py --attempts 5 --max-tokens 400000
```

Requires `ANTHROPIC_API_KEY` set (or `ant auth login`). It can spend up to `--max-tokens`
(400k by default) across up to `--attempts` rounds — lower both if you want to cap cost. This
path is the natural paid follow-up video once the free scripted match has an audience, not a
prerequisite for shipping anything.

## Troubleshooting

- **"port already in use"**: pass `--port 8431` (or any free port) to `escalation_round.py` /
  `faceoff.py`.
- **Scoreboard shows "Waiting for a round to start…" forever**: the round script has probably
  already finished and exited (the server only lives as long as the script does, unless you
  ran it in the background). Re-run the script, or serve a finished round's folder directly —
  see `scoreboard/server.py`'s `serve()` for how.
- **Tests**: `.\.venv\Scripts\python -m pytest -q` — full suite, about 6 minutes, no API key.
