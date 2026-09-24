"""Record the M6 escalation round for the video: four scripted attempts of increasing stealth,
built to a finale, no API key required.

    .venv/Scripts/python scripts/escalation_round.py            # 800 humans x 10 days, ~5 min
    .venv/Scripts/python scripts/escalation_round.py --small     # 250 humans x 6 days, faster

Same worlds as round-0 (ref seed 101, eval seed 202) by default, so this round's numbers are
directly comparable to `results/round-0.md`. The scripted farm runs reference_plans A, E, G,
then H (agent/reference_plans.py): naive, human text+timing with still-lockstep reactions,
15%/6h reactions, then 5%/12h reactions. The defender patches `target_overlap=true` live at the
start of attempt 3 (G), when reactions first spread out enough to start slipping the co-action
window (CLAUDE.md, "Known limits" (1)) — the M6 feature this round exists to show off, including
its real trade-off: it does not fix everything, and turning it on costs a little Bonferroni
budget from the other features.

While it runs: scoreboard at http://127.0.0.1:<port>/ (OBS Browser Source, 1920x1080), captioned
with each attempt's plan name so the arc reads without narration. Artifacts land in rounds/<id>/.
Replay without the model:
    .venv/Scripts/python scripts/replay_round.py rounds/<id>
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm import ScriptedLLM, scripted  # noqa: E402
from agent.reference_plans import reference_plans  # noqa: E402
from agent.sandbox import Budget, Sandbox  # noqa: E402
from agent.tools import ToolBox  # noqa: E402
from faceoff.round import FaceOff, WorldSpec  # noqa: E402
from scoreboard.server import serve  # noqa: E402

PLAN_ARC = ["A naive: fixed clock, canned text, lockstep likes",
            "E human text + timing, reactions still in lockstep",
            "G F with reactions at 15%, 6 h median",
            "H F with reactions at 5%, 12 h median"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--small", action="store_true", help="250 humans x 6 days (fast) instead of 800 x 10")
    ap.add_argument("--port", type=int, default=8430)
    ap.add_argument("--hold", type=float, default=0, help="keep the scoreboard up this many seconds after the round")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("rounds"))
    args = ap.parse_args()

    spec = WorldSpec(301, 302, 250, 6) if args.small else WorldSpec(101, 202, 800, 10)
    print(f"building worlds ({spec.n_humans} humans x {spec.days} days)...", flush=True)
    ref, world = spec.build()

    holder: dict = {}
    probe = ToolBox(Sandbox(world, Budget(), args.seed, 1), lambda s: {})
    targets = [t["account"] for t in probe.call("list_targets", {"limit": 3})["targets"]]
    plans = reference_plans(targets)

    def attempt(plan, first=None):
        return [first or scripted(("platform_overview", {}), ("list_targets", {"limit": 3})),
                scripted(("create_accounts", {"count": 40}), ("apply_profile", plan)),
                scripted(("launch_farm", {}))]

    def patch_target_overlap(messages):
        holder["fo"].console.patch("catch slow lockstep", target_overlap=True)
        return scripted(("platform_overview", {}))

    script = (attempt(plans[PLAN_ARC[0]])
              + attempt(plans[PLAN_ARC[1]])
              + attempt(plans[PLAN_ARC[2]], first=patch_target_overlap)
              + attempt(plans[PLAN_ARC[3]])
              + [scripted(("finish", {"reason": "escalation arc complete"}))])
    llm = ScriptedLLM(script)
    budget = Budget(max_attempts=len(PLAN_ARC))
    attempt_labels = {i: name for i, name in enumerate(PLAN_ARC, start=1)}

    round_id = time.strftime("%Y%m%d-%H%M%S") + "-escalation"
    out = args.out / round_id
    fo = FaceOff(llm, spec, out, budget, kept_fraction=0.10, max_human_fpr=0.10, seed=args.seed,
                 round_id=round_id, worlds=(ref, world), attempt_labels=attempt_labels)
    holder["fo"] = fo
    fo.start()
    server, _ = serve(out, args.port)
    print(f"scoreboard: http://127.0.0.1:{server.server_port}/   (round {round_id})", flush=True)

    result = fo.join()

    print(f"\nround over: {result.stop_reason}; {result.calls} model calls, {result.tokens_used:,} tokens")
    for name, o in zip(PLAN_ARC, result.outcomes):
        c = o.confusion
        prec = "n/a" if c.precision is None else f"{c.precision:.1%}"
        print(f"  attempt {o.attempt} [{name}] (detector {o.detector_version}): "
              f"bots {len(o.suspended)}/{len(o.commit.farm.accounts)} caught, humans flagged {c.fp}, "
              f"precision {prec}, recall {c.recall:.1%}, kept {o.engagement_kept:,}/{o.engagement_total:,}")
    print(f"artifacts: {out}")
    if args.hold:
        print(f"holding the scoreboard for {args.hold:.0f}s...", flush=True)
        threading.Event().wait(args.hold)
    server.shutdown()


if __name__ == "__main__":
    main()
