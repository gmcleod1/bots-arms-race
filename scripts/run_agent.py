"""Run a red-agent round against the detector.

    .venv/Scripts/python scripts/run_agent.py --dry-run        # scripted model, no API key
    .venv/Scripts/python scripts/run_agent.py                  # live: Claude Opus 5, needs credentials

Live mode needs ANTHROPIC_API_KEY or a logged-in profile (`ant auth login`), and spends tokens
up to --max-tokens. Dry-run drives the same runner, tools, referee, sealed log and traces with
a fixed two-attempt script (the naive plan, then a stealthy one), so it verifies everything
except the live model's behavior.

Each round writes rounds/<id>/: agent-log.jsonl (sealed until the round ends), attempt-N.jsonl
traces (replay without the model), and summary.md (the defender's view). The agent never sees
anything in the summary except its own suspended accounts and surviving engagement.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.agent import RedAgent  # noqa: E402
from agent.llm import AnthropicLLM, ScriptedLLM, scripted  # noqa: E402
from agent.log import SealedLog  # noqa: E402
from agent.reference_plans import reference_plans  # noqa: E402
from agent.referee import Referee  # noqa: E402
from agent.sandbox import Budget, Sandbox  # noqa: E402
from agent.tools import ToolBox  # noqa: E402
from detector.detector import Detector  # noqa: E402
from simulator.world import generate_world  # noqa: E402

REF_SEED, EVAL_SEED, N_HUMANS, DAYS = 101, 202, 800, 10


def dry_run_script(targets: list[str]):
    plans = reference_plans(targets)
    naive = next(v for k, v in plans.items() if k.startswith("A"))
    stealth = next(v for k, v in plans.items() if k.startswith("H"))

    def attempt(plan):
        return [scripted(("platform_overview", {}), ("list_targets", {"limit": 3})),
                scripted(("create_accounts", {"count": 40}), ("apply_profile", plan)),
                scripted(("launch_farm", {}))]

    return attempt(naive) + attempt(stealth) + [scripted(("finish", {"reason": "dry run complete"}))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="use a scripted model instead of the API")
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--out", type=Path, default=Path("rounds"))
    args = ap.parse_args()

    print("building worlds and calibrating the detector (about 45 s)...", flush=True)
    ref = generate_world(REF_SEED, N_HUMANS, DAYS)
    world = generate_world(EVAL_SEED, N_HUMANS, DAYS)
    detector = Detector(fpr_budget=0.05)
    detector.fit(ref.events)

    budget = Budget(max_attempts=args.attempts, max_tokens=args.max_tokens)
    if args.dry_run:
        probe = ToolBox(Sandbox(world, budget, args.seed, 1), lambda s: {})
        targets = [t["account"] for t in probe.call("list_targets", {"limit": 3})["targets"]]
        llm = ScriptedLLM(dry_run_script(targets))
        budget = Budget(max_attempts=2, max_tokens=args.max_tokens)
    else:
        llm = AnthropicLLM(effort=args.effort)
        print("live mode: claude-opus-5, adaptive thinking, server-side refusal fallbacks enabled", flush=True)

    round_id = time.strftime("%Y%m%d-%H%M%S") + ("-dry" if args.dry_run else "")
    out = args.out / round_id
    log = SealedLog(out / "agent-log.jsonl")
    agent = RedAgent(llm, world, Referee(detector), budget, seed=args.seed, log=log, trace_dir=out)
    result = agent.run()  # the log is sealed while this runs and unsealed when it returns

    lines = [f"# Round {round_id}", "",
             f"Stopped: {result.stop_reason}. Model calls: {result.calls}. Tokens: {result.tokens_used:,}. "
             f"Elapsed: {result.elapsed_s:.0f} s.", "",
             "Defender's view (the agent is never shown this). Precision and recall are reported separately.", "",
             "| Attempt | Bots suspended | Recall | Precision | Humans flagged | Engagement delivered | Kept after suspensions |",
             "|---|---|---|---|---|---|---|"]
    for o in result.outcomes:
        c, f = o.confusion, o.feedback
        prec = "n/a" if c.precision is None else f"{c.precision:.1%}"
        lines.append(f"| {o.attempt} | {f['accounts_suspended']}/{f['accounts_created']} | "
                     f"{c.recall:.1%} | {prec} | {c.fp} | {o.engagement_total:,} | {o.engagement_kept:,} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nartifacts: {out}  (agent log now unsealed: {not log.sealed})")


if __name__ == "__main__":
    main()
