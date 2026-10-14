#!/usr/bin/env python
"""Compare heuristic or custom schedulers on the simulation.

Usage: python baselines.py --config environment --schedulers MinT MCT Random
Custom policies: --schedulers schedulers/my_scheduler.py
Agent-generated policy: --schedulers agent
"""
import argparse
import json
import sys
from pathlib import Path

from agent.tools import ToolExecutor

SCHEDULERS = ["LeastLoaded", "MinT", "MCT", "DIANA"]


def label(name: str) -> str:
    if not name.endswith(".py"):
        return name
    return Path(name).stem


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="environment",
                    choices=["small_scale", "middle_scale", "large_scale",
                             "environment"])
    ap.add_argument("--schedulers", nargs="+", default=SCHEDULERS,
                    help=f"default: {' '.join(SCHEDULERS)}.  'agent' scores "
                         "schedulers/agent_scheduler.py")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="wall-clock seconds allowed per scheduler before it's "
                         "failed (default 180). Bigger configs (large_scale) or "
                         "more complex schedulers may need more.")
    args = ap.parse_args()

    e = ToolExecutor()
    rows = []
    for s in args.schedulers:
        out = e._run_experiment(
            s,
            config=args.config,
            wall_timeout_s=args.timeout,
        )
        if out.startswith("ERROR"):
            print(f"{s}: FAILED — {out[:200]}", file=sys.stderr)
            continue
        rows.append((label(s), json.loads(out)))

    if not rows:
        sys.exit(1)

    # Descending by the score, so the winner is the first line.
    rows.sort(key=lambda kv: -(kv[1].get("throughput_charged_gflops_s") or 0))

    w = max(10, max(len(s) for s, _ in rows))
    print(f"\nconfig={args.config}")
    print(f"{'sched':<{w}}  {'tputC GF/s':>10}  {'done/total':>12}")
    print("-" * (w + 28))
    for s, r in rows:
        print(
            f"{s:<{w}}  "
            f"{r['throughput_charged_gflops_s']:>10.3f}  "
            f"{r['jobs_done']:>5}/{r['jobs_total']:<6}"
        )


if __name__ == "__main__":
    main()
