"""CLI: run campaigns, inspect the result DB, launch Kaggle batches.

    python -m agent_scientist.cli run --campaign campaigns/variance_pilot.json --backend local
    python -m agent_scientist.cli run --campaign campaigns/variance_pilot.json --backend kaggle
    python -m agent_scientist.cli status --workdir runs/variance_pilot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _make_provider(name: str):
    from .providers import OllamaProvider, RuleBasedProvider

    if name == "ollama":
        return OllamaProvider()
    return RuleBasedProvider()


def _make_backend(name: str, workdir: Path):
    from .backends import KaggleBackend, LocalBackend

    if name == "kaggle":
        return KaggleBackend()
    return LocalBackend()


def cmd_run(args) -> int:
    from .campaign import CampaignRunner

    spec = json.loads(Path(args.campaign).read_text())
    workdir = Path(args.workdir or f"runs/{Path(args.campaign).stem}")
    runner = CampaignRunner(workdir, _make_provider(args.provider),
                            _make_backend(args.backend, workdir))
    summary = runner.run(spec)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_status(args) -> int:
    from .analyst import ResultDB, compute_stats, verdict_text

    db = ResultDB(Path(args.workdir) / "results.jsonl")
    print(f"runs: {len(db)} ok={len(db.ok_results())} "
          f"failed={len(db.failed())}")
    if db.ok_results():
        stats = compute_stats(db.all())
        print(verdict_text("status check", stats))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--campaign", required=True)
    r.add_argument("--backend", default="local", choices=["local", "kaggle"])
    r.add_argument("--provider", default="rule-based",
                   choices=["rule-based", "ollama"])
    r.add_argument("--workdir", default=None)
    s = sub.add_parser("status")
    s.add_argument("--workdir", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    return cmd_status(args)


if __name__ == "__main__":
    sys.exit(main())
