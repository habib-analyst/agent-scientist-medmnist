"""Experiment journal: auto-generated, human-readable campaign log.

Every round appends: hypotheses proposed, runs launched (with fingerprints),
results measured, analysis verdict, critic decision. The journal is the
audit trail that lets a human verify every claim traces to a logged run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class Journal:
    def __init__(self, path: Path, question: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(
                f"# Experiment Journal\n\n"
                f"**Research question:** {question}\n\n"
                f"**Opened:** {datetime.now(timezone.utc).isoformat()}\n\n"
                f"Every claim below traces to a logged run (fingerprint + config).\n\n---\n\n"
            )

    def _append(self, text: str) -> None:
        with self.path.open("a") as f:
            f.write(text + "\n")

    def round_start(self, round_no: int, rationale: str) -> None:
        self._append(
            f"## Round {round_no} — {datetime.now(timezone.utc).isoformat()}\n\n"
            f"**Hypothesizer:** {rationale}\n")

    def log_proposals(self, experiments: List[Dict[str, Any]]) -> None:
        for e in experiments:
            self._append(
                f"- proposed `{e['run_id']}` "
                f"(dataset={e['dataset']}, arch={e['arch']}, "
                f"augment={e['augment']}, seed={e['seed']}, "
                f"epochs={e['epochs']})")

    def log_results(self, results: List[Dict[str, Any]]) -> None:
        for r in results:
            if r.get("status") == "ok":
                m = r.get("metrics", {})
                acc = m.get("test_accuracy", float("nan"))
                self._append(
                    f"- `{r['run_id']}` OK: test_accuracy={acc:.4f} "
                    f"fp={r['fingerprint']} backend={r.get('backend')} "
                    f"duration={r.get('duration_s', 0):.0f}s")
            else:
                self._append(
                    f"- `{r['run_id']}` FAILED: {r.get('error', '?')[:200]}")

    def log_analysis(self, verdict: str) -> None:
        self._append(f"\n**Analyst verdict:**\n```\n{verdict}\n```\n")

    def log_critic(self, decision: str, reason: str) -> None:
        self._append(f"**Critic:** {decision.upper()} — {reason}\n\n---\n")

    def close(self, conclusion: str) -> None:
        self._append(
            f"\n# Campaign conclusion\n\n{conclusion}\n\n"
            f"**Closed:** {datetime.now(timezone.utc).isoformat()}\n")
