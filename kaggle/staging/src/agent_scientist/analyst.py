"""Analyst: result store + statistics.

ResultDB: append-only JSONL store of RunResults keyed by fingerprint
(dedupe: re-submitting the same config returns the stored result).

stats: per-cell mean + bootstrap 95% CIs, effect sizes, and a variance
decomposition that answers "what drives the variance — factor A, factor B,
or seed noise?" via a simple ANOVA-style sum-of-squares partition.
All statistics are computed from measured numbers only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


class ResultDB:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    self._records[rec["fingerprint"]] = rec

    def add(self, result: Dict[str, Any]) -> None:
        fp = result["fingerprint"]
        if fp in self._records:
            return
        self._records[fp] = result
        with self.path.open("a") as f:
            f.write(json.dumps(result, sort_keys=True) + "\n")

    def all(self) -> List[Dict[str, Any]]:
        return list(self._records.values())

    def ok_results(self) -> List[Dict[str, Any]]:
        return [r for r in self._records.values() if r.get("status") == "ok"]

    def failed(self) -> List[Dict[str, Any]]:
        return [r for r in self._records.values() if r.get("status") == "failed"]

    def __len__(self) -> int:
        return len(self._records)


def _bootstrap_ci(values: List[float], n_boot: int = 2000,
                  seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean. Pure Python (no scipy)."""
    import random

    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (values[0], values[0])
    means = []
    for _ in range(n_boot):
        s = sum(values[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return (lo, hi)


def cell_key(config: Dict[str, Any], group_by: List[str]) -> str:
    return "|".join(f"{k}={config.get(k)}" for k in group_by)


def compute_stats(results: List[Dict[str, Any]], metric: str = "test_accuracy",
                  group_by: List[str] | None = None) -> Dict[str, Any]:
    """Per-cell means + bootstrap CIs, effect sizes, variance decomposition."""
    group_by = group_by or ["dataset", "augment"]
    cells: Dict[str, List[float]] = {}
    for r in results:
        if r.get("status") != "ok":
            continue
        v = r.get("metrics", {}).get(metric)
        if v is None or not math.isfinite(v):
            continue
        cells.setdefault(cell_key(r["config"], group_by), []).append(float(v))

    per_cell = {}
    for cell, vals in sorted(cells.items()):
        lo, hi = _bootstrap_ci(vals)
        per_cell[cell] = {
            "mean": sum(vals) / len(vals),
            "ci95": [lo, hi],
            "ci_width": hi - lo,
            "n": len(vals),
        }

    # Effect sizes: for each grouping factor (except seed), max mean-gap
    # between its levels, holding other factors fixed.
    effects: Dict[str, float] = {}
    factor_names = [g for g in group_by if g != "seed"]
    for f in factor_names:
        others = [g for g in group_by if g != f]
        groups: Dict[str, Dict[str, List[float]]] = {}
        for cell, vals in cells.items():
            parts = dict(p.split("=", 1) for p in cell.split("|"))
            rest = "|".join(f"{k}={parts[k]}" for k in others)
            groups.setdefault(rest, {}).setdefault(parts[f], []).extend(vals)
        gaps = []
        for rest, lv in groups.items():
            if len(lv) >= 2:
                means = [sum(v) / len(v) for v in lv.values()]
                gaps.append(max(means) - min(means))
        if gaps:
            effects[f"augment_gap:{f}" if False else f] = sum(gaps) / len(gaps)

    # Variance decomposition (ANOVA-style): partition total sum of squares
    # into between-cell (factor-driven) and within-cell (seed noise).
    all_vals = [v for vals in cells.values() for v in vals]
    grand = sum(all_vals) / len(all_vals) if all_vals else float("nan")
    ss_between = sum(len(vals) * (sum(vals) / len(vals) - grand) ** 2
                     for vals in cells.values())
    ss_within = sum(sum((v - sum(vals) / len(vals)) ** 2 for v in vals)
                    for vals in cells.values())
    ss_total = ss_between + ss_within
    variance_decomposition = {
        "ss_factor_driven": ss_between,
        "ss_seed_noise": ss_within,
        "frac_factor_driven": ss_between / ss_total if ss_total > 0 else 0.0,
        "frac_seed_noise": ss_within / ss_total if ss_total > 0 else 0.0,
    }

    ci_width_max = max((c["ci_width"] for c in per_cell.values()), default=0.0)
    return {
        "metric": metric,
        "per_cell": per_cell,
        "effects": effects,
        "variance_decomposition": variance_decomposition,
        "ci_width_max": ci_width_max,
        "n_ok": len([r for r in results if r.get("status") == "ok"]),
        "n_failed": len([r for r in results if r.get("status") == "failed"]),
    }


def verdict_text(question: str, stats: Dict[str, Any]) -> str:
    """Human-readable verdict from computed stats (no new inference)."""
    vd = stats["variance_decomposition"]
    lines = [f"Research question: {question}", ""]
    for cell, s in sorted(stats["per_cell"].items()):
        lines.append(
            f"  {cell}: mean={s['mean']:.4f} "
            f"95%CI=[{s['ci95'][0]:.4f},{s['ci95'][1]:.4f}] (n={s['n']})")
    lines.append("")
    for eff, v in sorted(stats["effects"].items()):
        lines.append(f"  mean |Δ| across {eff} levels: {v:.4f}")
    lines.append("")
    lines.append(
        f"  Variance decomposition: {vd['frac_factor_driven']:.1%} factor-driven "
        f"vs {vd['frac_seed_noise']:.1%} seed noise.")
    if vd["frac_seed_noise"] > vd["frac_factor_driven"]:
        lines.append("  => Seed noise DOMINATES the measured variance.")
    else:
        lines.append("  => Experimental factors DOMINATE the measured variance.")
    return "\n".join(lines)
