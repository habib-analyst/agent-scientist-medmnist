"""LLM providers for the agents' reasoning steps.

The default is RuleBasedProvider: fully deterministic, auditable, $0 cost.
It implements the exact decision logic each agent needs (hypothesis
generation from a grammar, statistical verdicts, critic decisions) as
explicit rules rather than free-form text — every "decision" is reproducible
from the same inputs.

OllamaProvider optionally routes the same prompts to a local Ollama model
for free-form reasoning; it degrades gracefully to rule-based logic when
Ollama is unreachable.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List, Optional


class LLMProvider:
    """Interface: given a named task and a JSON-serializable context, return a
    JSON-serializable decision."""

    name = "base"

    def decide(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class RuleBasedProvider(LLMProvider):
    """Deterministic decision logic. Each task has an explicit rule; no
    sampling, no hidden state."""

    name = "rule-based"

    def decide(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        handler = getattr(self, f"_task_{task}", None)
        if handler is None:
            raise ValueError(f"unknown task: {task}")
        return handler(context)

    # -- Hypothesizer -----------------------------------------------------
    def _task_propose_experiments(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Propose the next batch of experiments from the hypothesis grammar.

        Context: question, factors {name: [levels]}, asked (already-run
        factor combos as list of dicts), budget {runs_remaining}, round.
        Returns: {"experiments": [factor-dict, ...], "rationale": str}.

        Optional budget key ``per_round`` caps how many experiments one
        round may propose, so a campaign iterates over several rounds
        (analyze -> critique -> propose more) instead of consuming the
        whole budget in round 0. Absent the key, behaviour is unchanged:
        propose every fresh combo the remaining budget allows.
        """
        factors: Dict[str, List[Any]] = ctx["factors"]
        asked = {json.dumps(e, sort_keys=True) for e in ctx.get("asked", [])}
        remaining: int = ctx.get("budget", {}).get("runs_remaining", 0)

        # Full factorial over the grammar, minus already-asked combos.
        names = list(factors.keys())
        combos: List[Dict[str, Any]] = [{}]
        for n in names:
            combos = [{**c, n: v} for c in combos for v in factors[n]]
        fresh = [c for c in combos if json.dumps(c, sort_keys=True) not in asked]
        cap = max(0, remaining)
        per_round = ctx.get("budget", {}).get("per_round")
        if per_round:
            cap = min(cap, int(per_round))
        batch = fresh[:cap]
        return {
            "experiments": batch,
            "rationale": (
                f"Round {ctx.get('round', 0)}: full-factorial proposal over "
                f"{len(names)} factors ({len(combos)} combos); {len(fresh)} "
                f"unasked, proposing {len(batch)} within budget"
                + (f" (per-round cap {int(per_round)})." if per_round else ".")
            ),
        }

    # -- Critic -----------------------------------------------------------
    def _task_critique(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Decide whether to continue, refine, or stop.

        Context: analysis summary {cells, ci_width_max, failed_runs},
        budget {runs_remaining, max_retries}, thresholds {ci_width_target,
        min_runs_per_cell}.
        Returns: {"decision": "continue"|"refine"|"stop", "reason": str,
                  "followups": [factor-dict, ...]}.
        """
        analysis = ctx.get("analysis", {})
        budget = ctx.get("budget", {})
        thr = ctx.get("thresholds", {})
        failed = analysis.get("failed_runs", 0)
        remaining = budget.get("runs_remaining", 0)
        ci_max = analysis.get("ci_width_max", float("inf"))
        ci_target = thr.get("ci_width_target", 0.05)

        if failed > 0 and budget.get("retries_remaining", 0) > 0:
            return {
                "decision": "refine",
                "reason": f"{failed} run(s) failed; retrying within retry budget.",
                "followups": analysis.get("failed_configs", []),
            }
        if remaining <= 0:
            return {
                "decision": "stop",
                "reason": "Compute budget exhausted; reporting on completed runs.",
                "followups": [],
            }
        if ci_max > ci_target and remaining > 0:
            return {
                "decision": "continue",
                "reason": (
                    f"Widest 95% CI is {ci_max:.3f} > target {ci_target:.3f}; "
                    "more runs will tighten the variance estimates."
                ),
                "followups": [],
            }
        return {
            "decision": "stop",
            "reason": (
                f"All 95% CIs within target width ({ci_max:.3f} <= "
                f"{ci_target:.3f}); research question answered."
            ),
            "followups": [],
        }

    # -- Analyst ----------------------------------------------------------
    def _task_summarize(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Turn computed statistics into a human-readable verdict.

        Context: stats {per_cell: {mean, ci95, n}, effects: {...}}.
        Pure formatting — no new inference beyond the numbers.
        """
        stats = ctx.get("stats", {})
        lines = []
        for cell, s in sorted(stats.get("per_cell", {}).items()):
            lines.append(
                f"{cell}: mean={s['mean']:.4f} 95%CI=[{s['ci95'][0]:.4f}, "
                f"{s['ci95'][1]:.4f}] (n={s['n']})"
            )
        for eff, v in sorted(stats.get("effects", {}).items()):
            lines.append(f"effect[{eff}] = {v:+.4f}")
        return {"verdict": "\n".join(lines)}


class OllamaProvider(LLMProvider):
    """Routes decisions to a local Ollama model; falls back to rule-based
    logic on any failure (never blocks the loop)."""

    name = "ollama"

    def __init__(self, model: str = "llama3.1:8b",
                 base_url: str = "http://localhost:11434",
                 fallback: Optional[LLMProvider] = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.fallback = fallback or RuleBasedProvider()

    def decide(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "You are a sub-agent in an autonomous ML experiment loop. "
            f"Task: {task}. Context (JSON):\n{json.dumps(context)}\n"
            "Reply with ONLY a JSON object matching the expected schema."
        )
        try:
            req = urllib.request.Request(
                self.base_url + "/api/generate",
                data=json.dumps(
                    {"model": self.model, "prompt": prompt,
                     "format": "json", "stream": False}
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode())
            return json.loads(body["response"])
        except Exception as exc:  # graceful degradation, never block the loop
            out = self.fallback.decide(task, context)
            out["_provider_fallback"] = f"ollama unreachable: {exc}"
            return out
