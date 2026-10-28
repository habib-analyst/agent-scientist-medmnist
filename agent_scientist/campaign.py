"""CampaignRunner: wires Hypothesizer -> Executor -> Analyst -> Critic into the
state graph and runs the closed loop.

State keys:
  round, question, factors, budget, thresholds, provider, backend,
  asked (list of proposed factor dicts), pending (RunSpecs),
  results_db_path, analysis, critic_decision, done, conclusion.

The loop is fully autonomous: it stops only when the Critic says stop
(question answered / budget exhausted) or max_rounds is hit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .analyst import ResultDB, compute_stats, verdict_text
from .backends import ExperimentBackend, RunResult, RunSpec
from .graph import END, StateGraph
from .journal import Journal
from .providers import LLMProvider


class CampaignRunner:
    def __init__(self, workdir: Path, provider: LLMProvider,
                 backend: ExperimentBackend) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.backend = backend
        self.db = ResultDB(self.workdir / "results.jsonl")

    # -- graph nodes ----------------------------------------------------
    def _node_hypothesize(self, state: dict) -> dict:
        asked_cfgs = [json.loads(json.dumps(e)) for e in state["asked"]]
        decision = self.provider.decide("propose_experiments", {
            "question": state["question"],
            "factors": state["factors"],
            "asked": asked_cfgs,
            "budget": state["budget"],
            "round": state["round"],
        })
        experiments = decision["experiments"]
        specs = []
        for i, e in enumerate(experiments):
            fields = dict(
                dataset=e["dataset"], arch=e["arch"], augment=e["augment"],
                seed=int(e["seed"]),
                epochs=int(e.get("epochs", state["defaults"].get("epochs", 5))),
                batch_size=int(e.get("batch_size",
                                 state["defaults"].get("batch_size", 128))),
                lr=float(e.get("lr", state["defaults"].get("lr", 1e-3))),
            )
            # The run_id MUST be unique per distinct config: the backend
            # names artefact files by run_id and results are re-attached to
            # specs by run_id, so two configs sharing an id silently
            # overwrite each other's files and corrupt each other's
            # recorded config. The human-readable prefix does not cover
            # lr/epochs/batch_size, so append a short hash of ALL fields.
            tag = hashlib.sha256(
                json.dumps(fields, sort_keys=True).encode()).hexdigest()[:6]
            run_id = (f"r{state['round']:02d}-{fields['dataset']}-"
                      f"{fields['arch']}-{fields['augment']}-"
                      f"s{fields['seed']}-{tag}")
            specs.append(RunSpec(run_id=run_id, **fields))
        state["_journal"].round_start(state["round"], decision["rationale"])
        state["_journal"].log_proposals([s.config_dict() | {"run_id": s.run_id}
                                         for s in specs])
        return {"pending": [s.config_dict() | {"run_id": s.run_id}
                            for s in specs],
                "_specs": specs,
                "asked": asked_cfgs + experiments}

    def _node_execute(self, state: dict) -> dict:
        specs: List[RunSpec] = state["_specs"]
        results: List[RunResult] = self.backend.submit_batch(specs, self.workdir)
        spec_by_id = {s.run_id: s for s in specs}
        for r in results:
            rec = r.to_dict()
            rec["config"] = spec_by_id[r.run_id].config_dict()
            self.db.add(rec)
        state["_journal"].log_results([r.to_dict() for r in results])
        return {"results": [r.to_dict() for r in results]}

    def _node_analyze(self, state: dict) -> dict:
        stats = compute_stats(self.db.all(),
                              metric=state.get("metric", "test_accuracy"),
                              group_by=state.get("group_by"))
        verdict = verdict_text(state["question"], stats)
        summary = self.provider.decide("summarize", {"stats": stats})
        state["_journal"].log_analysis(verdict + "\n" + summary["verdict"])
        failed = self.db.failed()
        analysis = {
            "stats": stats,
            "failed_runs": len(failed),
            "failed_configs": [f["config"] for f in failed],
            "ci_width_max": stats["ci_width_max"],
        }
        return {"analysis": analysis}

    def _node_critique(self, state: dict) -> dict:
        # Consume this round's spend BEFORE the Critic decides: it must
        # judge the budget that remains after the spend, not the budget
        # at the round's start — otherwise a campaign whose budget ends
        # exactly with its last run needs a trailing empty round (zero
        # proposals, zero runs) before the Critic notices exhaustion.
        spent = len(state.get("results", []))
        budget = dict(state["budget"])
        budget["runs_remaining"] = max(0, budget.get("runs_remaining", 0) - spent)
        decision = self.provider.decide("critique", {
            "analysis": state["analysis"],
            "budget": budget,
            "thresholds": state["thresholds"],
        })
        state["_journal"].log_critic(decision["decision"], decision["reason"])
        if decision["decision"] == "refine":
            budget["retries_remaining"] = max(
                0, budget.get("retries_remaining", 0) - 1)
        done = decision["decision"] == "stop"
        return {"critic_decision": decision,
                "budget": budget,
                "done": done}

    def _router_after_critique(self, state: dict) -> str:
        # One graph pass == exactly one round. Iteration across rounds is
        # owned by CampaignRunner.run()'s manual loop (which increments
        # state["round"]); routing back to "hypothesize" here as well made
        # the two loops fight — every cycle inside one graph pass was
        # journaled under the same round number.
        return END

    def _node_finalize(self, state: dict) -> dict:
        # reached only via max_rounds; synthesize conclusion from analysis
        stats = state["analysis"]["stats"]
        conclusion = verdict_text(state["question"], stats)
        state["_journal"].close(conclusion)
        return {"conclusion": conclusion, "done": True}

    # -- public API ------------------------------------------------------
    def build_graph(self) -> StateGraph:
        g = StateGraph()
        g.add_node("hypothesize", self._node_hypothesize)
        g.add_node("execute", self._node_execute)
        g.add_node("analyze", self._node_analyze)
        g.add_node("critique", self._node_critique)
        g.add_node("finalize", self._node_finalize)
        g.set_entry("hypothesize")
        g.add_edge("hypothesize", "execute")
        g.add_edge("execute", "analyze")
        g.add_edge("analyze", "critique")
        g.add_conditional_edges("critique", self._router_after_critique)
        # finalize is terminal bookkeeping when max_rounds hit via router END;
        # the router already ends the loop, so finalize runs inline below.
        return g

    def run(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        journal = Journal(self.workdir / "journal.md", spec["question"])
        state: dict = {
            "round": 0,
            "question": spec["question"],
            "factors": spec["factors"],
            "budget": dict(spec["budget"]),
            "thresholds": dict(spec.get("thresholds", {})),
            "defaults": dict(spec.get("defaults", {})),
            "metric": spec.get("metric", "test_accuracy"),
            "group_by": spec.get("group_by"),
            "max_rounds": int(spec.get("max_rounds", 10)),
            "asked": [],
            "_journal": journal,
            "_specs": [],
        }
        # Manual round loop (clearer than encoding round++ in the graph):
        g = self.build_graph()
        rounds = 0
        while True:
            state["round"] = rounds
            out = g.run({**state, "_trace": []})
            # g.run returns a fresh state; merge back the persistent bits
            state.update({k: v for k, v in out.items()
                          if not k.startswith("_") or k in ("_trace",)})
            state["_journal"] = journal
            state["_specs"] = []
            rounds += 1
            if out.get("done") or rounds >= state["max_rounds"]:
                break
        if "conclusion" not in state:
            stats = state["analysis"]["stats"]
            state["conclusion"] = verdict_text(state["question"], stats)
            journal.close(state["conclusion"])
        state["rounds"] = rounds
        # JSON-serializable summary for the caller
        return {
            "question": state["question"],
            "rounds": rounds,
            "n_runs": len(self.db),
            "n_ok": len(self.db.ok_results()),
            "n_failed": len(self.db.failed()),
            "conclusion": state["conclusion"],
            "journal": str(self.workdir / "journal.md"),
            "results_db": str(self.workdir / "results.jsonl"),
        }
