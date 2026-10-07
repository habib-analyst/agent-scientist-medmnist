"""Tests for the graph engine, providers, analyst, critic, and the full loop
with a fake backend (fast) — plus a real-training smoke test (slow, CPU)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_scientist.analyst import ResultDB, compute_stats, verdict_text
from agent_scientist.backends import ExperimentBackend, RunResult, RunSpec
from agent_scientist.campaign import CampaignRunner
from agent_scientist.graph import END, StateGraph
from agent_scientist.providers import OllamaProvider, RuleBasedProvider


# --- graph engine ---------------------------------------------------------
def test_graph_linear():
    g = StateGraph()
    g.add_node("a", lambda s: {"x": 1})
    g.add_node("b", lambda s: {"y": s["x"] + 1})
    g.set_entry("a").add_edge("a", "b")
    out = g.run({})
    assert out["x"] == 1 and out["y"] == 2
    assert out["_trace"][0]["node"] == "a"


def test_graph_conditional():
    g = StateGraph()
    g.add_node("a", lambda s: {"n": s.get("n", 0) + 1})
    g.add_node("b", lambda s: {"done": True})
    g.set_entry("a")
    g.add_conditional_edges("a", lambda s: END if s["n"] >= 3 else "a")
    g.add_edge("b", END)
    out = g.run({})
    assert out["n"] == 3
    assert out["_steps"] == 3


def test_graph_rejects_unknown_router_target():
    g = StateGraph()
    g.add_node("a", lambda s: {})
    g.set_entry("a")
    g.add_conditional_edges("a", lambda s: "nope")
    with pytest.raises(ValueError):
        g.run({})


# --- providers ------------------------------------------------------------
def test_rulebased_proposes_full_factorial_minus_asked():
    p = RuleBasedProvider()
    d = p.decide("propose_experiments", {
        "factors": {"augment": ["none", "basic"], "seed": [0, 1]},
        "asked": [{"augment": "none", "seed": 0}],
        "budget": {"runs_remaining": 10}, "round": 0})
    assert len(d["experiments"]) == 3
    assert {"augment": "none", "seed": 0} not in d["experiments"]


def test_rulebased_critic_stop_on_budget():
    p = RuleBasedProvider()
    d = p.decide("critique", {
        "analysis": {"failed_runs": 0, "ci_width_max": 0.01,
                     "failed_configs": []},
        "budget": {"runs_remaining": 0, "retries_remaining": 0},
        "thresholds": {"ci_width_target": 0.05}})
    assert d["decision"] == "stop"


def test_rulebased_critic_continue_on_wide_ci():
    p = RuleBasedProvider()
    d = p.decide("critique", {
        "analysis": {"failed_runs": 0, "ci_width_max": 0.2,
                     "failed_configs": []},
        "budget": {"runs_remaining": 5, "retries_remaining": 1},
        "thresholds": {"ci_width_target": 0.05}})
    assert d["decision"] == "continue"


def test_rulebased_critic_retry_failures():
    p = RuleBasedProvider()
    d = p.decide("critique", {
        "analysis": {"failed_runs": 2, "ci_width_max": 0.01,
                     "failed_configs": [{"seed": 0}]},
        "budget": {"runs_remaining": 5, "retries_remaining": 2},
        "thresholds": {"ci_width_target": 0.05}})
    assert d["decision"] == "refine"
    assert d["followups"] == [{"seed": 0}]


def test_ollama_falls_back_gracefully():
    p = OllamaProvider(model="no-such-model",
                       base_url="http://127.0.0.1:1")
    d = p.decide("critique", {
        "analysis": {"failed_runs": 0, "ci_width_max": 0.01,
                     "failed_configs": []},
        "budget": {"runs_remaining": 0, "retries_remaining": 0},
        "thresholds": {"ci_width_target": 0.05}})
    assert d["decision"] == "stop"
    assert "_provider_fallback" in d


# --- analyst --------------------------------------------------------------
def _fake_results():
    recs = []
    for aug in ("none", "basic"):
        for seed in range(4):
            acc = 0.80 + (0.02 if aug == "basic" else 0.0) + seed * 0.001
            recs.append({
                "fingerprint": f"fp-{aug}-{seed}", "status": "ok",
                "config": {"dataset": "pneumoniamnist", "augment": aug,
                           "seed": seed},
                "metrics": {"test_accuracy": acc}})
    return recs


def test_compute_stats_and_verdict(tmp_path):
    stats = compute_stats(_fake_results(), group_by=["augment"])
    assert stats["n_ok"] == 8
    assert abs(stats["per_cell"]["augment=none"]["mean"] - 0.8015) < 1e-9
    assert stats["effects"]["augment"] > 0.019
    vd = stats["variance_decomposition"]
    assert vd["frac_factor_driven"] > vd["frac_seed_noise"]
    text = verdict_text("q?", stats)
    assert "FACTOR" in text.upper() or "factors DOMINATE" in text


def test_resultdb_dedupes(tmp_path):
    db = ResultDB(tmp_path / "r.jsonl")
    rec = {"fingerprint": "abc", "status": "ok", "config": {},
           "metrics": {}}
    db.add(rec)
    db.add(rec)
    assert len(db) == 1


# --- full loop with a fake backend ----------------------------------------
class FakeBackend(ExperimentBackend):
    name = "fake"

    def submit(self, spec, workdir):
        # deterministic pseudo-measurement: no training, but exercises the
        # whole agent loop incl. journaling, analysis, critic decisions.
        acc = 0.75 + (0.03 if spec.augment == "basic" else 0.0) + spec.seed * 0.002
        return RunResult(run_id=spec.run_id,
                         fingerprint=spec.fingerprint(), status="ok",
                         metrics={"test_accuracy": acc}, duration_s=0.01,
                         backend=self.name)


def test_full_campaign_loop_fake_backend(tmp_path):
    runner = CampaignRunner(tmp_path / "work", RuleBasedProvider(),
                            FakeBackend())
    summary = runner.run({
        "question": "Does augmentation beat seed noise?",
        "factors": {"dataset": ["pneumoniamnist"], "arch": ["resnet18"],
                    "augment": ["none", "basic"], "seed": [0, 1, 2]},
        "budget": {"runs_remaining": 6, "retries_remaining": 1},
        "thresholds": {"ci_width_target": 0.5},  # loose: stop after round 1
        "defaults": {"epochs": 1},
        "max_rounds": 3,
    })
    assert summary["n_ok"] == 6
    assert summary["n_failed"] == 0
    assert "conclusion" in summary
    journal = Path(summary["journal"])
    assert journal.exists()
    text = journal.read_text()
    assert "Hypothesizer" in text and "Critic" in text
