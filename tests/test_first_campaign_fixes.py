"""Tests for the first-real-campaign fixes (2026-10-28):

1. RuleBasedProvider honours the optional ``per_round`` budget cap, so a
   campaign iterates over several rounds instead of consuming the whole
   budget in round 0 (previously multi-round campaigns were impossible:
   the Hypothesizer always proposed every fresh combo the budget allowed).
2. train_medmnist._auc_from_probs: per-column ROC AUC used to report
   test_auc for binary-class tasks (previously probabilities were only
   collected for multi-label tasks, so binary tasks got no AUC at all).
3. CampaignRunner run_id uniqueness: run_ids used to encode only
   dataset/arch/augment/seed, so configs differing in lr (or epochs /
   batch_size) collided — artefact files overwrote each other and results
   were re-attached to the WRONG config in the result DB. The first real
   campaign's round 0 stored an lr=0.001 run under an lr=0.003 config.
4. CampaignRunner round ownership: the graph router looped back to
   "hypothesize" while run()'s manual loop also iterated, so every cycle
   inside one graph pass was journaled as the same round. The router now
   ends each pass; run() owns round advancement.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_scientist.backends import ExperimentBackend, RunResult
from agent_scientist.campaign import CampaignRunner
from agent_scientist.providers import RuleBasedProvider
from agent_scientist.train_medmnist import _auc_from_probs

FACTORS = {"augment": ["none", "basic"], "seed": [0, 1, 2]}  # 6 combos


def _propose(budget, asked=None):
    p = RuleBasedProvider()
    return p.decide("propose_experiments", {
        "factors": FACTORS, "asked": asked or [],
        "budget": budget, "round": 0})


def test_per_round_cap_limits_batch():
    d = _propose({"runs_remaining": 6, "per_round": 2})
    assert len(d["experiments"]) == 2


def test_no_per_round_cap_unchanged_behaviour():
    d = _propose({"runs_remaining": 6})
    assert len(d["experiments"]) == 6


def test_per_round_cap_still_respects_budget_and_asked():
    asked = [{"augment": "none", "seed": 0}, {"augment": "none", "seed": 1}]
    d = _propose({"runs_remaining": 1, "per_round": 4}, asked=asked)
    assert len(d["experiments"]) == 1  # budget is the binding cap
    assert {"augment": "none", "seed": 0} not in d["experiments"]


def test_auc_from_probs_perfect_separation():
    P = np.array([[0.1], [0.9], [0.2], [0.8]])
    Y = np.array([[0], [1], [0], [1]])
    assert _auc_from_probs(P, Y) == [1.0]


def test_auc_from_probs_single_class_column_skipped():
    P = np.array([[0.1], [0.9], [0.2]])
    Y = np.array([[0], [0], [0]])
    assert _auc_from_probs(P, Y) == []


def test_auc_from_probs_multicolumn():
    # col 0: perfect; col 1: reversed (AUC 0.0)
    P = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    Y = np.array([[0, 0], [1, 1], [0, 0], [1, 1]])
    aucs = _auc_from_probs(P, Y)
    assert len(aucs) == 2
    assert aucs[0] == pytest.approx(1.0)
    assert aucs[1] == pytest.approx(0.0)


# --- campaign runner: run_id uniqueness + round ownership -----------------
class EchoBackend(ExperimentBackend):
    """Returns a metric computed FROM the spec, so a result recorded under
    the wrong config is detectable."""

    name = "echo"

    def submit(self, spec, workdir):
        acc = spec.lr * 100 + (0.03 if spec.augment == "basic" else 0.0) \
            + spec.seed * 0.002
        return RunResult(run_id=spec.run_id,
                         fingerprint=spec.fingerprint(), status="ok",
                         metrics={"test_accuracy": acc}, duration_s=0.01,
                         backend=self.name)


def test_run_ids_unique_and_configs_not_cross_attached(tmp_path):
    runner = CampaignRunner(tmp_path / "work", RuleBasedProvider(),
                            EchoBackend())
    summary = runner.run({
        "question": "lr effect?",
        "factors": {"dataset": ["pneumoniamnist"], "arch": ["tiny-cnn"],
                    "augment": ["none"], "lr": [0.001, 0.003], "seed": [0]},
        "budget": {"runs_remaining": 2, "retries_remaining": 0},
        "thresholds": {"ci_width_target": 0.5},
        "defaults": {"epochs": 1},
        "max_rounds": 2,
    })
    assert summary["n_ok"] == 2
    recs = runner.db.all()
    run_ids = [r["run_id"] for r in recs]
    assert len(set(run_ids)) == 2  # no collision despite same seed/augment
    for r in recs:
        # the metric must match the config it is recorded under
        expect = r["config"]["lr"] * 100
        assert r["metrics"]["test_accuracy"] == pytest.approx(expect)


def test_rounds_advance_and_journal_labels(tmp_path):
    runner = CampaignRunner(tmp_path / "work", RuleBasedProvider(),
                            EchoBackend())
    summary = runner.run({
        "question": "augment effect?",
        "factors": {"dataset": ["pneumoniamnist"], "arch": ["tiny-cnn"],
                    "augment": ["none", "basic"], "seed": [0, 1]},
        "budget": {"runs_remaining": 4, "retries_remaining": 0,
                   "per_round": 2},
        "thresholds": {"ci_width_target": 0.0001},  # continue while budget lasts
        "group_by": ["augment"],
        "defaults": {"epochs": 1},
        "max_rounds": 5,
    })
    assert summary["n_ok"] == 4
    assert summary["rounds"] == 2
    text = Path(summary["journal"]).read_text()
    assert text.count("## Round 0") == 1
    assert text.count("## Round 1") == 1
