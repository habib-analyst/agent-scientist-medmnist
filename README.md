# agent-scientist-medmnist

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)] [![GitHub stars](https://img.shields.io/github/stars/habib-analyst/agent-scientist-medmnist.svg)](https://github.com/habib-analyst/agent-scientist-medmnist)

**A closed-loop multi-agent system that designs ablation hypotheses, runs MedMNIST training jobs, analyzes results, and iterates** — compressing into days an ablation program that takes a researcher weeks. Every step is journaled for audit.

## Quickstart

```bash
git clone https://github.com/habib-analyst/agent-scientist-medmnist.git
cd agent-scientist-medmnist
pip install -r requirements.txt
python -m pytest tests/ -q     # 19 passed
python -m agent_scientist.cli run --campaign campaigns/variance_pilot.json --backend local
python -m agent_scientist.cli run --campaign campaigns/first_real_campaign.json --backend local
```

The run prints a JSON summary of the campaign — hypotheses asked, runs executed, measured effects, and the Critic's continue/refine/stop verdicts — and writes a human-readable journal under `runs/<campaign>/journal.md`.

## Measured results

CPU smoke test (2026-10-07, PneumoniaMNIST, tiny CNN, 2 epochs, 4 runs — real training, not simulation):

| Condition | Mean test accuracy | 95% CI |
|---|---|---|
| Augmentation: basic | 0.8421 | [0.8349, 0.8494] |
| Augmentation: none | 0.8101 | [0.8077, 0.8125] |
| **Effect** | **+0.0321** | |

Variance decomposition: 89.9% factor-driven, 10.1% seed noise — the loop correctly attributes the gain to augmentation, not luck. Full loop (hypothesize → execute → analyze → critique → journal) verified end-to-end. Test suite: 11/11 green.

## First real campaign (2026-10-28)

The first complete multi-round campaign, run on CPU with all artefacts committed under [`campaigns/first_real_campaign/`](campaigns/first_real_campaign/) (journal, result DB, per-run records, full report). Question: *on PneumoniaMNIST, what drives test-accuracy variance more — augmentation, learning rate, or seed noise?* Full factorial: augment {none, basic} × lr {0.001, 0.003} × seed {0, 1, 2} = 12 runs (tiny CNN, 3 epochs; train n=4708, test n=624), executed as 3 rounds × 4 runs. 12/12 runs ok, 20 min 18 s wall clock. Critic: CONTINUE → CONTINUE → STOP (budget exhausted).

| Condition (n=3 seeds) | Mean test accuracy | 95% CI | Mean AUC |
|---|---|---|---|
| Augmentation: basic, lr 0.001 | 0.8424 | [0.8381, 0.8446] | 0.9208 |
| Augmentation: basic, lr 0.003 | 0.8478 | [0.8205, 0.8734] | 0.9219 |
| Augmentation: none, lr 0.001 | 0.8237 | [0.7981, 0.8622] | 0.9265 |
| Augmentation: none, lr 0.003 | 0.8178 | [0.7853, 0.8718] | 0.9306 |

Augmentation effect +0.0243, lr effect +0.0056 — but the variance decomposition says **81.4% seed noise vs 18.6% factor-driven**: at this scale, seed noise dominates, and single-seed comparisons are unreliable. (The smoke test above found the opposite split under a different, smaller design — n=2/cell decompositions are unstable, which is the point.) Best single run: basic / lr 0.003 / seed 0 at 0.8734 accuracy, 0.9303 AUC.

Getting here fixed five real bugs the campaign exposed — including run-ID collisions that silently stored results under the wrong config, and a Hypothesizer that consumed the whole budget in round 0 so the loop never iterated. All five are documented, with regression tests (suite now 19 passing), in [`campaigns/first_real_campaign/REPORT.md`](campaigns/first_real_campaign/REPORT.md).

## How it works

Four agents wired by a minimal state-graph engine (`graph.py`, LangGraph-equivalent semantics but zero third-party dependencies, so the loop runs anywhere Python runs):

| Agent | Role |
|---|---|
| **Hypothesizer** | Proposes experiments from a hypothesis grammar (full-factorial minus already-asked, within budget) |
| **Executor** | Launches training via a pluggable backend: `LocalBackend` (this machine) or `KaggleBackend` (GPU kernels) |
| **Analyst** | Result DB (JSONL, fingerprint-deduped) → per-cell means, bootstrap 95% CIs, effect sizes, ANOVA-style variance decomposition |
| **Critic** | Continue / refine / stop on explicit rules: retry failures, tighten CIs, stop when the question is answered or the budget is exhausted |

The default LLM provider is deterministic and rule-based ($0 cost); an Ollama-backed provider plugs in for free-form reasoning.

## Limitations

- The 20-run GPU pilot is staged (Kaggle notebook included) but still untested — the loop's GPU-scale claims wait on the weekly quota reset. CPU operation is now validated by a full 12-run campaign (above).
- The rule-based Hypothesizer can't surprise you; it asks exactly what the grammar covers.
- The Critic's retries are nominal: it names failed configs as follow-ups, but already-asked configs are never re-proposed, so failed runs are not actually retried.
- Variance decompositions at n≤3 seeds per cell are themselves noisy — treat the factor/seed split as an estimate, not a constant of nature.

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
