# agent-scientist-medmnist

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)] [![GitHub stars](https://img.shields.io/github/stars/habib-analyst/agent-scientist-medmnist.svg)](https://github.com/habib-analyst/agent-scientist-medmnist)

**A closed-loop multi-agent system that designs ablation hypotheses, runs MedMNIST training jobs, analyzes results, and iterates** — compressing into days an ablation program that takes a researcher weeks. Every step is journaled for audit.

## Quickstart

```bash
git clone https://github.com/habib-analyst/agent-scientist-medmnist.git
cd agent-scientist-medmnist
pip install -r requirements.txt
python -m pytest tests/ -q     # 11 passed
python -m agent_scientist.cli run --campaign campaigns/variance_pilot.json --backend local
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

- Only validated on a CPU smoke campaign so far. The 20-run GPU pilot is staged (Kaggle notebook included) but was deferred to the weekly quota reset — the loop's GPU-scale claims are untested until then.
- The rule-based Hypothesizer can't surprise you; it asks exactly what the grammar covers.

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
