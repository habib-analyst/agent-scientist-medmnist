# agent-scientist-medmnist

A **closed-loop multi-agent system** that designs ablation hypotheses, runs MedMNIST training jobs, analyzes results, and iterates — completing in days an ablation program that would take a researcher weeks. Every step is logged for audit.

## Agents (LangGraph-equivalent semantics, zero dependencies)

| Agent | Role |
|---|---|
| **Hypothesizer** | Proposes experiments from a hypothesis grammar (full-factorial minus already-asked, within budget) |
| **Executor** | Launches training via a pluggable backend: `LocalBackend` (this machine) or `KaggleBackend` (GPU kernels) |
| **Analyst** | Result DB (JSONL, fingerprint-deduped) → per-cell means, bootstrap 95% CIs, effect sizes, ANOVA-style variance decomposition |
| **Critic** | Continues / refines / stops on explicit rules: retry failures, tighten CIs, stop when the question is answered or budget is exhausted |

The default reasoning provider is **deterministic and rule-based** ($0, fully reproducible). An `OllamaProvider` can be plugged in for free-form reasoning and degrades gracefully to rules when Ollama is unreachable.

## The novel bit

Not the training code — that's a standardized, auditable primitive. The novelty is the **autonomous experiment loop with statistical stopping rules**: the system decides *what* to run next based on measured uncertainty, and every claim in the auto-generated journal traces to a logged run (config fingerprint + metrics).

## Quickstart

```bash
pip install -r requirements.txt
# CPU smoke test of the full loop (fake backend, seconds):
python -m pytest tests/ -q
# Real pilot campaign on this machine (CPU, small):
python -m agent_scientist.cli run --campaign campaigns/variance_pilot.json --backend local
# GPU campaign on Kaggle (needs ~/.kaggle/kaggle.json):
python -m agent_scientist.cli run --campaign campaigns/variance_pilot.json --backend kaggle
# Inspect results:
python -m agent_scientist.cli status --workdir runs/variance_pilot
```

## Pilot campaign: what drives MedMNIST 2D variance?

`campaigns/variance_pilot.json` — 20 runs: 2 datasets (PathMNIST, PneumoniaMNIST) × ResNet-18 × {no augment, basic augment} × 5 seeds. The Analyst decomposes measured variance into factor-driven vs seed-noise components; the Critic stops when 95% CIs are tight or the 20-run budget is spent.

## Repo layout

```
agent_scientist/
  graph.py        # minimal state-graph engine (nodes/edges/conditional routing, full trace)
  providers.py    # RuleBasedProvider (default) / OllamaProvider
  backends.py     # LocalBackend (subprocess) / KaggleBackend (kernels API)
  train_medmnist.py  # standardized training primitive -> JSON result record
  campaign.py     # CampaignRunner: wires the four agents into the loop
  analyst.py      # ResultDB + bootstrap CIs + variance decomposition
  journal.py      # auto-generated audit-trail journal
  cli.py
campaigns/variance_pilot.json
tests/
```

## Results

See [`papers/PAPER.md`](papers/PAPER.md) for the pilot writeup with measured numbers.

## License

MIT
