# First real campaign — report (2026-10-28)

This campaign is the first time `agent-scientist-medmnist` ran a complete,
multi-round, real-training campaign end to end, with all artefacts committed:
[`journal.md`](journal.md) (the system's own audit log),
[`results.jsonl`](results.jsonl) (the result DB),
[`results_table.md`](results_table.md), [`summary.json`](summary.json), and
[`run_records/`](run_records/) (per-run config + result JSON).

## What the system actually was before this campaign

Audited module by module — the core loop was genuinely implemented, not stubbed:

| Component | State found |
|---|---|
| `graph.py` state-graph engine | Complete; conditional edges + decision trace |
| `providers.py` RuleBasedProvider | Complete: full-factorial hypothesizer, rule-based critic, summarizer |
| `campaign.py` CampaignRunner | Complete wiring, but with the latent bugs below |
| `backends.py` LocalBackend | Real: subprocess training. KaggleBackend real but credential-gated (raises `BackendUnavailable`, never fakes) |
| `train_medmnist.py` | Real PyTorch training; AUC only for multi-label tasks (gap below) |
| `analyst.py` | Complete: JSONL result DB with fingerprint dedupe, bootstrap CIs, effects, variance decomposition |
| `journal.py`, `cli.py` | Complete |
| Tests | 11 passing — but the full-loop test used a fake backend; no test ran real training |

What was missing: any committed real-campaign artefacts (the README's
2026-10-07 smoke test was 4 ad-hoc runs), a CPU-feasible campaign spec
(the shipped `variance_pilot.json` is 20 × ResNet-18 × PathMNIST — GPU-scale),
and multi-round iteration, which turned out to be impossible (fix 1).

## Fixes made (each with regression tests; suite now 19 passing)

1. **Per-round proposal cap** (`providers.py`). The Hypothesizer always
   proposed every fresh combo the budget allowed, so round 0 consumed the
   entire budget and the "loop" never iterated. It now honours an optional
   `budget.per_round` cap; behaviour without the key is unchanged.
2. **Run-ID collisions corrupted results** (`campaign.py`). Run IDs encoded
   only dataset/arch/augment/seed. Two configs differing only in `lr`
   shared an ID: artefact files overwrote each other, and results were
   re-attached to specs by ID, so a result was stored under the *wrong*
   config. Proven in the first launch's round 0: the (none, lr=0.001, seed 0)
   run's accuracy (0.8109) was recorded under lr=0.003. IDs now carry a
   6-char hash of *all* run fields.
3. **Round ownership** (`campaign.py`). The graph router looped back to
   "hypothesize" *and* `run()`'s manual loop iterated, so cycles inside one
   graph pass were all journaled as "Round 0". The router now ends each
   pass; `run()` owns round advancement (as its own comment always said).
4. **Critic judged a stale budget** (`campaign.py`). The Critic received
   the budget as of the round's *start*, so a campaign whose budget ended
   with its last run needed a trailing empty round to notice exhaustion.
   The spend is now consumed before the Critic decides.
5. **No AUC for binary tasks** (`train_medmnist.py`). Probabilities were
   only collected for multi-label tasks, so PneumoniaMNIST — a binary task,
   and MedMNIST's headline metric is AUC — reported accuracy only. The eval
   loop now keeps positive-class probabilities for `binary-class` tasks and
   reports `test_auc` via a unit-tested helper (`_auc_from_probs`, which
   also skips single-class columns explicitly; newer scikit-learn returns
   NaN there instead of raising).

## Campaign design (stated in advance)

- Question: *On PneumoniaMNIST, what drives test-accuracy variance more:
  augmentation, learning rate, or seed noise?*
- Full factorial: augment {none, basic} × lr {0.001, 0.003} × seed {0, 1, 2}
  = **12 experiment iterations**, tiny-cnn, 3 epochs, batch 128.
  Train n=4708, test n=624 (official splits).
- **3 rounds × 4 runs** (per-round cap 4), budget 12 runs, CI target 0.005.
- Wall-clock budget set in advance: **60 minutes** for the campaign process.
- Dataset pre-seeded via `curl` and MD5-verified against the official
  MedMNIST checksum (`28209eda62fecd6e6a2d98b1501bb15f`) because Python HTTP
  downloads are unreliable on this VM.

## What happened

**First launch (09:01 UTC)** was externally terminated ~4 minutes in,
after round 0's 4 runs — the log is empty and no traceback exists; the
process died silently mid-round. Its partial journal/results are what
exposed fixes 2–4 above (the mislabeled record and the duplicated
"Round 0" heading are visible in them verbatim).

**Second launch (09:19 UTC, detached session)** ran to completion:
3 rounds, **12/12 runs ok, 0 failed**, wall clock **20 min 18 s**
(09:19:07–09:39:25 UTC), sum of run durations 1217.5 s. Critic trajectory:
CONTINUE → CONTINUE → STOP (budget exhausted), exactly as designed.

## Results (all measured; see results_table.md)

| cell | mean acc (n=3) | 95% CI | mean AUC |
|---|---|---|---|
| basic, lr 0.001 | 0.8424 | [0.8381, 0.8446] | 0.9208 |
| basic, lr 0.003 | 0.8478 | [0.8205, 0.8734] | 0.9219 |
| none, lr 0.001 | 0.8237 | [0.7981, 0.8622] | 0.9265 |
| none, lr 0.003 | 0.8178 | [0.7853, 0.8718] | 0.9306 |

- Augmentation effect (mean |Δ|): **+0.0243** accuracy. LR effect: +0.0056.
- Variance decomposition: **18.6% factor-driven vs 81.4% seed noise** —
  at this scale (tiny CNN, 3 epochs, 3 seeds/cell), *seed noise dominates*.
  Single-seed comparisons in this regime are unreliable; that is the
  campaign's headline finding, and it is why the system exists.
- Best single run: basic, lr 0.003, seed 0 — accuracy **0.8734**, AUC **0.9303**.
- Determinism check: the (none, 0.001, seed 0) config reproduced accuracy
  0.8109 exactly across the standalone probe, the terminated launch, and
  this campaign.
- Note: the 2026-10-07 smoke test reported 89.9% factor-driven variance.
  Different design (2 epochs, augment-only grouping, n=2/cell) — small-n
  variance decompositions are unstable, which is precisely the phenomenon
  this campaign quantifies at n=3/cell. Both numbers stand as measured;
  they are not directly comparable.

## Known limitations left as-is (documented, not blocking)

- The Critic's "refine" verdict names failed configs as follow-ups, but
  the Hypothesizer never re-proposes already-asked configs, so failed
  runs are not actually retried. No run failed in this campaign.
- If the factorial exhausts before the budget, later rounds propose zero
  experiments until the Critic stops the campaign.
- `kaggle/staging/` still snapshots the pre-fix sources for the Kaggle
  kernel; refresh it before the GPU pilot.
- The GPU pilot (`variance_pilot.json`, 20 × ResNet-18) remains untested.
