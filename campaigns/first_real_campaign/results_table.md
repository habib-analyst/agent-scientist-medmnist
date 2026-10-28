# First real campaign — results table

Question: On PneumoniaMNIST, what drives test-accuracy variance more: augmentation, learning rate, or seed noise?

Design: full factorial augment {none, basic} x lr {0.001, 0.003} x seed {0, 1, 2} = 12 runs; tiny-cnn, 3 epochs, batch 128, 28x28. Train n=4708, test n=624 (official PneumoniaMNIST splits). Executed as 3 rounds x 4 runs (per-round budget cap 4), LocalBackend on a 2-core CPU VM. Campaign wall clock 20 min 18 s (2026-10-28 09:19:07-09:39:25 UTC); sum of run durations 1217.5 s; 12/12 runs ok, 0 failed.

## Per-run results

| run_id | augment | lr | seed | test_accuracy | test_auc | duration_s | fingerprint |
|---|---|---|---|---|---|---|---|
| r00-pneumoniamnist-tiny-cnn-none-s0-5d520a | none | 0.001 | 0 | 0.8109 | 0.9244 | 77 | 9c1907b9f59dc4ec |
| r00-pneumoniamnist-tiny-cnn-none-s0-f413bb | none | 0.003 | 0 | 0.7965 | 0.9221 | 99 | ad4b9fa517fcffa8 |
| r00-pneumoniamnist-tiny-cnn-none-s1-b07372 | none | 0.001 | 1 | 0.8622 | 0.9310 | 122 | 4a2a72c1cab40e0a |
| r00-pneumoniamnist-tiny-cnn-none-s2-3d5d09 | none | 0.001 | 2 | 0.7981 | 0.9241 | 111 | 02df54d2774297c7 |
| r01-pneumoniamnist-tiny-cnn-basic-s0-17991e | basic | 0.001 | 0 | 0.8446 | 0.9206 | 123 | 4286eb1b86b178c8 |
| r01-pneumoniamnist-tiny-cnn-basic-s1-bf0617 | basic | 0.001 | 1 | 0.8446 | 0.9212 | 113 | 884a7d1e776f7542 |
| r01-pneumoniamnist-tiny-cnn-none-s1-53d3f6 | none | 0.003 | 1 | 0.8718 | 0.9356 | 104 | 8a4ae0c953de1fda |
| r01-pneumoniamnist-tiny-cnn-none-s2-7b8b47 | none | 0.003 | 2 | 0.7853 | 0.9343 | 112 | 2bde7ab3ac75e594 |
| r02-pneumoniamnist-tiny-cnn-basic-s0-9e1807 | basic | 0.003 | 0 | 0.8734 | 0.9303 | 78 | 11b6a26605415a27 |
| r02-pneumoniamnist-tiny-cnn-basic-s1-c9be3b | basic | 0.003 | 1 | 0.8494 | 0.9168 | 87 | 75fe52917006758b |
| r02-pneumoniamnist-tiny-cnn-basic-s2-5e8cc3 | basic | 0.003 | 2 | 0.8205 | 0.9186 | 92 | ac77b0ba0ec6925e |
| r02-pneumoniamnist-tiny-cnn-basic-s2-f4b5a7 | basic | 0.001 | 2 | 0.8381 | 0.9206 | 99 | 0d038df5771b58da |

## Per-cell means (n=3 seeds each; accuracy CIs are the Analyst's bootstrap 95% CIs)

| cell | mean accuracy | 95% CI | mean test_auc |
|---|---|---|---|
| augment=basic \| lr=0.001 | 0.8424 | [0.8381, 0.8446] | 0.9208 |
| augment=basic \| lr=0.003 | 0.8478 | [0.8205, 0.8734] | 0.9219 |
| augment=none \| lr=0.001 | 0.8237 | [0.7981, 0.8622] | 0.9265 |
| augment=none \| lr=0.003 | 0.8178 | [0.7853, 0.8718] | 0.9306 |

## Analyst conclusion (from the campaign journal, computed by the system)

- mean |delta| across augment levels: 0.0243
- mean |delta| across lr levels: 0.0056
- Variance decomposition: 18.6% factor-driven vs 81.4% seed noise
- Verdict: **seed noise dominates** the measured variance at this scale.
- Best single run: augment=basic, lr=0.003, seed=0 — accuracy 0.8734, AUC 0.9303.
- Critic trajectory: CONTINUE (round 0, widest CI 0.064 > 0.005) -> CONTINUE (round 1, widest CI 0.087 > 0.005) -> STOP (round 2, budget exhausted).
