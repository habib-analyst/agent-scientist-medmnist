# Experiment Journal

**Research question:** On PneumoniaMNIST, what drives test-accuracy variance more: augmentation, learning rate, or seed noise?

**Opened:** 2026-10-28T09:19:07.529891+00:00

Every claim below traces to a logged run (fingerprint + config).

---

## Round 0 — 2026-10-28T09:19:07.551723+00:00

**Hypothesizer:** Round 0: full-factorial proposal over 5 factors (12 combos); 12 unasked, proposing 4 within budget (per-round cap 4).

- proposed `r00-pneumoniamnist-tiny-cnn-none-s0-5d520a` (dataset=pneumoniamnist, arch=tiny-cnn, augment=none, seed=0, epochs=3)
- proposed `r00-pneumoniamnist-tiny-cnn-none-s1-b07372` (dataset=pneumoniamnist, arch=tiny-cnn, augment=none, seed=1, epochs=3)
- proposed `r00-pneumoniamnist-tiny-cnn-none-s2-3d5d09` (dataset=pneumoniamnist, arch=tiny-cnn, augment=none, seed=2, epochs=3)
- proposed `r00-pneumoniamnist-tiny-cnn-none-s0-f413bb` (dataset=pneumoniamnist, arch=tiny-cnn, augment=none, seed=0, epochs=3)
- `r00-pneumoniamnist-tiny-cnn-none-s0-5d520a` OK: test_accuracy=0.8109 fp=9c1907b9f59dc4ec backend=local duration=77s
- `r00-pneumoniamnist-tiny-cnn-none-s1-b07372` OK: test_accuracy=0.8622 fp=4a2a72c1cab40e0a backend=local duration=122s
- `r00-pneumoniamnist-tiny-cnn-none-s2-3d5d09` OK: test_accuracy=0.7981 fp=02df54d2774297c7 backend=local duration=111s
- `r00-pneumoniamnist-tiny-cnn-none-s0-f413bb` OK: test_accuracy=0.7965 fp=ad4b9fa517fcffa8 backend=local duration=99s

**Analyst verdict:**
```
Research question: On PneumoniaMNIST, what drives test-accuracy variance more: augmentation, learning rate, or seed noise?

  augment=none|lr=0.001: mean=0.8237 95%CI=[0.7981,0.8622] (n=3)
  augment=none|lr=0.003: mean=0.7965 95%CI=[0.7965,0.7965] (n=1)

  mean |Δ| across lr levels: 0.0272

  Variance decomposition: 19.5% factor-driven vs 80.5% seed noise.
  => Seed noise DOMINATES the measured variance.
augment=none|lr=0.001: mean=0.8237 95%CI=[0.7981, 0.8622] (n=3)
augment=none|lr=0.003: mean=0.7965 95%CI=[0.7965, 0.7965] (n=1)
effect[lr] = +0.0272
```

**Critic:** CONTINUE — Widest 95% CI is 0.064 > target 0.005; more runs will tighten the variance estimates.

---

## Round 1 — 2026-10-28T09:25:56.417787+00:00

**Hypothesizer:** Round 1: full-factorial proposal over 5 factors (12 combos); 8 unasked, proposing 4 within budget (per-round cap 4).

- proposed `r01-pneumoniamnist-tiny-cnn-none-s1-53d3f6` (dataset=pneumoniamnist, arch=tiny-cnn, augment=none, seed=1, epochs=3)
- proposed `r01-pneumoniamnist-tiny-cnn-none-s2-7b8b47` (dataset=pneumoniamnist, arch=tiny-cnn, augment=none, seed=2, epochs=3)
- proposed `r01-pneumoniamnist-tiny-cnn-basic-s0-17991e` (dataset=pneumoniamnist, arch=tiny-cnn, augment=basic, seed=0, epochs=3)
- proposed `r01-pneumoniamnist-tiny-cnn-basic-s1-bf0617` (dataset=pneumoniamnist, arch=tiny-cnn, augment=basic, seed=1, epochs=3)
- `r01-pneumoniamnist-tiny-cnn-none-s1-53d3f6` OK: test_accuracy=0.8718 fp=8a4ae0c953de1fda backend=local duration=104s
- `r01-pneumoniamnist-tiny-cnn-none-s2-7b8b47` OK: test_accuracy=0.7853 fp=2bde7ab3ac75e594 backend=local duration=112s
- `r01-pneumoniamnist-tiny-cnn-basic-s0-17991e` OK: test_accuracy=0.8446 fp=4286eb1b86b178c8 backend=local duration=123s
- `r01-pneumoniamnist-tiny-cnn-basic-s1-bf0617` OK: test_accuracy=0.8446 fp=884a7d1e776f7542 backend=local duration=113s

**Analyst verdict:**
```
Research question: On PneumoniaMNIST, what drives test-accuracy variance more: augmentation, learning rate, or seed noise?

  augment=basic|lr=0.001: mean=0.8446 95%CI=[0.8446,0.8446] (n=2)
  augment=none|lr=0.001: mean=0.8237 95%CI=[0.7981,0.8622] (n=3)
  augment=none|lr=0.003: mean=0.8178 95%CI=[0.7853,0.8718] (n=3)

  mean |Δ| across augment levels: 0.0208
  mean |Δ| across lr levels: 0.0059

  Variance decomposition: 11.8% factor-driven vs 88.2% seed noise.
  => Seed noise DOMINATES the measured variance.
augment=basic|lr=0.001: mean=0.8446 95%CI=[0.8446, 0.8446] (n=2)
augment=none|lr=0.001: mean=0.8237 95%CI=[0.7981, 0.8622] (n=3)
augment=none|lr=0.003: mean=0.8178 95%CI=[0.7853, 0.8718] (n=3)
effect[augment] = +0.0208
effect[lr] = +0.0059
```

**Critic:** CONTINUE — Widest 95% CI is 0.087 > target 0.005; more runs will tighten the variance estimates.

---

## Round 2 — 2026-10-28T09:33:29.119019+00:00

**Hypothesizer:** Round 2: full-factorial proposal over 5 factors (12 combos); 4 unasked, proposing 4 within budget (per-round cap 4).

- proposed `r02-pneumoniamnist-tiny-cnn-basic-s2-f4b5a7` (dataset=pneumoniamnist, arch=tiny-cnn, augment=basic, seed=2, epochs=3)
- proposed `r02-pneumoniamnist-tiny-cnn-basic-s0-9e1807` (dataset=pneumoniamnist, arch=tiny-cnn, augment=basic, seed=0, epochs=3)
- proposed `r02-pneumoniamnist-tiny-cnn-basic-s1-c9be3b` (dataset=pneumoniamnist, arch=tiny-cnn, augment=basic, seed=1, epochs=3)
- proposed `r02-pneumoniamnist-tiny-cnn-basic-s2-5e8cc3` (dataset=pneumoniamnist, arch=tiny-cnn, augment=basic, seed=2, epochs=3)
- `r02-pneumoniamnist-tiny-cnn-basic-s2-f4b5a7` OK: test_accuracy=0.8381 fp=0d038df5771b58da backend=local duration=99s
- `r02-pneumoniamnist-tiny-cnn-basic-s0-9e1807` OK: test_accuracy=0.8734 fp=11b6a26605415a27 backend=local duration=78s
- `r02-pneumoniamnist-tiny-cnn-basic-s1-c9be3b` OK: test_accuracy=0.8494 fp=75fe52917006758b backend=local duration=87s
- `r02-pneumoniamnist-tiny-cnn-basic-s2-5e8cc3` OK: test_accuracy=0.8205 fp=ac77b0ba0ec6925e backend=local duration=92s

**Analyst verdict:**
```
Research question: On PneumoniaMNIST, what drives test-accuracy variance more: augmentation, learning rate, or seed noise?

  augment=basic|lr=0.001: mean=0.8424 95%CI=[0.8381,0.8446] (n=3)
  augment=basic|lr=0.003: mean=0.8478 95%CI=[0.8205,0.8734] (n=3)
  augment=none|lr=0.001: mean=0.8237 95%CI=[0.7981,0.8622] (n=3)
  augment=none|lr=0.003: mean=0.8178 95%CI=[0.7853,0.8718] (n=3)

  mean |Δ| across augment levels: 0.0243
  mean |Δ| across lr levels: 0.0056

  Variance decomposition: 18.6% factor-driven vs 81.4% seed noise.
  => Seed noise DOMINATES the measured variance.
augment=basic|lr=0.001: mean=0.8424 95%CI=[0.8381, 0.8446] (n=3)
augment=basic|lr=0.003: mean=0.8478 95%CI=[0.8205, 0.8734] (n=3)
augment=none|lr=0.001: mean=0.8237 95%CI=[0.7981, 0.8622] (n=3)
augment=none|lr=0.003: mean=0.8178 95%CI=[0.7853, 0.8718] (n=3)
effect[augment] = +0.0243
effect[lr] = +0.0056
```

**Critic:** STOP — Compute budget exhausted; reporting on completed runs.

---


# Campaign conclusion

Research question: On PneumoniaMNIST, what drives test-accuracy variance more: augmentation, learning rate, or seed noise?

  augment=basic|lr=0.001: mean=0.8424 95%CI=[0.8381,0.8446] (n=3)
  augment=basic|lr=0.003: mean=0.8478 95%CI=[0.8205,0.8734] (n=3)
  augment=none|lr=0.001: mean=0.8237 95%CI=[0.7981,0.8622] (n=3)
  augment=none|lr=0.003: mean=0.8178 95%CI=[0.7853,0.8718] (n=3)

  mean |Δ| across augment levels: 0.0243
  mean |Δ| across lr levels: 0.0056

  Variance decomposition: 18.6% factor-driven vs 81.4% seed noise.
  => Seed noise DOMINATES the measured variance.

**Closed:** 2026-10-28T09:39:25.330027+00:00

