# Jev vs Clef vs traditional ML — classifier benchmarks

Benchmark of decision models (Typesafe Jev, Cloudflare Clef / Clef-flash)
against traditional ML on four generated classifier datasets. Question-driven
detail lives further down; results and findings come first.

## Headline results

1. **Zero training beats 50 labels, on every dataset.** Decision models score
   93–99% on the choice task vs 78–85% for the best trained linear model and
   ~35% at n=10. No single model sweeps: Jev leads reviews (99.0%), Clef
   leads support-tickets (97.9%), Clef-flash leads topics (99.0%).
2. **Distribution shift inverts the classic systems.** Trained on one
   phrasing, tested on a paraphrased twin: logreg −60.4, keyword rules
   −39.6, while Jev −4.2 / Clef −6.3 / Clef-flash −10.4. If your input
   distribution moves, everything from the in-distribution tables points
   toward the decision models.
3. **Distillation works, capped by teacher quality.** A TF-IDF logreg
   distilled from 1000 Jev labels hits **99.0%** — beating gold-n=50
   (78.1%) and Jev itself (96.9%) — while the score task student is capped
   by its teacher's 67.8% agreement. Best teacher is per-task: Jev for
   graded judgments, Clef/Clef-flash for category-like ones.
4. **Confidence routing depends on which signal you use.** Jev's reported
   choice confidence is route-ready (ECE 0.04–0.05, 100% coverage at a 95%
   accuracy bar). Clef's reported confidence is miscalibrated (ECE 0.21–0.28)
   but its **max per-option probability** is nearly as good (ECE 0.079).
   On hard graded tasks, no decision model's confidence is usable (≤6%
   coverage at the 90% bar) while a 50-label logreg keeps 80%.
5. **Cheap systems win when the signal is lexical and labels are cheap.**
   Explicit-cue urgency: keyword rules 89.6% (and 0 ms); hard urgency:
   logreg (n=50) 83.3% beats all three decision models.
6. **Measured latency contradicts vendor claims here**: Jev ~15–17 ms/item
   vs Clef ~58–85 ms/item (Cloudflare claims Clef 2.5–13× faster). See
   caveat 14 before quoting either direction.

## Results

### Main benchmark — choice task accuracy (3 seeds, mean ± std, 32 test items/seed)

| Dataset | best ML (n=50) | keyword rules | Jev | Clef | Clef-flash |
| --- | --- | --- | --- | --- | --- |
| support-tickets | 85.4±6.4% | 74.0±5.3% | 96.9±2.6% | **97.9±1.5%** | 96.9±0.0% |
| tickets-hard | 82.3±5.9% | 69.8±1.5% | **97.9±1.5%** | 92.7±1.5% | 93.8±0.0% |
| reviews | 78.1±2.6% | **100.0±0.0%**¹ | **99.0±1.5%** | 94.8±7.4% | 84.4±9.2% |
| topics | 96.9±4.4% | 83.3±12.6% | 97.9±1.5% | 94.8±3.9% | **99.0±1.5%** |

¹ Keyword rules know the generator's sentiment vocabulary — leakage, see caveat 4.

Full learning curve (support-tickets): n=10 → ML ~35%, n=25 → ~56–59%,
n=50 → 83–85%. The decision-model gap is widest exactly where labels are
scarce. Macro-F1 tells the same story as accuracy everywhere (data in
`bench-results.json`).

### Score tasks (urgency 0–3 / rating 0–4, exact match)

| Dataset | best ML (n=50) | keyword rules | Jev | Clef | Clef-flash |
| --- | --- | --- | --- | --- | --- |
| support-tickets urgency | 78.1±7.7% | **89.6±1.5%** | **79.2±1.5%** | 75.0±6.8% | 75.0±2.6% |
| tickets-hard urgency | **83.3±2.9%** | 35.4±3.9% | 56.3±6.8% | **62.5±2.6%** | 57.3±5.9% |
| reviews rating | 50.0±14.2% | 28.1±6.8% | 56.3±13.5% | 55.2±5.3% | **58.3±12.8%** |

MAE (same order): explicit urgency 0.21–0.30; hard urgency 0.26 (logreg)
to 0.51 (Jev); rating 0.45–0.70. Ordinal error is small even where exact
match is poor — every system is usually within half a level.

### Distribution shift (`exp-shift.mjs`)

Train on original phrasing (n=50 gold), test on a paraphrased twin with
identical label logic; in-distribution control uses the original generator
with a different seed. 48 test items.

| System | choice in→shift (Δ) | score in→shift (Δ) |
| --- | --- | --- |
| keyword-rules | 66.7→27.1% (**−39.6**) | 85.4→54.2% (−31.3) |
| naive-bayes n=50 | 75.0→29.2% (−45.8) | 62.5→54.2% (−8.3) |
| logreg-tfidf n=50 | 85.4→25.0% (**−60.4**) | 70.8→56.3% (−14.6) |
| jev-zero-shot | 95.8→91.7% (−4.2) | 68.8→72.9% (+4.2) |
| clef | 89.6→83.3% (−6.3) | 66.7→75.0% (+8.3) |
| clef-flash | 93.8→83.3% (−10.4) | 68.8→68.8% (0.0) |

Clef-flash — the smallest, fastest model — is also the most
shift-sensitive of the three, but still ~5× more robust than any trained
system.

### Calibration & routing (`exp-calibration.mjs`)

80 test items, seed 42. cov@90 = fraction of items auto-acceptable while
keeping ≥90% accuracy on the accepted set; AURC = area under risk-coverage
(lower better; a perfect system gets 0).

| Dataset | System (signal) | choice: acc / ECE / cov@95 | score: acc / ECE / cov@90 |
| --- | --- | --- | --- |
| support-tickets | jev (confidence) | 96.3 / 0.040 / 100% | 75.0 / 0.069 / 66% |
| support-tickets | clef (max-prob) | 96.3 / 0.089 / 100% | 75.0 / 0.079 / 43% |
| support-tickets | clef (confidence) | 96.3 / 0.223 / 100% | 75.0 / 0.173 / 43% |
| support-tickets | clef-flash (confidence) | 97.5 / 0.277 / 100% | 70.0 / 0.193 / 30% |
| support-tickets | logreg (max softmax) | 85.0 / 0.303 / 84% | 76.3 / 0.137 / 57% |
| tickets-hard | jev (confidence) | 97.5 / 0.052 / 100% | 53.8 / 0.219 / 3% |
| tickets-hard | clef (max-prob) | 95.0 / 0.079 / 100% | 57.5 / 0.172 / 6% |
| tickets-hard | clef (confidence) | 95.0 / 0.210 / 100% | 57.5 / 0.169 / 6% |
| tickets-hard | clef-flash (confidence) | 97.5 / 0.274 / 100% | 55.0 / 0.202 / 5% |
| tickets-hard | logreg (max softmax) | 78.8 / 0.250 / 69% | 78.8 / 0.172 / **80%** |

- Route on Jev confidence or Clef's `probabilities` (not Clef's `confidence`).
- Logreg's raw softmax ranks well (low AURC) but is badly calibrated on
  choice — wrap in Platt/isotonic before thresholding.
- Where accuracy is near chance (hard urgency for all decision models),
  every semantic confidence signal fails; only the lexically-trained
  logreg can still route.

### Distillation (`exp-distill.mjs`)

1000-item pool labeled by each teacher (no human labels), distilled into
TF-IDF logreg, evaluated on 96 gold test items (3 seeds). Labeling cost:
Jev 9.6 ms/item, Clef-flash 32.5 ms/item, Clef 40.8 ms/item (concurrency 12).

| Deployment option | requests | choiceAcc | scoreAcc | scoreMAE |
| --- | --- | --- | --- | --- |
| logreg(50 gold) | 0 | 78.1% | **77.1%** | **0.29** |
| logreg(1000 jev-labels) | 1000 one-time | **99.0%** | 67.7% | 0.41 |
| logreg(1000 clef-labels) | 1000 one-time | 97.9% | 64.6% | 0.39 |
| logreg(1000 clef-flash-labels) | 1000 one-time | 96.9% | 56.3% | 0.47 |
| jev-zero-shot | 96 per batch | 96.9% | **78.1%** | **0.26** |
| clef-zero-shot | 96 per batch | **97.9%** | 75.0% | 0.27 |
| clef-flash-zero-shot | 96 per batch | 96.9% | 75.0% | 0.27 |

Teacher agreement with gold on the pool: Jev 97.8% choice / 69.8% score;
Clef 94.0% / 67.8%; Clef-flash 95.4% / 64.6%. Student accuracy orders with
teacher agreement on choice; on score, flash's weaker teacher drops the
student to 56.3%, and 50 gold labels stay competitive.

## Method

### Datasets

All corpora are seeded and reproducible (`datasets.mjs`; seeds 42/7/123,
plus 99 for shift, 500 for distillation). Labels are known by construction:
the generator picks labels first, then decorates text with vocabulary
consistent with them. Train/test draw from disjoint template picks.

| Dataset | Text | Choice task | Score task | Difficulty axis |
| --- | --- | --- | --- | --- |
| `support-tickets` | 1 sentence | category, 5 classes | urgency 0–3, explicit cues ("today") | baseline |
| `tickets-hard` | 1 sentence | same | urgency 0–3, oblique stakes only ("two teams paused work") | keyword brittleness |
| `reviews` | 1–3 sentences | sentiment, 3 classes | rating 0–4 | mixed/ambiguous class |
| `topics` | 2–3 sentences | team note topic, 6 classes | — | imbalance (8–25% priors) + distractor sentences |

### Systems

- **keyword-rules** — hand-written regexes per dataset (no learning)
- **naive-bayes** — multinomial NB, bag-of-words, pure JS
- **logreg-tfidf** — softmax regression on L2-normalized TF-IDF, pure JS
- **sk-\*** — sklearn cross-check (MultinomialNB / LogisticRegression /
  LinearSVC; `sklearn-results.txt`) validating the from-scratch JS baselines
- **jev** — Typesafe System One via `@typesafe-ai/sdk`, zero-shot
- **clef / clef-flash** — `@cf/cloudflare/clef[-flash]` on Workers AI via
  REST (`clef.mjs`). System One-compatible: identical dataset question
  objects go to both providers. Clef also returns per-option probabilities.

Metrics: accuracy, macro-F1, MAE for score tasks, training/inference time,
request counts.

## Caveats

1. **Synthetic, generator-vocabulary data.** Closed-vocabulary systems
   (TF-IDF, keyword rules) are near-ideal at memorizing a fixed vocabulary,
   so their numbers are upper bounds and likely lower bounds for the
   decision models. Read these as a controlled comparison of learning
   dynamics and robustness, not real-world accuracy estimates.
2. **Train/test are i.i.d. from the same generator** — no drift or new
   vocabulary over time. The shift experiment covers phrasing change only;
   real drift is worse for lexical systems.
3. **Small n, wide intervals.** 32 test items × 3 seeds; differences under
   ~5% are within noise. No significance tests.
4. **Keyword rules were written by the corpus author** — leakage in their
   favor (100% on `reviews` is the tell), and brittle: the `tickets-hard`
   collapse costs the same rules 54 points.
5. **No hyperparameter tuning for baselines** (fixed lr/epochs/L2, default
   sklearn settings, no ngram tuning). A tuned classic pipeline would close
   some gap. The JS implementations exist for same-process latency parity.
6. **Jev prompt sensitivity untested** — one prompt wording per task, no
   few-shot/context-augmented runs.
7. **Score tasks treat levels as nominal classes in the ML baselines**,
   discarding ordinal structure; Jev's `score` is ordinal-aware natively.
   MAE is reported to compensate.
8. **Calibration is measured only for choice+score confidence on two
   datasets, seed 42.** Reliability is dataset- and wording-specific;
   re-validate any threshold you deploy.
9. **Cost is request count only.** Clef is $0.24/M input tokens on Workers
   AI; Jev meters per Typesafe's terms. No $ accounting anywhere here.
10. **Latency is network-bound and single-host** (HTTPS round trips from
    one machine, concurrency 8–12). Training cost for ML (10–99 ms) recurs
    whenever labels change.
11. **Decision models are nondeterministic across runs**; JS baselines are
    deterministic given data. Averages blend model + sampling variability.
12. **Class imbalance is mild** (`topics` 8–25%); long-tail taxonomies not
    tested.
13. **JS Naive Bayes uses `indexOf` per token** — fine at this vocab size,
    not representative of production NB.
14. **Vendor latency claims conflict with our measurements.** Cloudflare's
    launch post: Clef median 209 ms vs Jev 524 ms (flash 39 ms). Here:
    Jev ~15–17 ms/item, Clef ~58–85 ms/item, flash ~56–78 ms/item. Jev runs
    on Typesafe's endpoint, Clef on Workers AI REST from the same host;
    region, SDK vs raw REST, and harness differences could each matter.
    Rerun in your own environment before capacity planning.
15. **Clef integration notes:** accepts System One question objects
    unchanged (including `null` criteria), returns full per-option
    probabilities (Jev's SDK does not), weights are Apache 2.0 — a
    self-hosted deployment could change the latency picture entirely.
16. **Cloudflare auth:** `clef.mjs` reads the wrangler OAuth token
    (`~/.config/.wrangler/config/default.toml`, `ai` scope) or
    `CLOUDFLARE_API_TOKEN`; account id auto-resolves via the API. Wrangler
    refreshes OAuth tokens on use — if a long run 401s, `npx wrangler
    login` and rerun.

## Bottom line

- No labels or shifting input distribution → a decision model, and it
  isn't close.
- Lexically stable signal + labels → trained linear model (or plain rules),
  free at inference, with retraining as maintenance.
- The strongest stack measured here: decision model labels a large synthetic
  pool once → distilled logreg serves the lexical core → decision model with
  calibrated confidence routing (Jev confidence / Clef max-prob) handles
  drift and hard cases.
- Pick the teacher per task: Jev for graded judgments, Clef/Clef-flash for
  category judgments; never assume one vendor's headline benchmark
  transfers to your distribution or your network.

## Reproduce

```
node bench.mjs            # main benchmark, all datasets × 3 seeds (Jev + Clef + Clef-flash)
python3 sklearn-check.py  # seed-42 sklearn cross-check
node exp-shift.mjs        # paraphrase-shift test
node exp-calibration.mjs  # ECE + risk-coverage
node exp-distill.mjs      # 1000-item distillation (Clef + Clef-flash teachers)
```

Raw outputs: `bench-results.json`, `exp-*-results.json`, `corpus.json`,
`sklearn-results.txt`.

Clef auth: wrangler login with the `ai` scope, or set `CLOUDFLARE_API_TOKEN`
and `CLOUDFLARE_ACCOUNT_ID` explicitly.
