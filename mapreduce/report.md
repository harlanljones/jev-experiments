# Semantic MapReduce benchmark — results

Fixture: fixtures/corpus.json (90 docs, 30/tier, labels planted by construction, seed 20260918).
Protocol: 3 seeds x full pipeline (map n=90 parallel requests + 1 batched reduce); A1/A2 single seed.

## Per-doc quality (mean ± sd across seeds)

| Tier | Jev topic acc | B0 keywords | A1 topic-only Jev | Jev sentiment MAE | B1 lexicon MAE |
|------|--------------|-------------|-------------------|-------------------|----------------|
| T1 | 0.867 ± 0.000 | 0.93 | 0.87 | 0.154 ± 0.001 | 0.33 |
| T2 | 0.933 ± 0.000 | 0.93 | 0.93 | 0.228 ± 0.002 | 0.37 |
| T3 | 0.967 ± 0.000 | 0.87 | 0.97 | 0.400 ± 0.001 | 0.67 |

Actionable noul (pooled over 3 seeds x 90 docs): Brier 0.05, AUC 0.99

## Reduce verdicts vs policy ground truth (per seed)

| Seed | act_now precision | act_now recall | verdict agreement | tp/fp/fn |
|------|-------------------|----------------|-------------------|----------|
| 0 | 1.00 | 1.00 | 0.80 | 3/0/0 |
| 1 | 1.00 | 1.00 | 0.80 | 3/0/0 |
| 2 | 1.00 | 1.00 | 0.80 | 3/0/0 |

## Ablation: batched vs per-group reduce (single seed)

Verdict agreement with policy truth (same map, same groups) — batched (1 request): 0.80, per-group (5 requests): 0.80
Per-group verdicts: {"support":"track","other":"track","pricing":"act_now","reliability":"act_now","performance":"act_now"}

## Latency (wall time, per seed)

| Seed | map_ms (90 parallel) | reduce_ms | total_ms | requests |
|------|----------------------|-----------|----------|----------|
| 0 | 2228 | 146 | 2374 | 91 |
| 1 | 404 | 169 | 573 | 91 |
| 2 | 285 | 177 | 462 | 91 |
