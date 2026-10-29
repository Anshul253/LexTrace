## Retrieval (ranking view)

| k | train baseline R / P | train final R / P | test baseline R / P | test final R / P |
|---|---|---|---|---|
| 1 | 0.001 / 0.002 | 0.445 / 0.309 | 0.000 / 0.000 | 0.443 / 0.295 |
| 2 | 0.006 / 0.005 | 0.558 / 0.206 | 0.003 / 0.001 | 0.527 / 0.171 |
| 4 | 0.036 / 0.011 | 0.652 / 0.126 | 0.034 / 0.007 | 0.583 / 0.106 |
| 8 | 0.112 / 0.013 | 0.749 / 0.074 | 0.087 / 0.011 | 0.689 / 0.071 |
| 16 | 0.208 / 0.014 | 0.841 / 0.046 | 0.161 / 0.013 | 0.768 / 0.042 |
| 32 | 0.309 / 0.011 | 0.903 / 0.029 | 0.250 / 0.010 | 0.804 / 0.022 |
| 64 | 0.438 / 0.008 | 0.951 / 0.021 | 0.386 / 0.008 | 0.924 / 0.017 |
| MRR | 0.082 | 0.620 | 0.069 | 0.607 |
| span-hit@4 | 0.107 | 0.729 | 0.093 | 0.667 |
| e2e R@4 | 0.033 | 0.602 | 0.034 | 0.526 |
| n answerable | 140 | 140 | 54 | 54 |

## Abstention

| | train baseline | train final | test baseline | test final |
|---|---|---|---|---|
| false abstentions / answerable | 14/140 | 14/140 | 6/54 | 7/54 |
| correct abstentions / unanswerable | 7/29 | 14/29 | 1/11 | 4/11 |
| false answers / unanswerable | 22/29 | 15/29 | 10/11 | 7/11 |
| correct-abstention 95% CI | [0.12, 0.42] | [0.31, 0.66] | [0.02, 0.38] | [0.15, 0.65] |
| gates (thresholds fitted on train) | top_score >= 42.67 | bm25_norm >= 0.3007 | top_score >= 42.67 | bm25_norm >= 0.3007 |
| p95 latency (ms) | 95.4 | 1433.6 | 25.0 | 902.3 |
| correct abstention: absent_clause | 0/18 | 9/18 | 0/8 | 4/8 |
| correct abstention: contract_not_in_corpus | 7/7 | 4/7 | 1/1 | 0/1 |
| correct abstention: out_of_scope | 0/4 | 1/4 | 0/2 | 0/2 |

## Method comparison (train)

| method (train) | R@1 | R@4 | R@16 | P@4 | MRR | correct abst. | p95 ms |
|---|---|---|---|---|---|---|---|
| bm25 | 0.223 | 0.403 | 0.688 | 0.073 | 0.388 | 14/29 | 12 |
| tfidf | 0.219 | 0.408 | 0.677 | 0.079 | 0.383 | 14/29 | 8 |
| dense | 0.325 | 0.639 | 0.849 | 0.128 | 0.556 | 14/29 | 46 |
| hybrid_bm25_dense_rrf | 0.325 | 0.551 | 0.815 | 0.097 | 0.540 | 14/29 | 45 |
| hybrid_bm25_dense_weighted | 0.326 | 0.562 | 0.820 | 0.101 | 0.534 | 14/29 | 38 |
| hybrid_bm25_tfidf_dense_rrf | 0.320 | 0.503 | 0.750 | 0.088 | 0.508 | 14/29 | 39 |
| hybrid_bm25_dense_rrf+rerank | 0.452 | 0.663 | 0.840 | 0.123 | 0.624 | 14/29 | 1000 |

## Iterations

| # | change | R@4 | dR@4 | P(gain) | P@4 | MRR | correct abst. | FA rate | decision | explanation |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline | 0.036 |  |  | 0.011 | 0.082 | 7/29 | 0.100 | **baseline** |  |
| 1 | route_hard | 0.317 | 0.281 | 1.00 | 0.067 | 0.366 | 8/29 | 0.100 | **accepted** | recall@4 0.036 -> 0.317 (+0.281, P(gain)=1.00); all guardrails pass |
| 2 | dense_minilm | 0.471 | 0.154 | 1.00 | 0.108 | 0.475 | 13/29 | 0.100 | **accepted** | recall@4 0.317 -> 0.471 (+0.154, P(gain)=1.00); all guardrails pass |
| 3 | hybrid_bm25_dense | 0.431 | -0.039 | 0.13 | 0.096 | 0.499 | 13/29 | 0.100 | **rejected** | recall@4 change -0.039 below +0.010; guardrail(s) failed: G1_precision |
| 4 | expansion_lexicon | 0.580 | 0.109 | 1.00 | 0.131 | 0.640 | 9/29 | 0.100 | **rejected** | guardrail(s) failed: G3_correct_abstention |
| 5 | rerank_cross_encoder | 0.552 | 0.081 | 0.99 | 0.119 | 0.617 | 12/29 | 0.100 | **accepted** | recall@4 0.471 -> 0.552 (+0.081, P(gain)=0.99); all guardrails pass |
| 6 | chunks_paragraph_800 | 0.626 | 0.074 | 0.99 | 0.111 | 0.581 | 13/29 | 0.100 | **rejected** | guardrail(s) failed: G2_mrr |
| 7 | chunks_clause_800 | 0.652 | 0.100 | 1.00 | 0.126 | 0.620 | 11/29 | 0.100 | **accepted** | recall@4 0.552 -> 0.652 (+0.100, P(gain)=1.00); all guardrails pass |
| 8 | tfidf | 0.597 | -0.055 | 0.00 | 0.101 | 0.597 | 10/29 | 0.100 | **rejected** | recall@4 change -0.055 below +0.010; guardrail(s) failed: G1_precision, G2_mrr |
| 9 | dense_bge_small | 0.654 | 0.002 | 0.55 | 0.120 | 0.629 | 11/29 | 0.100 | **rejected** | recall@4 change +0.002 below +0.010 and abstention gain +0 below +2; guardrail(s) failed: G5_latency |
| 10 | gate_bm25_norm | 0.652 | 0.000 | 0.00 | 0.126 | 0.620 | 14/29 | 0.100 | **accepted** | retrieval unchanged (delta +0.000); correct abstentions 11 -> 14; guardrails pass |
| 11 | expansion_lexicon | 0.658 | 0.006 | 0.64 | 0.126 | 0.628 | 12/29 | 0.100 | **rejected** | recall@4 change +0.006 below +0.010; guardrail(s) failed: G5_latency |
| 12 | chunks_fixed_1000 | 0.679 | 0.027 | 0.92 | 0.113 | 0.651 | 13/29 | 0.100 | **rejected** | guardrail(s) failed: G1_precision |
| 13 | expand_100 | 0.619 | -0.033 | 0.02 | 0.088 | 0.639 | 14/29 | 0.100 | **rejected** | recall@4 change -0.033 below +0.010; guardrail(s) failed: G1_precision |
| 14 | gate_route_and_norm | 0.652 | 0.000 | 0.00 | 0.126 | 0.620 | 15/29 | 0.100 | **rejected** | recall@4 change +0.000 below +0.010 and abstention gain +1 below +2; guardrail(s) failed: G5_latency |
| 15 | gate_route_and_dense | 0.652 | 0.000 | 0.00 | 0.126 | 0.620 | 12/29 | 0.100 | **rejected** | recall@4 change +0.000 below +0.010 and abstention gain -2 below +2 |
| 16 | gate_route_and_coverage | 0.652 | 0.000 | 0.00 | 0.126 | 0.620 | 9/29 | 0.100 | **rejected** | recall@4 change +0.000 below +0.010 and abstention gain -5 below +2; guardrail(s) failed: G3_correct_abstention |

Skipped without evaluation (no effect on the current parent): stem_light (lexical settings have no effect: no lexical retriever or lexical gate is used); hybrid_weight_lexical (hybrid settings have no effect: the pipeline is not hybrid)

Test ledger entries: 2
