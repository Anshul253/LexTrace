# LexTrace — Evaluation Report

All figures below are copied from files in this repository and can be regenerated: `experiments/results/*.json`
(per-split summaries), `experiments/experiment_log.jsonl` (loop), `experiments/method_comparison.json`,
`experiments/error_analysis.json`, `experiments/test_runs.jsonl` (test ledger), `experiments/dataset_validation.json`.
`python -m lextrace tables` renders the tables. "n" is always the number of queries behind a figure.

## 1. Summary
LexTrace is a local clause-evidence search system over the 29 CUAD contracts of LegalBench-RAG-mini. For a question it
returns ranked passages with their contract and exact character offsets, or abstains. The baseline (500-character
windows, BM25 over all contracts, gate on the raw BM25 score) reaches **test recall@4 = 0.034** (n = 54). A train-only
self-improvement loop (16 evaluated iterations: 5 accepted, 11 rejected) selected contract routing, a MiniLM bi-encoder,
a cross-encoder reranker, clause-aware chunks and a normalised-BM25 abstention gate. The frozen final configuration
scores **test recall@4 = 0.583** (95% bootstrap CI 0.46–0.70), recall@1 0.443, recall@64 0.924, precision@4 0.106,
MRR 0.607, versus 0.652 recall@4 on train (n = 140). **Abstention remains the weak part:** on held-out data it
correctly refuses **4 of 11** unanswerable questions (baseline 1/11) and wrongly refuses **7 of 54** answerable ones.
No LLM is used. The test split was evaluated **twice in total — once for the baseline and once for the final
configuration**, both after the final configuration was frozen (commit `0aecedb`).

## 2. System design
**Pipeline.** question → validate → split off the named contract ("Consider the …; …") → route to the best-matching
contract (character 3–5-gram TF-IDF cosine between the named title and each contract's file name + first 2,000
characters) → score that contract's chunks with the configured retriever → optional cross-encoder rerank → drop
passages overlapping a higher-ranked one → ranked candidates → confidence signals → abstention gate → passages or an
explicit abstention. Chunks are `[start, end)` ranges over the untouched contract text, so
`document[start:end] == passage` holds by construction; it is asserted again before display.

| Decision | Final choice (train evidence) | Alternatives tried / considered |
|---|---|---|
| Contract selection | hard routing to the named contract; retrieve with the question part only | none (baseline: the party names pull every contract's cover page to rank 1); soft score boost and top-2 routing were in the catalogue but never reached |
| First stage | dense `all-MiniLM-L6-v2` | BM25, TF-IDF, `bge-small-en-v1.5`, hybrid RRF / weighted fusion (§6, §7) |
| Reranking | cross-encoder `ms-marco-MiniLM-L-6-v2` over the top 32 | none |
| Chunking | clause/heading-aware, ≤ 800 chars | fixed 500 / 1000, paragraph-aware ≤ 800, ±100-char passage widening |
| Query handling | the question as asked | concept-lexicon expansion (`configs/lexicons/legal_concepts.yaml`, written before any experiment) |
| Abstention | one gate: rank-1 BM25 score within the routed contract / summed idf of the question's terms ≥ 0.3007 | raw top score, dense cosine, query-term coverage, contract-match similarity, two-gate combinations |
| Loop proposer | rule catalogue keyed on the diagnosed failure type | grid/Bayesian search (more freedom to over-fit 140 queries), LLM proposer (cost, reproducibility) |
| Frameworks | rank-bm25, scikit-learn, sentence-transformers only | LangChain/LlamaIndex would hide the offsets and scores that must be explainable |

**Messy contracts.** Clause chunking splits at numbered headings ("12.", "12.3", "Section 4", "ARTICLE V", "(a)") and
keeps each numbered clause in its own chunk; over-long clauses fall back to word-boundary windows. Definitions and
cross-references ("as defined in Section 4.2") are **not** resolved (limitation, §9). Questions about contracts that
are not indexed are the job of the abstention gate, which handles them poorly (§5).

## 3. LLM use
No LLM is used anywhere at run time (retrieval, ranking, abstention or loop proposals). Reasons: the task is retrieval
and the measured non-LLM pipeline shows where it fails (vocabulary mismatch, missing abstention signals, §8), which can
be attacked more cheaply first; real contracts are confidential — a hosted model (e.g. Gemini) would send contract text
to a third party and need a privacy/legal review, while a local model (e.g. Qwen via Ollama) keeps data on-premise but
adds seconds per query on CPU on top of the reranker's ~1 s; and reproducing results would require caching every
output. **Because no LLM variant was run, I make no claim about whether one would help.** An LLM reranker or query
rewriter, evaluated against the final pipeline with cached outputs, is the first next step (§9). The code was written
with an AI coding assistant (README, "AI tool usage"); that assistant is not part of the system.

## 4. Evaluation setup
- **Benchmark:** LegalBench-RAG CUAD, mirror `awinml/legalbench-rag` @ `2b9c248b`; 30/30 downloaded files match the
  mirror's SHA256SUMS. The prescribed `mini()` selection gives **194 queries over 29 contracts** (1,211,773 characters;
  only these are indexed). **272/272** gold snippets satisfy `document[start:end] == answer`; no overlapping gold spans;
  no duplicate query strings.
- **Split:** the prescribed `assign_split`: **train 140 / test 54** benchmark queries. All 29 contracts appear in train.
- **Unanswerable set:** 40 hand-written queries in the repository's first commit (`8cb2850`), before any configuration
  was evaluated; SHA-256 `57a11890…` is locked in `data/unanswerable_queries.sha256` and checked on every load; never
  edited. Same split function: **train 29 / test 11**. Categories (train / test): clause absent from a named indexed
  contract 18 / 8, contract not in the corpus 7 / 1, outside what a contract records 4 / 2. Each query records keyword
  patterns that must have zero hits in its target contract (or the whole corpus); `lextrace validate` re-runs them
  (0 failures). The two shortest contracts were read in full; for the others the matched context of each pattern was
  read. Three drafts flagged by the check were rewritten before the freeze (one was actually answerable).
- **Retrieval metrics** (the LegalBench-RAG authors' code semantics; macro mean over answerable queries; k ∈ {1, 2, 4,
  8, 16, 32, 64}): precision@k = |returned ∩ gold| / |returned| characters (text from other contracts counts against
  precision); recall@k = |returned ∩ gold| / |gold|; nothing returned → 0. Returned passages never overlap, so the
  union-based implementation equals the authors' pairwise formula (unit-tested against a port). Also span-hit@k (any
  overlap with gold in the top k), document-hit@k and MRR (reciprocal rank of the first overlapping passage in the top
  64). *Ranking view* = candidates before the gate (all retrieval tables); *end-to-end* recall scores abstained
  answerable queries as 0.
- **Abstention metrics** (decision level): false-abstention rate = answerable queries returning nothing / answerable
  queries; correct-abstention rate = unanswerable queries returning nothing / unanswerable queries; false-answer rate =
  1 − correct-abstention rate. Wilson 95% intervals.
- **Threshold calibration (train only):** at most 10% of train answerable queries may be refused (budget declared before
  tuning); among thresholds within budget, the one refusing most train unanswerable queries (ties → the larger
  threshold). Thresholds are frozen into the YAML; test evaluation refuses an uncalibrated config.
- **Test usage:** `evaluate --split test` requires `--confirm-final` and appends to `experiments/test_runs.jsonl`. The
  ledger holds **2 entries** (baseline, final), run once each after the final config was committed. No test number was
  seen before that, and none was used for any decision.

## 5. Results
**Character-level retrieval, answerable queries (ranking view), recall / precision**

| k | train baseline | train final | test baseline | test final |
|---|---|---|---|---|
| 1 | 0.001 / 0.002 | 0.445 / 0.309 | 0.000 / 0.000 | **0.443** / 0.295 |
| 2 | 0.006 / 0.005 | 0.558 / 0.206 | 0.003 / 0.001 | 0.527 / 0.171 |
| 4 | 0.036 / 0.011 | 0.652 / 0.126 | 0.034 / 0.007 | **0.583** / 0.106 |
| 8 | 0.112 / 0.013 | 0.749 / 0.074 | 0.087 / 0.011 | 0.689 / 0.071 |
| 16 | 0.208 / 0.014 | 0.841 / 0.046 | 0.161 / 0.013 | 0.768 / 0.042 |
| 32 | 0.309 / 0.011 | 0.903 / 0.029 | 0.250 / 0.010 | 0.804 / 0.022 |
| 64 | 0.438 / 0.008 | 0.951 / 0.021 | 0.386 / 0.008 | **0.924** / 0.017 |
| MRR | 0.082 | 0.620 | 0.069 | 0.607 |
| span-hit@4 | 0.107 | 0.729 | 0.093 | 0.667 |
| end-to-end recall@4 | 0.033 | 0.602 | 0.034 | 0.526 |
| n | 140 | 140 | 54 | 54 |

Percentile bootstrap (B = 2000, seed 0) for final recall@4: train [0.581, 0.724], test [0.462, 0.704].

**Abstention**

| | train baseline | train final | test baseline | test final |
|---|---|---|---|---|
| false abstentions / answerable | 14/140 | 14/140 | 6/54 | 7/54 |
| correct abstentions / unanswerable | 7/29 | 14/29 | 1/11 | **4/11** |
| false answers / unanswerable | 22/29 | 15/29 | 10/11 | 7/11 |
| correct-abstention 95% CI (Wilson) | [0.12, 0.42] | [0.31, 0.66] | [0.02, 0.38] | [0.15, 0.65] |
| — clause absent from a named contract | 0/18 | 9/18 | 0/8 | 4/8 |
| — contract not in corpus | 7/7 | 4/7 | 1/1 | 0/1 |
| — outside what a contract records | 0/4 | 1/4 | 0/2 | 0/2 |
| gate (threshold fitted on train) | raw BM25 top ≥ 42.67 | bm25_norm ≥ 0.3007 | same as train | same as train |
| AUROC of the gate signal (answerable vs unanswerable) | — | 0.740 | — | 0.650 |
| p95 query latency, warm, CPU | 95 ms | 1,434 ms | 25 ms | 902 ms |

The baseline's 7/7 on "contract not in corpus" is an accident of its design: invented party names match nothing, so
the raw BM25 score is low. Routing always selects *some* contract, which is why the final configuration loses on that
category (§8, F12–F13). Latency is dominated by the cross-encoder (≈1 s for 32 passages on this CPU); the BM25 and
dense first stages take < 50 ms (method comparison, §6). Building the final index (2,442 chunks) took 13.7 s
including model loading, with chunk embeddings already cached in `artifacts/`.

**Train/test gap.** Final recall@4 drops from 0.652 (train) to 0.583 (test), −0.069; recall@1 is the same (0.445 vs
0.443) and MRR nearly so (0.620 vs 0.607). The test interval [0.46, 0.70] contains the train value, so with 54 test
queries the gap is within noise, but some optimism is expected: 16 candidates were compared on the same 140 train
queries, and the accepted gains are biased upwards. Train and test queries ask about the *same contracts*, so this
measures generalisation to new questions, not to new contracts. Abstention shows the expected optimism more clearly:
the gate was chosen on the 29 train unanswerables (14/29 = 0.48) and gets 4/11 = 0.36 on test, and the signal's AUROC
falls from 0.74 to 0.65. With 11 test unanswerables these figures cannot support firm conclusions.

## 6. Iteration history
**Objective (declared in `optimization/objective.py` before the loop ran):** primary metric = train recall@4 (ranking
view, n = 140): a reviewer reads a handful of passages, and missing the clause is the costly error in contract review.
**Guardrails** (candidate vs current accepted parent, thresholds re-calibrated for each at the same 10% budget):
G1 precision@4 ≥ 0.9 × parent; G2 MRR ≥ parent − 0.02; G3 correct abstentions ≥ parent − 2; G4 false-abstention rate
≤ budget + 0.02; G5 p95 latency ≤ 1.5 s. **Accept** if (A) Δrecall@4 ≥ +0.010, paired-bootstrap P(gain) ≥ 0.90 and
G1–G5 pass, or (B) retrieval unchanged (|Δ| < 0.005), correct abstentions ≥ parent + 2 and G1, G2, G4, G5 pass;
otherwise reject (ties are rejected, keeping the simpler parent). The trade-off: recall leads; abstention is protected
by G3 and improved by rule B, so a retrieval gain cannot be bought by silently answering more unanswerable questions.

**How the loop works** (`python -m lextrace improve`; `optimization/loop.py`): evaluate the parent on train →
categorise every train failure (`wrong_document`, `wrong_section`, `partial_span`, `false_abstention`, `false_answer`,
plus a diagnostic "rank-1 passage is the contract's cover page") → take the most frequent category that still has an
untried proposal in a pre-registered catalogue (`optimization/proposer.py`, 19 rules) → apply it → evaluate on train →
judge → log → repeat. Rules that cannot affect the current pipeline are skipped and logged (2 such skips). A proposal
rejected *only* by G3 despite a real retrieval gain is retried once after an abstention-gate change is accepted.
For pipelines whose rank-1 score is fusion-scaled (RRF, min-max), the raw-score gate is swapped for an absolute signal
and the swap is logged. Proposals come from rules, not from an LLM.

| # | change (proposer target) | R@4 | ΔR@4 | P(gain) | P@4 | MRR | correct abst. | decision and reason |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline | 0.036 | | | 0.011 | 0.082 | 7/29 | |
| 1 | hard routing, question-only query (cover-page dominance) | 0.317 | +0.281 | 1.00 | 0.067 | 0.366 | 8/29 | **accepted** |
| 2 | dense MiniLM first stage (wrong section) | 0.471 | +0.154 | 1.00 | 0.108 | 0.475 | 13/29 | **accepted** |
| 3 | hybrid BM25 + dense, RRF | 0.431 | −0.039 | 0.13 | 0.096 | 0.499 | 13/29 | rejected: worse recall, G1 |
| 4 | concept-lexicon expansion | 0.580 | +0.109 | 1.00 | 0.131 | 0.640 | 9/29 | rejected: G3 (−4 abstentions); queued for retry |
| 5 | cross-encoder rerank top 32 | 0.552 | +0.081 | 0.99 | 0.119 | 0.617 | 12/29 | **accepted** |
| 6 | paragraph chunks ≤ 800 (partial span) | 0.626 | +0.074 | 0.99 | 0.111 | 0.581 | 13/29 | rejected: G2 (MRR −0.036) |
| 7 | clause chunks ≤ 800 | 0.652 | +0.100 | 1.00 | 0.126 | 0.620 | 11/29 | **accepted** |
| 8 | TF-IDF first stage | 0.597 | −0.055 | 0.00 | 0.101 | 0.597 | 10/29 | rejected: worse, G1, G2 |
| 9 | bge-small embeddings | 0.654 | +0.002 | 0.55 | 0.120 | 0.629 | 11/29 | rejected: no gain |
| 10 | gate on normalised BM25 (false answers) | 0.652 | 0.000 | — | 0.126 | 0.620 | 14/29 | **accepted** (rule B, +3 abstentions) |
| 11 | lexicon expansion, retry after gate change | 0.658 | +0.006 | 0.64 | 0.126 | 0.628 | 12/29 | rejected: gain below +0.010 |
| 12 | fixed 1000-char chunks | 0.679 | +0.027 | 0.92 | 0.113 | 0.651 | 13/29 | rejected: G1 (precision 0.126 → 0.113) |
| 13 | widen passages ±100 chars | 0.619 | −0.033 | 0.02 | 0.088 | 0.639 | 14/29 | rejected: worse, G1 |
| 14 | gates: contract match + normalised BM25 | 0.652 | 0.000 | — | 0.126 | 0.620 | 15/29 | rejected: +1 abstention < +2 |
| 15 | gates: contract match + dense cosine | 0.652 | 0.000 | — | 0.126 | 0.620 | 12/29 | rejected: −2 abstentions |
| 16 | gates: contract match + query coverage | 0.652 | 0.000 | — | 0.126 | 0.620 | 9/29 | rejected: −5, G3 |

Skipped without evaluation: light stemming and lexical hybrid weights (no lexical retriever in the parent). The
false-abstention rate is 0.100 in every row because each candidate is calibrated to the same budget. G5 (latency)
also failed in iterations 9, 11 and 14 (p95 1.5–1.7 s, close to the limit and noisy under machine load), but each of
those was already rejected on the primary rule, so latency decided nothing.

**What the history says.** The two largest wins were structural (routing +0.28, dense retrieval +0.15). Lexicon
expansion gave the biggest single retrieval gain on the dense parent (+0.109), but after the reranker and clause
chunks were in place its retry added only +0.006: the reranker already resolves most of the vocabulary mismatch it
targeted. Fixed 1000-char chunks raised recall (+0.027) at a precision cost the guardrail refused; this is a genuine
recall/precision trade-off, not a failure. Iteration 14 (contract-match gate) missed acceptance by one query.

**Method comparison** (`lextrace compare`, train, n = 140, all with the final routing, chunking and gate; rerank off
unless stated):

| method | R@1 | R@4 | R@16 | P@4 | MRR | p95 ms |
|---|---|---|---|---|---|---|
| BM25 | 0.223 | 0.403 | 0.688 | 0.073 | 0.388 | 12 |
| TF-IDF | 0.219 | 0.408 | 0.677 | 0.079 | 0.383 | 8 |
| dense (MiniLM) | 0.325 | 0.639 | 0.849 | 0.128 | 0.556 | 46 |
| hybrid BM25 + dense, RRF | 0.325 | 0.551 | 0.815 | 0.097 | 0.540 | 45 |
| hybrid BM25 + dense, weighted min-max | 0.326 | 0.562 | 0.820 | 0.101 | 0.534 | 38 |
| hybrid BM25 + TF-IDF + dense, RRF | 0.320 | 0.503 | 0.750 | 0.088 | 0.508 | 39 |
| hybrid BM25 + dense RRF + rerank | 0.452 | 0.663 | 0.840 | 0.123 | 0.624 | 1000 |
| *final (dense + rerank)* | *0.445* | *0.652* | *0.841* | *0.126* | *0.620* | *1,434* |

Dense retrieval beats both lexical methods by ~0.23 recall@4 once routing is in place. Fusing in BM25 *hurts* (lexical
rankings inside one contract are poor and RRF gives them equal say). The reranker's main contribution is at k = 1
(0.325 → 0.445); at k = 4 it adds only +0.013 over dense alone on clause chunks — at ~30× the latency. A deployment that
cares about latency should drop it first.

## 7. Error analysis
Cases are from the **final configuration on train** (`experiments/error_analysis.json`; no test data was inspected).
Train failure counts at k = 4 (n = 140): hit 75, wrong section 33, partial span 18, false abstention 14; and 15 false
answers on the 29 unanswerable questions. Only 4 of 65 misses still have the cover page at rank 1 (6%, down from 121 of 138 = 88% for the baseline,
before routing). Diagnoses are my reading of the text, not experiments.

| # | Cause | Query | What happened | Diagnosis |
|---|---|---|---|---|
| F1 | Concept worded differently | cuad-mini-006, *expiration date* (HPIL cooperation) | gold "The term of this Agreement shall be one (1) year…" (5604–5739) first appears at rank 8; rank 1 is "20. No Reliance" | "expiration date" shares neither words nor much meaning with a term clause; expiry questions are the most common type |
| F2 | same | cuad-mini-024, *expiration date* (Vitamin Shoppe) | gold "Unless terminated earlier… the term of this Agreement…" at rank 8; rank 1 is the *adjacent* termination-for-breach clause (34203) | right neighbourhood, wrong clause: the termination clause looks more like "expiration" to both models |
| F3 | Generic boilerplate outranks the clause | cuad-mini-009, *anti-assignment* (HPIL) | gold "…shall not be assignable by any Party" (91 chars) at rank 5; rank 1 is the entire-agreement clause | the gold sentence is very short inside a long boilerplate section; longer boilerplate chunks score higher |
| F4 | CUAD label loosely tied to the question | cuad-mini-019, *covenant not to sue* (Theravance) | gold is an employment-termination-on-reconstruction clause (27816–28397); rank 11 | the gold text does not read as a covenant not to sue; no retriever would rank it first (hypothesis) |
| F5 | Exception hidden in a sibling clause | cuad-mini-003, *exceptions to competitive restrictions* (Bicycle) | gold "save that nothing in this paragraph 13.2 shall prevent you…" at rank 16; rank 1 is the near-identical 14.4 carve-out | two carve-outs with the same wording; the model cannot tell which restriction the question means |
| F6 | Partial span: answer split across chunks | cuad-mini-039, *IP ownership* (Duos/Rubicon) | R@4 = 0.07; rank 1 is "5.1 ISA's Intellectual Property", gold is the Rubicon-created works sentence (14401) in the next sub-clause | the right section is found but the clause chunker cut 5.1/5.2 apart and the wrong sub-clause wins |
| F7 | Partial span: gold longer than the passage | cuad-mini-013, *expiration date* (Theravance) | R@4 = 0.57; rank 1 (4469–4653) is the second half of the gold sentence (4382–4705) | the chunk boundary falls inside the sentence; widening passages (iteration 13) cost too much precision |
| F8 | Partial span: second span missed | cuad-mini-010, *cap on liability* (HPIL) | R@4 = 0.5; the "sole and exclusive remedy" span (11380) is found at rank 4, the rest of the gold is not in the top 4 | multi-span answers need several passages; k = 4 is tight |
| F9 | False abstention on a correct hit | cuad-mini-026, *licenses granted* (Vitamin Shoppe) | rank 1 *is* the gold licence clause (38021), but bm25_norm = 0.291 < 0.301, so nothing is shown | the gate measures lexical overlap, while the passage was found by meaning; gate and ranker disagree |
| F10 | False abstention: terse fee heading | cuad-mini-073, *revenue/profit sharing* (Roanoke franchise) | rank 1 is the heading "3. FRANCHISE FEE." (17 chars), bm25_norm 0.083; gold (the percentage-share formula) at rank 2 | a bare heading became its own chunk; "revenue" vs "fee" gives little lexical overlap, so the gate refuses |
| F11 | False abstention after a near miss | cuad-mini-056, *revenue/profit sharing* (Premier) | gold "Royalty payments… 5% of the Fair Market Value" (11296); rank 1 is the adjacent paragraph (11679); bm25_norm 0.238 | "royalty" is never said in the question; the expansion lexicon covers it but was rejected (iteration 11) |
| F12 | False answer: invented contract | unans-028, *Bluewave Media / Harbor Lane Films distribution* | routed to the Igene JV contract with contract-match similarity **0.11** (real contracts score 0.34–0.47) and answered with a "Third Party Offer" clause | the routing similarity separates this case clearly, but the two-gate variant that uses it (iteration 14) gained only +1 abstention and was rejected |
| F13 | False answer: no contract named | unans-033, *Halcyon Digital / Orbit TV streaming licence* | no "Consider the …;" clause, so no routing; answered from the Glu amendment's opening paragraph | without a named contract the system searches everything and a generic governing-law question matches somewhere |
| F14 | False answer: clause absent, topic present | unans-004, *right of first refusal over future seasons* (Cano sponsorship) | answered with the recital "Sponsor desires to acquire the exclusive right to be the lead sponsor…" (bm25_norm 0.580) | the passage is on topic (sponsorship rights) but does not contain the asked-for right; evidence strength ≠ answer presence |
| F15 | False answer: out of scope | unans-038, *games the Jaguars won in 2018* (ARC sponsorship) | answered with the ticket-allocation clause (13539) | word overlap ("2018 season", the club) without the asked-for fact; no signal measures "does this passage answer the question" |

**Patterns.** Retrieval misses are mostly vocabulary/meaning mismatch inside the right contract (F1–F5) and chunk
boundaries (F6–F8). Abstention errors come from a single lexical signal that disagrees with a semantic ranker (F9–F11)
and from the lack of any "does the passage answer the question" signal (F14–F15). Unknown contracts (F12) have a good
signal that the loop's +2 rule narrowly declined. By question type (train, mean R@4, n in brackets): governing law
0.85 (17), expiration date 0.81 (19), licences granted 0.78 (10), cap on liability 0.77 (10), anti-assignment 0.75
(10), audit rights 0.71 (8) — but termination for convenience 0.40 (5), exceptions to competitive restrictions 0.26
(6) and exclusivity 0.10 (5). Frequent, formulaic clauses are found; rarer, negotiated ones (carve-outs, exclusivity
wording) are not. Individual misses of common types (F1, F2) still occur.

## 8. Limitations and next steps
- **Abstention is not reliable.** Held-out: 4/11 unanswerable questions refused, 7/54 answerable questions wrongly
  refused. An empty result must not be read as "no such clause", and a non-empty one not as "the clause exists".
  Next: combine the contract-match similarity (F12) with a semantic answerability signal — a cross-encoder or NLI
  model scoring whether the passage answers the question — trained on far more labelled negatives than 29.
- **Retrieval:** recall@4 = 0.58 on test means about 40% of gold text is outside the top four; recall@64 = 0.92.
  Next: overlapping or sub-clause chunks to fix boundary cuts (F6–F8), definition-aware indexing, an LLM reranker or
  query rewriter with cached outputs, evaluated against this pipeline with the same guardrails.
- **Routing** relies on the user naming the contract in the benchmark's phrasing; vague or wrong names search the wrong
  or all contracts (F13). A real product needs a contract picker (the web app has one) or metadata filters.
- **Statistical power:** 54 test and 11 unanswerable test queries give wide intervals; 16 candidates were compared on
  the same train queries; the abstention rule B and the retry rule were refined during development (two earlier loop
  runs with bugs are archived in `experiments/archive/` and were discarded, not used for selection). The test numbers
  are the honest figures.
- **Unanswerable labels** rely on keyword checks plus partial reading; a paraphrased clause could have been missed.
- **Documents:** plain-text SEC exhibits only; no PDFs/OCR, tables, amendments that modify other documents, or
  cross-reference resolution.
- **Latency:** the reranker puts p95 near 1–1.5 s on CPU; acceptable for review work, not for high-volume use.
- **Before legal users rely on it:** evaluation on JioStar's own agreements and question types (licensing, territory,
  windows, sublicensing to platforms), a much better abstention mechanism, mandatory human review of every passage,
  access control and audit logs, versioned ingestion, and a privacy review. Everything runs locally and no data leaves
  the machine, which is the right starting point for confidential contracts.

## 9. Reproducibility
Python 3.9.11, CPU only, versions in `requirements.txt`. Commands (see README):
`lextrace prepare` → `validate` → `evaluate --split train --config baseline --save-calibrated` →
`improve --start baseline --iterations 16` (≈ 30 min) → `evaluate --split train --config final` → `analyze` →
`compare --base final` → `evaluate --split test --config baseline_calibrated --tag baseline --confirm-final` →
`evaluate --split test --config final --confirm-final` → `tables`. All steps are deterministic on CPU; dense and
cross-encoder scores may differ in the last decimals across hardware or library versions.
