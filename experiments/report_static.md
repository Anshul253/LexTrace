## 2. System design

**Pipeline.** question → validate → split off the named contract ("Consider the …; …") → route to the best-matching
contract (character 3–5-gram TF-IDF cosine between the named title and each contract's file name + first 2,000
characters) → score that contract's chunks with the configured retriever → drop passages that overlap a higher-ranked
one → ranked candidates → confidence signals → abstention gate(s) → passages or an explicit abstention. Chunks are
`[start, end)` ranges over the untouched contract text, so `document[start:end] == passage` holds by construction and
is asserted again before display.

| Decision | Chosen (by train evidence) | Alternatives considered |
|---|---|---|
| Contract selection | hard routing to the named contract; retrieve with the question part only | no routing (baseline: party names pull every contract's cover page to rank 1); soft score boost; top-2 routing |
| First-stage retrieval | see §6 — chosen by the loop | BM25, TF-IDF, dense (MiniLM, bge-small), hybrid RRF / weighted fusion |
| Reranking | evaluated, see §6 | cross-encoder `ms-marco-MiniLM-L-6-v2` over the top 32 |
| Chunking | see §6 | fixed 500 / 1000, paragraph-aware ≤ 800, clause/heading-aware ≤ 800, passage widening |
| Query handling | evaluated: concept-lexicon expansion | LLM query rewriting (not used, §4) |
| Abstention | gate(s) on calibrated confidence signals | raw top score, normalised BM25, dense cosine, query coverage, contract-match similarity, two-gate combinations |
| Improvement loop | rule catalogue keyed on diagnosed failure type | grid/Bayesian search (more freedom to over-fit 140 queries), LLM proposer (reproducibility, cost) |
| Frameworks | none beyond rank-bm25, scikit-learn, sentence-transformers | LangChain/LlamaIndex: would hide offsets and scoring that must be explainable |

**Legal-document handling.** Defined terms and cross-references are not resolved (limitation); headings are used by
the clause chunker; long clauses fall back to word-boundary windows; questions about contracts that are not indexed
are handled by the contract-match signal (§5, abstention).

## 3. LLM use
No LLM is used anywhere at run time (retrieval, ranking, abstention or loop proposals). Reasons: (1) the task is
retrieval, and the measured non-LLM pipeline already shows where it fails (vocabulary mismatch, §8), which can be
attacked with cheaper means first; (2) real JioStar contracts are confidential — a hosted model (e.g. Gemini) would
send contract text to a third party and needs a privacy/legal review, while a local model (e.g. Qwen via Ollama)
keeps data on-premise but adds seconds of latency per query on CPU; (3) reproducibility would require caching every
output. Because no LLM variant was run, **I make no claim about whether one would help**; an LLM reranker or query
rewriter evaluated against the final pipeline with cached outputs is the first next step (§9). The code itself was
written with an AI coding assistant (README, "AI tool usage").

## 4. Evaluation setup
- **Benchmark:** LegalBench-RAG CUAD, mirror `awinml/legalbench-rag` @ `2b9c248b`, 30/30 downloaded files verified
  against the mirror's SHA256SUMS. The prescribed `mini()` selection yields **194 queries over 29 contracts**
  (1,211,773 characters; only these are indexed). Offset check: **272/272** gold snippets satisfy
  `document[start:end] == answer`; no overlapping gold spans; no duplicate query strings.
- **Split:** the prescribed `assign_split`: **train 140 / test 54** benchmark queries.
- **Unanswerable set:** 40 hand-written queries, committed in the repository's first commit (`8cb2850`) before any
  configuration was evaluated, SHA-256 `57a11890…` locked and checked on every load; never edited. Same split
  function: **train 29 / test 11**. Categories (train/test): clause absent from a named indexed contract 18/8,
  contract not in the corpus 7/1, outside what a contract records 4/2. Each query stores keyword patterns that must
  have zero hits in its target contract (or the whole corpus); `lextrace validate` re-runs them (0 failures). Three
  drafts that the check flagged were rewritten before the freeze (one turned out to be answerable: the ALCO agency
  agreement does mention sublicensing software).
- **Retrieval metrics** (authors' definition, macro mean over answerable queries, k ∈ {1,2,4,8,16,32,64}):
  precision@k = |returned ∩ gold| / |returned| characters (text from other contracts counts against precision),
  recall@k = |returned ∩ gold| / |gold|; nothing returned → 0. Returned passages never overlap, so the union-based
  implementation equals the authors' pairwise formula (unit-tested). Also span-hit@k (any overlap with gold in the top
  k), document-hit@k, MRR (first overlapping rank in the top 64).
  *Ranking view* = candidates before the gate (all retrieval tables); *end-to-end* = abstained answerable queries
  score 0.
- **Abstention metrics** (decision level): false-abstention rate = answerable queries returning nothing / answerable
  queries; correct-abstention rate = unanswerable queries returning nothing / unanswerable queries; false-answer rate
  = 1 − correct-abstention rate. Wilson 95% intervals. Thresholds are fitted on train only (≤ 10% of train
  answerable queries refused; among those, most train unanswerables refused) and frozen into the config; test
  evaluation refuses uncalibrated configs.
- **Test usage:** test evaluation requires `--confirm-final` and appends to `experiments/test_runs.jsonl`.
