# Retrieval benchmark v1

`retrieval_queries.jsonl` is the Phase 1 development benchmark, authored before indexing.
It contains 90 queries: 27 exact references, 36 French paraphrases, 18 English queries and
9 cross-regulation queries. Each regulation has 27 single-regulation queries and appears
in 6 comparisons. All evidence is French; `language` describes the question, not the corpus.

Status: **AI-assisted draft, awaiting human relevance review**. Questions and positive labels
were authored against the existing local source text, with verbatim anchors checked by code.
Validation establishes source consistency, not legal expertise or exhaustive relevance judgments.
No retrieval accuracy, latency or model quality has been measured yet.

## Record format

Each UTF-8 JSONL line is one object. The schema lives in
`src/verilegal/evaluation/retrieval_dataset.py`.

| Field | Meaning |
|---|---|
| `id` | Stable query identifier, such as `rgpd-001` or `cross-001` |
| `query` | User question; this is what gets encoded and searched |
| `language` | `fr` or `en` |
| `category` | `exact_reference`, `fr_paraphrase`, `en_query` or `cross_regulation` |
| `regulations` | Search scope using registry keys: `rgpd`, `dora`, `aiact` |
| `relevant_chunk_ids` | Positive judgments at child-chunk level, not parent article IDs |
| `evidence` | One verbatim quote and source `text_hash` for every labeled chunk |

Only `query` and the independently specified `regulations` scope are inputs to retrieval.
Keep IDs, labels, quotes and hashes out of query encoding, prompts and reranking inputs.
Map scope keys to the corpus payload names using `SOURCES`; e.g. `aiact` maps to `AI Act`.
Evaluate every retrieval configuration on the same scopes and candidate corpus. Labels do not
justify fetching their exact chunk IDs as a retrieval strategy.

## Corpus and label rules

- Corpus: the committed French RGPD, DORA and AI Act snapshots identified by
  `data/raw/regulations/manifest.fr.json`. These are the stored original-act snapshots, not
  a claim about current consolidated legislation. AI Act annexes are not indexed.
- Chunking budget: **1,800 characters**, matching the existing processed files. IDs can change
  when the splitting threshold changes; freeze this budget before retrieval comparisons.
- Positive labels target article children. Recitals may be candidates, but this initial set
  has no recital-specific questions; it does not measure recital retrieval quality.
- Exact-reference questions specify a paragraph, point or narrow topic rather than asking
  for an entire multi-paragraph article. The quoted anchor identifies the target passage.
- Single-regulation rows have one positive chunk; comparisons have two, one from each act.
  Both sides are required for complete cross-regulation evidence. Returning only one is
  partial coverage, even though it can count as a first relevant hit for MRR.
- Labels are binary, selected positive judgments, not an exhaustive annotation of every
  related passage. Review other retrieved passages before treating them as irrelevant.
- Related questions can share evidence. Treat the entire v1 set as development data; it is
  not an independent held-out test set. Group by topic/source article if splitting later,
  and author a separate reviewed holdout before making generalization claims.
- Unsupported questions, follow-ups and answer quality belong in the later Q&A dataset;
  they are not mixed into this positive-only retrieval set.

## Validate and inspect

Run from the repository root:

```bash
uv run verilegal eval validate-retrieval
uv run pytest tests/unit/test_retrieval_dataset.py -v
```

Validation regenerates chunks in a temporary directory from the committed snapshots, verifies
their manifest hashes and structure, and checks the query schema, uniqueness, source scope,
retrievable roles, evidence hashes and exact quote matches. It does not need processed JSONL,
Qdrant, model downloads, GPU access or LLM calls. Missing or altered snapshots fail explicitly.
`--dataset path/to/file.jsonl` selects another dataset with the same schema.

Inspect a complete query and its evidence (if `jq` is available):

```bash
jq 'select(.id == "rgpd-001")' eval/retrieval_queries.jsonl
jq 'select(.chunk_id == "32016R0679_art5_p1_ptc")' data/processed/rgpd.fr.jsonl
```

For human review, compare each question and anchor with the full chunk and its parent article.
Confirm that the passage answers the actual question, respects qualifications/exceptions and
is in the intended scope; add other equally valid labels where appropriate. Cross-regulation
questions ask for separate evidence, not an assumption that the acts impose identical duties.
Record review changes and benchmark versions before tuning. Do not silently refresh stored hashes
after changing sources or chunking; review the affected labels first.

Later retrieval runs will report Recall@1/3/5, MRR@10 and nDCG@10 overall and by category/language/
regulation. Report cross-regulation coverage separately. Validation counts are dataset inventory,
not retrieval performance. `verilegal eval retrieval --ablation` is not implemented in this task.
