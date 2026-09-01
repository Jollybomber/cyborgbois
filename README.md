# CyborgBoiS: Context-Aware Shopping Copilot

TechJam Conversational E-Commerce Search Challenge submission. An AI shopping
agent that asks useful follow-up questions and recommends the customer's
hidden target product within at most 10 turns.

## Project Overview

Real shoppers rarely start with the right keywords, and their intent evolves
across a conversation — they add constraints, drop others, or change their
mind entirely. CyborgBoiS converts a multi-turn conversation into an active
search plan instead of treating every message as an isolated query.

On every turn, the agent:

1. appends the message to session history and updates active structured facts;
2. removes superseded soft facts when the customer changes intent;
3. optionally asks Groq or Anthropic for a strict JSON search plan;
4. generates exact-phrase, grouped FTS/BM25, broad lexical, and BGE dense
   candidate routes;
5. fuses the candidate union using route rank, category and constraint
   coverage, exclusions, price, and a small quality/profile signal;
6. optionally asks the LLM to rerank the short fused list; and
7. returns the current Top 10 while independently choosing at most one useful
   clarification from candidate-pool information gain.

Clarification and recommendation are deliberately concurrent: a response can
return ten recommendations and ask one question. The agent never reads
`public_set.jsonl`, labels, intent cards, ground truth, or target identifiers,
and works fully offline via a deterministic fallback — an optional LLM
planner and reranker are used only when credentials are available, and any
model, network, or JSON failure falls back to the deterministic path.

### What you receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry`
  category of Amazon Reviews 2023.
- 200 labeled public sessions for local development (the organizer keeps 800
  additional sessions private for final evaluation).
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

### Agent interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`,
`brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See
`docs/agent_api_contract.json`.

### Technical metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the
  team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

Only exact `parent_asin` equality produces a hit. Core metrics are also
reported by scenario.

| Metric | Weak BM25 baseline | CyborgBoiS |
|---|---:|---:|
| Hit Rate@10 | 0.125 | **0.840** |
| MRR | 0.068034 | **0.542938** |
| MTTC | 9.810 | **3.855** |
| Technical Score | 0.106710 | **0.725781** |

## Setup and Installation

Python 3.10 or later is recommended. The deterministic lexical fallback uses
only the standard library.

```bash
git clone <this repository>
cd cyborgbois
python3 -m pip install -r requirements.txt
```

### Download the catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this
repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

### Optional: LLM planner and reranker

The pipeline works without a model API. To enable the planner and semantic
reranker, set either provider in the ignored repository-root `.env` file:

```bash
# Preferred when both keys exist
GROQ_API_KEY=...
AGENT_LLM_MODEL=openai/gpt-oss-20b

# Or use Anthropic
# ANTHROPIC_API_KEY=...
# AGENT_LLM_MODEL=claude-haiku-4-5-20251001
```

Set `AGENT_LLM_RERANK=0` to retain LLM planning but disable the second model
call. Any model, network, or JSON failure falls back to the deterministic
plan and ranking path.

### Optional: local BGE embeddings

The agent also supports an offline SentenceTransformers index using the
retrieval-trained `BAAI/bge-small-en-v1.5` model.

```bash
python -m scripts.build_local_embeddings \
  --model BAAI/bge-small-en-v1.5 \
  --model-dir data/bge-small-en-v1.5 \
  --output data/catalog_bge_embeddings.npz \
  --batch-size 128
```

This downloads and saves the BGE model, then embeds the catalog into a local
index. It requires internet access on the first run; subsequent evaluation
runs use the saved local model and index. If the embeddings or model
directory are already committed with Git LFS, run `git lfs pull` instead of
rebuilding.

The build stores both `data/catalog_bge_embeddings.npz` and a local copy of
the embedding model under `data/bge-small-en-v1.5/`. Runtime scoring loads
both locally and does not contact Hugging Face. Remove or override
`LOCAL_EMBEDDING_INDEX` to run the lexical-only fallback.

### Optional: Gemini embedding experiment

A standalone script can build a resumable SQLite embedding store using
Gemini Embedding 1, avoiding server-based vector database infrastructure.
This is an experimentation utility; the runtime agent uses the local BGE
index above.

```bash
# Add GEMINI_API_KEY=... to .env
python3 scripts/build_vector_store.py build
python3 scripts/build_vector_store.py query "lightweight waterproof hiking shoe"
```

The builder uses `ratelimit` plus exponential `backoff` and strictly caps
calls to Gemini at 100 requests per rolling minute (one embedding per
request), so a full 50,000-product build takes at least about 8 hours and 20
minutes; it resumes safely after interruption.

## Reproduce the Results

Run the full evaluator to write per-session results and aggregate metrics to
`results.json`. Do not edit the evaluator or public labels when reporting a
local score.

```bash
python3 -m evaluator.local_evaluator
```

The documented `0.84` Hit Rate@10 result is the deterministic ablation with
no LLM and no dense index. To reproduce it exactly, unset model credentials
and point the optional index at a nonexistent file:

```bash
env -u GROQ_API_KEY -u ANTHROPIC_API_KEY \
  LOCAL_EMBEDDING_INDEX=data/disabled-index.npz \
  python3 -m evaluator.local_evaluator \
  --output results.json
```

Compare the aggregate metrics in `results.json` with
`docs/pipeline_results.json`. The included weak BM25 starter scores are in
`docs/baseline_results.json`.

Run the focused tests with:

```bash
python3 -m unittest discover -s tests -v
```

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage
their own credentials and must never commit API keys. Model choice, estimated
cost, token usage, and latency must be disclosed. Token usage is a
feasibility metric, not part of the core technical score.

## Limitations and Future Improvements

- Deterministic extraction cannot normalize every synonym, sizing convention,
  or semi-structured catalog feature.
- Candidate information gain uses lightweight catalog signals rather than a
  learned question-value model.
- Local dense encoding adds model startup, memory, and per-turn CPU cost and
  did not outperform the documented lexical ablation on the complete public
  set.
- Groq and Anthropic planning require external credentials and may add
  latency and cost, although the complete deterministic fallback requires
  neither.
- The 200-session public set is small, so tuning can still be distribution
  sensitive even though the runtime agent never reads labels or target IDs.

With more time, we would add a compact offline cross-encoder, field-specific
embeddings, learned question value, quantized inference, calibrated
stopping, and constraint-based explanations for each recommendation.

## Team Contributions

- **Evan** — Bootstrapped the project: initial repo scaffolding, competition data/docs, the evaluator, and the first version of the starter agent with intent routing and adaptive retrieval.
- **Gladwin** — Built the offline local embedding pipeline (`local_embeddings.py`, `scripts/build_local_embeddings.py`), integrated BGE dense retrieval into the agent, and generated the committed catalog embedding index.
- **Lincoln** — Implemented the Gemini vector store experiment (`scripts/build_vector_store.py`) and extended the agent's structured state-filling logic across turns.
- **Mingyang** — Wired up the deterministic context-planning pipeline end-to-end, added the test suite (`tests/test_agent.py`), and produced the pipeline benchmark results.
- **Waylon** — Coordinated the team, reviewed and tested the agent across sessions, and drove documentation and submission readiness.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
docs/pipeline_results.json        current deterministic benchmark
starter/agent.py                  context-aware hybrid shopping agent
local_embeddings.py               local BGE index and cosine retrieval
evaluator/local_evaluator.py      public-set simulator and scorer
tests/test_agent.py               focused pipeline and contract tests
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Competition specification: `docs/competition_specification.md`
- Agent API contract: `docs/agent_api_contract.json`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab,
UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core
leave-last-out split and joined to the frozen catalog.
