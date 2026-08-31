# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

By default, the simulator answers up to three clarification questions per
session. Adjust that local limit with `--max-questions`, for example:

```bash
python3 -m evaluator.local_evaluator --max-questions 2
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

### Optional LLM assist

The agent uses deterministic retrieval by default when no compatible model is
available. Select the optional LLM backend with `AGENT_LLM_PROVIDER` in the
repository-root `.env` file (which is ignored by Git). The default is a local
Hugging Face model:

```bash
AGENT_LLM_PROVIDER=huggingface  # or hf
# Optional; defaults to Qwen/Qwen2.5-1.5B-Instruct
AGENT_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

To use the restored Groq backend instead, set the provider and API key:

```bash
AGENT_LLM_PROVIDER=groq
GROQ_API_KEY=...
# Optional; defaults to openai/gpt-oss-20b
AGENT_LLM_MODEL=openai/gpt-oss-120b
```

Values already set in your shell take precedence over `.env`. Then run:

```bash
python3 -m evaluator.local_evaluator
```

To print each customer question, retrieval trace, and any LLM calls during a
local run, add `AGENT_DEBUG=1` to `.env`.

## Optional local semantic index

For dense or hybrid retrieval, build the local, resumable SQLite embedding store.
This uses Gemini Embedding 1 and intentionally avoids server-based vector
database infrastructure. Install the small client dependencies first; the
generated index is ignored by Git.

```bash
# Add GEMINI_API_KEY=... to .env
python3 -m pip install -r requirements.txt
python3 scripts/build_vector_store.py build
python3 scripts/build_vector_store.py query "lightweight waterproof hiking shoe"
```

The builder uses `ratelimit` plus exponential `backoff` and strictly caps calls
to Gemini at 100 requests per rolling minute. Its default is one product per
request, so the cap also maps directly to at most 100 product embeddings per
minute. A full 50,000-product build therefore takes at least about 8 hours and
20 minutes; it resumes safely after interruption. If Gemini responds with a
429 quota rejection, the script logs it, waits 30 seconds, and retries the
same unsaved batch.

Use the dense results as a candidate source alongside the starter's lexical
FTS/BM25 retrieval; embeddings are not a replacement for
exact category, brand, color, size, or budget filters.

### Offline local embeddings

The agent also supports an offline SentenceTransformers index using the
retrieval-trained `BAAI/bge-small-en-v1.5` model. Run the following setup
commands once from the repository root when the model and catalog index are
not included in the checkout:

```bash
python -m pip install -r requirements.txt
python -m scripts.build_local_embeddings \
  --model BAAI/bge-small-en-v1.5 \
  --model-dir data/bge-small-en-v1.5 \
  --output data/catalog_bge_embeddings.npz \
  --batch-size 128
```

The first command installs the local embedding dependencies. The second
downloads and saves the BGE model, then embeds the catalog into the local
index. It requires internet access on the first run; subsequent evaluation
runs use the saved local model and index:

```bash
python -m evaluator.local_evaluator
```

If `data/catalog_bge_embeddings.npz` is committed but the model directory is
not, download only the model after installing dependencies:

```bash
python -m pip install -r requirements.txt
hf download BAAI/bge-small-en-v1.5 \
  --local-dir data/bge-small-en-v1.5
python -m evaluator.local_evaluator
```

If both `data/catalog_bge_embeddings.npz` and
`data/bge-small-en-v1.5/` are committed with Git LFS, retrieve the large
files and run the evaluator:

```bash
git lfs pull
python -m pip install -r requirements.txt
python -m evaluator.local_evaluator
```

The build stores both `data/catalog_bge_embeddings.npz` and a local copy of the
embedding model under `data/bge-small-en-v1.5/`. Runtime scoring loads both
locally and does not contact Hugging Face. Remove or override
`LOCAL_EMBEDDING_INDEX` to run the lexical-only fallback.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Agent Interface

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

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer may reimburse model costs through prizes instead of issuing API keys.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Participant release checklist: `docs/participant_release_checklist.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
