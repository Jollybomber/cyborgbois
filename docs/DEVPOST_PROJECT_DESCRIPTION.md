# CyborgBoiS: Context-Aware Shopping Copilot

> Devpost-ready project description. Replace every `TODO` item before final
> submission, then copy the relevant sections into Devpost.

## Submission links

- Public repository: <https://github.com/Jollybomber/cyborgbois>
- Demo video: **TODO — add the public YouTube URL**
- Team name: **CyborgBoiS**
- Team members: **TODO — add names and Devpost handles**

## Tagline

An intent-aware conversational shopping agent that remembers evolving customer
needs, asks high-value questions, and combines exact, lexical, and semantic
retrieval to return a better Top 10 in fewer turns.

## Inspiration and problem

Traditional e-commerce search works best when shoppers already know the right
keywords. Real customers rarely start that way. Someone may begin with “I need
shoes for a trip,” later reveal that they expect long walks, then add a budget
or replace an earlier preference entirely. A useful shopping assistant must
understand this evolving intent instead of treating every message as an
isolated query.

The TechJam challenge makes that problem measurable. An agent must navigate a
50,000-product catalog, support Buying, Browsing, Intent Override, and Boundary
conversations, and place the hidden purchased product in the first ten results
within at most ten turns. Success depends on recall, ranking precision, and
conversational efficiency—not merely generating fluent text.

## What we built

CyborgBoiS is a stateful shopping copilot that converts a multi-turn
conversation into an active search plan. It remembers valid requirements,
separates hard constraints from soft preferences, removes superseded context
after an intent change, and preserves explicit exclusions and budgets.

On every turn, the agent can both ask one useful clarification and return its
current best ten recommendations. This is important because clarification and
retrieval are complementary: shoppers should receive useful options while the
agent continues reducing uncertainty.

The system works fully offline through a deterministic fallback. When model
credentials are available, an optional LLM planner distils the active context
into validated JSON, and an optional LLM reranker improves the short fused
candidate list. Network or model failures never prevent the offline pipeline
from responding.

## How it works

### 1. Stateful conversational memory

Each session stores the conversation history, category, material, color, size,
style, brand, use case, features, budget, exclusions, declined attributes, and
anonymized profile preferences. Answers containing multiple requirements are
kept as separate facts instead of being flattened into a lossy summary.

Intent overrides selectively remove superseded soft preferences while
preserving independently stated hard constraints and the product category.
This prevents both stale-context retrieval and over-aggressive memory resets.

### 2. Intent routing and search planning

The agent identifies whether the customer is Buying with explicit constraints
or Browsing more openly. A deterministic planner always produces a semantic
query, hard constraints, soft preferences, exclusions, and a clarification
decision.

If Groq or Anthropic is configured, the conversation and active state are sent
to an LLM that returns the same strict JSON schema. All fields are validated,
and malformed or unavailable model responses fall back automatically.

### 3. Multi-route retrieval

The frozen catalog is indexed in memory through SQLite FTS5 and an optional
local BGE embedding matrix. The agent retrieves a recall-oriented candidate
pool from several complementary routes:

- exact phrases for highly specific product details;
- grouped precision queries that combine category and active constraints;
- recent-constraint precision queries;
- balanced category-plus-preference BM25 retrieval;
- broad lexical retrieval for recall; and
- dense semantic retrieval using `BAAI/bge-small-en-v1.5`.

All candidate identifiers originate from the frozen catalog.

### 4. Constraint-aware fusion and reranking

Candidates are deduplicated and reranked using route position, exact phrase
evidence, category match, hard-constraint coverage, soft-preference coverage,
budget compliance, exclusion penalties, and small rating/profile signals.

Dense retrieval is deliberately recall-only. Cosine similarity can introduce
a product into the candidate union, but it cannot displace a product with
stronger exact catalog evidence merely because the score scales differ.

When enabled, the LLM reranks only the short fused list and is restricted to
the supplied catalog identifiers.

### 5. High-value clarification

The question policy is not tied to public labels or a fixed evaluator-specific
order. It examines the live candidate pool and estimates how much each
attribute would divide the remaining products. The highest-information
relevant attribute that has not already been answered or declined is selected.

The agent still returns its current Top 10 while asking that question, reducing
the mean number of turns needed to find the target.

## How the solution addresses the challenge

### Core architecture: intent routing and hybrid retrieval

- Buying and Browsing context is tracked explicitly.
- Exact, BM25, broad lexical, and dense semantic routes improve recall across
  both highly constrained and open-ended requests.
- Constraint-aware fusion protects exact requirements while retaining semantic
  discovery.

### Dialog strategy: multi-turn scenario evolution

- Structured session memory accumulates useful information across turns.
- Intent overrides erase obsolete preferences without forgetting valid hard
  facts.
- Candidate information gain drives proactive clarification.
- Recommendations and clarification happen concurrently.

### Self-evolution: dynamic context programming

- The active plan is rebuilt every turn from the latest valid context.
- Retrieval routes and query specificity change as confidence and constraints
  evolve.
- Optional LLM planning provides semantic interpretation while deterministic
  logic guarantees reproducibility and offline operation.

### Product and efficiency metrics

The unmodified public evaluator measures exact `parent_asin` hits. Our
generalization-safe deterministic ablation—without an LLM or dense retrieval—
achieved:

| Metric | Weak BM25 baseline | CyborgBoiS |
|---|---:|---:|
| Hit Rate@10 | 0.125 | **0.840** |
| MRR | 0.068034 | **0.542938** |
| MTTC | 9.810 | **3.855** |
| Technical Score | 0.106710 | **0.725781** |

This is a 6.7× improvement in Hit Rate@10, an approximately 8× improvement in
MRR, and a 60.7% reduction in MTTC relative to the supplied weak baseline.
Detailed results are stored in `docs/pipeline_results.json`.

## Development tools

- Python command-line tooling for implementation and evaluation
- Codex desktop for code development, repository analysis, testing, and
  documentation
- Git and GitHub for version control and collaboration
- SQLite tooling through Python’s standard library
- **TODO — add VS Code, PyCharm, Colab, Jupyter, or other tools if the team used
  them**

## APIs and models

- Groq OpenAI-compatible Chat Completions API for optional planning and
  reranking
- Anthropic Messages API as an optional alternative model provider
- Gemini Embedding API in a standalone, resumable embedding experiment; it is
  not required by the runtime agent
- Hugging Face `BAAI/bge-small-en-v1.5` for local catalog and query embeddings

No external API is required for the deterministic submission path. API keys
are read only from environment variables or an ignored `.env` file and are
never committed.

## Libraries and frameworks

- Python 3.10+
- SQLite FTS5 and standard-library `sqlite3`
- NumPy for normalized embedding matrices and cosine search
- Sentence Transformers
- Hugging Face Transformers
- PyTorch
- Google Gen AI SDK for the optional Gemini experiment
- `backoff` and `ratelimit` for resumable, quota-aware embedding generation
- Python `unittest` for pipeline tests

## Datasets and assets

- Frozen 50,000-product `Clothing_Shoes_and_Jewelry` catalog derived from
  Amazon Reviews 2023 by McAuley Lab, UCSD
- 200 labeled public development sessions used only through the official local
  evaluator
- 800 organizer-held private evaluation sessions that are never available to
  the agent or repository
- Precomputed 50,000-row BGE embedding matrix generated from the frozen catalog

At runtime, the agent does not read public labels, intent cards, ground truth,
sample IDs, target ASINs, raw reviews, or private evaluation data.

## Challenges we encountered

### Remembering meaning without retaining stale intent

Keeping every message verbatim improves recall, but it can also preserve a
preference the customer explicitly replaced. We solved this with structured
facts, provenance by turn, and selective override handling rather than a
single ever-growing prompt.

### Combining lexical and dense scores

BM25 and cosine similarity have unrelated scales. Directly adding them caused
semantic candidates to displace exact matches. We therefore use route-based
candidate generation and catalog-evidence reranking, treating dense retrieval
as a recall source rather than unquestioned ranking authority.

### Asking fewer, better questions

A fixed questionnaire wastes turns and can overfit a simulator. The final
policy estimates attribute information gain from the current candidates and
adapts the next question to the active search space.

### Remaining reliable without network access

External LLMs improve semantic planning but introduce latency, cost, and
availability risk. Every LLM operation is optional and validated, with a
deterministic offline fallback covering the complete Agent contract.

## Accomplishments we are proud of

- Increased public Hit Rate@10 from `0.125` to `0.840` without runtime access
  to public labels or ground truth.
- Reduced MTTC from `9.81` turns to `3.855` turns.
- Built a complete offline path that requires neither an API key nor a vector
  database server.
- Loaded and queried a local 50,000-product BGE index through a normalized
  in-memory matrix.
- Supported abrupt intent changes, negative preferences, multi-value answers,
  and boundary “no preference” responses.
- Added automated tests for memory, overrides, exclusions, LLM-plan validation,
  dense candidate fusion, embedding metadata, and response-contract behavior.

## Limitations

- Catalog attributes are semi-structured, so deterministic extraction cannot
  normalize every possible material, feature, sizing convention, or synonym.
- Candidate-information-gain estimation uses lightweight catalog signals rather
  than a learned question-value model.
- Dense encoding increases startup time, memory use, and per-turn latency; the
  deterministic lexical path remains slightly stronger on the documented full
  public ablation.
- LLM planning and reranking require external credentials and can add two model
  calls per turn when both features are enabled.
- The public development set is small, so all tuning risks some distribution
  sensitivity even though runtime code never consumes labels.
- The current system recommends only products already present in the fixed
  text catalog and does not process images or real-time inventory changes.

## What we would improve next

- Train or calibrate a compact offline cross-encoder on permitted development
  examples for stronger final reranking.
- Learn question value from candidate entropy, expected constraint coverage,
  and observed turn cost rather than relying on domain priors.
- Add field-specific or multi-vector embeddings for titles, categories,
  features, and descriptions to reduce long-document dilution.
- Quantize the embedding model and cache repeated state encodings to reduce
  CPU latency and memory.
- Calibrate confidence and stopping thresholds through cross-validation and
  broader adversarial paraphrase tests.
- Produce transparent recommendation explanations showing which active facts
  each product satisfies.

## Responsible data use

The competition data is derived from Amazon Reviews 2023. The agent uses only
the frozen catalog fields and the anonymized aggregate profile supplied by the
evaluator. It does not reconstruct user identities, consume raw purchase
history, or access private organizer data. Dataset attribution is documented
in `DATA_ATTRIBUTION.md`.

## Team contributions

Replace this section with concrete ownership before submission. Suggested
format:

- **TODO — Member 1:** conversational state, intent routing, and planner
- **TODO — Member 2:** lexical/dense retrieval, embeddings, and reranking
- **TODO — Member 3:** evaluation, experiments, documentation, and demo

For a solo submission, replace the list with:

> This project was designed, implemented, evaluated, and documented by
> **TODO — full name**.

## Demo video outline

The backend/NLP track permits an API and inference walkthrough. A concise demo
can follow this sequence:

1. **Problem, 15 seconds:** show a vague request and explain why static keyword
   search loses evolving intent.
2. **Architecture, 30 seconds:** show session memory, the planner, candidate
   routes, fusion, clarification, and Top 10 response.
3. **Live session, 60 seconds:** run a browsing request, answer a clarification,
   then demonstrate an intent override while displaying ranked ASINs.
4. **Offline resilience, 20 seconds:** show that the same evaluator works with
   no API key and that invalid model responses fall back safely.
5. **Results, 25 seconds:** compare the weak baseline and CyborgBoiS metrics.
6. **Limitations and next steps, 20 seconds:** discuss cross-encoder reranking,
   latency, and richer field-specific embeddings.

Before uploading, remove API keys and private information from the recording,
avoid unrelated copyrighted media, upload to YouTube with public visibility,
and add the URL at the top of this document and in Devpost.

## Final submission checklist

- [ ] Replace every `TODO` in this file
- [ ] Confirm the GitHub repository is public
- [ ] Add team members and contributions
- [ ] Run the documented evaluator command from a clean environment
- [ ] Confirm `results.json` matches the reported benchmark configuration
- [ ] Confirm no `.env`, API key, or private data is tracked by Git
- [ ] Record and upload the demo video publicly to YouTube
- [ ] Add the YouTube link to this file, the README, and Devpost
- [ ] Verify Amazon Reviews 2023 attribution and repository licensing
