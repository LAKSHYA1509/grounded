---
title: grounded
emoji: 🔎
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.26.0
python_version: '3.12'
app_file: space_app.py
pinned: false
---

# grounded

[![CI](https://github.com/LAKSHYA1509/grounded/actions/workflows/ci.yml/badge.svg)](https://github.com/LAKSHYA1509/grounded/actions/workflows/ci.yml)

A document Q&A service that refuses to answer without sources — and checks its own retrieval before it answers.

FastAPI · LangGraph · Qdrant · Gemini. Containerised, CI on every push.

> The YAML block above is Hugging Face Spaces metadata for the live demo. `app_file` points at `space_app.py` so it doesn't collide with the `app/` package.

---

## What it does

Most RAG implementations retrieve once and hope. If the retrieval was poor, the model either invents an answer or gives up, and either way the failure is invisible.

`grounded` adds a **grading step**: after retrieving, it asks the model whether the retrieved context is actually sufficient to answer the question. If it isn't, it goes back and retrieves again with a wider net.

That backward edge is why this is a graph and not a chain.

---

## Architecture

```
                    POST /ask
                        │
                        ▼
                     ┌─────┐
                     │START│
                     └──┬──┘
                        │
                        ▼
              ┌──────────────────┐
        ┌────▶│    retrieve      │  embed question, pull top-k from Qdrant
        │     └────────┬─────────┘
        │              │
        │              ▼
        │     ┌──────────────────┐
        │     │      grade       │  "is this context enough?"
        │     └────────┬─────────┘
        │              │
        │       ┌──────┴───────┐
        │       │              │
        │  insufficient    sufficient
        │  & attempts       (or out of
        │  remaining        attempts)
        │       │              │
        │       ▼              │
        │  ┌─────────┐         │
        └──┤  widen  │         │      ◀── THE CYCLE
           └─────────┘         │          k doubles each retry
                               ▼
                     ┌──────────────────┐
                     │    generate      │  answer from context only,
                     └────────┬─────────┘  cite sources, or say
                              │            "I don't know"
                              ▼
                           ┌─────┐
                           │ END │
                           └─────┘
```

State is persisted after every node by a checkpointer, keyed on `thread_id`.

---

## Running it

Requires Python 3.11+ and one model API key. The default provider is Google Gemini, whose free tier needs no credit card — get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

**No vector database to install**: with `QDRANT_URL` unset the store runs in-memory, so a fresh clone works with nothing but that key.

```bash
git clone https://github.com/LAKSHYA1509/grounded.git
cd grounded

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # then put your Gemini key in .env

uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive API.

### Try it

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text":"Refunds are processed within 5 working days. Payouts above INR 5000 require manual review.","source":"policy.md"}'

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How long do refunds take?"}'
```

The response carries the answer, the source chunks it used, how many retrieval attempts it took, and how the context was graded.

### Tests

```bash
pytest
```

24 tests, no model key and no running database required. They cover chunking, the routing logic, the auth gate, and the store against a real in-memory Qdrant.

---

## Deploying

**Live demo:** a Gradio front-end (`space_app.py`) runs on Hugging Face Spaces. It imports the same graph, chunker and store the API uses — nothing is reimplemented. That it took ~60 lines is the payoff from keeping the route handlers free of business logic: a completely different front-end was additive rather than a refactor.

**As a container**, the image reads `$PORT` from the environment, so the same build runs on Render or Cloud Run — both inject their own — without modification. Hardcoding a port is what makes an image host-specific.

```bash
docker build -t grounded .
docker run -p 8000:7860 --env-file .env grounded
```

For a real deployment set `QDRANT_URL` to a Qdrant Cloud cluster — a container filesystem is ephemeral, so an embedded store would be wiped on every redeploy.

**Set `API_KEY` on any public deployment.** `/ask` spends model quota; an open endpoint holding a provider key is an open wallet. With it set, every route except `/health` requires the `X-API-Key` header.

---

## Design notes

**Why a graph rather than a chain.** A chain is a straight line and cannot go backwards. Grading retrieval and retrying on a poor result is a loop, so a chain cannot express it. This is the one thing LangGraph is genuinely for.

**Why grade at all.** Retrieval quality is the single biggest determinant of RAG output quality, and it fails silently. Grading turns an invisible failure into a state the system can respond to. It costs one extra model call per question — a cheap, fast call, since the grader answers with one word.

**Why `k` doubles on retry.** Retrying with the same `k` retrieves the same chunks and grades them the same way — an infinite loop that also achieves nothing. Each retry has to change something, and widening the net is the cheapest useful change.

**Why attempts are capped at 2.** Every cycle in a graph needs a termination guarantee. Some questions cannot be answered from the indexed documents, and the system must be able to conclude that rather than loop. When attempts run out we generate anyway, and the prompt makes the model say "I don't know" rather than invent something.

**Why Qdrant instead of an embedded store.** This began on Chroma, which was right while it ran only on a laptop and wrong the moment it was containerised — a container filesystem is ephemeral, so the index would be wiped on every redeploy. Storage that must outlive the process has to live outside it. The second reason was a surprise: Chroma pulled ~235 MB of dependencies the project never used (a Kubernetes client, an ONNX inference runtime, gRPC, OpenTelemetry) because it *can* also embed locally and run distributed. Moving to a thin HTTP client cut the install from 422 MB to 187 MB.

**Why 800-character chunks with 120 overlap.** Big enough to hold a complete thought, small enough that retrieval stays precise. The overlap matters more than it looks: without it, an idea spanning a chunk boundary is cut in half and no single chunk contains it — and no later stage can recover information that chunking destroyed.

**Why the response returns sources.** An answer you cannot verify is an answer you cannot trust. Returning the chunks lets a caller check whether the model actually used the documents.

**Why the provider is a config string.** This is not a hypothetical benefit. The project started on OpenAI and moved to Google Gemini because Gemini's free tier needs no credit card. That migration was two lines of `.env` and one dependency — no application code changed at all, because both chat and embeddings go through provider-agnostic factories. Depend on the interface, not the vendor.

**Why the vector dimension is measured, not configured.** Qdrant needs the vector size to create a collection and rejects anything else, so a wrong number breaks every insert. Putting it in config means config can disagree with reality — swap the embedding model, forget the dimension, and the failure surfaces later somewhere else as a confusing database rejection. So `llm.embedding_dimension()` asks the model instead. When a value can be derived from the system, deriving it beats configuring it: configuration is for choices, and this is a fact.

**Why a shared secret rather than user accounts.** There is exactly one thing to decide — may you call this at all — and no per-user data to authorise between. Real accounts would be more code, more attack surface, and no more security. Choosing the control that is proportionate to the risk is the point.

---

## Known limitations

Written down deliberately. A system whose failure modes you haven't named is one you haven't finished thinking about.

- **Ingestion is synchronous.** Embedding a large document blocks the HTTP request. The correct shape is accept → enqueue → return a job id, with a worker doing the embedding. Fine for pasted text, wrong for a 300-page PDF.
- **Semantic search only.** Embeddings are fuzzy and miss exact tokens — error codes, product IDs, names. Hybrid search (vector + BM25 keyword) would fix this and is the most valuable single upgrade here.
- **No re-ranking.** We take the top `k` by cosine similarity and use them in that order. Retrieving wider and re-ranking with a cross-encoder would put the genuinely best chunk first, which matters because context position affects the answer.
- **The grader is the same model as the generator.** Cheap and convenient, but a model is not a neutral judge of context it is about to use. A smaller dedicated grader, or a heuristic on similarity scores, would be more honest.
- **`InMemorySaver` loses conversation state on restart.** Deliberate, to keep setup to one command. Swapping in the SQLite or Postgres checkpointer is an interface-level change, not a rewrite.
- **No evaluation set.** There is no way to tell whether a prompt change made things better or worse. For anything real this is the first thing I would add: a fixed list of questions with known-good answers, run on every change.
- **No rate limiting.** The API key gates *who* can call, not *how much*. A leaked key is still an uncapped bill.
- **Single-tenant.** The tenant filter would have to be pushed *into* the Qdrant query, not applied to its results — filtering afterwards silently returns fewer chunks than requested and risks leaking across tenants. Noted in `app/store.py`.

---

## Layout

```
app/
  main.py       FastAPI routes — thin, no business logic
  graph.py      the LangGraph: state, nodes, router, cycle   ← start here
  store.py      Qdrant: index and similarity search
  chunking.py   splitting documents, and why the splits matter
  llm.py        model access + measured vector dimension, provider-agnostic
  auth.py       the API key gate, and why it's proportionate
  models.py     request/response schemas
  config.py     all configuration, read from the environment
tests/          24 tests — no model key, no database required
.github/
  workflows/ci.yml   pytest + docker build on every push
```
