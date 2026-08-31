# grounded

A document Q&A service that refuses to answer without sources — and checks its own retrieval before it answers.

Built with FastAPI, LangGraph and Chroma.

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
        ┌────▶│    retrieve      │  embed question, pull top-k chunks
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

Requires Python 3.11+ and an OpenAI API key.

```bash
git clone https://github.com/Lakshya1509/grounded.git
cd grounded

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # then put your key in .env

uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive API.

### Try it

```bash
# index a document
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text":"Refunds are processed within 5 working days. Payouts above INR 5000 require manual review.","source":"policy.md"}'

# ask about it
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How long do refunds take?"}'
```

The response includes the answer, the source chunks it used, how many retrieval attempts it took, and how the context was graded.

### Tests

```bash
pytest
```

The tests cover chunking and the routing logic — the two pure-logic pieces. They need no API key and run in under a second.

---

## Design notes

**Why a graph rather than a chain.** A chain is a straight line and cannot go backwards. Grading retrieval and retrying on a poor result is a loop, so a chain cannot express it. This is the one thing LangGraph is genuinely for.

**Why grade at all.** Retrieval quality is the single biggest determinant of RAG output quality, and it fails silently. Grading turns an invisible failure into a state the system can respond to. It costs one extra model call per question — a cheap, fast call, since the grader answers with one word.

**Why `k` doubles on retry.** Retrying with the same `k` retrieves the same chunks and grades them the same way — an infinite loop that also achieves nothing. Each retry has to change something, and widening the net is the cheapest useful change.

**Why attempts are capped at 2.** Every cycle in a graph needs a termination guarantee. Some questions simply cannot be answered from the indexed documents, and the system must be able to conclude that rather than loop. When attempts run out we generate anyway, and the prompt makes the model say "I don't know" rather than invent something.

**Why 800-character chunks with 120 overlap.** Big enough to hold a complete thought, small enough that retrieval stays precise. The overlap matters more than it looks: without it, an idea spanning a chunk boundary is cut in half and no single chunk contains it — and no later stage can recover information that chunking destroyed.

**Why the response returns sources.** An answer you cannot verify is an answer you cannot trust. Returning the chunks lets a caller check whether the model actually used the documents.

**Why the provider is a config string.** `CHAT_MODEL=openai:gpt-4o-mini` can become `anthropic:...` with no code change. Depend on the interface, not the vendor.

---

## Known limitations

Written down deliberately. A system whose failure modes you haven't named is one you haven't finished thinking about.

- **Ingestion is synchronous.** Embedding a large document blocks the HTTP request. The correct shape is accept → enqueue → return a job id, with a worker doing the embedding. Fine for pasted text, wrong for a 300-page PDF.
- **Semantic search only.** Embeddings are fuzzy and miss exact tokens — error codes, product IDs, names. Hybrid search (vector + BM25 keyword) would fix this and is the most valuable single upgrade here.
- **No re-ranking.** We take the top `k` by cosine similarity and use them in that order. Retrieving wider and re-ranking with a cross-encoder would put the genuinely best chunk first, which matters because context position affects the answer.
- **The grader is the same model as the generator.** Cheap and convenient, but a model is not a neutral judge of context it is about to use. A smaller dedicated grader, or a non-model heuristic on similarity scores, would be more honest.
- **`InMemorySaver` loses state on restart.** Deliberate, to keep setup to one command. Swapping in the SQLite or Postgres checkpointer is an interface-level change, not a rewrite.
- **No evaluation set.** There is no way to tell whether a prompt change made things better or worse. For anything real this is the first thing I would add: a fixed list of questions with known-good answers, run on every change.
- **Single-tenant.** The tenant filter would have to be pushed *into* the vector search, not applied to its results — filtering afterwards silently returns fewer chunks than requested and risks leaking across tenants. Noted in `app/store.py`.

---

## Layout

```
app/
  main.py       FastAPI routes — thin, no business logic
  graph.py      the LangGraph: state, nodes, router, cycle   ← start here
  store.py      vector store: index and similarity search
  chunking.py   splitting documents, and why the splits matter
  llm.py        model access in one place, provider-agnostic
  models.py     request/response schemas
  config.py     all configuration, read from the environment
tests/
  test_chunking.py   pure logic, no API key needed
  test_routing.py    the cycle's termination guarantee
```
