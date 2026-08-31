# Read this first

This file is for you, not for an interviewer. It's the reading order, what each file is *for*, and the questions to ask yourself as you go.

**This is a scaffold, not a finished thing.** It's a working skeleton with the reasoning written into the comments so you can read it rather than reverse-engineer it. Tomorrow you'll change things, break things, and add the pieces that are deliberately missing — and that's what makes it yours. You should be able to explain every line before you put it on your GitHub. That's the whole point of reading it first.

Don't try to run it yet. Just read.

---

## Reading order

Roughly 90 minutes if you go slowly, which you should.

### 1. `app/config.py` — 5 minutes

The easiest file. Every knob the system has, in one place.

Ask yourself: *why is the API key in an environment variable rather than in the code?* Two reasons, and the second one is the one people forget.

### 2. `app/chunking.py` — 15 minutes

This is where RAG most commonly fails, so it gets read early.

The idea to sit with: **chunking destroys information, and nothing downstream can recover it.** If the answer to a question spans a boundary and no single chunk contains it whole, then no amount of better retrieval or a smarter model will help. It isn't there to be found.

That's why overlap exists. And it's why the splitter tries paragraph breaks before sentence breaks before cutting mid-word — split at the least damaging available place.

Ask yourself: *what happens if I set `CHUNK_OVERLAP=0`? What specifically breaks, and would I notice?*

### 3. `app/store.py` — 15 minutes

What a vector store is and what it's doing.

The mechanism: text becomes a vector, similar meanings land near each other, and "find relevant chunks" becomes "find nearby vectors." Nearness is cosine similarity — the *angle* between vectors, not the distance, because we care about direction (meaning) and not magnitude (roughly, length).

Read the long comment in `search()` twice. The multi-tenancy point in it is the single strongest thing you can say about this project in an interview, because it comes from work you actually do — and it's a genuine bug class, not a style opinion.

Ask yourself: *why must the same embedding model be used for ingestion and for queries?* The answer is in `llm.py`.

### 4. `app/graph.py` — 40 minutes. This is the file.

Everything else is plumbing. This is the project.

Read the docstring at the top, then the state, then the three nodes, then the router, then `build_graph()`. Then read it again in that order, because the second pass is where it actually lands.

The four ideas to come away with:

- **State** is a shared scratchpad. Every node reads it and returns a partial update; LangGraph merges the update in. That's the whole contract.
- **A node is just a function.** There is nothing magic about them. If that feels anticlimactic, good — it means you've understood it.
- **A conditional edge is a function that returns the name of the next node.** That's how branching happens, and it's how the loop closes.
- **Every cycle needs a way to stop.** `MAX_ATTEMPTS` is that. A graph that can loop and has no termination guarantee is a graph that can burn money forever.

Ask yourself, and be honest about whether you can answer without looking:

1. Why can't a chain do this? (If you can answer this one cleanly you've got the most important thing.)
2. What would happen if `widen()` didn't exist — if a retry used the same `k`?
3. Why does `grade()` explicitly say "do NOT answer the question"?
4. What does `thread_id` actually do?

### 5. `app/models.py` and `app/main.py` — 10 minutes

The HTTP boundary. Notice how thin `main.py` is: routes parse input, call one function, shape the response. No logic.

Ask yourself: *why is that separation worth having?* Hint — how would you test the graph if the logic lived in the route handler?

### 6. `tests/` — 10 minutes

Two files, and notice what they test: chunking and routing. Both are pure functions — no model, no network, no API key.

That isn't laziness about the rest. It's the design working: **the riskiest part of the system (the cycle) is also the cheapest part to test**, because it was deliberately kept free of I/O. That's a real engineering argument and a good thing to be able to make.

---

## What's deliberately missing

Things for you to add. Each one is a real improvement, not busywork:

- **Hybrid search.** Semantic search misses exact tokens — error codes, IDs, names. Combining vector similarity with keyword matching is the highest-value upgrade here.
- **A `/ingest/file` endpoint** that takes an actual upload instead of pasted text.
- **Async ingestion.** Accept, enqueue, return a job id. This is the shape you already know from your OTA bundle pipeline.
- **A tiny eval set.** Ten questions with known-good answers, and a script that runs them. This is the thing that most impresses people and almost nobody does.
- **The SQLite checkpointer** instead of `InMemorySaver`, so threads survive a restart.
- **Streaming** on `/ask`, so the answer appears token by token instead of after a five-second wait.

Pick one or two. Finishing two properly beats starting six.

---

## Questions worth bringing back

These are the ones where I'd rather you ask than guess. Genuinely — asking any of these is a sign the reading worked, not that it didn't:

- Anything in `graph.py` where you can follow *what* the code does but not *why* it's shaped that way.
- The reducer concept — it's mentioned in the runbook and this project doesn't actually need one. Why not? When would it?
- What "temperature=0" is doing in `llm.py` and why it matters for a grader specifically.
- Whether the grading step is actually a good idea, or whether it's just an expensive way to be wrong twice. (This is a real critique. Have a view on it — an interviewer might push exactly there, and "here's the trade-off and here's why I chose it anyway" is a much better answer than a defence.)
- How any of this would change if it had to serve seventeen tenants. You know more about that than most people would.

---

## Before it goes public

- [ ] You can explain every file without opening this document
- [ ] `.env` is **not** committed — check `git status` before every push
- [ ] You've run it once and asked it a real question
- [ ] `pytest` passes
- [ ] You've added at least one thing from the missing list
- [ ] You've cloned it fresh into a new folder and followed the README from scratch

That last one catches more problems than all the others combined.
