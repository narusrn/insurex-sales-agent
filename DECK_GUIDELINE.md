# Deck guideline — InsureX Sales Agent

Not the deck itself — a brief for whoever builds it (you, or Claude via the
Artifact/Design tool in a later session). Goal: a short technical presentation
for the take-home review, built from what's *actually* in this repo (real
code, real eval numbers, real transcripts, real screenshots), not invented
content.

**Audience**: whoever is reviewing the take-home — assume they can read code
but haven't read this repo. Optimize for "what did you decide and why," not a
code walkthrough.

**Length**: ~10-12 slides. Deep enough to show engineering judgment, short
enough that nothing gets skipped for time.

**Source note**: this project doesn't have a separate `WRITEUP.md` (unlike
the `support-ticket-triage-service` reference this guideline is modeled on,
which pulls slide reasoning from its own `WRITEUP.md`) — every citation below
points at `README.md`'s own section headers instead, since that's where the
equivalent "why" content actually lives in this repo. If a `WRITEUP.md` gets
written later, update these citations to point at it.

## Suggested slide order

1. **Title** — "InsureX Sales Agent" + one line: a LangGraph agent behind a
   FastAPI REST API that answers insurance product questions via agentic RAG
   and collects sales leads mid-conversation, from the take-home brief.

2. **The problem, in one breath** — answer a customer's product questions
   accurately (not guess when the knowledge base doesn't have the answer),
   and when they signal buying interest, collect their contact details — but
   never save incomplete or unconfirmed personal data, and never leak one
   customer's conversation into another's.

3. **Architecture overview** — `assets/graph.png`. Walk the happy path left
   to right: Supervisor → (RAG subgraph, when product info is needed) →
   `extract_lead_info` → `save_lead`, every tool routing back to Supervisor
   via `Command(goto=...)`, never a static edge (`README.md` "Command-routing
   caveats" — a static edge out of a `Command`-returning tool's node was
   found to fire *in addition to* the `Command`'s own `goto`, confirmed by a
   duplicated trace line during development; fixed by removing every static
   edge out of a tool node). One sentence worth saying out loud: Supervisor
   is the **only** node that ever composes a user-facing reply, on every
   path — QA answers, lead acknowledgements, and plain conversation all go
   through it (`README.md` "Architecture overview").

4. **Safety boundary #1 — no hallucinated certainty** (`README.md` "No
   hallucinated certainty"). The concept slide: `grade_documents`' relevance
   verdict is threaded back to Supervisor specifically so it can tell "here's
   what I found" apart from "retries ran out and this still wasn't relevant."
   State the real bug this closed: Supervisor once stated a **specific,
   confident, wrong number** pulled from snippets that had just been graded
   irrelevant — found during evaluation, not invented for the slide. Fixed
   with an explicit rule: a note prefixed `⚠️` may never be used to state a
   specific number or condition; the model must say "ไม่พบข้อมูลที่ชัดเจน"
   (no clear information found) instead. Re-verified across the full 21-case
   suite with zero regressions. The real transcript proof — `qa-06`'s full
   two-rewrite retry loop ending in an honest fallback, not a guess — lives
   on the eval slide (6), not duplicated here.

5. **Safety boundary #2 — won't save data it hasn't confirmed**
   (`README.md` "Lead collection"). `save_lead` has two separate,
   deterministic gates, not one: **completeness** (all 4 required fields
   present and non-blank, or it returns `"Missing: X"`) and **confirmation**
   (even with all 4 fields present, the *first* call never persists — it asks
   Supervisor to restate the values and get an explicit "yes" first). Every
   field is `Optional` on purpose: a *required* tool argument pressures the
   model to fabricate a value it doesn't have, so the real gate lives in the
   tool body, not the schema. The proof — not a clean run, a real **correction
   case**: `lead-06`'s transcript, where a customer's phone-number correction
   on turn 2 was *not* misread as a confirmation; Supervisor re-extracted,
   re-stated, and asked again, and the saved DB row holds the corrected
   number, not the original wrong one. This is the one case in the suite with
   zero code-level fallback left to catch a misread, so a pass here is a real
   result, not a formality.

6. **Eval harness — real numbers, not cherry-picked.** `eval/golden_dataset.json`
   (13 `qa_cases` across easy/hard/out_of_scope/multi_hop + 6 `lead_cases` +
   2 `session_cases`) run via `python -m eval.run_full_eval`, full transcripts
   + pass/fail in `eval/demo_log.txt`. Report the real last-run numbers:
   - **16/21 overall.** `lead_cases` and `session_cases` both **100%** (6/6,
     2/2) — the completeness/confirmation/isolation gates held on every case.
     `qa_cases` **8/13 (61.5%)** on strict keypoint substring-match — worth
     stating plainly, not left implicit: **4 of the 5 `qa` failures
     (`qa-06`, `qa-09`, `qa-11`, `qa-12`) are the honest-fallback behavior
     from slide 4 firing correctly**, not wrong answers — the agent ran its
     full retry budget and said "I don't know, contact an agent" instead of
     guessing. The 5th (`qa-02`) actually retrieved the right premium figure
     (6,800 บาท, matching the expected keypoint) but missed a secondary
     phrasing keypoint — a strict-grading near-miss, not a factual error.
   - **Retrieval quality, separately**: Hit Rate@k = 0.91, MRR = 0.86 over the
     11 `qa_cases` with a known expected source doc — a more lenient measure
     of "did it find the right document" than the strict keypoint match
     above, and the number that actually isolates retrieval quality from
     generation quality.

7. **The agentic RAG retriever** — `assets/graph-retriever.png` (the same
   nested subgraph visible inside `graph.png`'s xray view, isolated on its
   own for this slide). Not a single fire-and-forget similarity search:
   retrieve → `grade_documents` (an LLM judges relevance against the
   **original** question, not the possibly-rewritten search query) →
   rewrite-and-retry if not relevant, capped at `RETRIEVER_MAX_REWRITES`
   (default 2) (`README.md` "RAG subgraph"). Use `qa-06`'s or `qa-11`'s real
   transcript here as the concrete "watch it retry and still refuse to
   guess" proof — both show the full two-rewrite loop verbatim, straight
   from `eval/demo_log.txt`, not a cleaned-up paraphrase.

8. **A real conversation, start to finish** — `assets/lead-collection-conversation.png`,
   a live-captured run (not staged): greeting → product question → stated
   interest → step-by-step info gathering (Supervisor asks specifically for
   the one missing field, not all 4 again) → confirm → saved. Pair it with
   `session-01`/`session-02`'s transcript (`README.md` "Demo transcripts") if
   there's room: a second session has zero knowledge of the first's name
   (`sess-B` can't answer "what's my name"), and the *same* session correctly
   recalls "the cheapest plan" refers to Health Plus from two turns earlier —
   both verified directly against `checkpoints.sqlite`, not just the reply
   text.

9. **It runs, with real usage data** — `assets/dashboard-screenshot.png` +
   `assets/dashboard-expanded.png` (click-to-expand trace + per-node token
   breakdown). One line: every `/api/v1/chat` turn logs real `usage_metadata`
   from every LLM call in that turn — not just Supervisor's own, also
   `grade_documents`, `extract_lead_info`'s structured-output call, every
   call inside the RAG subgraph — summed and priced against a small
   hand-maintained rate table, explicitly labeled approximate rather than
   presented as exact (`README.md` "Usage dashboard").

10. **Two ways to drive it** — `assets/chat-test-screenshot.png`
    (`/api/v1/chat-test`, a self-contained HTML page with real token-by-token
    SSE streaming, no backend framework needed to open it) and
    `/api/v1/docs` (Swagger, standardized to match
    `narusrn/support-ticket-triage-service`'s own `api/main.py` pattern —
    `/api/v1` prefix on every real route + docs, one shared checkpointer
    connection for the app's lifetime instead of one per request, structured
    `{"error": {"code","message"}}` responses on every failure, `GET
    /health`). Worth a one-line mention: an earlier Streamlit app was built,
    then deliberately removed in favor of this simpler static page — see
    slide 12's scope note.

11. **What I'd change for production** — not pulled from a `WRITEUP.md`
    (none exists yet), stated plainly here instead:
    - Real per-token pricing (an API lookup or a maintained table with a
      freshness check), not the hand-maintained static rates
      `_PRICING_PER_1M_TOKENS` uses today — already labeled "approximate" in
      the dashboard, but worth being explicit about the gap on this slide.
    - Lead PII (name, phone, income) sits in `db/leads.sqlite` as plaintext,
      unencrypted, with no access control beyond the single shared
      `X-API-Key` — fine for a demo, not for real customer data.
    - SQLite under concurrent write load — swap for Postgres once traffic
      justifies it; nothing in the schema assumes SQLite specifically.
    - Real per-customer API keys + rate limiting, instead of one shared demo
      key.
    - Point the trace/usage logs (`db/runs.jsonl`) at a real tracing backend
      (e.g. LangSmith) instead of an in-house JSON-lines file, once volume
      justifies it.

12. **Close** — repo link, one line on scope: what the brief asked for vs.
    what was built on request beyond it — the usage dashboard, the live chat
    test page (replacing an earlier Streamlit app, removed on request), the
    per-turn cost/trace logging, and standardizing the API layer
    (`/api/v1` everywhere, shared lifespan connection, structured errors,
    `/health`) against a second reference project partway through — not
    just a feature list, the reasoning for each lives in the relevant
    `README.md` section cited above.

## What to leave out

- Don't walk source code line by line — the diagrams + the two safety-boundary
  slides (4, 5) carry the technical weight. Point to `README.md` for anyone
  who wants the detail afterward instead of reading it aloud.
- Don't re-derive eval numbers live — pull them from `eval/demo_log.txt`'s
  last real run, don't eyeball new ones under presentation pressure. If
  they've gone stale by presentation time, regenerate with
  `python -m eval.run_full_eval` rather than quoting old numbers from memory.
- Don't show `implementation-plan.md`, `discussion-supervisor-redesign.md`,
  `agent-architecture-patterns.md`, or this file itself — all four are
  working documents, not deck content (same exception the reference repo's
  own guideline makes for its `PLAN.md`).
- Slide 8's transcripts are real, captured examples — don't invent a
  cleaner-looking one. If they've gone stale by presentation time, re-run the
  same conversation and recapture it rather than hand-writing dialogue that
  looks plausible.

## Asset reference

| File | Use for |
|---|---|
| `assets/graph.png` | Slide 3 (architecture) |
| `assets/graph-retriever.png` | Slide 7 (agentic RAG) |
| `assets/lead-collection-conversation.png` | Slide 8 (a real conversation) |
| `assets/dashboard-screenshot.png`, `assets/dashboard-expanded.png` | Slide 9 (it runs) |
| `assets/chat-test-screenshot.png` | Slide 10 (two ways to drive it) |
| `agents/visualize.py` (`python -m agents.visualize`) | Re-run if the graph structure changes before presenting — regenerates both `graph.png` and `graph-retriever.png` from the real compiled graphs, keeping the diagrams honest instead of stale |

## If building this as a Claude Artifact/Design deck

Reference this file plus `README.md` as source material — don't re-summarize
from memory of the conversation. Use the PNGs in `assets/` directly (via
`files` on the Artifact publish call) rather than redrawing the diagrams;
they're already generated from the real compiled graphs, which is the whole
point of citing them. Pull transcript text verbatim from `eval/demo_log.txt`
and `README.md`'s "Demo transcripts" section rather than paraphrasing — the
exact Thai wording (including the honest "ไม่พบข้อมูลที่ชัดเจน" fallback
phrasing) is part of the proof, not decoration.
