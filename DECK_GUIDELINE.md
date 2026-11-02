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

3. **System design template (reference only)** — `assets/example system
   design'.jpg` (a generic "System Design Master Template" from
   DesignGurus.io: DNS/load balancer/API gateway/caching/queues/workers —
   none of it specific to this project). State this plainly on the slide,
   don't let it pass as InsureX's own diagram: it's the framework used to
   *think through* what this system needed before narrowing down to the
   actual, much smaller shape — a single FastAPI process, one LangGraph
   agent, SQLite, no queues, no separate workers, no CDN, because none of
   that scale is called for here. The real answer to "what's the system
   design" is the very next slide, `assets/graph.png` — this one's only job
   is showing the starting checklist got actively narrowed down, not
   applied wholesale.

4. **Architecture overview** — `assets/graph.png`. Walk the happy path left
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

5. **Safety boundary #1 — no hallucinated certainty** (`README.md` "No
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

6. **Safety boundary #2 — won't save data it hasn't confirmed**
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

7. **Eval harness — three separate layers, real numbers, not cherry-picked.**
   `eval/golden_dataset.json` (13 `qa_cases` across
   easy/hard/out_of_scope/multi_hop + 6 `lead_cases` + 2 `session_cases`) run
   via `python -m eval.run_full_eval`, full transcripts + pass/fail in
   `eval/demo_log.txt`. Worth stating plainly that `qa_cases` are graded
   three *different* ways, answering three different questions — not one
   blended score (`README.md` "Eval harness — three separate layers"):
   - **Pass/fail, overall: 16/21.** `lead_cases` and `session_cases` both
     **100%** (6/6, 2/2) — the completeness/confirmation/isolation gates held
     on every case. `qa_cases` **8/13 (61.5%)** on strict keypoint
     substring-match — **4 of the 5 `qa` failures (`qa-06`, `qa-09`,
     `qa-11`, `qa-12`) are the honest-fallback behavior from slide 5 firing
     correctly**, not wrong answers — the agent ran its full retry budget and
     said "I don't know, contact an agent" instead of guessing.
   - **Retrieval accuracy** (`eval/metrics.py`): Hit Rate@k = 0.91, MRR =
     0.86 over the 11 `qa_cases` with a known expected source doc — isolates
     "did it find the right document" from generation quality.
   - **LLM-as-judge groundedness** (`eval/judge.py`, new): a second,
     independent LLM call grades whether the reply is faithful to what was
     *actually retrieved that run* — no pre-written expected answer, since a
     correct reply can be phrased many ways. **11/12** qa cases that
     retrieved something. The one miss is the honest, interesting result for
     this slide: `qa-02` reads the premium for a 35-year-old off the
     retrieved "age 31–40" bracket row — arithmetically correct, but the
     judge flagged it as not *explicitly* stated for age 35 — a real
     strictness call about what "grounded" means, kept as-is rather than
     loosened to make the number look better. (This is also the same case
     the strict keypoint check above fails, for an unrelated reason — a
     missed secondary phrase — worth not conflating the two on the slide.)

8. **The agentic RAG retriever** — `assets/graph-retriever.png` (the same
   nested subgraph visible inside `graph.png`'s xray view, isolated on its
   own for this slide). Not a single fire-and-forget similarity search:
   retrieve → `grade_documents` (an LLM judges relevance against the
   **original** question, not the possibly-rewritten search query) →
   rewrite-and-retry if not relevant, capped at `RETRIEVER_MAX_REWRITES`
   (default 2) (`README.md` "RAG subgraph"). Use `qa-06`'s or `qa-11`'s real
   transcript here as the concrete "watch it retry and still refuse to
   guess" proof — both show the full two-rewrite loop verbatim, straight
   from `eval/demo_log.txt`, not a cleaned-up paraphrase.

9. **A real conversation, start to finish** — `assets/lead-collection-conversation.png`,
   a live-captured run (session `fda66787-98d8-4e79-95aa-c110f82e94f6`, not
   staged — reproduce fresh via `GET /api/v1/chat/fda66787-...` or by replaying
   the same 6 messages through `/api/v1/chat-test` if `db/checkpoints.sqlite`
   has moved on by presentation time). The actual speaker-labeled trace, same
   convention as `README.md`'s "Demo transcripts", straight from that
   session's own `db/runs.jsonl` record — don't invent a cleaner-looking one:
   ```
   Supervisor      สวัสดีครับ! มีอะไรให้ช่วยเหลือเกี่ยวกับประกันหรือผลิตภัณฑ์
                   ของ InsureX ไหมครับ?
   Supervisor      ลูกค้าสอบถามเกี่ยวกับแผนประกันที่มี -- ค้นข้อมูลในฐานความรู้
   RAG Agent       ค้นข้อมูลด้วยคำค้น "แผนประกัน" -- พบ 2 ข้อความ, เกี่ยวข้อง ✅
   Supervisor      เรามีแผนประกันหลักๆ ดังนี้ครับ: 1. InsureX Life Secure...
                   2. InsureX Health Plus...
   Lead Extractor  {"name":null,"occupation":null,"income":null,"phone":null,
                    "interested_product":"InsureX Health Plus", ...}
   Supervisor      คุณสนใจประกันสุขภาพ InsureX Health Plus ครับ ขอข้อมูล
                   เพิ่มเติมหน่อยนะครับ: 1. ชื่อ 2. อาชีพ 3. รายได้ 4. เบอร์โทร
   Lead Extractor  {"name":"สมชาย ใจดี","occupation":"วิศวกรซอฟต์แวร์",
                    "income":"45,000 บาทต่อเดือน","phone":null, ...}
   Supervisor      ข้อมูลที่ได้จากคุณ: ชื่อ สมชาย ใจดี, อาชีพ วิศวกรซอฟต์แวร์,
                   รายได้ 45,000 บาทต่อเดือน -- ยังขาดเบอร์โทรศัพท์ครับ
   Lead Saver      Data complete but not yet confirmed (name=สมชาย ใจดี,
                   occupation=วิศวกรซอฟต์แวร์, income=45,000 บาทต่อเดือน,
                   phone=0812345678). Restate and ask to confirm.
   Supervisor      ข้อมูลที่เรามีตอนนี้คือ: ... กรุณายืนยันข้อมูลเหล่านี้ว่า
                   ถูกต้องหรือไม่ครับ?
   Lead Saver      Lead saved (id=5)
   Supervisor      ข้อมูลของคุณได้ถูกบันทึกเรียบร้อยแล้วครับ!
   ```
   The point for the slide: `extract_lead_info` ran twice (fields arrive
   incrementally — product first, then name/occupation/income, phone still
   missing), `save_lead` was only ever called *after* `extract_lead_info`
   reported complete data, and it refused to persist on the first complete
   call — exactly the two gates from slide 6, visibly firing in a real run,
   not asserted in the abstract. Pair with `session-01`/`session-02`'s
   transcript (`README.md` "Demo transcripts") if there's room: a second
   session has zero knowledge of the first's name (`sess-B` can't answer
   "what's my name"), and the *same* session correctly recalls "the cheapest
   plan" refers to Health Plus from two turns earlier — both verified
   directly against `checkpoints.sqlite`, not just the reply text.

10. **It runs, with real usage data** — `assets/dashboard-screenshot.png` +
   `assets/dashboard-expanded.png` (click-to-expand trace + per-node token
   breakdown). Every `/api/v1/chat` turn appends one record to
   `db/runs.jsonl` — this is the final turn of the same session as slide 9's
   transcript (`db/runs.jsonl`, trimmed to the fields worth putting on a
   slide):
   ```json
   {
     "session_id": "fda66787-98d8-4e79-95aa-c110f82e94f6",
     "message": "ยืนยันครับ ถูกต้อง",
     "tools_used": ["save_lead"],
     "latency": 2.75,
     "llm_calls": 2,
     "usage": {"input_tokens": 5123, "output_tokens": 129, "total_tokens": 5252},
     "estimated_cost_usd": 0.000846,
     "cost_note": "approximate -- hardcoded gpt-4o-mini per-token rates, may not reflect current pricing"
   }
   ```
   One line: `usage.total_tokens` is summed from real `usage_metadata` on
   every LLM call that turn — not just Supervisor's own, also
   `grade_documents`, `extract_lead_info`'s structured-output call, every
   call inside the RAG subgraph when it runs — priced against a small
   hand-maintained rate table, explicitly labeled approximate rather than
   presented as exact (`README.md` "Usage dashboard"). The full `trace` array
   (what slide 9's transcript is built from) and `usage.by_node` breakdown
   are both in the same record, one click away in the dashboard's expanded
   row view — that's what `dashboard-expanded.png` is a screenshot of.

11. **Three ways to drive it** — `assets/chat-test-screenshot.png`
    (`/api/v1/chat-test`, a self-contained HTML page with real token-by-token
    SSE streaming against a *running FastAPI server*, no backend framework
    needed to open it), `/api/v1/docs` (Swagger, standardized to match
    `narusrn/support-ticket-triage-service`'s own `api/main.py` pattern —
    `/api/v1` prefix on every real route + docs, one shared checkpointer
    connection for the app's lifetime instead of one per request, structured
    `{"error": {"code","message"}}` responses on every failure, `GET
    /health`), and `streamlit_app.py` — a separate chat UI deployable
    standalone to Streamlit Cloud, embedding the agent *in-process* instead
    of calling the API over HTTP (no server needed on Cloud; `st.cache_
    resource` plays the same one-shared-connection role api/main.py's
    lifespan does). If deployed by presentation time, this is the natural
    "here's a link, go try it yourself" slide — put the live Streamlit Cloud
    URL here once it exists, don't invent one.

12. **What I'd change for production** — not pulled from a `WRITEUP.md`
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

13. **Close** — repo link, one line on scope: what the brief asked for vs.
    what was built on request beyond it — the usage dashboard, the live chat
    test page, the standalone Streamlit Cloud deploy, the per-turn
    cost/trace logging, the LLM-as-judge eval layer, and standardizing the
    API layer (`/api/v1` everywhere, shared lifespan connection, structured
    errors, `/health`) against a second reference project partway through —
    not just a feature list, the reasoning for each lives in the relevant
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
- Slide 9's transcripts are real, captured examples — don't invent a
  cleaner-looking one. If they've gone stale by presentation time, re-run the
  same conversation and recapture it rather than hand-writing dialogue that
  looks plausible.

## Asset reference

| File | Use for |
|---|---|
| `assets/example system design'.jpg` | Slide 3 (system design template, reference only — not InsureX's own diagram) |
| `assets/graph.png` | Slide 4 (architecture) |
| `assets/graph-retriever.png` | Slide 8 (agentic RAG) |
| `assets/lead-collection-conversation.png` | Slide 9 (a real conversation) |
| `assets/dashboard-screenshot.png`, `assets/dashboard-expanded.png` | Slide 10 (it runs) |
| `assets/chat-test-screenshot.png` | Slide 11 (two ways to drive it) |
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
