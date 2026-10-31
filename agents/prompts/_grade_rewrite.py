"""GRADE_PROMPT / REWRITE_PROMPT — reused near-verbatim from BBLAgent's
agents/retriever.py. See implementation-plan.md §3.2.
"""

GRADE_PROMPT = """You are grading whether a retrieved document is relevant to a user question.

Retrieved document:
{context}

User question: {question}

If the document contains information related to the question, grade it 'yes', else 'no'."""

REWRITE_PROMPT = """This question didn't retrieve a relevant document from the knowledge base:
-------
{question}
-------
Reason about the underlying intent and rewrite it as a clearer, more specific
question likely to retrieve a better match. Return only the rewritten question."""
