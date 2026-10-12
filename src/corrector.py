"""Targeted correction: turn unsupported claims into new retrieval queries."""
from src import llm


def followup_queries(question, bad_claims, passages, max_queries=3):
    titles = ", ".join(p["title"] for p in passages[:10])
    bad = "\n".join(f"- {c}" for c in bad_claims) or "- (the answer could not be grounded at all)"
    msg = (
        "A question-answering system could not find evidence for some statements. Write up to "
        f"{max_queries} short, specific search queries that would find the missing evidence. "
        "Mention concrete entities. Return JSON: {\"queries\": [\"...\"]}\n\n"
        f"Question: {question}\nUnsupported statements:\n{bad}\nPassages already retrieved: {titles}"
    )
    out = llm.chat_json([{"role": "user", "content": msg}])
    queries = [q.strip() for q in out.get("queries", []) if isinstance(q, str) and q.strip()]
    if not queries:
        queries = list(bad_claims)[:max_queries] or [question]
    return queries[:max_queries]
