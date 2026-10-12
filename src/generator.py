"""Grounded answer generation."""
from src import llm

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

SYSTEM = (
    "You are a careful question-answering assistant. Answer ONLY using the numbered context passages. "
    "Never use outside knowledge. If the context does not contain enough information, set answer to "
    f"\"{INSUFFICIENT}\". Respond with a JSON object: "
    "{\"answer\": \"<short answer, a few words>\", "
    "\"explanation\": \"<one or two complete sentences stating the answer and the facts it relies on>\"}"
)


def format_context(passages):
    return "\n\n".join(f"[{i + 1}] {p['title']}: {' '.join(p['sentences'])}" for i, p in enumerate(passages))


def generate(question, passages, flagged_claims=None):
    user = f"Context:\n{format_context(passages)}\n\nQuestion: {question}"
    if flagged_claims:
        bad = "\n".join(f"- {c}" for c in flagged_claims)
        user += (
            "\n\nA previous draft contained statements that the evidence did NOT support:\n"
            f"{bad}\nDo not repeat them unless the context above clearly supports them. "
            "Include only statements you can ground in the context."
        )
    out = llm.chat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    answer = str(out.get("answer", "")).strip() or INSUFFICIENT
    explanation = str(out.get("explanation", "")).strip()
    return {"answer": answer, "explanation": explanation or answer}
