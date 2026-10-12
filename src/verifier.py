"""Adaptive evidence verification.

1. Split the generated explanation into atomic claims.
2. For each claim, pick the most relevant evidence sentences (local embeddings).
   The number of sentences per claim is adaptive (see `pipeline.adaptive_depth`).
3. Near-verbatim matches skip the LLM (fast path); everything else is judged by the LLM
   as SUPPORTED / CONTRADICTED / UNSUPPORTED using only the selected evidence.
"""
import numpy as np

import config
from src import embedder, llm

SUPPORTED, CONTRADICTED, UNSUPPORTED = "SUPPORTED", "CONTRADICTED", "UNSUPPORTED"


def split_claims(question, explanation, max_claims=6):
    msg = (
        "Break the TEXT into atomic, self-contained factual claims. Replace pronouns with the entities "
        f"they refer to. At most {max_claims} claims. Return JSON: {{\"claims\": [\"...\"]}}\n\n"
        f"Question: {question}\nTEXT: {explanation}"
    )
    out = llm.chat_json([{"role": "user", "content": msg}])
    claims = [c.strip() for c in out.get("claims", []) if isinstance(c, str) and c.strip()]
    return claims[:max_claims] or ([explanation] if explanation else [])


def _sentence_pool(passages):
    pool = []
    for p in passages:
        for s in p["sentences"]:
            pool.append(f"{p['title']}: {s}")
    return pool


def select_evidence(claims, passages, n_evidence):
    pool = _sentence_pool(passages)
    if not pool or not claims:
        return [[] for _ in claims], [0.0 for _ in claims]
    pool_emb = embedder.embed(pool)
    claim_emb = embedder.embed(claims)
    sims = claim_emb @ pool_emb.T
    evidence, best = [], []
    for row in sims:
        idx = np.argsort(-row)[:n_evidence]
        evidence.append([pool[i] for i in idx])
        best.append(float(row[idx[0]]))
    return evidence, best


def verify_claims(question, claims, passages, n_evidence):
    """Return a list of dicts: claim, verdict, reason, evidence."""
    if not claims:
        return []
    evidence, best = select_evidence(claims, passages, n_evidence)
    results = [None] * len(claims)

    to_judge = []
    for i, claim in enumerate(claims):
        if best[i] >= config.FAST_PATH_SIM:
            results[i] = {"claim": claim, "verdict": SUPPORTED,
                          "reason": "near-verbatim match with evidence (fast path)",
                          "evidence": evidence[i][:1]}
        else:
            to_judge.append(i)

    if to_judge:
        blocks = []
        for i in to_judge:
            ev = "\n".join(f"  - {e}" for e in evidence[i])
            blocks.append(f"Claim {i + 1}: {claims[i]}\nEvidence:\n{ev}")
        msg = (
            "You are a strict fact-checker. For each claim, use ONLY its evidence lines (they may need to be "
            "combined). Verdict must be one of: SUPPORTED (evidence together entails the claim), "
            "CONTRADICTED (evidence says otherwise), UNSUPPORTED (evidence is insufficient). "
            "Return JSON: {\"results\": [{\"id\": <claim number>, \"verdict\": \"...\", \"reason\": \"<short>\"}]}\n\n"
            f"Question context: {question}\n\n" + "\n\n".join(blocks)
        )
        out = llm.chat_json([{"role": "user", "content": msg}], max_tokens=900)
        judged = {}
        for r in out.get("results", []):
            try:
                judged[int(r.get("id"))] = r
            except (TypeError, ValueError):
                continue
        for i in to_judge:
            r = judged.get(i + 1, {})
            verdict = str(r.get("verdict", UNSUPPORTED)).upper()
            if verdict not in (SUPPORTED, CONTRADICTED, UNSUPPORTED):
                verdict = UNSUPPORTED
            results[i] = {"claim": claims[i], "verdict": verdict,
                          "reason": str(r.get("reason", "")), "evidence": evidence[i]}
    return results


def faithfulness(results):
    if not results:
        return 0.0
    return sum(r["verdict"] == SUPPORTED for r in results) / len(results)
