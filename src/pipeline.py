"""Self-Corrective RAG with Adaptive Evidence Verification.

Loop:  retrieve -> generate -> split into claims -> verify claims -> (pass?) else
       generate follow-up queries -> retrieve more evidence -> regenerate with flagged claims.
Adaptive parts:
  * verification threshold depends on retrieval confidence (weak retrieval => stricter)
  * evidence sentences per claim depends on retrieval confidence (weak retrieval => look deeper)
  * the retrieval depth grows each correction round
Finally: verified answer, low-confidence answer, or abstention.
"""
import time

import numpy as np

import config
from src import corrector, generator, llm, verifier


def retrieval_confidence(passages):
    return float(np.mean([p["score"] for p in passages[:3]])) if passages else 0.0


def adaptive_threshold(conf, base=config.BASE_THRESHOLD):
    if conf >= 0.60:
        return max(0.6, base - 0.1)
    if conf < 0.35:
        return min(1.0, base + 0.1)
    return base


def adaptive_depth(conf):
    """Evidence sentences inspected per claim."""
    if conf >= 0.60:
        return 4
    if conf >= 0.35:
        return 6
    return 8


def _merge(old, new, cap):
    by_id = {p["id"]: p for p in old}
    for p in new:
        if p["id"] not in by_id or p["score"] > by_id[p["id"]]["score"]:
            by_id[p["id"]] = p
    return sorted(by_id.values(), key=lambda p: -p["score"])[:cap]


class SelfCorrectiveRAG:
    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, question, top_k=config.TOP_K, max_iterations=config.MAX_ITERATIONS,
            base_threshold=config.BASE_THRESHOLD, abstain_below=config.ABSTAIN_BELOW):
        t0, calls0 = time.time(), llm.calls
        passages = self.retriever.search(question, top_k)
        conf = retrieval_confidence(passages)
        threshold = adaptive_threshold(conf, base_threshold)
        depth = adaptive_depth(conf)

        iterations, flagged = [], None
        for it in range(max_iterations + 1):
            gen = generator.generate(question, passages, flagged)
            abstained_by_model = gen["answer"] == generator.INSUFFICIENT
            if abstained_by_model:
                claims, results, faith = [], [], 0.0
            else:
                claims = verifier.split_claims(question, gen["explanation"])
                results = verifier.verify_claims(question, claims, passages, depth)
                faith = verifier.faithfulness(results)
            passed = (not abstained_by_model) and faith >= threshold
            record = {
                "iteration": it, "answer": gen["answer"], "explanation": gen["explanation"],
                "claims": results, "faithfulness": faith, "passed": passed,
                "num_passages": len(passages), "queries": [],
                "passages": [{"title": p["title"], "score": p["score"]} for p in passages],
            }
            iterations.append(record)
            if passed or it == max_iterations:
                break

            # ---- correction round ----
            flagged = [r["claim"] for r in results if r["verdict"] != verifier.SUPPORTED]
            queries = corrector.followup_queries(question, flagged, passages)
            record["queries"] = queries
            new = []
            for q in queries:
                new += self.retriever.search(q, top_k)
            passages = _merge(passages, new, min(config.MAX_CONTEXT_PASSAGES, top_k + 3 * (it + 1)))

        # ---- choose final result ----
        passed_runs = [r for r in iterations if r["passed"]]
        best = passed_runs[-1] if passed_runs else max(iterations, key=lambda r: (r["faithfulness"], -r["iteration"]))
        answered = best["answer"] != generator.INSUFFICIENT
        if best["passed"]:
            status = "verified"
        elif answered and best["faithfulness"] >= abstain_below:
            status = "low_confidence"
        else:
            status = "abstained"
        final_answer = best["answer"] if status != "abstained" else config.ABSTAIN_MESSAGE

        return {
            "question": question, "status": status, "final_answer": final_answer,
            "final_explanation": best["explanation"] if status != "abstained" else "",
            "faithfulness": best["faithfulness"], "threshold": threshold,
            "retrieval_confidence": conf, "evidence_depth": depth,
            "iterations": iterations, "baseline": iterations[0],
            "llm_calls": llm.calls - calls0, "latency": time.time() - t0,
        }
