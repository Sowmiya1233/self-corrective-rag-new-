"""Compare plain RAG (iteration 0 of the pipeline) against the full self-corrective pipeline.

Progress is saved to results/eval_progress.jsonl after every question, so a crash or a Groq
daily-limit stop never loses work. Re-running the same command resumes where it stopped.
To start completely fresh, delete results/eval_progress.jsonl first.
"""
import json

import pandas as pd
from tqdm import tqdm

import config
from src import metrics
from src.generator import INSUFFICIENT
from src.llm import DailyLimitError
from src.pipeline import SelfCorrectiveRAG
from src.retriever import Retriever


def _mean(series):
    return float(series.mean()) if len(series) else 0.0


def _make_row(q, res):
    base, final_ans = res["baseline"], res["final_answer"]
    abstained = res["status"] == "abstained"
    pred_final = "" if abstained else final_ans
    base_abstained = base["answer"] == INSUFFICIENT
    pred_base = "" if base_abstained else base["answer"]
    return {
        "id": q["id"], "question": q["question"], "gold": q["answer"],
        "type": q["type"], "level": q["level"],
        "base_answer": base["answer"], "base_abstained": base_abstained,
        "base_faith": base["faithfulness"],
        "base_em": metrics.exact_match(pred_base, q["answer"]),
        "base_f1": metrics.f1_score(pred_base, q["answer"]),
        "final_answer": final_ans, "status": res["status"], "final_faith": res["faithfulness"],
        "final_em": metrics.exact_match(pred_final, q["answer"]),
        "final_f1": metrics.f1_score(pred_final, q["answer"]),
        "iterations_used": len(res["iterations"]), "llm_calls": res["llm_calls"],
        "latency": res["latency"],
    }


def evaluate(n=50, top_k=config.TOP_K, max_iterations=config.MAX_ITERATIONS):
    retriever = Retriever()
    rag = SelfCorrectiveRAG(retriever)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = config.RESULTS_DIR / "eval_progress.jsonl"

    questions = retriever.questions[:n]
    wanted = {q["id"] for q in questions}

    rows, done = [], set()
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    rows.append(r)
                    done.add(r["id"])
        print(f"Resuming: {len(done & wanted)} questions already done.")

    todo = [q for q in questions if q["id"] not in done]
    stopped_early = False
    try:
        for q in tqdm(todo, desc="Evaluating"):
            res = rag.run(q["question"], top_k=top_k, max_iterations=max_iterations)
            row = _make_row(q, res)
            rows.append(row)
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
    except DailyLimitError as e:
        stopped_early = True
        print("\nGroq daily token limit reached. Progress is saved.")
        print("Run the same command again later (the quota refills over time) to continue.")
        print(f"Details: {e}\n")

    rows = [r for r in rows if r["id"] in wanted]
    if not rows:
        print("No questions completed yet.")
        return pd.DataFrame(), {}

    df = pd.DataFrame(rows)
    base_ans = df[~df.base_abstained]
    final_ans = df[df.status != "abstained"]
    summary = {
        "n_questions": len(df),
        "requested_questions": len(questions),
        "stopped_early": stopped_early,
        "baseline": {
            "EM_all": _mean(df.base_em), "F1_all": _mean(df.base_f1),
            "n_answered": len(base_ans),
            "abstain_rate": float(df.base_abstained.mean()),
            "EM_on_answered": _mean(base_ans.base_em),
            "avg_faithfulness_on_answered": _mean(base_ans.base_faith),
            "unfaithful_answer_rate": float((base_ans.base_faith < config.ABSTAIN_BELOW).mean()) if len(base_ans) else 0.0,
        },
        "self_corrective": {
            "EM_all": _mean(df.final_em), "F1_all": _mean(df.final_f1),
            "n_answered": len(final_ans),
            "abstain_rate": float((df.status == "abstained").mean()),
            "EM_on_answered": _mean(final_ans.final_em),
            "avg_faithfulness_on_answered": _mean(final_ans.final_faith),
            "unfaithful_answer_rate": float((final_ans.final_faith < config.ABSTAIN_BELOW).mean()) if len(final_ans) else 0.0,
            "verified_rate": float((df.status == "verified").mean()),
            "avg_iterations": _mean(df.iterations_used),
            "avg_llm_calls": _mean(df.llm_calls), "avg_latency_s": _mean(df.latency),
        },
    }
    df.to_csv(config.RESULTS_DIR / "eval_results.csv", index=False)
    with open(config.RESULTS_DIR / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return df, summary