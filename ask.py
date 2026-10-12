"""Command-line demo:  python ask.py "your question"   (or no argument = first dataset question)"""
import sys

from src.pipeline import SelfCorrectiveRAG
from src.retriever import Retriever

if __name__ == "__main__":
    retriever = Retriever()
    question = " ".join(sys.argv[1:]) or retriever.questions[0]["question"]
    res = SelfCorrectiveRAG(retriever).run(question)
    print(f"\nQ: {question}")
    print(f"A: {res['final_answer']}   [{res['status']}, faithfulness={res['faithfulness']:.2f}, threshold={res['threshold']:.2f}]")
    for it in res["iterations"]:
        print(f"\n-- iteration {it['iteration']}: answer='{it['answer']}' faithfulness={it['faithfulness']:.2f} passed={it['passed']}")
        for c in it["claims"]:
            print(f"   [{c['verdict']}] {c['claim']}")
        if it["queries"]:
            print(f"   follow-up queries: {it['queries']}")
    print(f"\nLLM calls: {res['llm_calls']}, latency: {res['latency']:.1f}s")
