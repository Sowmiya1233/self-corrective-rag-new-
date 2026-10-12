"""Step 3: python run_eval.py --n 50   -> results/eval_results.csv + eval_summary.json"""
import argparse
import json

from src.evaluate import evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of questions")
    ap.add_argument("--iters", type=int, default=2, help="max correction rounds")
    args = ap.parse_args()
    _, summary = evaluate(args.n, max_iterations=args.iters)
    print(json.dumps(summary, indent=2, default=float))
