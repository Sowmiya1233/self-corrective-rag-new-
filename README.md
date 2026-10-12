# Self-Corrective RAG with Adaptive Evidence Verification for Hallucination-Resistant Answer Generation

A RAG system that does not trust its own first answer. Every answer is split into atomic claims,
each claim is checked against retrieved evidence, and if too many claims are unsupported the system
retrieves more evidence and regenerates. If it still cannot ground the answer, it abstains.

- **LLM:** Groq (default `llama-3.3-70b-versatile`)
- **Embeddings:** local `all-MiniLM-L6-v2` (no API)
- **Data:** HotpotQA `dev_distractor` (paragraph corpus + evaluation questions)
- **Front end:** Streamlit

## Pipeline

```
question -> retrieve top-k -> generate (answer + explanation)
         -> split explanation into atomic claims
         -> verify every claim against the best evidence sentences
         -> faithfulness = supported claims / all claims
              >= adaptive threshold ?  yes -> VERIFIED answer
              no -> flagged claims -> follow-up queries -> retrieve more -> regenerate (max N rounds)
         -> best attempt: verified / low-confidence / abstain
```

### What is "adaptive"
| Component | Behaviour |
|---|---|
| Verification threshold | Weak retrieval (low similarity) -> stricter threshold; strong retrieval -> relaxed |
| Evidence depth | Weak retrieval -> inspect more evidence sentences per claim (8), strong -> fewer (4) |
| Retrieval depth | Grows every correction round, using queries targeted at the unsupported claims |
| Verification path | Near-verbatim claim/evidence matches skip the LLM (saves calls); the rest are LLM-judged |

## Setup (Windows / VS Code)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then paste your GROQ_API_KEY into .env
```

## Run

```
python build_index.py                 # 1. downloads HotpotQA + embeds paragraphs (one time)
python ask.py "your question"         # 2. CLI demo with the correction trace
streamlit run app.py                  # 3. front end
python run_eval.py --n 50             # 4. baseline RAG vs self-corrective, saved in results/
pytest                                # unit tests (no API needed)
```

If the automatic download fails, download `hotpot_dev_distractor_v1.json` manually from
https://hotpotqa.github.io/ and place it in `data/`.

## Folder structure

```
self_corrective_rag/
├── app.py              Streamlit front end
├── ask.py              CLI demo
├── build_index.py      build the embedding index
├── run_eval.py         evaluation entry point
├── config.py           all settings (override in .env)
├── src/
│   ├── data_loader.py  HotpotQA download + corpus building
│   ├── embedder.py     local embeddings
│   ├── indexer.py      saves embeddings/docs/questions to index/
│   ├── retriever.py    cosine-similarity retrieval
│   ├── llm.py          Groq wrapper (retries, JSON parsing)
│   ├── generator.py    grounded answer generation
│   ├── verifier.py     claim splitting + adaptive evidence verification
│   ├── corrector.py    follow-up query generation
│   ├── pipeline.py     the self-corrective loop
│   ├── metrics.py      EM / F1
│   └── evaluate.py     baseline vs self-corrective comparison
├── tests/
├── data/  index/  results/
└── requirements.txt
```

## Metrics reported (results/eval_summary.json)
EM, F1, average faithfulness, unfaithful-answer rate (answers with <50% supported claims),
abstain rate, verified rate, average iterations, LLM calls and latency per question.
Baseline = iteration 0 of the pipeline (plain RAG), so both systems are verified by the same judge.

## Tuning tips
- Groq free-tier rate limits: set `REQUEST_DELAY=2` in `.env` for evaluation runs.
- `MAX_QUESTIONS` controls corpus size (500 questions ≈ 5,000 paragraphs, indexes in a few minutes on CPU).
  Re-run `build_index.py` after changing it.
- Raise `BASE_THRESHOLD` for stricter hallucination control (more abstentions), lower it for higher coverage.
- Limitation: the LLM judge can itself make errors, and the fast path assumes a very high embedding
  similarity means support (set `FAST_PATH_SIM=2` to disable it).
