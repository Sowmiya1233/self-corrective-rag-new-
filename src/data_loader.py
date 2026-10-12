"""Download / load HotpotQA dev_distractor and turn it into a paragraph corpus."""
import json

import requests

import config

HF_PARQUET_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet"
)


def _stream_to_file(url, dest):
    with requests.get(url, stream=True, timeout=180, allow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def _download_from_cmu(path):
    _stream_to_file(config.HOTPOT_URL, path)


def _download_from_huggingface(path):
    """Fallback: download only the validation parquet with plain requests, then convert to HotpotQA JSON."""
    import pandas as pd

    parquet_path = path.with_suffix(".parquet")
    print(f"Downloading validation parquet (~27 MB) to {parquet_path} ...")
    _stream_to_file(HF_PARQUET_URL, parquet_path)

    df = pd.read_parquet(parquet_path)
    data = []
    for _, ex in df.iterrows():
        ctx = ex["context"]
        data.append({
            "_id": str(ex["id"]),
            "question": str(ex["question"]),
            "answer": str(ex["answer"]),
            "type": str(ex.get("type", "")),
            "level": str(ex.get("level", "")),
            "context": [[str(t), [str(x) for x in s]] for t, s in zip(ctx["title"], ctx["sentences"])],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    parquet_path.unlink(missing_ok=True)


def download_hotpot(path=config.HOTPOT_PATH):
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading HotpotQA dev_distractor to {path} ...")
        _download_from_cmu(path)
    except Exception as e:
        print(f"CMU download failed ({e}). Trying Hugging Face instead...")
        if path.exists():
            path.unlink()
        _download_from_huggingface(path)
    return path


def load_hotpot(path=config.HOTPOT_PATH):
    download_hotpot(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_corpus(data, max_questions=config.MAX_QUESTIONS):
    """Return (docs, questions). Each doc is one Wikipedia paragraph."""
    data = data[:max_questions]
    docs, seen = [], set()
    for ex in data:
        for title, sents in ex["context"]:
            sents = [s.strip() for s in sents if s.strip()]
            key = title + "||" + " ".join(sents)
            if key in seen:
                continue
            seen.add(key)
            docs.append({
                "id": len(docs),
                "title": title,
                "sentences": sents,
                "text": f"{title}. " + " ".join(sents),
            })
    questions = [{
        "id": ex["_id"], "question": ex["question"], "answer": ex["answer"],
        "type": ex.get("type"), "level": ex.get("level"),
    } for ex in data]
    return docs, questions