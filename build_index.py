"""Step 1: download HotpotQA (if needed) and build the local embedding index."""
from src.indexer import build_index

if __name__ == "__main__":
    build_index()
