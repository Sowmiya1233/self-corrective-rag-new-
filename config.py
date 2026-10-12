"""Central configuration. Override anything via the .env file."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ---- LLM (Groq) ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.0"))  # seconds between calls (rate limits)

# ---- Embeddings (local) ----
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---- Paths ----
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
RESULTS_DIR = ROOT / "results"
HOTPOT_PATH = DATA_DIR / "hotpot_dev_distractor_v1.json"
HOTPOT_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

# ---- Corpus ----
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "500"))

# ---- Retrieval ----
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_PASSAGES = int(os.getenv("MAX_CONTEXT_PASSAGES", "10"))

# ---- Self-correction loop ----
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "2"))     # correction rounds after the first answer
BASE_THRESHOLD = float(os.getenv("BASE_THRESHOLD", "0.8"))  # required fraction of supported claims
ABSTAIN_BELOW = float(os.getenv("ABSTAIN_BELOW", "0.5"))    # below this faithfulness -> abstain
FAST_PATH_SIM = float(os.getenv("FAST_PATH_SIM", "0.95"))   # near-verbatim claim/evidence skips LLM (set 2 to disable)

ABSTAIN_MESSAGE = "I don't have enough supported evidence to answer this reliably."
