"""Explicit, recorded configuration. Every run stores a copy of CONFIG_SNAPSHOT."""
import os
import pathlib

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"
SNAPSHOTS = DATA / "snapshots"
DB_PATH = ROOT / "runs.db"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
# GEMMA_BACKEND: "gemini_api" (hosted Gemma 4, online) or "ollama" (local Gemma 4 E2B/E4B)
GEMMA_BACKEND = os.getenv("GEMMA_BACKEND", "gemini_api")
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-31b-it" if GEMMA_BACKEND == "gemini_api" else "gemma4:e2b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MEDGEMMA_MODEL = os.getenv("MEDGEMMA_MODEL", "medgemma:4b")

CALL_TIMEOUT_S = float(os.getenv("CALL_TIMEOUT_S", "120"))
RUN_BUDGET_S = float(os.getenv("RUN_BUDGET_S", "420"))

GENERATION = {"temperature": 0.1, "top_p": 0.9, "seed": 7, "num_predict": 900, "num_ctx": 8192}

LIMITS = {
    "initial_lookups": 3,
    "reconcile_lookups": 1,
    "max_passages": 6,
    "schema_repairs": 1,
    "max_questions": 2,
    "max_secondary_dx": 2,
    "max_discrepancies": 3,
}

SCHEMA_VERSION = "1"
PROMPT_VERSION = "p1"
RULE_VERSION = "r1"

CONFIG_SNAPSHOT = {
    "gemma_backend": GEMMA_BACKEND,
    "gemma_model": GEMMA_MODEL,
    "medgemma_model": MEDGEMMA_MODEL,
    "generation": GENERATION,
    "limits": LIMITS,
    "call_timeout_s": CALL_TIMEOUT_S,
    "run_budget_s": RUN_BUDGET_S,
    "schema_version": SCHEMA_VERSION,
    "prompt_version": PROMPT_VERSION,
    "rule_version": RULE_VERSION,
}
