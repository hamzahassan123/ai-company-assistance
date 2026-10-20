import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vectorDB"
SALES_DATA_PATH = PROJECT_ROOT / "data" / "sales"


def require_gemini_api_key() -> str:
    """Return the configured Gemini API key or raise an actionable error."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            f"GEMINI_API_KEY is not configured. Add it to {PROJECT_ROOT / '.env'}."
        )
    return GEMINI_API_KEY




