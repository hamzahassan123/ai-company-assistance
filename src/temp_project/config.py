import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VECTOR_DB_PATH = PROJECT_ROOT / 'data' /  'vectorDB'
SALES_DATA_PATH = PROJECT_ROOT / 'data' / 'sales'








