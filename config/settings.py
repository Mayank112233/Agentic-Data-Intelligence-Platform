import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Output directory for agent runs (data, plots, reports)
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Portions of output folder
CLEANED_DATA_DIR = OUTPUT_DIR / "cleaned_data"
CLEANED_DATA_DIR.mkdir(exist_ok=True)

PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# LLM Configs
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini").lower()
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-3.5-flash")

# Logging configurations
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

def get_api_key(provider: str = None) -> str:
    """Retrieve API key for provider (gemini or openai)"""
    if provider is None:
        provider =  provider = os.getenv("DEFAULT_LLM_PROVIDER", DEFAULT_LLM_PROVIDER).lower()
        
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY", "")
    elif provider == "openai":
        return os.getenv("OPENAI_API_KEY", "")
    return ""
