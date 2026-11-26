"""Configuration, provider endpoints, and environment-backed secret access.

Migrated from Cell 01 and Cell 15 URL constants of the source notebook.
Importing this module does not require API keys or perform network calls.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import List

import matplotlib
import pandas as pd
import urllib3

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
FRED_BASE_URL = "https://api.stlouisfed.org/fred"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"


def _parse_api_keys_list(raw_value: str) -> List[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


ALPHA_VANTAGE_API_KEYS: List[str] = _parse_api_keys_list(
    os.getenv("ALPHA_VANTAGE_API_KEYS", "")
)
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "").strip()


def get_alpha_vantage_api_keys() -> List[str]:
    """Return configured Alpha Vantage API keys in their exact rotation order."""
    if ALPHA_VANTAGE_API_KEYS:
        return list(ALPHA_VANTAGE_API_KEYS)
    env_keys = _parse_api_keys_list(os.getenv("ALPHA_VANTAGE_API_KEYS", ""))
    if env_keys:
        return env_keys
    single_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    return [single_key] if single_key else []


def get_gemini_api_key() -> str:
    """Return configured Gemini API key (empty string if unset)."""
    if GEMINI_API_KEY:
        return GEMINI_API_KEY
    return os.getenv("GEMINI_API_KEY", "").strip()


def get_fred_api_key() -> str:
    """Return configured FRED API key (empty string if unset)."""
    if FRED_API_KEY:
        return FRED_API_KEY
    return os.getenv("FRED_API_KEY", "").strip()


def _redact_secret(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "***REDACTED***")
    return text


def configure_runtime(print_banner: bool = False) -> None:
    """Apply the source notebook's Cell 01 runtime settings on explicit request."""
    os.environ["PYTENSOR_FLAGS"] = "cxx="
    matplotlib.use("Agg")

    warnings.filterwarnings("ignore", category=UserWarning, module="pytensor")
    warnings.filterwarnings("ignore", message=".*g\\+\\+.*")
    warnings.filterwarnings("ignore", message=".*compiler.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    logging.basicConfig(
        level=logging.ERROR,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    if print_banner:
        print("\n" + "=" * 70)
        print("🚀 SISTEMA DE OPTIMIZACIÓN Y PROYECCIÓN DE PORTFOLIO")
        print("=" * 70)
        print("Modo: XGBoost | Shrinkage: LW/OAS | Régimen: 70/30 | Stress: corr=1 + vol + sistémico")
        print("=" * 70 + "\n")
