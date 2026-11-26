"""Server-side provider settings, runtime setup, and central config exports.

Only active server-side technical credentials (currently FRED_API_KEY as an
optional risk-free rate fallback) are read from environment variables and are
never exposed to client payloads or required at import time.
"""

from __future__ import annotations

import logging
import os

import matplotlib

from .schemas import (
    AnalysisResult,
    BlackLittermanConfig,
    BlackLittermanResult,
    BrokerConfig,
    BrokerPosition,
    CashFlowEvent,
    ConstraintsConfig,
    DCFViewConfig,
    DCFViewResult,
    FixedIncomeMetricsResult,
    FixedIncomeTerms,
    InstrumentConfig,
    IOLConfig,
    IOLQuoteResult,
    OmittedTrade,
    OptimizationConfig,
    PortfolioAnalysisConfig,
    ProposedOrder,
    RebalancingConfig,
    RebalancingResult,
    RiskModelConfig,
    RiskModelResult,
    StrategyAllocationResult,
    VixRegimeConfig,
)

FRED_BASE_URL = "https://api.stlouisfed.org/fred"

FRED_API_KEY: str = os.getenv("FRED_API_KEY", "").strip()


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
    """Configure non-interactive matplotlib backend and logging cleanly on pandas 3.x."""
    matplotlib.use("Agg")
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    if print_banner:
        print("=" * 72)
        print("PORTFOLIO OPTIMIZATION ENGINE — STATELESS MULTI-ASSET EDITION")
        print("DCF (Annualized) -> Black-Litterman | Weekly W-FRI VIX-Regime Cov")
        print("Optimizers: Max Sharpe | MinVar | MinCVaR | ERC | HRP")
        print("=" * 72)


__all__ = [
    "FRED_BASE_URL",
    "FRED_API_KEY",
    "get_fred_api_key",
    "_redact_secret",
    "configure_runtime",
    "AnalysisResult",
    "BlackLittermanConfig",
    "BlackLittermanResult",
    "BrokerConfig",
    "BrokerPosition",
    "CashFlowEvent",
    "ConstraintsConfig",
    "DCFViewConfig",
    "DCFViewResult",
    "FixedIncomeMetricsResult",
    "FixedIncomeTerms",
    "InstrumentConfig",
    "IOLConfig",
    "IOLQuoteResult",
    "OmittedTrade",
    "OptimizationConfig",
    "PortfolioAnalysisConfig",
    "ProposedOrder",
    "RebalancingConfig",
    "RebalancingResult",
    "RiskModelConfig",
    "RiskModelResult",
    "StrategyAllocationResult",
    "VixRegimeConfig",
]
