"""Smoke, stateless isolation, dead-code/ML/country-risk removal, and security tests."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import portfolio_optimization as po
from portfolio_optimization import (
    ConfigurationError,
    ConstraintsConfig,
    DCFViewConfig,
    InstrumentConfig,
    OptimizationConfig,
    PortfolioAnalysisConfig,
    RiskModelConfig,
    analyze_portfolio,
    ejecutar_analisis_completo,
    run_portfolio_analysis,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
PKG_DIR = SRC_DIR / "portfolio_optimization"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "portfolio_optimization.ipynb"


def test_clean_headless_import_without_secrets_or_side_effects():
    """Importing portfolio_optimization must require no secrets, network, GUI, or global user state."""
    env = os.environ.copy()
    for var in ("FRED_API_KEY", "GEMINI_API_KEY", "ALPHA_VANTAGE_API_KEYS", "ALPHA_VANTAGE_API_KEY"):
        env.pop(var, None)
    env["PYTHONPATH"] = str(SRC_DIR)

    code = (
        "import sys, portfolio_optimization as po; "
        "assert 'ipywidgets' not in sys.modules; "
        "assert 'tkinter' not in sys.modules; "
        "assert 'xgboost' not in sys.modules; "
        "assert not hasattr(po, 'state'); "
        "assert not hasattr(po, 'GEMINI_API_KEY'); "
        "assert not hasattr(po, 'ALPHA_VANTAGE_API_KEYS'); "
        "assert po.get_fred_api_key() == ''; "
        "print('OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "OK"


def test_removed_modules_symbols_and_dependencies_are_absent():
    """Verify complete removal of ML, country risk, stubs, global state, and unused keys."""
    assert not (PKG_DIR / "country_risk.py").exists()
    assert not (PKG_DIR / "ml.py").exists()
    assert not (PKG_DIR / "state.py").exists()

    removed_exports = [
        "state",
        "GEMINI_API_KEY",
        "get_gemini_api_key",
        "ALPHA_VANTAGE_API_KEYS",
        "ALPHA_VANTAGE_BASE_URL",
        "get_alpha_vantage_api_keys",
        "GOOGLE_NEWS_RSS_URL",
        "obtener_riesgo_pais_argentina",
        "calcular_primas_riesgo_pais",
        "country_etf_mapping",
        "DataIngestor",
        "IngestedData",
        "PortfolioProcessor",
        "PortfolioPredictor",
        "_train_xgb_robust",
        "_train_short_term_model_robust",
        "_train_long_term_model_robust",
        "_process_macro_with_lag",
        "build_ml_views_for_bl",
        "mostrar_proyecciones_ml",
        "crear_interfaz_proyecciones_ml",
        "forward_looking_cov_adjustment",
        "_build_static_figure",
        "_build_historical_performance_plot",
        "_build_correlation_heatmap",
        "_build_mcr_chart",
        "_build_real_portfolio_performance",
        "_build_real_portfolio_with_sales",
        "_build_real_individual_returns_plot",
        "_build_patrimonio_plot",
        "_build_portfolio_strategies_performance_plot",
        "_build_individual_returns_plot",
    ]
    for sym in removed_exports:
        assert not hasattr(po, sym), f"Removed symbol '{sym}' is still exported."

    forbidden_patterns = [
        r"\bxgboost\b",
        r"\bTimeSeriesSplit\b",
        r"\b_train_xgb_robust\b",
        r"\bpredicciones_ml\b",
        r"\bobtener_riesgo_pais_argentina\b",
        r"\bcalcular_primas_riesgo_pais\b",
        r"\b_fetch_alpha_vantage_financials\b",
        r"\b_process_macro\b",
        r"\b_build_static_figure\b",
        r"\bGEMINI_API_KEY\b",
        r"verify\s*=\s*False",
        r"fillna\(\s*method\s*=",
        r"\blocalStorage\b",
        r"\bsessionStorage\b",
        r"\bIndexedDB\b",
        r"\bsqlite3\b",
    ]
    for py_file in PKG_DIR.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for pat in forbidden_patterns:
            if pat == r"\bpredicciones_ml\b" and py_file.name == "schemas.py":
                continue
            assert re.search(pat, text) is None, f"Forbidden pattern {pat} in {py_file}"

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    reqs = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for dep in ("xgboost", "tqdm", "feedparser", "tkhtmlview"):
        assert dep not in pyproject
        assert dep not in reqs

    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "FRED_API_KEY=" in env_example
    assert "GEMINI_API_KEY" not in env_example
    assert "ALPHA_VANTAGE_API_KEYS" not in env_example


def test_legacy_ml_and_country_risk_arguments_are_rejected():
    """Removed legacy parameters must raise ConfigurationError instead of silent shims."""
    with pytest.raises(ConfigurationError, match="Removed legacy"):
        InstrumentConfig.from_dict({"instrument_id": "GGAL", "ticker": "GGAL", "pais_riesgo": "Argentina"})

    cfg = PortfolioAnalysisConfig(
        variable_income=(InstrumentConfig(instrument_id="AAPL", ticker="AAPL"),),
    )
    with pytest.raises(ConfigurationError, match="Removed legacy"):
        ejecutar_analisis_completo(cfg, ejecutar_ml=True)

    with pytest.raises(ConfigurationError, match="Removed legacy"):
        analyze_portfolio({"variable_income": [{"instrument_id": "AAPL", "ticker": "AAPL"}], "country_risk_argentina": 0.06})


def test_stateless_request_isolation_and_json_serializable_output():
    """Independent calls must not leak state, persist files, or share mutable results."""
    rng = np.random.default_rng(101)
    dates = pd.bdate_range("2022-01-03", periods=420)
    prices_a = pd.DataFrame({
        "EQ1": 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, len(dates)))),
        "GLD": 180.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.007, len(dates)))),
    }, index=dates)
    prices_b = pd.DataFrame({
        "BTC": 30000.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.025, len(dates)))),
        "SHY": 82.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.002, len(dates)))),
    }, index=dates)

    cfg_a = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(
                instrument_id="EQ1",
                ticker="EQ1",
                section="variable_income",
                asset_subtype="equity",
                country="Argentina",
                dcf_view=DCFViewConfig(valuation_basis="terminal_target", total_upside=0.40, horizon_years=5.0, confidence=0.5),
            ),
            InstrumentConfig(instrument_id="GLD", ticker="GLD", section="variable_income", asset_subtype="commodity_gold"),
        ),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        black_litterman=po.BlackLittermanConfig(rf_rate=0.04),
        optimization=OptimizationConfig(selected_strategy="erc"),
    )
    cfg_b = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(instrument_id="BTC", ticker="BTC", section="variable_income", asset_subtype="crypto"),
        ),
        fixed_income=(
            InstrumentConfig(instrument_id="SHY", ticker="SHY", section="fixed_income", asset_subtype="fixed_income_etf"),
        ),
        constraints=ConstraintsConfig(class_bounds={"crypto": (0.0, 0.10)}),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        black_litterman=po.BlackLittermanConfig(rf_rate=0.04),
        optimization=OptimizationConfig(selected_strategy="hrp"),
    )

    res_a = run_portfolio_analysis(cfg_a, prices_usd_override=prices_a, render_ui=False)
    res_b = run_portfolio_analysis(cfg_b, prices_usd_override=prices_b, render_ui=False)

    assert res_a.tickers == ["EQ1", "GLD"]
    assert res_b.tickers == ["BTC", "SHY"]
    assert res_a.selected_strategy == "erc"
    assert res_b.selected_strategy == "hrp"
    assert res_b.strategies["hrp"].class_exposure["crypto"] <= 0.10 + 1e-6

    payload_a = json.dumps(res_a.to_dict())
    payload_b = json.dumps(res_b.to_dict())
    assert "EQ1" in payload_a and "BTC" not in payload_a
    assert "BTC" in payload_b and "EQ1" not in payload_b


def test_sanitized_notebook_has_no_outputs_widgets_or_secrets():
    """Published notebook adapter must have zero outputs, widget states, ML, country risk, or secrets."""
    assert NOTEBOOK_PATH.exists()
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert "widgets" not in nb.get("metadata", {})
    assert "colab" not in nb.get("metadata", {})
    assert len(nb["cells"]) > 0

    all_sources = []
    for cell in nb["cells"]:
        assert not cell.get("attachments")
        if cell["cell_type"] == "code":
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
        all_sources.append("".join(cell.get("source", [])))

    joined = "\n".join(all_sources)
    assert "import portfolio_optimization" in joined
    assert "!pip install" not in joined
    assert "riesgo_pais" not in joined
    assert "proyecciones_ml" not in joined


def test_repository_contains_no_hardcoded_secrets():
    """Scan all project files for real API keys, spreadsheet IDs, or .env files."""
    assert not (REPO_ROOT / ".env").exists()
    assert (REPO_ROOT / ".env.example").exists()
    assert (REPO_ROOT / ".gitignore").exists()

    secret_regexes = [
        re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
        re.compile(r"ALPHA_VANTAGE_API_KEYS\s*=\s*\[\s*['\"][A-Z0-9]{10,}['\"]"),
        re.compile(r"FRED_API_KEY\s*=\s*['\"][0-9a-f]{32}['\"]"),
        re.compile(r"1ApjrJTrI4R8RXufL6UllBZw73c6HJApIHKL4htooSvs"),
    ]
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache", ".venv"} for part in path.parts):
            continue
        if path.name == "test_smoke_and_security.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for rx in secret_regexes:
            assert rx.search(text) is None, f"Sensitive pattern matched in {path}"
