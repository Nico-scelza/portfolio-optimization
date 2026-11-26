"""Side-by-side numerical and AST parity tests against the original notebook.

These tests execute only when the untouched source notebook is available locally
(outside the repository) and never print or leak any secret values.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import requests
import urllib3
import xgboost as xgb
import yfinance as yf
from requests.adapters import HTTPAdapter
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf, OAS as OAS_estimator
from tqdm import tqdm
from urllib3.util.retry import Retry

import portfolio_optimization as po
from portfolio_optimization import market_data, ml, optimization, pipeline, state

_SOURCE_NOTEBOOK_ENV = os.getenv("SOURCE_NOTEBOOK_PATH", "").strip()
SOURCE_NOTEBOOK = Path(_SOURCE_NOTEBOOK_ENV) if _SOURCE_NOTEBOOK_ENV else None

pytestmark = pytest.mark.skipif(
    SOURCE_NOTEBOOK is None or not SOURCE_NOTEBOOK.exists(),
    reason="Set SOURCE_NOTEBOOK_PATH to run side-by-side parity tests against the untouched source notebook.",
)


@pytest.fixture(scope="module")
def notebook_namespace():
    """Load calculation cells (08-13, 15-17, 19) from source notebook with dummy secrets."""
    import sys
    import types

    nb = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 37

    mod = types.ModuleType("__notebook_oracle__")
    sys.modules["__notebook_oracle__"] = mod
    ns = mod.__dict__
    ns.update({
        "np": np,
        "pd": pd,
        "xgb": xgb,
        "yf": yf,
        "requests": requests,
        "urllib3": urllib3,
        "HTTPAdapter": HTTPAdapter,
        "Retry": Retry,
        "LedoitWolf": LedoitWolf,
        "OAS_estimator": OAS_estimator,
        "minimize": minimize,
        "tqdm": tqdm,
        "logging": logging,
        "time": time,
        "tempfile": tempfile,
        "Path": Path,
        "dataclass": dataclass,
        "field": field,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "Iterable": Iterable,
        # Strictly dummy secrets (never read Cell 01 real keys)
        "ALPHA_VANTAGE_API_KEYS": ["DUMMY_AV_1", "DUMMY_AV_2"],
        "FRED_API_KEY": "DUMMY_FRED",
        "GEMINI_API_KEY": "DUMMY_GEMINI",
        "prima_riesgo_mas_reciente": 6.5,
        "country_risk_premia": {"Brazil": 1.8},
        # Headless display stubs
        "display": lambda *a, **k: None,
        "HTML": lambda x: x,
        "Markdown": lambda x: x,
    })

    for cell_idx in (8, 9, 10, 11, 12, 13, 15, 16, 17, 19):
        src = "".join(nb["cells"][cell_idx]["source"])
        exec(compile(src, f"<cell_{cell_idx:02d}>", "exec"), ns)

    yield ns
    sys.modules.pop("__notebook_oracle__", None)


def _make_prices(n_days: int = 320, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = ["T1", "T2", "T3", "T4", "T5", "T6"]
    rets = rng.normal(0.0003, 0.014, size=(n_days, len(tickers)))
    rets[:, 1] = 0.6 * rets[:, 0] + 0.8 * rets[:, 1]
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def test_exact_ast_parity_for_verbatim_calculation_functions():
    """Verify verbatim migrated functions have identical ASTs to the original notebook."""
    nb = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

    def get_func_ast(source: str, func_name: str, occurrence: int = -1) -> str:
        tree = ast.parse(source)
        matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name]
        assert matches, f"{func_name} not found"
        return ast.dump(matches[occurrence], include_attributes=False)

    opt_src = Path(optimization.__file__).read_text(encoding="utf-8")
    mkt_src = Path(market_data.__file__).read_text(encoding="utf-8")
    ml_src = Path(ml.__file__).read_text(encoding="utf-8")
    pipe_src = Path(pipeline.__file__).read_text(encoding="utf-8")

    verbatim_checks = [
        (8, "_download_prices", mkt_src),
        (9, "optimize_sharpe_hhi", opt_src),
        (10, "_optimize_min_cvar", opt_src),
        (10, "_optimize_min_variance", opt_src),
        (10, "_portfolio_metrics", opt_src),
        (10, "_calc_mcr", opt_src),
        (10, "_calc_historical_performance", opt_src),
        (10, "_is_existing_portfolio", opt_src),
        (10, "_calc_cvar", opt_src),
        (10, "_simulate_portfolios", opt_src),
        (11, "obtener_tasa_libre_riesgo", mkt_src),
        (11, "descargar_precios_historicos", mkt_src),
        (11, "metricas_portfolio", mkt_src),
        (12, "get_market_caps", opt_src),
        (12, "compute_market_weights", opt_src),
        (12, "compute_equilibrium_returns", opt_src),
        (12, "black_litterman_returns", opt_src),
        (12, "build_ml_views_for_bl", opt_src),
        (12, "shrinkage_covariance", opt_src),
        (12, "regime_adjusted_covariance", opt_src),
        (12, "get_current_vix", opt_src),
        (12, "forward_looking_cov_adjustment", opt_src),
        (12, "stress_test_corr_one", opt_src),
        (12, "stress_test_vol_shock", opt_src),
        (12, "stress_test_systemic", opt_src),
        (12, "run_all_stress_tests", opt_src),
        (12, "get_sector_map", opt_src),
        (12, "compute_stress_weight_caps", opt_src),
        (12, "optimize_with_sector_and_stress", opt_src),
        (15, "_normalizar_df_yf", ml_src),
        (16, "_train_xgb_robust", ml_src),
        (16, "_train_short_term_model_robust", ml_src),
        (16, "_train_long_term_model_robust", ml_src),
        (17, "_process_macro_with_lag", ml_src),
        (19, "_calcular_rebalanceo", pipe_src),
    ]

    for cell_idx, fn_name, mod_src in verbatim_checks:
        nb_src = "".join(nb["cells"][cell_idx]["source"])
        assert get_func_ast(nb_src, fn_name) == get_func_ast(mod_src, fn_name), (
            f"AST mismatch for {fn_name} (Cell {cell_idx:02d})"
        )


def test_numerical_parity_optimization_bl_covariance_stress(notebook_namespace, monkeypatch):
    prices = _make_prices()
    daily = prices.pct_change().dropna()
    tickers = list(prices.columns)
    rf = 0.04

    # Covariance shrinkage (LW + OAS) and regime blend
    for method in ("ledoit_wolf", "oas"):
        c_nb, s_nb = notebook_namespace["shrinkage_covariance"](daily, method=method)
        c_pk, s_pk = optimization.shrinkage_covariance(daily, method=method)
        np.testing.assert_allclose(c_pk.values, c_nb.values, rtol=1e-12, atol=1e-12)
        assert s_pk == pytest.approx(s_nb, rel=1e-12)

    cov_nb, calm_nb, cri_nb = notebook_namespace["regime_adjusted_covariance"](daily)
    cov_pk, calm_pk, cri_pk = optimization.regime_adjusted_covariance(daily)
    np.testing.assert_allclose(cov_pk.values, cov_nb.values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(calm_pk.values, calm_nb.values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cri_pk.values, cri_nb.values, rtol=1e-12, atol=1e-12)

    # Expected returns Pipeline A & Pipeline B (with matching country risk globals)
    monkeypatch.setattr(state, "country_risk_premia", {"Brazil": 1.8})
    monkeypatch.setattr(state, "prima_riesgo_mas_reciente", 6.5)

    caps = {t: 1e10 * (i + 1) for i, t in enumerate(tickers)}
    notebook_namespace["get_market_caps"] = lambda ts: caps
    monkeypatch.setattr(optimization, "get_market_caps", lambda ts: caps)

    ml_views_in = {t: {"scenarios": {"base": {"predicted_return": 0.08 * (i + 1)}}} for i, t in enumerate(tickers)}
    cmap = {"T1": "Argentina", "T2": "Argentina", "T3": "Brazil"}

    mu_a_nb, src_a_nb = notebook_namespace["compute_expected_returns_pipeline"](
        tickers, daily, cov_nb, {"T1": 0.18}, None, cmap, 0.065
    )
    mu_a_pk, src_a_pk = optimization.compute_expected_returns_pipeline(
        tickers, daily, cov_pk, {"T1": 0.18}, None, cmap, 0.065
    )
    np.testing.assert_allclose(mu_a_pk.values, mu_a_nb.values, rtol=1e-12, atol=1e-12)
    assert src_a_pk == src_a_nb

    mu_b_nb, src_b_nb = notebook_namespace["compute_expected_returns_pipeline"](
        tickers, daily, cov_nb, {}, ml_views_in, cmap, 0.065
    )
    mu_b_pk, src_b_pk = optimization.compute_expected_returns_pipeline(
        tickers, daily, cov_pk, {}, ml_views_in, cmap, 0.065
    )
    np.testing.assert_allclose(mu_b_pk.values, mu_b_nb.values, rtol=1e-12, atol=1e-12)
    assert src_b_pk == src_b_nb

    # Optimizers & stress tests
    sector_map = {t: f"Sec{i % 3}" for i, t in enumerate(tickers)}
    w_sh_nb = notebook_namespace["optimize_sharpe_hhi"](mu_b_nb, cov_nb, rf)
    w_sh_pk = optimization.optimize_sharpe_hhi(mu_b_pk, cov_pk, rf)
    np.testing.assert_allclose(w_sh_pk, w_sh_nb, rtol=1e-12, atol=1e-12)

    w_sec_nb = notebook_namespace["optimize_with_sector_and_stress"](mu_b_nb, cov_nb, rf, tickers, sector_map=sector_map)
    w_sec_pk = optimization.optimize_with_sector_and_stress(mu_b_pk, cov_pk, rf, tickers, sector_map=sector_map)
    np.testing.assert_allclose(w_sec_pk, w_sec_nb, rtol=1e-12, atol=1e-12)

    w_mv_nb = notebook_namespace["_optimize_min_variance"](mu_b_nb, cov_nb)
    w_mv_pk = optimization._optimize_min_variance(mu_b_pk, cov_pk)
    np.testing.assert_allclose(w_mv_pk, w_mv_nb, rtol=1e-12, atol=1e-12)

    w_cv_nb = notebook_namespace["_optimize_min_cvar"](mu_b_nb, cov_nb, daily)
    w_cv_pk = optimization._optimize_min_cvar(mu_b_pk, cov_pk, daily)
    np.testing.assert_allclose(w_cv_pk, w_cv_nb, rtol=1e-12, atol=1e-12)

    st_nb = notebook_namespace["run_all_stress_tests"](w_sec_nb, cov_nb, tickers)
    st_pk = optimization.run_all_stress_tests(w_sec_pk, cov_pk, tickers)
    for k in ("normal_vol", "stress_vol", "diversification_ratio", "vol_increase_pct"):
        assert st_pk["corr_one"][k] == pytest.approx(st_nb["corr_one"][k], rel=1e-12)
    assert st_pk["systemic"]["max_daily_loss_99"] == pytest.approx(st_nb["systemic"]["max_daily_loss_99"], rel=1e-12)
    assert st_pk["vol_shock"]["stress_vol"] == pytest.approx(st_nb["vol_shock"]["stress_vol"], rel=1e-12)


def test_numerical_parity_analyze_portfolio_and_rebalance(notebook_namespace, monkeypatch):
    prices = _make_prices()
    tickers = list(prices.columns)
    assets = [
        {"ticker": t, "mode": "existing", "quantity": 10 + i, "buy_price": 90.0, "pais_riesgo": "Argentina" if i == 1 else "United States"}
        for i, t in enumerate(tickers)
    ]
    sector_map = {t: f"Sec{i % 3}" for i, t in enumerate(tickers)}
    caps = {t: 1e10 * (i + 1) for i, t in enumerate(tickers)}

    # Mock identically on both sides
    notebook_namespace["_download_prices"] = lambda ts, period="10y": (prices[ts].copy(), [])
    notebook_namespace["get_current_vix"] = lambda: 27.5
    notebook_namespace["get_sector_map"] = lambda ts: sector_map
    notebook_namespace["get_market_caps"] = lambda ts: caps

    monkeypatch.setattr(optimization, "_download_prices", lambda ts, period="10y": (prices[ts].copy(), []))
    monkeypatch.setattr(optimization, "get_current_vix", lambda: 27.5)
    monkeypatch.setattr(optimization, "get_sector_map", lambda ts: sector_map)
    monkeypatch.setattr(optimization, "get_market_caps", lambda ts: caps)
    monkeypatch.setattr(state, "prima_riesgo_mas_reciente", 6.5)
    monkeypatch.setattr(state, "country_risk_premia", {"Brazil": 1.8})

    np.random.seed(999)
    res_nb = notebook_namespace["analyze_portfolio"](assets=assets, rf_rate=0.04, samples=200)
    np.random.seed(999)
    res_pk = optimization.analyze_portfolio(assets=assets, rf_rate=0.04, samples=200)

    assert res_pk.tickers == res_nb.tickers
    assert res_pk.cov_inflation_factor == res_nb.cov_inflation_factor
    assert res_pk.mu_sources == res_nb.mu_sources
    np.testing.assert_allclose(res_pk.expected_returns.values, res_nb.expected_returns.values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_pk.cov_matrix.values, res_nb.cov_matrix.values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_pk.weights_sharpe, res_nb.weights_sharpe, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_pk.weights_minvar, res_nb.weights_minvar, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_pk.weights_cvar, res_nb.weights_cvar, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_pk.mcr_sharpe.values, res_nb.mcr_sharpe.values, rtol=1e-12, atol=1e-12)

    prices_last = prices.ffill().iloc[-1]
    reb_nb = notebook_namespace["_calcular_rebalanceo"](assets, tickers, res_nb.weights_sharpe, prices_last)
    reb_pk = pipeline._calcular_rebalanceo(assets, tickers, res_pk.weights_sharpe, prices_last)
    assert len(reb_pk) == len(reb_nb)
    for r_pk, r_nb in zip(reb_pk, reb_nb):
        assert r_pk["ticker"] == r_nb["ticker"]
        for k in ("peso_actual", "peso_objetivo", "cantidad_objetivo", "diferencia_cantidad", "diferencia_valor", "ganancia_perdida_pct"):
            assert r_pk[k] == pytest.approx(r_nb[k], rel=1e-12, abs=1e-12)


def test_numerical_parity_ml_processor_and_predictor(notebook_namespace):
    dates = pd.bdate_range("2015-01-01", periods=1550)
    rng = np.random.default_rng(77)
    prices = pd.DataFrame({
        "date": dates,
        "SYM": 80.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, len(dates)))),
    })
    inc = pd.DataFrame({
        "fiscalDateEnding": pd.to_datetime(["2020-12-31", "2019-12-31"]),
        "totalRevenue": [1200.0, 1000.0],
        "netIncome": [240.0, 180.0],
    })
    bal = pd.DataFrame({
        "fiscalDateEnding": pd.to_datetime(["2020-12-31", "2019-12-31"]),
        "totalAssets": [4000.0, 3800.0],
        "totalShareholderEquity": [2000.0, 1900.0],
        "totalLiabilities": [2000.0, 1900.0],
        "totalCurrentAssets": [1000.0, 950.0],
        "totalCurrentLiabilities": [500.0, 480.0],
    })
    cf = pd.DataFrame({
        "fiscalDateEnding": pd.to_datetime(["2020-12-31", "2019-12-31"]),
        "operatingCashflow": [310.0, 260.0],
        "capitalExpenditure": [-60.0, -50.0],
    })
    fin = {"SYM": {"income_statement": {"annual": inc}, "balance_sheet": {"annual": bal}, "cash_flow": {"annual": cf}}}
    econ = {"fed_rate": pd.DataFrame({"date": pd.to_datetime(["2015-01-01", "2019-06-01"]), "fed_rate": [0.25, 2.0]})}

    ing_nb = notebook_namespace["IngestedData"](financials=fin, prices=prices.copy(), economic=econ)
    ing_pk = ml.IngestedData(financials=fin, prices=prices.copy(), economic=econ)

    proc_nb = notebook_namespace["PortfolioProcessor"](ing_nb).process_all()
    proc_pk = ml.PortfolioProcessor(ing_pk).process_all()
    pd.testing.assert_frame_equal(proc_pk["SYM"], proc_nb["SYM"])

    pred_nb = notebook_namespace["PortfolioPredictor"](proc_nb)
    pred_pk = ml.PortfolioPredictor(proc_pk)

    res1q_nb = pred_nb.train_short_term_model("SYM")
    res1q_pk = pred_pk.train_short_term_model("SYM")
    assert res1q_pk["predicted_return"] == pytest.approx(res1q_nb["predicted_return"], rel=1e-12)
    assert res1q_pk["predicted_price"] == pytest.approx(res1q_nb["predicted_price"], rel=1e-12)
    assert res1q_pk["metrics"] == res1q_nb["metrics"]

    res5y_nb = pred_nb.train_long_term_model("SYM")
    res5y_pk = pred_pk.train_long_term_model("SYM")
    for sc in ("base", "bull", "bear"):
        assert res5y_pk["scenarios"][sc]["predicted_return"] == pytest.approx(
            res5y_nb["scenarios"][sc]["predicted_return"], rel=1e-12
        )
        assert res5y_pk["scenarios"][sc]["predicted_price"] == pytest.approx(
            res5y_nb["scenarios"][sc]["predicted_price"], rel=1e-12
        )
    assert res5y_pk["metrics"] == res5y_nb["metrics"]
