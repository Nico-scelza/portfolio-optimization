"""Preserved and modernized offline engine characterization tests.

Covers risk-free rate retrieval paths (^TNX -> FRED DGS10 -> default),
_download_prices FX normalization, stress tests (corr=1, vol shock, systemic),
metricas_portfolio / CVaR edge cases, and explicit fixed-income YTM BL mode.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization import (
    BlackLittermanConfig,
    CashFlowEvent,
    FixedIncomeTerms,
    InstrumentConfig,
    PortfolioAnalysisConfig,
    RiskModelConfig,
    analyze_portfolio,
    calcular_cvar_historico,
    config,
    market_data,
    metricas_portfolio,
    run_all_stress_tests,
)


def test_download_prices_usd_inverse_direct_and_subunit_fx(monkeypatch):
    dates = pd.bdate_range("2025-01-06", periods=10)
    usd_hist = pd.DataFrame({"Close": np.full(10, 100.0)}, index=dates)
    eur_hist = pd.DataFrame({"Close": np.full(10, 50.0)}, index=dates)
    brl_hist = pd.DataFrame({"Close": np.full(10, 500.0)}, index=dates)
    gbp_pence_hist = pd.DataFrame({"Close": np.full(10, 2000.0)}, index=dates)

    def fake_ticker(sym):
        t = MagicMock()
        if sym == "US1":
            t.info = {"currency": "USD"}
            t.history.return_value = usd_hist.copy()
        elif sym == "EU1":
            t.info = {"currency": "EUR"}
            t.history.return_value = eur_hist.copy()
        elif sym == "BR1":
            t.info = {"currency": "BRL"}
            t.history.return_value = brl_hist.copy()
        elif sym == "UK1.L":
            t.info = {"currency": "GBp"}
            t.history.return_value = gbp_pence_hist.copy()
        else:
            t.info = {"currency": "ZZZ"}
            t.history.return_value = usd_hist.copy()
        return t

    def fake_fx_fetcher(fx_sym, period="10y"):
        if fx_sym == "EURUSD=X":
            return pd.Series(np.full(10, 1.10), index=dates)
        if fx_sym == "BRL=X":
            return pd.Series(np.full(10, 5.00), index=dates)
        if fx_sym == "GBPUSD=X":
            return pd.Series(np.full(10, 1.25), index=dates)
        return pd.Series(dtype=float)

    monkeypatch.setattr(market_data.yf, "Ticker", fake_ticker)

    prices, dropped = market_data._download_prices(
        ["US1", "EU1", "BR1", "UK1.L", "BAD"],
        period="1y",
        fx_fetcher=fake_fx_fetcher,
    )
    assert dropped == ["BAD"]
    assert list(prices.columns) == ["US1", "EU1", "BR1", "UK1.L"]
    np.testing.assert_allclose(prices["US1"].values, 100.0)
    np.testing.assert_allclose(prices["EU1"].values, 55.0)
    np.testing.assert_allclose(prices["BR1"].values, 100.0)
    np.testing.assert_allclose(prices["UK1.L"].values, 25.0)


def test_get_risk_free_rate_paths(monkeypatch):
    t_ok = MagicMock()
    t_ok.history.return_value = pd.DataFrame({"Close": [4.25]})
    monkeypatch.setattr(market_data.yf, "Ticker", lambda _: t_ok)
    assert market_data.get_risk_free_rate() == pytest.approx(0.0425)

    t_bad = MagicMock()
    t_bad.history.side_effect = RuntimeError("no network")
    monkeypatch.setattr(market_data.yf, "Ticker", lambda _: t_bad)
    monkeypatch.setattr(config, "FRED_API_KEY", "")
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    mock_get = MagicMock()
    monkeypatch.setattr(market_data.requests, "get", mock_get)
    assert market_data.get_risk_free_rate(default=0.035) == pytest.approx(0.035)
    mock_get.assert_not_called()

    monkeypatch.setenv("FRED_API_KEY", "FAKE_FRED_KEY")
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"observations": [{"value": "."}, {"value": "4.10"}]}
    mock_get = MagicMock(return_value=resp)
    monkeypatch.setattr(market_data.requests, "get", mock_get)
    assert market_data.get_risk_free_rate() == pytest.approx(0.041)
    call_args, call_kwargs = mock_get.call_args
    assert "FAKE_FRED_KEY" not in call_args[0]
    assert call_kwargs["params"]["api_key"] == "FAKE_FRED_KEY"
    assert call_kwargs["verify"] is True


def test_stress_tests_and_metric_edge_cases():
    r, v, s = metricas_portfolio(None, None, None, 0.04)
    assert np.isnan(r) and np.isnan(v) and np.isnan(s)

    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"])
    mu = pd.Series([0.08, 0.12], index=["A", "B"])
    w = np.array([0.6, 0.4])
    ret, vol, sharpe = metricas_portfolio(w, mu, cov, 0.04)
    assert ret == pytest.approx(0.096)
    assert vol == pytest.approx(np.sqrt(w @ cov.values @ w))
    assert sharpe == pytest.approx((ret - 0.04) / vol)

    st = run_all_stress_tests(w, cov, ["A", "B"])
    assert st["corr_one"]["stress_vol"] >= st["corr_one"]["normal_vol"]
    assert st["vol_shock"]["stress_vol"] == pytest.approx(st["vol_shock"]["normal_vol"] * 2.0)
    assert st["systemic"]["max_weekly_loss_99"] > 0.0

    rets = pd.DataFrame({"A": [-0.05, -0.02, 0.01, 0.03, 0.02], "B": [-0.04, -0.01, 0.02, 0.01, 0.03]})
    cvar = calcular_cvar_historico(w, rets, alpha=0.95)
    assert cvar < 0.0


def test_sovereign_bond_with_explicit_proxy_and_metric_only_vs_ytm_view():
    """Sovereign bonds remain eligible fixed-income instruments with explicit proxy & YTM handling."""
    rng = np.random.default_rng(88)
    dates = pd.date_range("2022-01-07", periods=90, freq="W-FRI")
    prices = pd.DataFrame({
        "SPY": 400.0 * np.exp(np.cumsum(rng.normal(0.002, 0.02, len(dates)))),
        "EMB": 85.0 * np.exp(np.cumsum(rng.normal(0.001, 0.012, len(dates)))),
    }, index=dates)

    bond_terms = FixedIncomeTerms(
        cashflow_currency="USD",
        reference_nominal=100.0,
        settlement_date=date(2025, 1, 1),
        maturity_date=date(2027, 1, 1),
        price_convention="per_100_original_nominal",
        price_type="dirty",
        cashflows=(
            CashFlowEvent(payment_date=date(2026, 1, 1), coupon_amount=8.0, amortization_amount=0.0),
            CashFlowEvent(payment_date=date(2027, 1, 1), coupon_amount=8.0, amortization_amount=100.0),
        ),
    )

    cfg_metric = PortfolioAnalysisConfig(
        variable_income=(InstrumentConfig(instrument_id="SPY", ticker="SPY", section="variable_income", asset_subtype="equity_etf"),),
        fixed_income=(
            InstrumentConfig(
                instrument_id="SOV27",
                ticker="SOV27",
                section="fixed_income",
                asset_subtype="sovereign_bond",
                quote_source="manual",
                manual_price_usd=92.0,
                historical_proxy_ticker="EMB",
                fixed_income_terms=bond_terms,
            ),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04, fixed_income_bl_view_mode="metric_only"),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52, allow_explicit_proxies=True),
    )
    res_metric = analyze_portfolio(cfg_metric, prices_usd_override=prices)
    assert res_metric.fixed_income_metrics["SOV27"].status == "ok"
    assert res_metric.fixed_income_metrics["SOV27"].ytm_annual > 0.12
    assert res_metric.black_litterman.active_view_assets == []
    assert res_metric.risk_model.proxies_used == {"SOV27": "EMB"}

    cfg_ytm_view = PortfolioAnalysisConfig(
        variable_income=cfg_metric.variable_income,
        fixed_income=cfg_metric.fixed_income,
        black_litterman=BlackLittermanConfig(rf_rate=0.04, fixed_income_bl_view_mode="explicit_ytm_view", fixed_income_view_confidence=0.50),
        risk=cfg_metric.risk,
    )
    res_view = analyze_portfolio(cfg_ytm_view, prices_usd_override=prices)
    assert "SOV27" in res_view.black_litterman.active_view_assets
    assert res_view.expected_returns["SOV27"] > res_metric.expected_returns["SOV27"]
