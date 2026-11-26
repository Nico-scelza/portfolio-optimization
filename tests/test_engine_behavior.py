"""Offline characterization tests for all migrated portfolio_optimization modules."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import portfolio_optimization as po
from portfolio_optimization import analytics, config, country_risk, market_data, ml, optimization, pipeline, state


def _synthetic_prices(n_days: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    drifts = np.array([0.0004, 0.0003, 0.00025, 0.00035, 0.0002, 0.00045])
    vols = np.array([0.012, 0.015, 0.010, 0.018, 0.011, 0.022])
    common = rng.normal(0, 0.006, size=(n_days, 1))
    idio = rng.normal(0, 1.0, size=(n_days, len(tickers))) * vols
    rets = drifts + 0.5 * common + idio
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def test_country_risk_argentina_ambito_and_rava_fallback(monkeypatch):
    # 1. Ámbito success
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = [
        ["Fecha", "Puntos"],
        ["01-11-2026", "620,50"],
        ["02-11-2026", "615,25"],
    ]
    fake_session = MagicMock()
    fake_session.get.return_value = fake_resp
    monkeypatch.setattr(country_risk.requests, "Session", lambda: fake_session)

    res = country_risk.obtener_riesgo_pais_argentina(allow_manual_input=False, render_html=False)
    assert res["fuente_datos"] == "Ámbito Financiero"
    assert res["fecha_actualizacion"] == "02/11/2026"
    assert res["prima_riesgo_mas_reciente"] == pytest.approx(6.1525)
    assert state.prima_riesgo_mas_reciente == pytest.approx(6.1525)

    # 2. Ámbito fails -> Rava fallback
    resp_ambito_fail = MagicMock(status_code=503)
    resp_rava_ok = MagicMock(status_code=200, content=b"<div class='price'>650,40</div>")
    sessions = iter([MagicMock(get=MagicMock(return_value=resp_ambito_fail)),
                     MagicMock(get=MagicMock(return_value=resp_rava_ok))])
    monkeypatch.setattr(country_risk.requests, "Session", lambda: next(sessions))

    res2 = country_risk.obtener_riesgo_pais_argentina(allow_manual_input=False, render_html=False)
    assert res2["fuente_datos"] == "Rava Bursátil"
    assert res2["prima_riesgo_mas_reciente"] == pytest.approx(6.504)

    # 3. Both fail -> explicit RuntimeError without hanging on input()
    resp_fail = MagicMock(status_code=500, content=b"")
    monkeypatch.setattr(country_risk.requests, "Session", lambda: MagicMock(get=MagicMock(return_value=resp_fail)))
    with pytest.raises(RuntimeError, match="ERROR CRÍTICO"):
        country_risk.obtener_riesgo_pais_argentina(allow_manual_input=False, render_html=False)


def test_calcular_primas_riesgo_pais(monkeypatch):
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2023-01-01", periods=250)
    urth_close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 250))), index=dates)
    ewz_close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 250))), index=dates)
    short_close = pd.Series(np.linspace(10, 12, 50), index=dates[:50])

    def fake_ticker(sym):
        t = MagicMock()
        if sym == "URTH":
            t.history.return_value = pd.DataFrame({"Close": urth_close})
        elif sym == "EWZ":
            t.history.return_value = pd.DataFrame({"Close": ewz_close})
        else:
            t.history.return_value = pd.DataFrame({"Close": short_close})
        return t

    monkeypatch.setattr(country_risk.yf, "Ticker", fake_ticker)
    premia = country_risk.calcular_primas_riesgo_pais({"Brazil": "EWZ", "ShortCountry": "SHRT"})
    expected = (
        np.log(ewz_close / ewz_close.shift(1)).dropna().std() * np.sqrt(252)
    ) / (
        np.log(urth_close / urth_close.shift(1)).dropna().std() * np.sqrt(252)
    )
    assert "Brazil" in premia
    assert "ShortCountry" not in premia
    assert premia["Brazil"] == pytest.approx(expected, rel=1e-12)
    assert state.country_risk_premia["Brazil"] == pytest.approx(expected, rel=1e-12)


def test_download_prices_usd_inverse_and_direct_fx(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=10)
    usd_hist = pd.DataFrame({"Close": np.full(10, 100.0)}, index=dates)
    eur_hist = pd.DataFrame({"Close": np.full(10, 50.0)}, index=dates)
    brl_hist = pd.DataFrame({"Close": np.full(10, 500.0)}, index=dates)

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
        else:
            t.info = {"currency": "ZZZ"}
            t.history.return_value = usd_hist.copy()
        return t

    def fake_download(fx_sym, period="10y", progress=False):
        if fx_sym == "EURUSD=X":
            return pd.DataFrame({"Close": np.full(10, 1.1)}, index=dates)
        if fx_sym == "BRL=X":
            return pd.DataFrame({"Close": np.full(10, 5.0)}, index=dates)
        return pd.DataFrame()

    monkeypatch.setattr(market_data.yf, "Ticker", fake_ticker)
    monkeypatch.setattr(market_data.yf, "download", fake_download)

    prices, dropped = market_data._download_prices(["US1", "EU1", "BR1", "BAD"], period="1y")
    assert dropped == ["BAD"]
    assert list(prices.columns) == ["US1", "EU1", "BR1"]
    np.testing.assert_allclose(prices["US1"].values, 100.0)
    np.testing.assert_allclose(prices["EU1"].values, 55.0)
    np.testing.assert_allclose(prices["BR1"].values, 100.0)


def test_get_risk_free_rate_paths(monkeypatch):
    # Path 1: ^TNX works
    t_ok = MagicMock()
    t_ok.history.return_value = pd.DataFrame({"Close": [4.25]})
    monkeypatch.setattr(market_data.yf, "Ticker", lambda _: t_ok)
    assert market_data.get_risk_free_rate() == pytest.approx(0.0425)

    # Path 2: ^TNX fails, no FRED key -> skips HTTP and returns default
    t_bad = MagicMock()
    t_bad.history.side_effect = RuntimeError("no network")
    monkeypatch.setattr(market_data.yf, "Ticker", lambda _: t_bad)
    monkeypatch.setattr(config, "FRED_API_KEY", "")
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    mock_get = MagicMock()
    monkeypatch.setattr(market_data.requests, "get", mock_get)
    assert market_data.get_risk_free_rate(default=0.035) == pytest.approx(0.035)
    mock_get.assert_not_called()

    # Path 3: ^TNX fails, FRED key configured -> uses params dict (not raw URL key)
    monkeypatch.setenv("FRED_API_KEY", "FAKE_FRED_KEY")
    resp = MagicMock()
    resp.json.return_value = {"observations": [{"value": "."}, {"value": "4.10"}]}
    mock_get = MagicMock(return_value=resp)
    monkeypatch.setattr(market_data.requests, "get", mock_get)
    assert market_data.get_risk_free_rate() == pytest.approx(0.041)
    call_args, call_kwargs = mock_get.call_args
    assert "FAKE_FRED_KEY" not in call_args[0]
    assert call_kwargs["params"]["api_key"] == "FAKE_FRED_KEY"


def test_vix_cov_adjustment_brackets():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"])
    for vix, expected_factor in [
        (None, 1.0),
        (float("nan"), 1.0),
        (15.0, 1.0),
        (20.0, 1.0),
        (21.0, 1.05),
        (26.0, 1.15),
        (31.0, 1.25),
        (36.0, 1.40),
    ]:
        adj, factor = optimization.forward_looking_cov_adjustment(cov, vix)
        assert factor == pytest.approx(expected_factor)
        np.testing.assert_allclose(adj.values, cov.values * expected_factor)


def test_expected_returns_pipeline_a_sacred_dcf_and_pipeline_b_bl(monkeypatch):
    prices = _synthetic_prices(300)
    daily = prices.pct_change().dropna()
    cov, _ = optimization.shrinkage_covariance(daily, method="ledoit_wolf")
    tickers = list(prices.columns)

    monkeypatch.setattr(state, "country_risk_premia", {"Brazil": 2.0})

    # Pipeline A: DCF sacred, non-DCF gets historical mean + country risk adjustment
    mu_a, src_a = optimization.compute_expected_returns_pipeline(
        tickers_list=tickers,
        daily_returns=daily,
        cov_matrix=cov,
        dcf_upsides={"AAA": 0.25},
        country_risk_map={"AAA": "Argentina", "BBB": "Argentina", "CCC": "Brazil"},
        country_risk_value=0.06,
    )
    assert mu_a["AAA"] == pytest.approx(0.25)  # sacred, unaffected by Argentina country risk
    assert "DCF/Usuario" in src_a["AAA"]
    assert mu_a["BBB"] == pytest.approx(daily["BBB"].mean() * 252 - 0.06)
    assert mu_a["CCC"] == pytest.approx(daily["CCC"].mean() * 252 - 0.03)

    # Pipeline B: No DCF -> Black-Litterman + ML views
    monkeypatch.setattr(optimization, "get_market_caps", lambda ts: {t: 1e11 * (i + 1) for i, t in enumerate(ts)})
    ml_preds = {
        t: {"scenarios": {"base": {"predicted_return": 0.1 * (i + 1)}}}
        for i, t in enumerate(tickers)
    }
    mu_b, src_b = optimization.compute_expected_returns_pipeline(
        tickers_list=tickers,
        daily_returns=daily,
        cov_matrix=cov,
        dcf_upsides={},
        ml_predictions_5y=ml_preds,
        country_risk_map={},
        country_risk_value=0.0,
    )
    assert all("Black-Litterman (mercado + ML views" in s for s in src_b.values())
    assert np.all(np.isfinite(mu_b.values))


def test_ml_processor_90_day_lag_and_predictor_robust():
    dates = pd.bdate_range("2015-01-01", periods=1700)
    rng = np.random.default_rng(42)
    prices = pd.DataFrame({
        "date": dates,
        "AAA": 50.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(dates)))),
    })

    inc = pd.DataFrame({
        "fiscalDateEnding": pd.to_datetime(["2020-12-31", "2019-12-31"]),
        "totalRevenue": [1000.0, 900.0],
        "netIncome": [200.0, 150.0],
    })
    bal = pd.DataFrame({
        "fiscalDateEnding": pd.to_datetime(["2020-12-31", "2019-12-31"]),
        "totalAssets": [5000.0, 4500.0],
        "totalShareholderEquity": [2500.0, 2200.0],
        "totalLiabilities": [2500.0, 2300.0],
        "totalCurrentAssets": [1500.0, 1200.0],
        "totalCurrentLiabilities": [750.0, 600.0],
    })
    cf = pd.DataFrame({
        "fiscalDateEnding": pd.to_datetime(["2020-12-31", "2019-12-31"]),
        "operatingCashflow": [300.0, 250.0],
        "capitalExpenditure": [-80.0, -70.0],
    })
    macro = {"fed_rate": pd.DataFrame({"date": pd.to_datetime(["2015-01-01", "2020-01-01"]), "fed_rate": [0.25, 1.5]})}

    ingested = ml.IngestedData(
        financials={"AAA": {"income_statement": {"annual": inc}, "balance_sheet": {"annual": bal}, "cash_flow": {"annual": cf}}},
        prices=prices,
        economic=macro,
    )
    processor = ml.PortfolioProcessor(ingested)
    processed = processor.process_all()
    assert "AAA" in processed
    df = processed["AAA"]

    # Check 90-day look-ahead bias lag: 2020-12-31 + 90 days = 2021-03-31
    before_report = df[df["date"] == pd.Timestamp("2021-03-01")]
    after_report = df[df["date"] == pd.Timestamp("2021-04-05")]
    assert not before_report.empty and not after_report.empty
    assert before_report["profit_margin"].iloc[0] == pytest.approx(150.0 / 900.0)
    assert after_report["profit_margin"].iloc[0] == pytest.approx(200.0 / 1000.0)

    predictor = ml.PortfolioPredictor(processed)
    pred_1q = predictor.train_short_term_model("AAA")
    pred_5y = predictor.train_long_term_model("AAA")
    assert pred_1q is not None and "metrics" in pred_1q
    assert pred_5y is not None and "metrics" in pred_5y
    assert set(pred_5y["scenarios"].keys()) == {"base", "bull", "bear"}
    assert pred_5y["scenarios"]["bull"]["predicted_return"] == pytest.approx(
        pred_5y["scenarios"]["base"]["predicted_return"] * 1.15
    )


def test_run_portfolio_analysis_headless_and_analytics(monkeypatch):
    prices = _synthetic_prices(350)
    tickers = list(prices.columns)

    monkeypatch.setattr(optimization, "_download_prices", lambda ts, period="10y": (prices[ts].copy(), []))
    monkeypatch.setattr(market_data, "_download_prices", lambda ts, period="10y": (prices[ts].copy(), []))
    monkeypatch.setattr(optimization, "get_current_vix", lambda: 22.0)
    monkeypatch.setattr(optimization, "get_sector_map", lambda ts: {t: f"Sector{i % 3}" for i, t in enumerate(ts)})
    monkeypatch.setattr(optimization, "get_market_caps", lambda ts: {t: 1e10 * (i + 1) for i, t in enumerate(ts)})
    monkeypatch.setattr(pipeline, "obtener_tasa_libre_riesgo", lambda: 0.04)
    monkeypatch.setattr(analytics, "obtener_tasa_libre_riesgo", lambda: 0.04)
    monkeypatch.setattr(analytics, "descargar_precios_historicos", lambda ts, periodo="10y": prices[ts].copy())
    monkeypatch.setattr(
        analytics.yf,
        "download",
        lambda ts, period="10y", progress=False, threads=True: pd.concat({"Close": prices[ts]}, axis=1),
    )

    activos = [
        {"ticker": t, "cantidad": 10.0, "precio_compra": 95.0, "upside_esperado": 12.0 if i == 0 else 0.0}
        for i, t in enumerate(tickers)
    ]
    state.activos_portfolio = activos

    res = po.run_portfolio_analysis(tickers_list=tickers, activos_dict=activos, ejecutar_ml=False, render_ui=False)
    assert isinstance(res, po.AnalysisResult)
    assert res.tickers == tickers
    assert res.cov_inflation_factor == pytest.approx(1.05)
    assert np.isclose(res.weights_sharpe.sum(), 1.0, atol=1e-6)
    assert np.isclose(res.weights_minvar.sum(), 1.0, atol=1e-6)
    assert np.isclose(res.weights_cvar.sum(), 1.0, atol=1e-6)
    assert len(state.rebalanceo_sharpe) == len(tickers)

    # Headless execution of analytics cells (Cells 22, 24, 26, 30, 32, 34, 36)
    with po._display.suppress_display():
        analytics.generar_frontera_eficiente(n_samples=200)
        analytics.analizar_rendimiento_historico()
        analytics.generar_heatmap_correlaciones()
        analytics.generar_underwater_plot()
        analytics.analizar_contribucion_marginal_riesgo()
        analytics.generar_grafico_riesgo()
        alloc = analytics.armar_cartera(capital_disponible=10000.0)
        assert len(alloc) == 4
        for df_dist in alloc.values():
            assert df_dist["Monto (USD)"].sum() == pytest.approx(10000.0, rel=1e-3)


def test_edge_cases_nan_fallbacks_and_exceptions(monkeypatch):
    # Empty assets / <2 assets -> ValueError
    with pytest.raises(ValueError, match="No hay tickers"):
        optimization.analyze_portfolio(assets=[], rf_rate=0.03)

    single_price = _synthetic_prices(100)[["AAA"]]
    monkeypatch.setattr(optimization, "_download_prices", lambda ts, period="10y": (single_price.copy(), []))
    with pytest.raises(ValueError, match="retornos suficientes"):
        optimization.analyze_portfolio(assets=[{"ticker": "AAA"}], rf_rate=0.03)

    # Short series (<63 rows) -> regime_adjusted_covariance 1.5x fallback
    short_daily = _synthetic_prices(40).pct_change().dropna()
    cov_blend, cov_calm, cov_crisis = optimization.regime_adjusted_covariance(short_daily)
    np.testing.assert_allclose(cov_crisis.values, cov_calm.values * 1.5, rtol=1e-12)
    np.testing.assert_allclose(cov_blend.values, 0.7 * cov_calm.values + 0.3 * cov_crisis.values, rtol=1e-12)

    # Infeasible bounds (<5 assets with 0.20 max weight) -> equal weights fallback
    mu3 = pd.Series([0.10, 0.12, 0.08], index=["A", "B", "C"])
    cov3 = pd.DataFrame(np.eye(3) * 0.04, index=["A", "B", "C"], columns=["A", "B", "C"])
    w_fallback = optimization.optimize_sharpe_hhi(mu3, cov3, rf=0.03)
    np.testing.assert_allclose(w_fallback, np.array([1 / 3, 1 / 3, 1 / 3]), atol=1e-12)

    # NaN / zero edge cases in metrics, MCR, and rebalance
    r, v, s = market_data.metricas_portfolio(None, None, None, 0.03)
    assert np.isnan(r) and np.isnan(v) and np.isnan(s)
    zero_cov = pd.DataFrame(np.zeros((2, 2)), index=["A", "B"], columns=["A", "B"])
    mcr_zero = optimization._calc_mcr(np.array([0.5, 0.5]), zero_cov, ["A", "B"])
    np.testing.assert_allclose(mcr_zero.values, [0.0, 0.0])
    assert pipeline._calcular_rebalanceo([], ["A"], np.array([1.0]), pd.Series({"A": 100.0})) == []
    assert pipeline._calcular_rebalanceo([{"ticker": "A", "quantity": 1}], ["A"], np.array([1.0]), pd.Series(dtype=float)) == []

    # Missing FRED_API_KEY in DataIngestor -> returns None without HTTP call
    monkeypatch.setattr(config, "FRED_API_KEY", "")
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    ing = ml.DataIngestor(["AAA"])
    ing.session.get = MagicMock()
    assert ing._fetch_fred_series("GDP", "GDP") is None
    ing.session.get.assert_not_called()

    # PortfolioPredictor short series / missing ticker -> None
    pred = ml.PortfolioPredictor({"SHORT": pd.DataFrame({"price": [1.0, 2.0], "target_next_return": [0.1, 0.2]})})
    assert pred.train_short_term_model("MISSING") is None
    assert pred.train_short_term_model("SHORT") is None
    assert pred.train_long_term_model("SHORT") is None
