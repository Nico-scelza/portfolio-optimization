"""Regression suite for NEW-04: timezone-aware FX conversion, yfinance daily bars, and multi-exchange alignment."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization import (
    BlackLittermanConfig,
    InstrumentConfig,
    OptimizationConfig,
    PortfolioAnalysisConfig,
    RiskModelConfig,
    analyze_portfolio,
    compute_weekly_returns_w_fri,
    market_data,
)


def test_new04_tz_aware_yfinance_daily_intraday_fx_and_multitimezone_alignment(monkeypatch):
    """NEW-04: Preserve timezone semantics across convert_price_series_to_usd, _download_prices, and analyze_portfolio."""
    rng = np.random.default_rng(4044)

    bdays_ny = pd.date_range("2024-01-05", "2025-07-04", freq="B", tz="America/New_York")
    bdays_lon = pd.date_range("2024-01-05", "2025-07-04", freq="B", tz="Europe/London")
    aapl_vals = 150.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(bdays_ny))))
    azn_vals = 7500.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(bdays_lon))))
    gbpusd = pd.Series(np.full(len(bdays_lon), 1.25), index=bdays_lon.tz_localize(None))

    # 1. Real yfinance-style NY + London daily exchange-midnight bars under as_of_date=None:
    #    completed Friday 2025-07-04 MUST NOT be excluded and rows MUST align on exact calendar dates.
    def fake_ticker_daily(sym):
        t = MagicMock()
        if sym == "AAPL":
            t.info = {"currency": "USD"}
            t.history.side_effect = lambda period="10y", auto_adjust=True: pd.DataFrame(
                {"Close": aapl_vals}, index=bdays_ny
            )
        elif sym == "AZN.L":
            t.info = {"currency": "GBp"}
            t.history.side_effect = lambda period="10y", auto_adjust=True: pd.DataFrame(
                {"Close": azn_vals}, index=bdays_lon
            )
        else:
            raise AssertionError(f"Unexpected ticker {sym}")
        return t

    monkeypatch.setattr(market_data.yf, "Ticker", fake_ticker_daily)
    monkeypatch.setattr(market_data, "_download_fx_series", lambda sym, period="10y": gbpusd)

    cfg_multi = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(instrument_id="AAPL", ticker="AAPL"),
            InstrumentConfig(instrument_id="AZN.L", ticker="AZN.L"),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("erc",), selected_strategy="erc"),
        as_of_date=None,
    )
    res_multi = analyze_portfolio(cfg_multi)
    assert res_multi.risk_model.last_closed_friday == "2025-07-04"
    assert res_multi.risk_model.excluded_incomplete_week_bins == []
    assert len(res_multi.prices_hist) == len(bdays_ny)
    assert res_multi.quotes_and_data_quality["latest_prices_usd"]["AZN.L"] == pytest.approx(
        float(azn_vals[-1]) * 0.01 * 1.25, rel=1e-12
    )

    # 2. Positive-offset London daily bars under Thursday as_of_date=2025-07-03:
    #    Friday 2025-07-04 close MUST NOT leak into Thursday cutoff.
    azn_shocked = azn_vals.copy()
    azn_shocked[-1] = 99999.0
    expected_thu_usd = float(azn_shocked[-2]) * 0.01 * 1.25

    def fake_ticker_lon_shock(sym):
        t = MagicMock()
        t.info = {"currency": "GBp"}
        t.history.side_effect = lambda period="10y", auto_adjust=True: pd.DataFrame(
            {"Close": azn_shocked}, index=bdays_lon
        )
        return t

    monkeypatch.setattr(market_data.yf, "Ticker", fake_ticker_lon_shock)
    cfg_thu = PortfolioAnalysisConfig(
        variable_income=(InstrumentConfig(instrument_id="AZN.L", ticker="AZN.L"),),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("erc",), selected_strategy="erc"),
        as_of_date=date(2025, 7, 3),
    )
    res_thu = analyze_portfolio(cfg_thu)
    assert res_thu.quotes_and_data_quality["latest_prices_usd"]["AZN.L"] == pytest.approx(expected_thu_usd, rel=1e-12)
    assert res_thu.risk_model.last_closed_friday == "2025-06-27"
    assert "2025-07-04" in res_thu.risk_model.excluded_incomplete_week_bins

    # 3. Live market-hours intraday last bar (Fri 14:35 EDT appended to daily NY bars) STILL excludes open Friday week.
    idx_live = bdays_ny[:-1].append(pd.DatetimeIndex([pd.Timestamp("2025-07-04 14:35:00", tz="America/New_York")]))

    def fake_ticker_live(sym):
        t = MagicMock()
        t.info = {"currency": "USD"}
        t.history.side_effect = lambda period="10y", auto_adjust=True: pd.DataFrame(
            {"Close": aapl_vals}, index=idx_live
        )
        return t

    monkeypatch.setattr(market_data.yf, "Ticker", fake_ticker_live)
    cfg_live = PortfolioAnalysisConfig(
        variable_income=(InstrumentConfig(instrument_id="AAPL", ticker="AAPL"),),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("erc",), selected_strategy="erc"),
        as_of_date=None,
    )
    res_live = analyze_portfolio(cfg_live)
    assert res_live.risk_model.last_closed_friday == "2025-06-27"
    assert "2025-07-04" in res_live.risk_model.excluded_incomplete_week_bins

    # 4. Standalone convert_price_series_to_usd on intraday 16:00 EDT bars (target_tz=None and target_tz="America/New_York"):
    #    MUST NOT shift Friday 4 PM EDT prints (+1 week via stripped UTC 20:00).
    idx_4pm = pd.date_range("2024-01-05", "2025-07-04", freq="W-FRI", tz="America/New_York") + pd.Timedelta(hours=16)
    eur_px = pd.Series(np.linspace(100.0, 150.0, len(idx_4pm)), index=idx_4pm)
    fx_tz = pd.Series(np.full(len(idx_4pm), 1.10), index=idx_4pm.tz_convert("UTC"))
    fx_naive = pd.Series(np.full(len(idx_4pm), 1.10), index=idx_4pm.tz_localize(None).normalize())
    cfg_ny17 = RiskModelConfig(min_weekly_observations=52, weekly_cutoff_tz="America/New_York", weekly_cutoff_hour=17)

    for fx_in in (fx_tz, fx_naive):
        usd_utc, _ = market_data.convert_price_series_to_usd("SAP.DE", eur_px, "EUR", fx_in)
        assert str(usd_utc.index.tz) == "UTC"
        b_utc = compute_weekly_returns_w_fri(
            pd.DataFrame({"SAP.DE": usd_utc}),
            risk_config=cfg_ny17,
            as_of_cutoff=pd.Timestamp("2025-07-04 17:00:00", tz="America/New_York"),
        )
        assert b_utc.last_closed_friday == pd.Timestamp("2025-07-04")
        assert b_utc.excluded_incomplete_week_bins == []
        assert b_utc.weekly_prices_usd.loc[pd.Timestamp("2025-07-04"), "SAP.DE"] == pytest.approx(150.0 * 1.10, rel=1e-12)

        usd_ny, _ = market_data.convert_price_series_to_usd(
            "SAP.DE", eur_px, "EUR", fx_in, target_tz="America/New_York"
        )
        assert usd_ny.index.tz is None
        assert usd_ny.index[-1] == pd.Timestamp("2025-07-04 16:00:00")
        b_ny = compute_weekly_returns_w_fri(
            pd.DataFrame({"SAP.DE": usd_ny}),
            risk_config=cfg_ny17,
            as_of_cutoff=pd.Timestamp("2025-07-04 17:00:00", tz="America/New_York"),
        )
        assert b_ny.weekly_prices_usd.loc[pd.Timestamp("2025-07-04"), "SAP.DE"] == pytest.approx(150.0 * 1.10, rel=1e-12)

    # 5. DST fall-back overlap (2025-11-02 01:30 EDT vs 01:30 EST) preserves distinct UTC instants & exact FX match.
    dst_idx = pd.DatetimeIndex([
        pd.Timestamp("2025-11-02 01:30:00-04:00").tz_convert("America/New_York"),
        pd.Timestamp("2025-11-02 01:30:00-05:00").tz_convert("America/New_York"),
    ])
    usd_dst, _ = market_data.convert_price_series_to_usd(
        "SAP.DE",
        pd.Series([100.0, 110.0], index=dst_idx),
        "EUR",
        pd.Series([1.10, 1.20], index=dst_idx.tz_convert("UTC")),
    )
    assert len(usd_dst) == 2
    assert str(usd_dst.index.tz) == "UTC"
    np.testing.assert_allclose(usd_dst.values, [110.0, 132.0], rtol=1e-12)

    # 6. Mixed tz-aware prices_usd_override + yfinance download in analyze_portfolio + America/Chicago ^VIX daily bars.
    monkeypatch.setattr(market_data.yf, "Ticker", fake_ticker_daily)
    btc_df = pd.DataFrame(
        {"BTC": 50000.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, len(bdays_ny))))},
        index=bdays_ny + pd.Timedelta(hours=16),
    )
    cfg_mix = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(instrument_id="AAPL", ticker="AAPL"),
            InstrumentConfig(instrument_id="BTC", ticker="BTC", asset_subtype="crypto", risk_class="crypto"),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("erc",), selected_strategy="erc"),
        as_of_date=date(2025, 7, 4),
    )
    res_mix = analyze_portfolio(cfg_mix, prices_usd_override=btc_df)
    assert res_mix.risk_model.last_closed_friday == "2025-07-04"

    vix_idx = pd.date_range("2024-01-05", "2025-07-04", freq="B", tz="America/Chicago")

    def fake_vix(sym):
        t = MagicMock()
        t.history.side_effect = lambda period="10y", auto_adjust=False: pd.DataFrame(
            {"Close": np.full(len(vix_idx), 18.5)}, index=vix_idx
        )
        return t

    monkeypatch.setattr(market_data.yf, "Ticker", fake_vix)
    vix_s = market_data.fetch_vix_history(period="2y", target_tz="America/New_York")
    assert vix_s.index[-1] == pd.Timestamp("2025-07-04")
    assert bool((vix_s.index == vix_s.index.normalize()).all())
