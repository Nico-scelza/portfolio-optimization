"""Independent oracle regression suite for Round 2 remediation:
AUD-04 (residual tz-aware midnight/DST), NEW-01 (unified tradable debt prices),
NEW-02 (transaction costs vs real funding shortfall), NEW-03 (Max Sharpe n=1),
and cross-cutting multi-asset E2E variants under pandas 3.x.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization import (
    BlackLittermanConfig,
    BrokerConfig,
    BrokerPosition,
    CashFlowEvent,
    ConstraintsConfig,
    DataQualityError,
    DCFViewConfig,
    FixedIncomeTerms,
    InfeasibleConstraintsError,
    InstrumentConfig,
    InsufficientHistoryError,
    IOLPanelRow,
    IOLPanelSnapshot,
    OptimizationConfig,
    OptimizationConvergenceError,
    PortfolioAnalysisConfig,
    RebalancingConfig,
    RebalancingResult,
    RiskModelConfig,
    VixRegimeConfig,
    analyze_portfolio,
    compute_rebalancing_plan,
    compute_weekly_returns_w_fri,
    ejecutar_analisis_completo,
    market_data,
    optimize_erc,
    optimize_hrp,
    optimize_max_sharpe,
    validate_constraints_feasibility,
)


# =============================================================================
# ROUND 2 REMEDIATION: AUD-04 (RESIDUAL), NEW-01, NEW-02, NEW-03 & E2E VARIANTS
# =============================================================================

def test_aud04_round2_tz_aware_midnight_dst_subsecond_alt_tz_and_ewma_cutoff():
    """AUD-04 residual: tz-aware midnight (EDT/EST), sub-second precision, alt TZ/hour, and EWMA cutoff."""
    rng = np.random.default_rng(4042)

    # 1. Summer EDT fixture ending Wednesday 2025-07-02 (>= 52 closed weeks)
    bdays_sum = pd.bdate_range("2024-01-02", "2025-07-02")
    eq_sum = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(bdays_sum))))
    df_sum = pd.DataFrame({"EQ": eq_sum}, index=bdays_sum)
    cfg_ny17 = RiskModelConfig(min_weekly_observations=52, weekly_cutoff_tz="America/New_York", weekly_cutoff_hour=17)

    # Tz-aware Friday midnight NY / 04:00Z EDT, explicit time strings, and sub-second instants MUST NOT close 2025-07-04
    for cut in (
        pd.Timestamp("2025-07-04 00:00:00", tz="America/New_York"),
        pd.Timestamp("2025-07-04 04:00:00", tz="UTC"),
        datetime(2025, 7, 4, 0, 0, tzinfo=ZoneInfo("America/New_York")),
        datetime(2025, 7, 4, 4, 0, tzinfo=timezone.utc),
        "2025-07-04T04:00:00Z",
        "2025-07-04T00:00:00-04:00",
        "2025-07-04 00:00:00",
        pd.Timestamp("2025-07-04 00:00:00.000001"),
        pd.Timestamp("2025-07-04 00:00:00.000000001"),
        pd.Timestamp("2025-07-04 20:59:59.999999", tz="UTC"),
    ):
        b = compute_weekly_returns_w_fri(df_sum, risk_config=cfg_ny17, as_of_cutoff=cut)
        assert b.last_closed_friday == pd.Timestamp("2025-06-27")
        assert b.weekly_returns.index.max() == pd.Timestamp("2025-06-27")
        assert "2025-07-04" in b.excluded_incomplete_week_bins

    # Legacy date-only inputs (date, ISO date string, naive midnight Timestamp) preserve end-of-day contract
    # AND use the real last observed price (Wed 2025-07-02), never fabricating a Friday price
    wed_price = float(df_sum.loc[pd.Timestamp("2025-07-02"), "EQ"])
    for cut_eod in (
        date(2025, 7, 4),
        "2025-07-04",
        pd.Timestamp("2025-07-04"),
        pd.Timestamp("2025-07-04 21:00:00", tz="UTC"),  # exact 17:00 EDT close
    ):
        b_eod = compute_weekly_returns_w_fri(df_sum, risk_config=cfg_ny17, as_of_cutoff=cut_eod)
        assert b_eod.last_closed_friday == pd.Timestamp("2025-07-04")
        assert b_eod.weekly_returns.index.max() == pd.Timestamp("2025-07-04")
        assert b_eod.excluded_incomplete_week_bins == []
        assert b_eod.weekly_prices_usd.loc[pd.Timestamp("2025-07-04"), "EQ"] == pytest.approx(wed_price, rel=1e-12)

    # 2. Winter EST fixture ending Wednesday 2025-01-08 (>= 52 closed weeks)
    bdays_win = pd.bdate_range("2023-09-01", "2025-01-08")
    df_win = pd.DataFrame(
        {"EQ": 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(bdays_win))))},
        index=bdays_win,
    )
    for cut_est_open in (
        pd.Timestamp("2025-01-10 05:00:00", tz="UTC"),  # 00:00 EST
        pd.Timestamp("2025-01-10 00:00:00", tz="America/New_York"),
        pd.Timestamp("2025-01-10 21:59:59.999999", tz="UTC"),  # 16:59:59.999999 EST
    ):
        bw = compute_weekly_returns_w_fri(df_win, risk_config=cfg_ny17, as_of_cutoff=cut_est_open)
        assert bw.last_closed_friday == pd.Timestamp("2025-01-03")
        assert bw.weekly_returns.index.max() == pd.Timestamp("2025-01-03")
        assert "2025-01-10" in bw.excluded_incomplete_week_bins

    for cut_est_closed in (
        pd.Timestamp("2025-01-10 22:00:00", tz="UTC"),  # exact 17:00 EST
        pd.Timestamp("2025-01-10 22:00:00.000001", tz="UTC"),  # 17:00:00.000001 EST
    ):
        bw_c = compute_weekly_returns_w_fri(df_win, risk_config=cfg_ny17, as_of_cutoff=cut_est_closed)
        assert bw_c.last_closed_friday == pd.Timestamp("2025-01-10")
        assert bw_c.weekly_returns.index.max() == pd.Timestamp("2025-01-10")

    # 3. Intraday sub-second bin assignment + post-cutoff exclusion (summer EDT & winter EST)
    fris_sum = pd.date_range("2025-05-02", "2025-06-27", freq="W-FRI", tz="America/New_York") + pd.Timedelta(hours=16)
    idx_sub = fris_sum.append(pd.DatetimeIndex([
        pd.Timestamp("2025-07-04 16:59:59.999999", tz="America/New_York"),  # bin 2025-07-04 (<= 17:00)
        pd.Timestamp("2025-07-04 17:00:00.000000", tz="America/New_York"),  # bin 2025-07-04 (== 17:00)
        pd.Timestamp("2025-07-04 17:00:00.000001", tz="America/New_York"),  # bin 2025-07-11 (> 17:00!)
        pd.Timestamp("2025-07-05 12:00:00", tz="America/New_York"),         # future print
    ]))
    vals_sub = list(np.linspace(100.0, 120.0, len(fris_sum))) + [125.0, 126.0, 999.0, 888.0]
    df_sub = pd.DataFrame({"BTC": vals_sub}, index=idx_sub)

    b_sub_before = compute_weekly_returns_w_fri(
        df_sub,
        risk_config=RiskModelConfig(min_weekly_observations=6, weekly_cutoff_tz="America/New_York", weekly_cutoff_hour=17),
        as_of_cutoff=pd.Timestamp("2025-07-04 20:59:59.999999", tz="UTC"),
    )
    assert b_sub_before.last_closed_friday == pd.Timestamp("2025-06-27")
    assert b_sub_before.weekly_returns.index.max() == pd.Timestamp("2025-06-27")

    b_sub_exact = compute_weekly_returns_w_fri(
        df_sub,
        risk_config=RiskModelConfig(min_weekly_observations=6, weekly_cutoff_tz="America/New_York", weekly_cutoff_hour=17),
        as_of_cutoff=pd.Timestamp("2025-07-04 21:00:00", tz="UTC"),
    )
    assert b_sub_exact.last_closed_friday == pd.Timestamp("2025-07-04")
    assert b_sub_exact.weekly_prices_usd.loc[pd.Timestamp("2025-07-04"), "BTC"] == pytest.approx(126.0)

    b_sub_after = compute_weekly_returns_w_fri(
        df_sub,
        risk_config=RiskModelConfig(min_weekly_observations=6, weekly_cutoff_tz="America/New_York", weekly_cutoff_hour=17),
        as_of_cutoff=pd.Timestamp("2025-07-04 21:00:00.000001", tz="UTC"),
    )
    assert b_sub_after.last_closed_friday == pd.Timestamp("2025-07-04")
    # 17:00:00.000001 print (999.0) belongs to 2025-07-11 and MUST NOT overwrite 2025-07-04 (126.0)!
    assert b_sub_after.weekly_prices_usd.loc[pd.Timestamp("2025-07-04"), "BTC"] == pytest.approx(126.0)
    assert "2025-07-11" in b_sub_after.excluded_incomplete_week_bins

    # 4. Alternative timezone & cutoff hour: Europe/London 16:00 (BST UTC+1 in summer, GMT UTC+0 in winter)
    cfg_lon16 = RiskModelConfig(min_weekly_observations=52, weekly_cutoff_tz="Europe/London", weekly_cutoff_hour=16)
    # 2025-07-03 23:00Z == 2025-07-04 00:00 BST -> 16h before London 16:00 close -> 2025-06-27
    b_lon_mid = compute_weekly_returns_w_fri(df_sum, risk_config=cfg_lon16, as_of_cutoff=pd.Timestamp("2025-07-03 23:00:00", tz="UTC"))
    assert b_lon_mid.last_closed_friday == pd.Timestamp("2025-06-27")
    # 2025-07-04 14:59:59Z == 15:59:59 BST -> 2025-06-27; 15:00:00Z == 16:00:00 BST -> 2025-07-04
    b_lon_pre = compute_weekly_returns_w_fri(df_sum, risk_config=cfg_lon16, as_of_cutoff=pd.Timestamp("2025-07-04 14:59:59", tz="UTC"))
    b_lon_at = compute_weekly_returns_w_fri(df_sum, risk_config=cfg_lon16, as_of_cutoff=pd.Timestamp("2025-07-04 15:00:00", tz="UTC"))
    assert b_lon_pre.last_closed_friday == pd.Timestamp("2025-06-27")
    assert b_lon_at.last_closed_friday == pd.Timestamp("2025-07-04")
    # Winter GMT: 2025-01-10 00:00Z == 00:00 GMT -> 2025-01-03; 16:00Z == 16:00 GMT -> 2025-01-10
    b_lon_w0 = compute_weekly_returns_w_fri(df_win, risk_config=cfg_lon16, as_of_cutoff=pd.Timestamp("2025-01-10 00:00:00", tz="UTC"))
    b_lon_w16 = compute_weekly_returns_w_fri(df_win, risk_config=cfg_lon16, as_of_cutoff=pd.Timestamp("2025-01-10 16:00:00", tz="UTC"))
    assert b_lon_w0.last_closed_friday == pd.Timestamp("2025-01-03")
    assert b_lon_w16.last_closed_friday == pd.Timestamp("2025-01-10")

    # 5. as_of_cutoff=None with tz-aware exchange-midnight daily bars (e.g. yfinance) keeps closed Friday
    fris_daily_tz = pd.date_range("2024-01-05", periods=60, freq="W-FRI", tz="America/New_York")
    df_daily_tz = pd.DataFrame({"EQ": 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.015, len(fris_daily_tz))))}, index=fris_daily_tz)
    b_daily_tz = compute_weekly_returns_w_fri(df_daily_tz, risk_config=cfg_ny17, as_of_cutoff=None)
    assert b_daily_tz.last_closed_friday == pd.Timestamp(fris_daily_tz[-1].tz_localize(None))

    # 6. Invalid cutoff raises DataQualityError
    for bad_cut in (pd.NaT, "", "not-a-timestamp"):
        with pytest.raises(DataQualityError):
            compute_weekly_returns_w_fri(df_sum, risk_config=cfg_ny17, as_of_cutoff=bad_cut)

    # 7. Master pipeline with as_of_date + EWMA (AUD-14 + AUD-04 interaction): future data has ZERO leakage
    bdays_ext = pd.bdate_range("2024-01-02", "2025-07-18")
    p_a = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(bdays_ext))))
    p_b = 50.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(bdays_ext))))
    df_full = pd.DataFrame({"A": p_a, "B": p_b}, index=bdays_ext)
    df_shocked_future = df_full.copy()
    df_shocked_future.loc[df_shocked_future.index > pd.Timestamp("2025-07-02"), "A"] *= 3.5

    cfg_ewma = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(instrument_id="A", ticker="A"),
            InstrumentConfig(instrument_id="B", ticker="B"),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="ewma", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("min_variance", "erc", "hrp"), selected_strategy="erc"),
        as_of_date=date(2025, 7, 2),
    )
    res_base = analyze_portfolio(cfg_ewma, prices_usd_override=df_full)
    res_fut = ejecutar_analisis_completo(cfg_ewma, render_ui=False, prices_usd_override=df_shocked_future)
    assert res_base.risk_model.last_closed_friday == "2025-06-27"
    assert "2025-07-04" in res_base.risk_model.excluded_incomplete_week_bins
    np.testing.assert_allclose(res_base.cov_matrix.values, res_fut.cov_matrix.values, atol=1e-14, rtol=1e-14)
    assert res_base.quotes_and_data_quality["latest_prices_usd"]["A"] == pytest.approx(
        float(df_full.loc[pd.Timestamp("2025-07-02"), "A"]), rel=1e-12
    )


def test_new01_unified_tradable_price_source_for_individual_debt_ytm_valuation_and_rebalancing(monkeypatch):
    """NEW-01: Single authoritative tradable price for individual debt across YTM, latest_prices, execution & rebalancing."""
    dates = pd.date_range("2024-01-05", periods=70, freq="W-FRI")
    tbill_terms = FixedIncomeTerms(
        cashflow_currency="USD",
        reference_nominal=100.0,
        settlement_date=date(2025, 5, 2),
        maturity_date=date(2026, 5, 2),
        price_convention="per_100_original_nominal",
        price_type="dirty",
        coupon_structure="zero_coupon",
        day_count_convention="ACT/365",
        cashflows=(CashFlowEvent(payment_date=date(2026, 5, 2), coupon_amount=0.0, amortization_amount=100.0),),
    )
    rng = np.random.default_rng(101)
    adj_series = np.linspace(80.0, 85.50, len(dates)) + rng.normal(0.0, 0.25, len(dates))
    adj_series[-1] = 85.50
    unadj_series = np.linspace(90.0, 95.00, len(dates))
    unadj_series[-1] = 95.00
    aapl_series = np.linspace(180.0, 200.0, len(dates))
    azn_gbp_series = np.linspace(7000.0, 7801.33, len(dates))

    # --- Oracle A: yfinance auto_adjust=False (95.00) vs auto_adjust=True (85.50), 1000 VN ---
    def fake_ticker_oracle_a(sym):
        t = MagicMock()
        if sym == "BOND_YF":
            t.info = {"currency": "USD"}
            def _hist(period="10y", auto_adjust=True):
                return pd.DataFrame({"Close": adj_series if auto_adjust else unadj_series}, index=dates)
            t.history.side_effect = _hist
        elif sym == "BOND_NO_UNADJ":
            t.info = {"currency": "USD"}
            def _hist_no_unadj(period="10y", auto_adjust=True):
                if auto_adjust:
                    return pd.DataFrame({"Close": adj_series}, index=dates)
                return pd.DataFrame()
            t.history.side_effect = _hist_no_unadj
        elif sym == "BOND_EUR":
            t.info = {"currency": "EUR"}
            def _hist_eur(period="10y", auto_adjust=True):
                vals = np.linspace(75.0, 80.0, len(dates)) if auto_adjust else np.linspace(85.0, 90.0, len(dates))
                return pd.DataFrame({"Close": vals}, index=dates)
            t.history.side_effect = _hist_eur
        elif sym == "AZN.L":
            t.info = {"currency": "GBp"}
            t.history.return_value = pd.DataFrame({"Close": azn_gbp_series}, index=dates)
        else:
            t.info = {"currency": "USD"}
            t.history.return_value = pd.DataFrame({"Close": aapl_series}, index=dates)
        return t

    def fake_fx_dl(fx_sym, period="10y", auto_adjust=True, progress=False):
        if fx_sym == "GBPUSD=X":
            return pd.DataFrame({"Close": np.full(len(dates), 1.25)}, index=dates)
        if fx_sym == "EURUSD=X":
            return pd.DataFrame({"Close": np.full(len(dates), 1.10)}, index=dates)
        return pd.DataFrame()

    monkeypatch.setattr(market_data.yf, "Ticker", fake_ticker_oracle_a)
    monkeypatch.setattr(market_data.yf, "download", fake_fx_dl)

    cfg_a = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="BOND_YF", ticker="BOND_YF", section="fixed_income", asset_subtype="tbill",
                quote_source="yfinance", historical_price_series_type="market_price",
                fixed_income_terms=tbill_terms, default_buy_broker_id="B1",
            ),
        ),
        brokers=(BrokerConfig(broker_id="B1", fee_structure="percentage", buy_percentage_fee=0.0, sell_percentage_fee=0.0),),
        positions=(BrokerPosition(instrument_id="BOND_YF", broker_id="B1", quantity=1000.0),),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
    )
    res_a = analyze_portfolio(cfg_a)
    res_a_master = ejecutar_analisis_completo(cfg_a, render_ui=False)

    expected_ytm = 100.0 / 95.0 - 1.0  # 0.05263157894736842
    fi_a = res_a.fixed_income_metrics["BOND_YF"]
    assert fi_a.status == "ok"
    assert fi_a.quoted_price == pytest.approx(95.0, abs=1e-12)
    assert res_a.quotes_and_data_quality["latest_prices_usd"]["BOND_YF"] == pytest.approx(95.0, abs=1e-12)
    assert fi_a.dirty_price_usd_per_original_nominal == pytest.approx(0.95, abs=1e-12)
    assert res_a.quotes_and_data_quality["execution_prices_usd_per_unit"]["BOND_YF"] == pytest.approx(0.95, abs=1e-12)
    assert fi_a.dirty_price_usd_per_original_nominal == pytest.approx(
        res_a.quotes_and_data_quality["execution_prices_usd_per_unit"]["BOND_YF"], abs=1e-12
    )
    assert fi_a.ytm_annual == pytest.approx(expected_ytm, abs=1e-10)
    assert res_a.rebalancing["min_variance"].initial_holdings_value_usd == pytest.approx(950.0, abs=1e-9)
    assert res_a.rebalancing["min_variance"].status == "ok"
    assert res_a.quotes_and_data_quality["price_resolution"]["BOND_YF"]["series_type"] == "yfinance_unadjusted_close"
    assert res_a_master.rebalancing["min_variance"].initial_holdings_value_usd == pytest.approx(950.0, abs=1e-9)
    # Risk series still retains the auto_adjust=True total-return series
    expected_adj_ret = pd.Series(adj_series, index=dates).pct_change().dropna()
    np.testing.assert_allclose(res_a.risk_model.weekly_returns["BOND_YF"].values, expected_adj_ret.values, rtol=1e-12, atol=1e-12)

    # --- Oracle B: adjusted_total_return=140.0 without IOL/manual -> YTM & rebalancing explicitly blocked ---
    tr_series = np.linspace(120.0, 140.0, len(dates))
    cfg_b = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="BOND_TR", ticker="BOND_TR", section="fixed_income", asset_subtype="sovereign_bond",
                quote_source="yfinance", historical_price_series_type="adjusted_total_return",
                manual_price_usd=None, fixed_income_terms=tbill_terms, default_buy_broker_id="B1",
            ),
        ),
        brokers=(BrokerConfig(broker_id="B1"),),
        positions=(BrokerPosition(instrument_id="BOND_TR", broker_id="B1", quantity=1000.0),),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
    )
    res_b = analyze_portfolio(cfg_b, prices_usd_override=pd.DataFrame({"BOND_TR": tr_series}, index=dates))
    assert res_b.fixed_income_metrics["BOND_TR"].status == "not_evaluable_under_available_method"
    assert res_b.fixed_income_metrics["BOND_TR"].ytm_annual is None
    assert "BOND_TR" not in res_b.quotes_and_data_quality["latest_prices_usd"]
    assert "BOND_TR" not in res_b.quotes_and_data_quality["execution_prices_usd_per_unit"]
    assert "BOND_TR" in res_b.quotes_and_data_quality["unpriced_instruments"]
    assert res_b.quotes_and_data_quality["price_resolution"]["BOND_TR"]["tradable"] is False
    assert res_b.rebalancing["min_variance"].status == "missing_tradable_market_price"
    assert res_b.rebalancing["min_variance"].feasible is False
    assert res_b.rebalancing["min_variance"].orders == []
    assert res_b.rebalancing["min_variance"].initial_holdings_value_usd == 0.0
    assert "BOND_TR" in res_b.rebalancing["min_variance"].unpriced_instruments
    # Risk model & optimizer still ran cleanly on the adjusted_total_return series
    assert res_b.risk_model.cov_matrix.shape == (1, 1)
    assert res_b.strategies["min_variance"].converged is True

    # --- Oracle B1: Same adjusted_total_return risk series + valid manual_price_usd=95.0 -> valuation enabled ---
    cfg_b_manual = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="BOND_TR", ticker="BOND_TR", section="fixed_income", asset_subtype="sovereign_bond",
                quote_source="manual", historical_price_series_type="adjusted_total_return",
                manual_price_usd=95.0, fixed_income_terms=tbill_terms, default_buy_broker_id="B1",
            ),
        ),
        brokers=(BrokerConfig(broker_id="B1"),),
        positions=(BrokerPosition(instrument_id="BOND_TR", broker_id="B1", quantity=1000.0),),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
    )
    res_b_man = analyze_portfolio(cfg_b_manual, prices_usd_override=pd.DataFrame({"BOND_TR": tr_series}, index=dates))
    assert res_b_man.fixed_income_metrics["BOND_TR"].status == "ok"
    assert res_b_man.fixed_income_metrics["BOND_TR"].ytm_annual == pytest.approx(expected_ytm, abs=1e-10)
    assert res_b_man.quotes_and_data_quality["execution_prices_usd_per_unit"]["BOND_TR"] == pytest.approx(0.95, abs=1e-12)
    assert res_b_man.rebalancing["min_variance"].status == "ok"
    assert res_b_man.rebalancing["min_variance"].feasible is True
    assert res_b_man.rebalancing["min_variance"].initial_holdings_value_usd == pytest.approx(950.0, abs=1e-9)

    # --- Oracle B2: Same adjusted_total_return risk series + valid IOL quote (95.0/100 VN) -> valuation enabled ---
    iol_snap = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at="2025-05-02T15:00:00+00:00", column_headers=(),
        rows_by_ticker={"BOND_TRD": IOLPanelRow(ticker="BOND_TRD", name="Bono TR", last_price=95.0, market_timestamp="02/05/2025 15:00")},
    )
    cfg_b_iol = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="BOND_TR", ticker="BOND_TR", section="fixed_income", asset_subtype="sovereign_bond",
                quote_source="iol_bonos", historical_price_series_type="adjusted_total_return",
                fixed_income_terms=tbill_terms, default_buy_broker_id="B1",
            ),
        ),
        brokers=(BrokerConfig(broker_id="B1"),),
        positions=(BrokerPosition(instrument_id="BOND_TR", broker_id="B1", quantity=1000.0),),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
    )
    res_b_iol = analyze_portfolio(
        cfg_b_iol,
        prices_usd_override=pd.DataFrame({"BOND_TR": tr_series}, index=dates),
        iol_snapshots_override={"iol_bonos": iol_snap},
    )
    assert res_b_iol.fixed_income_metrics["BOND_TR"].status == "ok"
    assert res_b_iol.fixed_income_metrics["BOND_TR"].ytm_annual == pytest.approx(expected_ytm, abs=1e-10)
    assert res_b_iol.rebalancing["min_variance"].initial_holdings_value_usd == pytest.approx(950.0, abs=1e-9)
    assert res_b_iol.rebalancing["min_variance"].status == "ok"

    # --- Variant: yfinance unadjusted_closes absent -> NEVER fall back to auto_adjust=True ---
    cfg_no_unadj = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="BOND_NO_UNADJ", ticker="BOND_NO_UNADJ", section="fixed_income", asset_subtype="tbill",
                quote_source="yfinance", historical_price_series_type="market_price",
                fixed_income_terms=tbill_terms, default_buy_broker_id="B1",
            ),
        ),
        brokers=(BrokerConfig(broker_id="B1"),),
        positions=(BrokerPosition(instrument_id="BOND_NO_UNADJ", broker_id="B1", quantity=1000.0),),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
    )
    res_no_unadj = analyze_portfolio(cfg_no_unadj)
    assert res_no_unadj.fixed_income_metrics["BOND_NO_UNADJ"].status == "not_evaluable_under_available_method"
    assert "BOND_NO_UNADJ" not in res_no_unadj.quotes_and_data_quality["latest_prices_usd"]
    assert res_no_unadj.rebalancing["min_variance"].status == "missing_tradable_market_price"
    assert res_no_unadj.rebalancing["min_variance"].feasible is False

    # --- Variant: mixed portfolio (AAPL + AZN.L GBp + unpriced adjusted bond) blocks global rebalancing, keeps equity/FX intact ---
    cfg_mix = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(instrument_id="AAPL", ticker="AAPL", default_buy_broker_id="B1"),
            InstrumentConfig(instrument_id="AZN.L", ticker="AZN.L", default_buy_broker_id="B1"),
        ),
        fixed_income=(
            InstrumentConfig(
                instrument_id="BOND_TR", ticker="BOND_TR", section="fixed_income", asset_subtype="corporate_bond_on",
                quote_source="manual", historical_price_series_type="adjusted_total_return",
                fixed_income_terms=tbill_terms, default_buy_broker_id="B1",
            ),
        ),
        brokers=(BrokerConfig(broker_id="B1"),),
        positions=(
            BrokerPosition(instrument_id="AAPL", broker_id="B1", quantity=5.0),
            BrokerPosition(instrument_id="BOND_TR", broker_id="B1", quantity=1000.0),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
    )
    res_mix = analyze_portfolio(cfg_mix, prices_usd_override=pd.DataFrame({"BOND_TR": tr_series}, index=dates))
    assert res_mix.quotes_and_data_quality["latest_prices_usd"]["AAPL"] == pytest.approx(200.0, rel=1e-12)
    assert res_mix.quotes_and_data_quality["latest_prices_usd"]["AZN.L"] == pytest.approx(7801.33 * 0.01 * 1.25, rel=1e-12)
    assert "BOND_TR" not in res_mix.quotes_and_data_quality["latest_prices_usd"]
    assert res_mix.rebalancing["min_variance"].status == "missing_tradable_market_price"
    assert res_mix.rebalancing["min_variance"].feasible is False
    assert res_mix.rebalancing["min_variance"].orders == []

    # --- Variant: clean + accrued + residual nominal (0.6) + reference_nominal=1000 + historical cutoff ---
    amort_terms = FixedIncomeTerms(
        cashflow_currency="USD",
        reference_nominal=1000.0,
        residual_nominal=600.0,
        settlement_date=date(2025, 4, 25),
        maturity_date=date(2026, 4, 25),
        price_convention="per_100_residual_nominal",
        price_type="clean",
        accrued_interest=2.0,
        accrued_interest_basis="quote_convention",
        coupon_structure="fixed",
        day_count_convention="ACT/365",
        cashflows=(CashFlowEvent(payment_date=date(2026, 4, 25), coupon_amount=30.0, amortization_amount=600.0),),
    )
    # Historical series has 78.0 on 2025-04-25 and a future 40.0 on 2025-05-02 -> cutoff 2025-04-25 ignores 40.0!
    clean_vals = np.linspace(72.0, 78.0, len(dates) - 1).tolist() + [40.0]
    cfg_amort = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="AMORT", ticker="AMORT", section="fixed_income", asset_subtype="corporate_bond_on",
                quote_source="manual", historical_price_series_type="market_price",
                fixed_income_terms=amort_terms, default_buy_broker_id="B1",
            ),
        ),
        brokers=(BrokerConfig(broker_id="B1"),),
        positions=(BrokerPosition(instrument_id="AMORT", broker_id="B1", quantity=1000.0),),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
        as_of_date=date(2025, 4, 25),
    )
    res_amort = analyze_portfolio(cfg_amort, prices_usd_override=pd.DataFrame({"AMORT": clean_vals}, index=dates))
    # Clean 78 + accrued 2 = 80 per 100 residual -> 80 * 0.6 / 100 = 0.48 USD/VN -> 1000 VN = $480; YTM = 630/480 - 1 = 31.25%
    assert res_amort.fixed_income_metrics["AMORT"].quoted_price == pytest.approx(78.0, abs=1e-12)
    assert res_amort.quotes_and_data_quality["latest_prices_usd"]["AMORT"] == pytest.approx(78.0, abs=1e-12)
    assert res_amort.fixed_income_metrics["AMORT"].dirty_price_usd_per_original_nominal == pytest.approx(0.48, abs=1e-12)
    assert res_amort.quotes_and_data_quality["execution_prices_usd_per_unit"]["AMORT"] == pytest.approx(0.48, abs=1e-12)
    assert res_amort.fixed_income_metrics["AMORT"].ytm_annual == pytest.approx(630.0 / 480.0 - 1.0, abs=1e-10)
    assert res_amort.rebalancing["min_variance"].initial_holdings_value_usd == pytest.approx(480.0, abs=1e-9)

    # --- Variant: Non-USD yfinance bond (EUR * 1.10 = 99.0 USD) has ZERO double FX / nominal conversion ---
    cfg_eur = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="BOND_EUR", ticker="BOND_EUR", section="fixed_income", asset_subtype="tbill",
                quote_source="yfinance", historical_price_series_type="market_price",
                fixed_income_terms=tbill_terms, default_buy_broker_id="B1",
            ),
        ),
        brokers=(BrokerConfig(broker_id="B1"),),
        positions=(BrokerPosition(instrument_id="BOND_EUR", broker_id="B1", quantity=1000.0),),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
    )
    res_eur = analyze_portfolio(cfg_eur)
    assert res_eur.quotes_and_data_quality["latest_prices_usd"]["BOND_EUR"] == pytest.approx(99.0, abs=1e-10)
    assert res_eur.quotes_and_data_quality["execution_prices_usd_per_unit"]["BOND_EUR"] == pytest.approx(0.99, abs=1e-10)
    assert res_eur.fixed_income_metrics["BOND_EUR"].dirty_price_usd_per_original_nominal == pytest.approx(0.99, abs=1e-10)
    assert res_eur.rebalancing["min_variance"].initial_holdings_value_usd == pytest.approx(990.0, abs=1e-8)

    # --- Variant: Proxy-only bond without IOL/manual blocks tradable valuation; InsufficientHistoryError preserves standalone YTM ---
    cfg_proxy_only = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="BOND_PRX", ticker="BOND_PRX", section="fixed_income", asset_subtype="sovereign_bond",
                quote_source="manual", manual_price_usd=None, historical_proxy_ticker="EMB",
                fixed_income_terms=tbill_terms, default_buy_broker_id="B1",
            ),
        ),
        brokers=(BrokerConfig(broker_id="B1"),),
        positions=(BrokerPosition(instrument_id="BOND_PRX", broker_id="B1", quantity=1000.0),),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52, allow_explicit_proxies=True),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
    )
    res_prx = analyze_portfolio(cfg_proxy_only, prices_usd_override=pd.DataFrame({"EMB": np.linspace(80.0, 90.0, len(dates))}, index=dates))
    assert res_prx.fixed_income_metrics["BOND_PRX"].status == "not_evaluable_under_available_method"
    assert res_prx.quotes_and_data_quality["price_resolution"]["BOND_PRX"]["series_type"] == "proxy_only"
    assert res_prx.rebalancing["min_variance"].status == "missing_tradable_market_price"

    cfg_insuff = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="BOND_NOHIST", ticker="BOND_NOHIST", section="fixed_income", asset_subtype="tbill",
                quote_source="manual", manual_price_usd=95.0, fixed_income_terms=tbill_terms,
            ),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
    )
    with pytest.raises(InsufficientHistoryError) as exc_ih:
        analyze_portfolio(cfg_insuff)
    assert exc_ih.value.details["fixed_income_ytm_available"]["BOND_NOHIST"] == pytest.approx(expected_ytm, abs=1e-10)


def _assert_rebalancing_ledger_reconciliation(reb: RebalancingResult) -> None:
    """Independent accounting identity oracle for every broker ledger and aggregate totals."""
    for bid, led in reb.broker_ledgers.items():
        # Per-broker cash identity: init + flow + sell_gross - buy_gross - total_tx_cost == ending_cash
        expected_end = (
            led["initial_cash_usd"]
            + led["cash_flow_usd"]
            + led["sell_gross_usd"]
            - led["buy_gross_usd"]
            - led["total_transaction_cost_usd"]
        )
        assert led["ending_cash_usd"] == pytest.approx(expected_end, abs=1e-8), f"Cash tie-out failed for {bid}"
        assert led["ending_cash_usd"] == pytest.approx(reb.ending_cash_by_broker[bid], abs=1e-9)
        if reb.status != "insufficient_cash":
            assert led["ending_cash_usd"] >= -1e-9, f"Negative cash in {bid}: {led['ending_cash_usd']}"
        # Per-broker causal unexecuted buy decomposition identity
        unexec = max(0.0, led["desired_buy_gross_usd"] - led["buy_gross_usd"])
        decomp_sum = (
            led["unfunded_buy_shortfall_usd"]
            + led["cost_adjusted_buy_reduction_usd"]
            + led["threshold_omitted_buy_usd"]
            + led["lot_rounding_residual_usd"]
        )
        assert decomp_sum == pytest.approx(unexec, abs=1e-7), f"Buy decomposition failed for {bid}: {led}"

    assert sum(led["ending_cash_usd"] for led in reb.broker_ledgers.values()) == pytest.approx(reb.ending_cash_usd, abs=1e-8)
    assert sum(led["unfunded_buy_shortfall_usd"] for led in reb.broker_ledgers.values()) == pytest.approx(
        reb.unfunded_buy_shortfall_usd, abs=1e-8
    )
    assert reb.deployable_capital_usd == pytest.approx(reb.initial_portfolio_value_usd + reb.cash_flow_usd, abs=1e-8)
    assert reb.final_portfolio_value_usd == pytest.approx(
        reb.deployable_capital_usd - reb.total_transaction_cost_usd, abs=1e-7
    )


def test_new02_transaction_cost_friction_vs_real_funding_shortfall_and_ledger_reconciliation():
    """NEW-02: Separate broker transaction-cost friction from real funding shortfall with exact ledger tie-outs."""
    # --- Oracle A: Single broker IBKR, 10 A @ $100, A=0%, B=100%, cash=0, 0.5% sell/buy, spread=0 ---
    ibkr = BrokerConfig(broker_id="IBKR", fee_structure="percentage", buy_percentage_fee=0.005, sell_percentage_fee=0.005)
    inst_ab = {
        "A": InstrumentConfig(instrument_id="A", ticker="A", default_buy_broker_id="IBKR"),
        "B": InstrumentConfig(instrument_id="B", ticker="B", default_buy_broker_id="IBKR"),
    }
    reb_a = compute_rebalancing_plan(
        strategy_name="max_sharpe",
        tickers=["A", "B"],
        target_weights={"A": 0.0, "B": 1.0},
        prices_usd={"A": 100.0, "B": 100.0},
        instruments_by_id=inst_ab,
        brokers_by_id={"IBKR": ibkr},
        positions=[BrokerPosition(instrument_id="A", broker_id="IBKR", quantity=10.0)],
        rebalancing_config=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=25.0),
    )
    exp_buy_gross = 995.0 / 1.005  # 990.0497512437812
    exp_buy_comm = exp_buy_gross * 0.005  # 4.950248756218906
    exp_total_cost = 5.0 + exp_buy_comm  # 9.950248756218906
    assert reb_a.status == "ok"
    assert reb_a.feasible is True
    assert reb_a.unfunded_buy_shortfall_usd == pytest.approx(0.0, abs=1e-10)
    assert reb_a.total_sell_gross_usd == pytest.approx(1000.0, abs=1e-9)
    assert reb_a.total_buy_gross_usd == pytest.approx(exp_buy_gross, abs=1e-7)
    assert reb_a.total_commission_usd == pytest.approx(exp_total_cost, abs=1e-7)
    assert reb_a.total_transaction_cost_usd == pytest.approx(exp_total_cost, abs=1e-7)
    assert reb_a.ending_cash_usd == pytest.approx(0.0, abs=1e-7)
    assert reb_a.achievable_weights["A"] == pytest.approx(0.0, abs=1e-10)
    assert reb_a.achievable_weights["B"] == pytest.approx(1.0, abs=1e-9)
    assert reb_a.broker_ledgers["IBKR"]["cost_adjusted_buy_reduction_usd"] == pytest.approx(exp_total_cost, abs=1e-7)
    assert any("absorb broker transaction costs" in w for w in reb_a.warnings)
    assert any("settlement timing" in w.lower() for w in reb_a.warnings)
    _assert_rebalancing_ledger_reconciliation(reb_a)

    # --- Oracle B (AUD-03): X in B_SELL (0.3% sell fee -> $997), Y in B_BUY ($0 cash) -> shortfall=$1000 ---
    brokers_sep = {
        "B_SELL": BrokerConfig(broker_id="B_SELL", fee_structure="percentage", sell_percentage_fee=0.003),
        "B_BUY": BrokerConfig(broker_id="B_BUY", fee_structure="percentage", buy_percentage_fee=0.005),
    }
    inst_xy = {
        "X": InstrumentConfig(instrument_id="X", ticker="X", default_buy_broker_id="B_SELL"),
        "Y": InstrumentConfig(instrument_id="Y", ticker="Y", default_buy_broker_id="B_BUY"),
    }
    reb_b = compute_rebalancing_plan(
        strategy_name="max_sharpe",
        tickers=["X", "Y"],
        target_weights={"X": 0.0, "Y": 1.0},
        prices_usd={"X": 100.0, "Y": 100.0},
        instruments_by_id=inst_xy,
        brokers_by_id=brokers_sep,
        positions=[BrokerPosition(instrument_id="X", broker_id="B_SELL", quantity=10.0)],
        rebalancing_config=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=25.0),
    )
    assert reb_b.status == "partially_funded"
    assert reb_b.feasible is False
    assert reb_b.unfunded_buy_shortfall_usd == pytest.approx(1000.0, abs=1e-9)
    assert reb_b.broker_ledgers["B_SELL"]["ending_cash_usd"] == pytest.approx(997.0, abs=1e-9)
    assert reb_b.broker_ledgers["B_BUY"]["ending_cash_usd"] == pytest.approx(0.0, abs=1e-9)
    assert all(o.broker_id != "B_BUY" for o in reb_b.orders)
    _assert_rebalancing_ledger_reconciliation(reb_b)

    # --- Extra 1: Own-cash buy without sell ---
    reb_cash_only = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["A", "B"],
        target_weights={"A": 0.5, "B": 0.5},
        prices_usd={"A": 100.0, "B": 100.0},
        instruments_by_id=inst_ab,
        brokers_by_id={"IBKR": ibkr},
        positions=[],
        rebalancing_config=RebalancingConfig(available_cash_usd=1000.0, min_weight_diff=0.01, min_notional_usd=25.0),
    )
    assert reb_cash_only.status == "ok" and reb_cash_only.feasible is True
    assert reb_cash_only.unfunded_buy_shortfall_usd == pytest.approx(0.0, abs=1e-10)
    assert reb_cash_only.broker_ledgers["IBKR"]["cost_adjusted_buy_reduction_usd"] == pytest.approx(1000.0 - 1000.0 / 1.005, abs=1e-7)
    _assert_rebalancing_ledger_reconciliation(reb_cash_only)

    # --- Extra 2: Two brokers BOTH funded with own commissions -> status='ok', shortfall=0 ---
    brokers_two = {
        "B1": BrokerConfig(broker_id="B1", fee_structure="percentage", buy_percentage_fee=0.005, sell_percentage_fee=0.005),
        "B2": BrokerConfig(broker_id="B2", fee_structure="percentage", buy_percentage_fee=0.01, sell_percentage_fee=0.01),
    }
    inst_two = {
        "A": InstrumentConfig(instrument_id="A", ticker="A", default_buy_broker_id="B1"),
        "B": InstrumentConfig(instrument_id="B", ticker="B", default_buy_broker_id="B2"),
    }
    reb_both_ok = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["A", "B"],
        target_weights={"A": 0.5, "B": 0.5},
        prices_usd={"A": 100.0, "B": 100.0},
        instruments_by_id=inst_two,
        brokers_by_id=brokers_two,
        positions=[],
        rebalancing_config=RebalancingConfig(
            available_cash_by_broker={"B1": 500.0, "B2": 500.0},
            min_weight_diff=0.01,
            min_notional_usd=25.0,
        ),
    )
    assert reb_both_ok.status == "ok" and reb_both_ok.feasible is True
    assert reb_both_ok.unfunded_buy_shortfall_usd == pytest.approx(0.0, abs=1e-10)
    _assert_rebalancing_ledger_reconciliation(reb_both_ok)

    # --- Extra 3: Two brokers — B1 funded with fees, B2 genuinely short $500 (costs in B1 do NOT offset/inflate B2 shortfall) ---
    reb_one_short = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["A", "B"],
        target_weights={"A": 0.5, "B": 0.5},
        prices_usd={"A": 100.0, "B": 100.0},
        instruments_by_id=inst_two,
        brokers_by_id=brokers_two,
        positions=[],
        rebalancing_config=RebalancingConfig(
            available_cash_by_broker={"B1": 1000.0, "B2": 0.0},
            min_weight_diff=0.01,
            min_notional_usd=25.0,
        ),
    )
    assert reb_one_short.status == "partially_funded" and reb_one_short.feasible is False
    assert reb_one_short.broker_ledgers["B1"]["unfunded_buy_shortfall_usd"] == pytest.approx(0.0, abs=1e-10)
    assert reb_one_short.broker_ledgers["B2"]["unfunded_buy_shortfall_usd"] == pytest.approx(500.0, abs=1e-9)
    assert reb_one_short.unfunded_buy_shortfall_usd == pytest.approx(500.0, abs=1e-9)
    _assert_rebalancing_ledger_reconciliation(reb_one_short)

    # --- Extra 4: Mixed real funding shortfall ($800) + own fees ($1.9900) in SAME broker (no tolerance erasure!) ---
    inst_mixed = {
        "X": InstrumentConfig(instrument_id="X", ticker="X", default_buy_broker_id="B_SELL", sell_broker_priority=("B_SELL", "B_BUY")),
        "Y": InstrumentConfig(instrument_id="Y", ticker="Y", default_buy_broker_id="B_BUY"),
    }
    brokers_mixed = {
        "B_SELL": BrokerConfig(broker_id="B_SELL", fee_structure="percentage", sell_percentage_fee=0.005),
        "B_BUY": BrokerConfig(broker_id="B_BUY", fee_structure="percentage", buy_percentage_fee=0.005, sell_percentage_fee=0.005),
    }
    reb_mixed = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["X", "Y"],
        target_weights={"X": 0.0, "Y": 1.0},
        prices_usd={"X": 100.0, "Y": 100.0},
        instruments_by_id=inst_mixed,
        brokers_by_id=brokers_mixed,
        positions=[
            BrokerPosition(instrument_id="X", broker_id="B_SELL", quantity=8.0),
            BrokerPosition(instrument_id="X", broker_id="B_BUY", quantity=2.0),
        ],
        rebalancing_config=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=25.0),
    )
    exp_mix_gross = 199.0 / 1.005
    exp_mix_tx = 1.0 + exp_mix_gross * 0.005
    assert reb_mixed.status == "partially_funded" and reb_mixed.feasible is False
    assert reb_mixed.broker_ledgers["B_BUY"]["unfunded_buy_shortfall_usd"] == pytest.approx(800.0, abs=1e-9)
    assert reb_mixed.broker_ledgers["B_BUY"]["cost_adjusted_buy_reduction_usd"] == pytest.approx(exp_mix_tx, abs=1e-7)
    _assert_rebalancing_ledger_reconciliation(reb_mixed)

    # --- Extra 5: Combined fixed + percentage + spread (price_includes_spread False vs True, no double counting) ---
    for incl_sp, exp_sp in ((False, True), (True, False)):
        brk_sp = BrokerConfig(
            broker_id="SP", fee_structure="fixed_plus_percentage",
            buy_fixed_fee=5.0, sell_fixed_fee=5.0, buy_percentage_fee=0.005, sell_percentage_fee=0.005,
            estimated_spread_bps=10.0, price_includes_spread=incl_sp,
        )
        inst_sp = {
            "A": InstrumentConfig(instrument_id="A", ticker="A", default_buy_broker_id="SP"),
            "B": InstrumentConfig(instrument_id="B", ticker="B", default_buy_broker_id="SP"),
        }
        reb_sp = compute_rebalancing_plan(
            strategy_name="erc",
            tickers=["A", "B"],
            target_weights={"A": 0.0, "B": 1.0},
            prices_usd={"A": 100.0, "B": 100.0},
            instruments_by_id=inst_sp,
            brokers_by_id={"SP": brk_sp},
            positions=[BrokerPosition(instrument_id="A", broker_id="SP", quantity=10.0)],
            rebalancing_config=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=25.0),
        )
        assert reb_sp.status == "ok" and reb_sp.feasible is True
        assert reb_sp.unfunded_buy_shortfall_usd == pytest.approx(0.0, abs=1e-10)
        if exp_sp:
            assert reb_sp.total_spread_cost_usd > 1.9
            assert reb_sp.total_buy_gross_usd == pytest.approx(984.0 / 1.006, abs=1e-7)
        else:
            assert reb_sp.total_spread_cost_usd == pytest.approx(0.0, abs=1e-12)
            assert reb_sp.total_buy_gross_usd == pytest.approx(985.0 / 1.005, abs=1e-7)
        _assert_rebalancing_ledger_reconciliation(reb_sp)

    # --- Extra 6: Fixed fee > cash ($10 cash vs $25 fixed fee) -> genuinely unable to execute -> partially_funded ---
    brk_fix = BrokerConfig(broker_id="FIX", fee_structure="fixed", buy_fixed_fee=25.0, sell_fixed_fee=25.0)
    reb_fix_fail = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["A"],
        target_weights={"A": 1.0},
        prices_usd={"A": 10.0},
        instruments_by_id={"A": InstrumentConfig(instrument_id="A", ticker="A", default_buy_broker_id="FIX")},
        brokers_by_id={"FIX": brk_fix},
        positions=[],
        rebalancing_config=RebalancingConfig(available_cash_usd=10.0, use_weight_threshold=False, min_notional_usd=1.0),
    )
    assert reb_fix_fail.status == "partially_funded" and reb_fix_fail.feasible is False
    assert reb_fix_fail.unfunded_buy_shortfall_usd == pytest.approx(10.0, abs=1e-9)
    _assert_rebalancing_ledger_reconciliation(reb_fix_fail)

    # --- Extra 7: Scaled buy < min_notional strictly due to fee -> status='ok' vs due to real cash deficit -> partially_funded ---
    brk_1usd = BrokerConfig(broker_id="B1", fee_structure="fixed", buy_fixed_fee=1.0, sell_fixed_fee=1.0)
    reb_thresh_fee = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["A", "B"],
        target_weights={"A": 0.51, "B": 0.49},
        prices_usd={"A": 100.0, "B": 100.0},
        instruments_by_id={
            "A": InstrumentConfig(instrument_id="A", ticker="A", default_buy_broker_id="B1"),
            "B": InstrumentConfig(instrument_id="B", ticker="B", default_buy_broker_id="B1"),
        },
        brokers_by_id={"B1": brk_1usd},
        positions=[
            BrokerPosition(instrument_id="A", broker_id="B1", quantity=25.0),
            BrokerPosition(instrument_id="B", broker_id="B1", quantity=25.0),
        ],
        rebalancing_config=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=50.0, threshold_comparison=">="),
    )
    assert reb_thresh_fee.status == "ok" and reb_thresh_fee.feasible is True
    assert reb_thresh_fee.unfunded_buy_shortfall_usd == pytest.approx(0.0, abs=1e-10)
    assert reb_thresh_fee.broker_ledgers["B1"]["threshold_omitted_buy_usd"] == pytest.approx(50.0, abs=1e-9)
    _assert_rebalancing_ledger_reconciliation(reb_thresh_fee)

    # --- Extra 8: Discrete integer lot rounding + cash reserve + positive/negative cash flows ---
    reb_int = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["A"],
        target_weights={"A": 1.0},
        prices_usd={"A": 100.0},
        instruments_by_id={"A": InstrumentConfig(instrument_id="A", ticker="A", fractional_allowed=False, default_buy_broker_id="IBKR")},
        brokers_by_id={"IBKR": ibkr},
        positions=[],
        rebalancing_config=RebalancingConfig(available_cash_usd=800.0, cash_flow_usd=200.0, use_weight_threshold=False, min_notional_usd=25.0),
    )
    assert reb_int.status == "ok" and reb_int.feasible is True
    assert reb_int.initial_portfolio_value_usd == pytest.approx(800.0, abs=1e-9)
    assert reb_int.deployable_capital_usd == pytest.approx(1000.0, abs=1e-9)
    assert reb_int.orders[0].quantity == pytest.approx(9.0, abs=1e-10)
    assert reb_int.broker_ledgers["IBKR"]["cost_adjusted_buy_reduction_usd"] == pytest.approx(4.5, abs=1e-9)
    assert reb_int.broker_ledgers["IBKR"]["lot_rounding_residual_usd"] == pytest.approx(95.5, abs=1e-9)
    assert reb_int.ending_cash_usd == pytest.approx(95.5, abs=1e-9)
    _assert_rebalancing_ledger_reconciliation(reb_int)

    # Cash reserve (target weights sum to 0.90) + positive/negative multi-broker flows
    reb_res = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["A"],
        target_weights={"A": 0.90},
        prices_usd={"A": 100.0},
        instruments_by_id={"A": InstrumentConfig(instrument_id="A", ticker="A", default_buy_broker_id="IBKR")},
        brokers_by_id={"IBKR": ibkr},
        positions=[],
        rebalancing_config=RebalancingConfig(available_cash_usd=1000.0, use_weight_threshold=False, min_notional_usd=25.0),
    )
    assert reb_res.status == "ok" and reb_res.feasible is True
    assert reb_res.ending_cash_usd == pytest.approx(95.5, abs=1e-9)
    _assert_rebalancing_ledger_reconciliation(reb_res)

    # --- Extra 9: Status precedence (insufficient_cash > infeasible_hard_limits > partially_funded > ok) ---
    reb_prec_hard = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["X", "Y"],
        target_weights={"X": 0.0, "Y": 1.0},
        prices_usd={"X": 100.0, "Y": 100.0},
        instruments_by_id=inst_xy,
        brokers_by_id=brokers_sep,
        positions=[BrokerPosition(instrument_id="X", broker_id="B_SELL", quantity=10.0)],
        rebalancing_config=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=25.0),
        constraints_config=ConstraintsConfig(asset_bounds={"Y": (0.5, 1.0)}),
    )
    # Achievable Y=0.0 violates hard min 0.5 AND B_BUY has $1000 shortfall -> hard limit takes precedence!
    assert reb_prec_hard.status == "infeasible_hard_limits"
    assert reb_prec_hard.feasible is False

    reb_prec_neg = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["X", "Y"],
        target_weights={"X": 0.0, "Y": 1.0},
        prices_usd={"X": 100.0, "Y": 100.0},
        instruments_by_id=inst_xy,
        brokers_by_id=brokers_sep,
        positions=[BrokerPosition(instrument_id="X", broker_id="B_SELL", quantity=10.0)],
        rebalancing_config=RebalancingConfig(
            available_cash_by_broker={"B_SELL": 0.0, "B_BUY": 0.0},
            cash_flow_by_broker={"B_SELL": 0.0, "B_BUY": -50.0},
            min_weight_diff=0.01,
            min_notional_usd=25.0,
        ),
        constraints_config=ConstraintsConfig(asset_bounds={"Y": (0.5, 1.0)}),
    )
    # Negative ending cash in B_BUY takes precedence over hard limit & shortfall!
    assert reb_prec_neg.status == "insufficient_cash"
    assert reb_prec_neg.feasible is False


def test_new03_max_sharpe_single_asset_non_positive_excess_consistency_and_rebalancing_skip():
    """NEW-03: Unify n==1 Max Sharpe non-positive excess contract with n>=2 and skip invalid targets in rebalancing."""
    rc1 = validate_constraints_feasibility(["A"])
    cov1 = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    mu_neg = pd.Series([0.02], index=["A"])

    # 1. Exact oracle: mu=[0.02], cov=[[0.04]], rf=0.04 under block & allow_with_warning
    res_block = optimize_max_sharpe(mu_neg, cov1, 0.04, rc1, non_positive_excess_return_policy="block")
    assert res_block.converged is False
    assert res_block.status == "blocked_non_positive_excess"
    assert res_block.diagnostics["is_valid_target"] is False
    assert len(res_block.warnings) >= 1
    assert res_block.sharpe_ratio == pytest.approx(-0.10, abs=1e-12)

    res_allow = optimize_max_sharpe(mu_neg, cov1, 0.04, rc1, non_positive_excess_return_policy="allow_with_warning")
    assert res_allow.converged is True
    assert res_allow.status == "suboptimal_non_positive_excess"
    assert res_allow.diagnostics["is_valid_target"] is False
    assert len(res_allow.warnings) >= 1
    assert res_allow.weights_by_asset["A"] == pytest.approx(1.0, abs=1e-12)
    assert res_allow.sharpe_ratio == pytest.approx(-0.10, abs=1e-12)

    # 2. Equality mu == rf and tiny boundary around 1e-12
    for mu_val in (0.04, 0.04 + 1e-13):
        r_b = optimize_max_sharpe(pd.Series([mu_val], index=["A"]), cov1, 0.04, rc1, non_positive_excess_return_policy="block")
        r_a = optimize_max_sharpe(pd.Series([mu_val], index=["A"]), cov1, 0.04, rc1, non_positive_excess_return_policy="allow_with_warning")
        assert r_b.status == "blocked_non_positive_excess" and r_b.diagnostics["is_valid_target"] is False
        assert r_a.status == "suboptimal_non_positive_excess" and r_a.diagnostics["is_valid_target"] is False

    # 3. Positive excess (just above tolerance & normal positive) -> optimal & valid target
    for mu_pos in (0.04 + 1e-11, 0.10):
        r_pos = optimize_max_sharpe(pd.Series([mu_pos], index=["A"]), cov1, 0.04, rc1)
        assert r_pos.converged is True
        assert r_pos.status == "optimal"
        assert r_pos.diagnostics["is_valid_target"] is True
        assert r_pos.warnings == []

    # 4. Zero variance, negative variance, non-finite inputs, and infeasible constraints
    r_zero = optimize_max_sharpe(pd.Series([0.10], index=["A"]), pd.DataFrame([[0.0]], index=["A"], columns=["A"]), 0.04, rc1)
    assert r_zero.converged is False
    assert r_zero.status == "singular_zero_variance"
    assert r_zero.diagnostics["is_valid_target"] is False

    with pytest.raises(OptimizationConvergenceError):
        optimize_max_sharpe(pd.Series([0.10], index=["A"]), pd.DataFrame([[-0.01]], index=["A"], columns=["A"]), 0.04, rc1)

    with pytest.raises(OptimizationConvergenceError):
        optimize_max_sharpe(pd.Series([float("nan")], index=["A"]), cov1, 0.04, rc1)

    with pytest.raises(InfeasibleConstraintsError):
        validate_constraints_feasibility(["A"], ConstraintsConfig(asset_bounds={"A": (0.0, 0.5)}))
    with pytest.raises(InfeasibleConstraintsError):
        validate_constraints_feasibility(["A"], ConstraintsConfig(hhi_limit=0.5))

    # 5. Parity with n>=2 AND analyze_portfolio skipping rebalancing for both block & allow_with_warning
    dates = pd.date_range("2023-01-06", periods=80, freq="W-FRI")
    rng = np.random.default_rng(303)
    p_a = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.015, len(dates))))
    p_b = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.020, len(dates))))
    prices_df = pd.DataFrame({"A": p_a, "B": p_b}, index=dates)

    for n_mode in (1, 2):
        vi = (
            InstrumentConfig(
                instrument_id="A", ticker="A", default_buy_broker_id="B1",
                dcf_view=DCFViewConfig(valuation_basis="terminal_target", total_upside=-0.10, horizon_years=5.0, confidence=1.0),
            ),
        )
        if n_mode == 2:
            vi = vi + (
                InstrumentConfig(
                    instrument_id="B", ticker="B", default_buy_broker_id="B1",
                    dcf_view=DCFViewConfig(valuation_basis="terminal_target", total_upside=-0.05, horizon_years=5.0, confidence=1.0),
                ),
            )
        for policy, exp_st in (("block", "blocked_non_positive_excess"), ("allow_with_warning", "suboptimal_non_positive_excess")):
            cfg = PortfolioAnalysisConfig(
                variable_income=vi,
                brokers=(BrokerConfig(broker_id="B1"),),
                positions=(BrokerPosition(instrument_id="A", broker_id="B1", quantity=10.0),),
                black_litterman=BlackLittermanConfig(rf_rate=0.05),
                risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
                optimization=OptimizationConfig(
                    strategies_to_run=("max_sharpe",),
                    selected_strategy="max_sharpe",
                    non_positive_excess_return_policy=policy,
                ),
            )
            ar = analyze_portfolio(cfg, prices_usd_override=prices_df[["A"]] if n_mode == 1 else prices_df)
            st = ar.strategies["max_sharpe"]
            reb = ar.rebalancing["max_sharpe"]
            assert st.status == exp_st
            assert st.diagnostics["is_valid_target"] is False
            assert len(st.warnings) >= 1
            assert reb.status == "skipped_invalid_strategy_target"
            assert reb.feasible is False
            assert reb.orders == []


def test_e2e_round2_cross_cutting_variants_dst_debt_prices_cost_vs_shortfall_and_invalid_n1(monkeypatch):
    """Cross-cutting E2E verification combining AUD-04, NEW-01, NEW-02, NEW-03 with multi-asset & ERC/HRP invariants."""
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2023-07-07", periods=104, freq="W-FRI")  # Ends Friday 2025-06-27
    dates_plus_wed = dates.append(pd.DatetimeIndex([pd.Timestamp("2025-07-02")]))
    n_pts = len(dates_plus_wed)

    aapl = 150.0 * np.exp(np.cumsum(rng.normal(0.002, 0.020, n_pts)))
    azn_gbp = 7200.0 * np.exp(np.cumsum(rng.normal(0.001, 0.015, n_pts)))
    azn_gbp[-1] = 7801.33
    btc = 40000.0 * np.exp(np.cumsum(rng.normal(0.003, 0.045, n_pts)))
    bond_adj = 80.0 * np.exp(np.cumsum(rng.normal(0.001, 0.012, n_pts)))
    bond_adj[-1] = 85.50
    bond_unadj = np.linspace(90.0, 95.0, n_pts)
    gbpusd = np.full(n_pts, 1.25)
    vix = pd.Series(np.where(np.arange(n_pts) % 3 == 0, 31.0, 14.0), index=dates_plus_wed, name="^VIX")

    def fake_ticker(sym):
        t = MagicMock()
        if sym == "AZN.L":
            t.info = {"currency": "GBp"}
            t.history.return_value = pd.DataFrame({"Close": azn_gbp}, index=dates_plus_wed)
        elif sym == "AAPL":
            t.info = {"currency": "USD"}
            t.history.return_value = pd.DataFrame({"Close": aapl}, index=dates_plus_wed)
        elif sym == "BTC-USD":
            t.info = {"currency": "USD"}
            t.history.return_value = pd.DataFrame({"Close": btc}, index=dates_plus_wed)
        elif sym == "BOND_YF":
            t.info = {"currency": "USD"}
            t.history.side_effect = lambda period="10y", auto_adjust=True: pd.DataFrame(
                {"Close": bond_adj if auto_adjust else bond_unadj}, index=dates_plus_wed
            )
        else:
            t.info = {"currency": "USD"}
            t.history.return_value = pd.DataFrame()
        return t

    monkeypatch.setattr(market_data.yf, "Ticker", fake_ticker)
    monkeypatch.setattr(
        market_data.yf,
        "download",
        lambda sym, **kw: pd.DataFrame({"Close": gbpusd}, index=dates_plus_wed) if sym == "GBPUSD=X" else pd.DataFrame(),
    )

    bond_terms = FixedIncomeTerms(
        cashflow_currency="USD",
        reference_nominal=100.0,
        settlement_date=date(2025, 7, 2),
        maturity_date=date(2026, 7, 2),
        price_convention="per_100_original_nominal",
        price_type="dirty",
        coupon_structure="zero_coupon",
        day_count_convention="ACT/365",
        cashflows=(CashFlowEvent(payment_date=date(2026, 7, 2), coupon_amount=0.0, amortization_amount=100.0),),
    )

    # 1. Full multi-asset E2E with unadjusted vs adjusted yfinance bond, Wed cutoff, 2 brokers with fees
    cfg = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(
                instrument_id="AAPL", ticker="AAPL", asset_subtype="equity", default_buy_broker_id="IBKR",
                dcf_view=DCFViewConfig(valuation_basis="terminal_target", total_upside=0.40, confidence=0.5),
            ),
            InstrumentConfig(instrument_id="AZN.L", ticker="AZN.L", asset_subtype="equity", default_buy_broker_id="IBKR"),
            InstrumentConfig(instrument_id="BTC", ticker="BTC-USD", asset_subtype="crypto", default_buy_broker_id="IBKR"),
        ),
        fixed_income=(
            InstrumentConfig(
                instrument_id="BOND_YF", ticker="BOND_YF", section="fixed_income", asset_subtype="sovereign_bond",
                quote_source="yfinance", historical_price_series_type="market_price",
                fixed_income_terms=bond_terms, default_buy_broker_id="IOL",
            ),
        ),
        brokers=(
            BrokerConfig(broker_id="IBKR", fee_structure="percentage", buy_percentage_fee=0.005, sell_percentage_fee=0.005),
            BrokerConfig(broker_id="IOL", fee_structure="percentage", buy_percentage_fee=0.005, sell_percentage_fee=0.005),
        ),
        positions=(
            BrokerPosition(instrument_id="AAPL", broker_id="IBKR", quantity=20.0),
            BrokerPosition(instrument_id="AZN.L", broker_id="IBKR", quantity=20.0),
            BrokerPosition(instrument_id="BTC", broker_id="IBKR", quantity=0.02),
            BrokerPosition(instrument_id="BOND_YF", broker_id="IOL", quantity=2000.0),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04, default_dcf_horizon_years=4.0),
        risk=RiskModelConfig(
            covariance_estimator="vix_regime",
            vix_regime=VixRegimeConfig(min_calm_weeks=20, min_crisis_weeks=10),
            min_weekly_observations=52,
        ),
        constraints=ConstraintsConfig(class_bounds={"crypto": (0.0, 0.15), "fixed_income": (0.15, 0.60)}),
        optimization=OptimizationConfig(
            strategies_to_run=("max_sharpe", "min_variance", "min_cvar", "erc", "hrp"),
            selected_strategy="erc",
        ),
        rebalancing=RebalancingConfig(
            available_cash_by_broker={"IBKR": 500.0, "IOL": 1500.0},
            min_weight_diff=0.01,
            min_notional_usd=25.0,
        ),
        as_of_date=date(2025, 7, 2),
    )

    res = analyze_portfolio(cfg, vix_history_override=vix, current_vix_override=(25.0, dates[-1], False))
    res_master = ejecutar_analisis_completo(cfg, render_ui=False, vix_history_override=vix, current_vix_override=(25.0, dates[-1], False))

    # Temporal W-FRI cutoff check: Wed 2025-07-02 -> last closed Friday 2025-06-27
    assert res.risk_model.last_closed_friday == "2025-06-27"
    assert "2025-07-04" in res.risk_model.excluded_incomplete_week_bins
    # FX & NEW-01 unified tradable bond price check
    assert res.quotes_and_data_quality["latest_prices_usd"]["AZN.L"] == pytest.approx(7801.33 * 0.01 * 1.25, rel=1e-12)
    assert res.quotes_and_data_quality["latest_prices_usd"]["BOND_YF"] == pytest.approx(95.0, abs=1e-12)
    assert res.quotes_and_data_quality["execution_prices_usd_per_unit"]["BOND_YF"] == pytest.approx(0.95, abs=1e-12)
    assert res.fixed_income_metrics["BOND_YF"].dirty_price_usd_per_original_nominal == pytest.approx(0.95, abs=1e-12)
    assert res.fixed_income_metrics["BOND_YF"].ytm_annual == pytest.approx(100.0 / 95.0 - 1.0, abs=1e-10)

    # Parity between analyze_portfolio and ejecutar_analisis_completo + risk contribution identity
    for sname in ("max_sharpe", "min_variance", "min_cvar", "erc", "hrp"):
        st = res.strategies[sname]
        assert st.converged is True and st.feasible is True
        assert float(st.risk_contributions_abs.sum()) == pytest.approx(st.volatility, abs=1e-10)
        np.testing.assert_allclose(st.weights, res_master.strategies[sname].weights, atol=1e-12)

    # Rebalancing uses 2000 VN * $0.95 = $1,900.00 (NOT $1,710.00) and reconciles every ledger!
    reb_erc = res.rebalancing["erc"]
    assert reb_erc.broker_ledgers["IOL"]["initial_holdings_value_usd"] == pytest.approx(1900.0, abs=1e-8)
    _assert_rebalancing_ledger_reconciliation(reb_erc)

    # ERC & HRP remain invariant to mu changes
    resolved = validate_constraints_feasibility(
        res.tickers, cfg.constraints,
        risk_class_by_ticker={i.instrument_id: str(i.risk_class) for i in cfg.instruments},
    )
    mu_alt = res.expected_returns * 2.5 + 0.12
    np.testing.assert_allclose(
        optimize_erc(res.cov_matrix, resolved, mu=res.expected_returns, rf=0.04).weights,
        optimize_erc(res.cov_matrix, resolved, mu=mu_alt, rf=0.04).weights,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        optimize_hrp(res.cov_matrix, resolved, mu=res.expected_returns, rf=0.04).weights,
        optimize_hrp(res.cov_matrix, resolved, mu=mu_alt, rf=0.04).weights,
        atol=1e-12,
    )
