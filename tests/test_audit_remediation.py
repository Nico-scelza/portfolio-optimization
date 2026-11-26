"""Independent oracle regression suite for audit findings AUD-01 through AUD-15
and full multi-asset end-to-end verification under pandas 3.x.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import threading
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization import (
    AnalysisResult,
    BlackLittermanConfig,
    BrokerConfig,
    BrokerPosition,
    CashFlowEvent,
    ConfigurationError,
    ConstraintsConfig,
    DataQualityError,
    DCFViewConfig,
    FixedIncomeTerms,
    FixedIncomeValidationError,
    FXConversionError,
    InfeasibleConstraintsError,
    InstrumentConfig,
    InsufficientHistoryError,
    IOLConfig,
    IOLPanelRow,
    IOLPanelSnapshot,
    IOLQuoteError,
    OptimizationConfig,
    PortfolioAnalysisConfig,
    ProposedOrder,
    RebalancingConfig,
    RebalancingResult,
    RiskModelConfig,
    VixRegimeConfig,
    analyze_portfolio,
    annualize_dcf_view,
    compute_black_litterman,
    compute_dirty_price_per_reference,
    compute_implicit_mep_al30,
    compute_pure_hrp_weights,
    compute_rebalancing_plan,
    compute_weekly_returns_w_fri,
    config,
    descargar_precios_historicos,
    ejecutar_analisis_completo,
    evaluate_fixed_income_instrument,
    ewma_covariance_weekly,
    fetch_iol_panel,
    get_fred_api_key,
    iol,
    is_display_enabled,
    market_data,
    mostrar_rebalanceo,
    normalize_execution_price_to_usd_per_unit,
    notebook_ui,
    optimization,
    optimize_erc,
    optimize_hrp,
    optimize_max_sharpe,
    optimize_min_cvar,
    optimize_min_variance,
    parse_iol_fixed_income_table,
    resolve_iol_bond_quote_usd,
    resolve_provider_currency,
    run_portfolio_analysis,
    validate_constraints_feasibility,
)
from portfolio_optimization._display import suppress_display


# =============================================================================
# AUD-01 & AUD-10: Currency detection, subunit GBp scaling, conflict blocking,
# and preservation of FXConversionError causes
# =============================================================================

def test_aud01_default_currency_none_converts_azn_gbp_via_analyze_portfolio_and_master(monkeypatch):
    """AUD-01: AZN.L with default currency=None uses Yahoo GBp metadata -> 7801.33 * 0.01 * 1.25 = 97.516625 USD."""
    dates = pd.date_range("2024-01-05", periods=70, freq="W-FRI")
    azn_gbp_prices = np.linspace(7000.0, 7801.33, len(dates))
    us_prices = np.linspace(180.0, 220.0, len(dates))
    # Time-varying GBPUSD=X ending at exact 1.25
    gbpusd_series = pd.Series(np.linspace(1.20, 1.25, len(dates)), index=dates)

    fx_calls: list[str] = []

    def fake_ticker(sym):
        t = MagicMock()
        if sym == "AZN.L":
            t.info = {"currency": "GBp"}
            t.history.return_value = pd.DataFrame({"Close": azn_gbp_prices}, index=dates)
        elif sym == "AAPL":
            t.info = {"currency": "USD"}
            t.history.return_value = pd.DataFrame({"Close": us_prices}, index=dates)
        elif sym == "NO_CCY":
            t.info = {}
            t.history.return_value = pd.DataFrame({"Close": us_prices}, index=dates)
        elif sym == "SAP.DE":
            t.info = {"currency": "EUR"}
            t.history.return_value = pd.DataFrame({"Close": us_prices}, index=dates)
        else:
            t.info = {"currency": "ARS"}
            t.history.return_value = pd.DataFrame({"Close": us_prices * 1000.0}, index=dates)
        return t

    def fake_yf_download(fx_sym, period="10y", auto_adjust=True, progress=False):
        fx_calls.append(fx_sym)
        if fx_sym == "GBPUSD=X":
            return pd.DataFrame({"Close": gbpusd_series.values}, index=dates)
        return pd.DataFrame()

    monkeypatch.setattr(market_data.yf, "Ticker", fake_ticker)
    monkeypatch.setattr(market_data.yf, "download", fake_yf_download)

    # 1. Default currency=None on AZN.L and AAPL via analyze_portfolio and ejecutar_analisis_completo
    cfg = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(instrument_id="AZN.L", ticker="AZN.L"),
            InstrumentConfig(instrument_id="AAPL", ticker="AAPL"),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
    )
    assert cfg.instrument_map["AZN.L"].currency is None
    res = analyze_portfolio(cfg)
    res_master = ejecutar_analisis_completo(cfg, render_ui=False)

    expected_azn_usd = 7801.33 * 0.01 * 1.25  # 97.516625
    assert res.quotes_and_data_quality["latest_prices_usd"]["AZN.L"] == pytest.approx(expected_azn_usd, rel=1e-12)
    assert round(res.quotes_and_data_quality["latest_prices_usd"]["AZN.L"], 2) == 97.52
    assert res_master.quotes_and_data_quality["latest_prices_usd"]["AZN.L"] == pytest.approx(expected_azn_usd, rel=1e-12)
    assert "GBPUSD=X" in fx_calls
    assert res.quotes_and_data_quality["fx_diagnostics"]["AAPL"]["fx_symbol"] is None
    assert res.quotes_and_data_quality["fx_diagnostics"]["AZN.L"]["unit_scale"] == pytest.approx(0.01)

    # Verify USD weekly returns used historical time-varying FX (not static FX)
    expected_usd_series = pd.Series(azn_gbp_prices * 0.01 * gbpusd_series.values, index=dates)
    expected_ret = expected_usd_series.pct_change().dropna()
    np.testing.assert_allclose(
        res.risk_model.weekly_returns["AZN.L"].values,
        expected_ret.values,
        rtol=1e-12,
        atol=1e-12,
    )

    # 2. Explicit economic currency="GBP" with provider GBp still applies 0.01 scale
    cfg_gbp = PortfolioAnalysisConfig(
        variable_income=(InstrumentConfig(instrument_id="AZN.L", ticker="AZN.L", currency="GBP"),),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
    )
    res_gbp = analyze_portfolio(cfg_gbp)
    assert res_gbp.quotes_and_data_quality["latest_prices_usd"]["AZN.L"] == pytest.approx(expected_azn_usd, rel=1e-12)

    # 3. Explicit currency="USD" against GBp, EUR, or ARS MUST be rejected (never silently mislabeled USD)
    for tkr in ("AZN.L", "SAP.DE", "GGAL.BA"):
        cfg_conflict = PortfolioAnalysisConfig(
            variable_income=(InstrumentConfig(instrument_id=tkr, ticker=tkr, currency="USD"),),
            black_litterman=BlackLittermanConfig(rf_rate=0.04),
            risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        )
        with pytest.raises(FXConversionError, match="Material currency conflict"):
            analyze_portfolio(cfg_conflict)

    # 4. Missing provider currency metadata with currency=None MUST NOT assume USD
    cfg_missing_ccy = PortfolioAnalysisConfig(
        variable_income=(InstrumentConfig(instrument_id="NO_CCY", ticker="NO_CCY"),),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
    )
    with pytest.raises(FXConversionError, match="Missing currency metadata"):
        analyze_portfolio(cfg_missing_ccy)


def test_aud10_fx_conversion_error_preserved_across_all_download_and_pipeline_apis(monkeypatch):
    """AUD-10: GGAL.BA ARS with empty ARS=X raises typed FXConversionError (never hidden under InsufficientHistoryError)."""
    dates = pd.date_range("2024-01-05", periods=70, freq="W-FRI")

    def fake_ticker(sym):
        t = MagicMock()
        if sym == "AAPL":
            t.info = {"currency": "USD"}
            t.history.return_value = pd.DataFrame({"Close": np.linspace(150.0, 190.0, len(dates))}, index=dates)
        elif sym == "GGAL.BA":
            t.info = {"currency": "ARS"}
            t.history.return_value = pd.DataFrame({"Close": np.linspace(3000.0, 5000.0, len(dates))}, index=dates)
        elif sym == "EMPTY_HIST":
            t.info = {"currency": "USD"}
            t.history.return_value = pd.DataFrame()
        else:
            t.info = {"currency": "UNKNOWN_XYZ"}
            t.history.return_value = pd.DataFrame({"Close": np.linspace(10.0, 20.0, len(dates))}, index=dates)
        return t

    monkeypatch.setattr(market_data.yf, "Ticker", fake_ticker)
    monkeypatch.setattr(market_data.yf, "download", lambda *a, **kw: pd.DataFrame())

    # 1. _download_prices
    with pytest.raises(FXConversionError, match="Missing historical FX series 'ARS=X'") as e1:
        market_data._download_prices(["AAPL", "GGAL.BA"], period="1y")
    assert e1.value.asset_id == "GGAL.BA"

    # 2. descargar_precios_historicos
    with pytest.raises(FXConversionError, match="Missing historical FX series 'ARS=X'") as e2:
        descargar_precios_historicos(["AAPL", "GGAL.BA"], periodo="1y")
    assert e2.value.asset_id == "GGAL.BA"

    # 3. analyze_portfolio and ejecutar_analisis_completo
    cfg = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(instrument_id="AAPL", ticker="AAPL"),
            InstrumentConfig(instrument_id="GGAL.BA", ticker="GGAL.BA"),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
    )
    with pytest.raises(FXConversionError, match="Missing historical FX series 'ARS=X'") as e3:
        analyze_portfolio(cfg)
    assert e3.value.asset_id == "GGAL.BA"

    with pytest.raises(FXConversionError, match="Missing historical FX series 'ARS=X'") as e4:
        ejecutar_analisis_completo(cfg, render_ui=False)
    assert e4.value.asset_id == "GGAL.BA"

    # 4. Non-FX empty history still raises InsufficientHistoryError with exact asset_id
    with pytest.raises(InsufficientHistoryError) as e5:
        descargar_precios_historicos(["AAPL", "EMPTY_HIST"], periodo="1y")
    assert e5.value.asset_id == "EMPTY_HIST"


# =============================================================================
# AUD-02 & AUD-15(per_unit): Bond nominal unit normalization in valuation,
# YTM, and rebalancing
# =============================================================================

def test_aud02_bond_per_100_valuation_orders_residual_clean_dirty_and_equivalence():
    """AUD-02: AL30 at 49.16/100 VN, 1000 VN = $491.60; buy $500 -> 1017 VN ($499.9572), cash $0.0428."""
    terms_al30 = FixedIncomeTerms(
        cashflow_currency="USD",
        reference_nominal=100.0,
        settlement_date=date(2025, 1, 1),
        maturity_date=date(2027, 1, 1),
        price_convention="per_100_original_nominal",
        price_type="dirty",
        cashflows=(
            CashFlowEvent(payment_date=date(2026, 1, 1), coupon_amount=4.0, amortization_amount=0.0),
            CashFlowEvent(payment_date=date(2027, 1, 1), coupon_amount=4.0, amortization_amount=100.0),
        ),
    )
    broker = BrokerConfig(broker_id="IOL", fee_structure="percentage", buy_percentage_fee=0.0, sell_percentage_fee=0.0)
    inst_al30 = InstrumentConfig(
        instrument_id="AL30",
        ticker="AL30",
        section="fixed_income",
        asset_subtype="sovereign_bond",
        quote_source="iol_bonos",
        fractional_allowed=False,
        lot_size=1.0,
        min_nominal=100.0,
        default_buy_broker_id="IOL",
        fixed_income_terms=terms_al30,
    )
    inst_spy = InstrumentConfig(
        instrument_id="SPY",
        ticker="SPY",
        section="variable_income",
        asset_subtype="equity_etf",
        default_buy_broker_id="IOL",
    )

    # 1. Valuation: 1000 VN AL30 ($491.60) + 1 SPY ($500) = $991.60, AL30 weight ~49.5764%
    reb_val = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["AL30", "SPY"],
        target_weights={"AL30": 491.60 / 991.60, "SPY": 500.0 / 991.60},
        prices_usd={"AL30": 49.16, "SPY": 500.0},
        instruments_by_id={"AL30": inst_al30, "SPY": inst_spy},
        brokers_by_id={"IOL": broker},
        positions=[
            BrokerPosition(instrument_id="AL30", broker_id="IOL", quantity=1000.0),
            BrokerPosition(instrument_id="SPY", broker_id="IOL", quantity=1.0),
        ],
    )
    assert reb_val.initial_holdings_value_usd == pytest.approx(991.60, abs=1e-9)
    assert reb_val.current_weights["AL30"] == pytest.approx(491.60 / 991.60, abs=1e-9)
    assert round(reb_val.current_weights["AL30"] * 100.0, 2) == 49.58

    # 2. Buy $500 AL30 with 0 costs, integer VN, min_nominal=100 -> 1017 VN, $499.9572, cash $0.0428
    reb_buy = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["AL30"],
        target_weights={"AL30": 1.0},
        prices_usd={"AL30": 49.16},
        instruments_by_id={"AL30": inst_al30},
        brokers_by_id={"IOL": broker},
        positions=[],
        rebalancing_config=RebalancingConfig(available_cash_usd=500.0, min_weight_diff=0.0, min_notional_usd=0.0),
    )
    assert len(reb_buy.orders) == 1
    ord_buy = reb_buy.orders[0]
    assert ord_buy.quantity == pytest.approx(1017.0, abs=1e-12)
    assert ord_buy.price_usd == pytest.approx(0.4916, abs=1e-12)
    assert ord_buy.gross_amount_usd == pytest.approx(499.9572, abs=1e-10)
    assert reb_buy.ending_cash_usd == pytest.approx(0.0428, abs=1e-10)
    assert ord_buy.quantity_unit == "original_nominal_units"

    # 3. Economic equivalence across per_100_residual_nominal (factor=0.6, ref=1000),
    #    per_100_original_nominal, per_unit (AUD-15), and clean+accrued vs dirty!
    cfs_1000 = (
        CashFlowEvent(payment_date=date(2026, 1, 1), coupon_amount=60.0, amortization_amount=0.0),
        CashFlowEvent(payment_date=date(2027, 1, 1), coupon_amount=60.0, amortization_amount=600.0),
    )
    t_res = FixedIncomeTerms(
        cashflow_currency="USD", reference_nominal=1000.0, amortization_factor=0.6,
        settlement_date=date(2025, 1, 1), maturity_date=date(2027, 1, 1),
        price_convention="per_100_residual_nominal", price_type="dirty", cashflows=cfs_1000,
    )
    t_orig = FixedIncomeTerms(
        cashflow_currency="USD", reference_nominal=1000.0, amortization_factor=0.6,
        settlement_date=date(2025, 1, 1), maturity_date=date(2027, 1, 1),
        price_convention="per_100_original_nominal", price_type="dirty", cashflows=cfs_1000,
    )
    t_unit = FixedIncomeTerms(
        cashflow_currency="USD", reference_nominal=1000.0, amortization_factor=0.6,
        settlement_date=date(2025, 1, 1), maturity_date=date(2027, 1, 1),
        price_convention="per_unit", price_type="dirty", cashflows=cfs_1000,
    )
    t_clean = FixedIncomeTerms(
        cashflow_currency="USD", reference_nominal=1000.0, amortization_factor=0.6,
        settlement_date=date(2025, 1, 1), maturity_date=date(2027, 1, 1),
        price_convention="per_100_original_nominal", price_type="clean",
        accrued_interest=1.5, accrued_interest_basis="quote_convention", cashflows=cfs_1000,
    )
    # 80 per 100 residual (0.6 factor) == 48 per 100 original == 480 per 1000-VN bond unit == clean 46.5 + 1.5 accrued
    m_res = evaluate_fixed_income_instrument("B1", t_res, quoted_price=80.0)
    m_orig = evaluate_fixed_income_instrument("B1", t_orig, quoted_price=48.0)
    m_unit = evaluate_fixed_income_instrument("B1", t_unit, quoted_price=480.0)
    m_clean = evaluate_fixed_income_instrument("B1", t_clean, quoted_price=46.5)

    for m in (m_res, m_orig, m_unit, m_clean):
        assert m.status == "ok"
        assert m.dirty_price_per_reference == pytest.approx(480.0, abs=1e-12)
        assert m.dirty_price_usd_per_original_nominal == pytest.approx(0.48, abs=1e-12)
        assert m.ytm_annual == pytest.approx(m_res.ytm_annual, abs=1e-12)

    # Verify normalize_execution_price_to_usd_per_unit produces exact 0.48 across all 4 representations
    for terms_i, qpx in ((t_res, 80.0), (t_orig, 48.0), (t_unit, 480.0), (t_clean, 46.5)):
        inst_i = InstrumentConfig(
            instrument_id="B1", ticker="B1", section="fixed_income",
            asset_subtype="corporate_bond_on", fixed_income_terms=terms_i,
        )
        px_vn, _ = normalize_execution_price_to_usd_per_unit(inst_i, qpx)
        assert px_vn == pytest.approx(0.48, abs=1e-12)

    # 4. Lot size > 1 (lot_size=100), partial sell, commissions
    broker_fee = BrokerConfig(broker_id="IOL", fee_structure="percentage", buy_percentage_fee=0.005, sell_percentage_fee=0.005)
    inst_lot100 = InstrumentConfig(
        instrument_id="AL30", ticker="AL30", section="fixed_income", asset_subtype="sovereign_bond",
        fractional_allowed=False, lot_size=100.0, min_nominal=100.0, default_buy_broker_id="IOL",
        fixed_income_terms=terms_al30,
    )
    reb_lot = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["AL30", "SPY"],
        target_weights={"AL30": 0.25, "SPY": 0.75},
        prices_usd={"AL30": 49.16, "SPY": 500.0},
        instruments_by_id={"AL30": inst_lot100, "SPY": inst_spy},
        brokers_by_id={"IOL": broker_fee},
        positions=[
            BrokerPosition(instrument_id="AL30", broker_id="IOL", quantity=1000.0),
            BrokerPosition(instrument_id="SPY", broker_id="IOL", quantity=1.0),
        ],
    )
    sell_ord = [o for o in reb_lot.orders if o.instrument_id == "AL30" and o.side == "SELL"][0]
    assert sell_ord.quantity % 100.0 == 0.0 and sell_ord.quantity == 400.0
    assert sell_ord.gross_amount_usd == pytest.approx(400.0 * 0.4916, abs=1e-10)
    assert sell_ord.commission_usd == pytest.approx(sell_ord.gross_amount_usd * 0.005, abs=1e-10)


# =============================================================================
# AUD-03 & AUD-11: Segregated broker cash ledgers, post-rebalance HHI/sector
# hard limits, scaled min_notional revalidation, omitted SELL broker attribution
# =============================================================================

def test_aud03_segregated_broker_cash_no_implicit_interbroker_transfer():
    """AUD-03: Selling X ($1000) in B_SELL with 0 cash in B_BUY must NOT fund buying Y in B_BUY."""
    brokers = {
        "B_SELL": BrokerConfig(broker_id="B_SELL", fee_structure="percentage", sell_percentage_fee=0.005),
        "B_BUY": BrokerConfig(broker_id="B_BUY", fee_structure="percentage", buy_percentage_fee=0.01),
    }
    insts = {
        "X": InstrumentConfig(instrument_id="X", ticker="X", default_buy_broker_id="B_SELL"),
        "Y": InstrumentConfig(instrument_id="Y", ticker="Y", default_buy_broker_id="B_BUY"),
    }

    # Case 1: Zero cash in B_BUY -> sell stays in B_SELL, ZERO buy in B_BUY!
    reb = compute_rebalancing_plan(
        strategy_name="max_sharpe",
        tickers=["X", "Y"],
        target_weights={"X": 0.0, "Y": 1.0},
        prices_usd={"X": 100.0, "Y": 100.0},
        instruments_by_id=insts,
        brokers_by_id=brokers,
        positions=[BrokerPosition(instrument_id="X", broker_id="B_SELL", quantity=10.0)],
        rebalancing_config=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=25.0),
    )
    assert reb.status == "partially_funded"
    assert reb.feasible is False
    assert [o.side for o in reb.orders] == ["SELL"]
    assert reb.orders[0].broker_id == "B_SELL"
    assert reb.ending_cash_by_broker["B_SELL"] == pytest.approx(995.0, abs=1e-9)
    assert reb.ending_cash_by_broker["B_BUY"] == pytest.approx(0.0, abs=1e-9)
    assert reb.ending_cash_usd == pytest.approx(995.0, abs=1e-9)
    assert reb.broker_ledgers["B_SELL"]["sell_net_proceeds_usd"] == pytest.approx(995.0, abs=1e-9)
    assert reb.broker_ledgers["B_BUY"]["buy_total_outflow_usd"] == pytest.approx(0.0, abs=1e-9)
    assert any(o.instrument_id == "Y" and o.broker_id == "B_BUY" for o in reb.omitted_operations)

    # Case 2: Scalar available_cash_usd with multiple brokers MUST raise ConfigurationError
    with pytest.raises(ConfigurationError, match="ambiguous with multiple brokers"):
        compute_rebalancing_plan(
            strategy_name="max_sharpe",
            tickers=["X", "Y"],
            target_weights={"X": 0.0, "Y": 1.0},
            prices_usd={"X": 100.0, "Y": 100.0},
            instruments_by_id=insts,
            brokers_by_id=brokers,
            positions=[BrokerPosition(instrument_id="X", broker_id="B_SELL", quantity=10.0)],
            rebalancing_config=RebalancingConfig(available_cash_usd=500.0),
        )

    # Case 3: Explicit own funds & cash flows per broker -> reconciles every ledger and aggregate exactly
    # Holdings X = 10.15 * $100 = $1,015 in B_SELL; init cash = $50 (B_SELL) + $635 (B_BUY) = $685
    # Flows = -$50 (B_SELL) + $350 (B_BUY) = +$300 -> deployable = $2,000
    # Target X=0.515 (within 1pp -> HOLD, B_SELL cash=$0), Y=0.485 ($970 + $9.70 fee = $979.70 <= $985 in B_BUY)
    reb_funded = compute_rebalancing_plan(
        strategy_name="max_sharpe",
        tickers=["X", "Y"],
        target_weights={"X": 0.515, "Y": 0.485},
        prices_usd={"X": 100.0, "Y": 100.0},
        instruments_by_id=insts,
        brokers_by_id=brokers,
        positions=[BrokerPosition(instrument_id="X", broker_id="B_SELL", quantity=10.15)],
        rebalancing_config=RebalancingConfig(
            available_cash_by_broker={"B_SELL": 50.0, "B_BUY": 635.0},
            cash_flow_by_broker={"B_SELL": -50.0, "B_BUY": 350.0},
            min_weight_diff=0.01,
            min_notional_usd=25.0,
        ),
    )
    assert reb_funded.status == "ok"
    assert reb_funded.feasible is True
    assert reb_funded.initial_portfolio_value_usd == pytest.approx(1700.0, abs=1e-9)
    assert reb_funded.deployable_capital_usd == pytest.approx(2000.0, abs=1e-9)
    assert reb_funded.ending_cash_by_broker["B_SELL"] == pytest.approx(0.0, abs=1e-9)
    assert reb_funded.ending_cash_by_broker["B_BUY"] == pytest.approx(5.30, abs=1e-9)
    # Ledger tie-out per broker: ending = init + flow + sell_net - buy_out
    for bid, led in reb_funded.broker_ledgers.items():
        expected_end = led["initial_cash_usd"] + led["cash_flow_usd"] + led["sell_net_proceeds_usd"] - led["buy_total_outflow_usd"]
        assert led["ending_cash_usd"] == pytest.approx(expected_end, abs=1e-9)
        assert led["ending_cash_usd"] >= -1e-9
    assert sum(led["ending_cash_usd"] for led in reb_funded.broker_ledgers.values()) == pytest.approx(reb_funded.ending_cash_usd, abs=1e-9)


def test_aud11_hhi_sector_hard_limits_scaled_min_notional_and_omitted_sell_broker():
    """AUD-11: Post-rebalance HHI & sector checks, scaled buy < min_notional omitted, unified initial value, SELL broker."""
    broker = BrokerConfig(broker_id="B1", fee_structure="percentage", buy_percentage_fee=0.01, sell_percentage_fee=0.01)
    insts = {
        "A": InstrumentConfig(instrument_id="A", ticker="A", default_buy_broker_id="B1"),
        "B": InstrumentConfig(instrument_id="B", ticker="B", default_buy_broker_id="B1"),
    }

    # 1. HHI hard limit 0.582 vs achievable HHI ~0.586528 (trade omitted by min_weight_diff=0.01)
    reb_hhi = compute_rebalancing_plan(
        strategy_name="min_variance",
        tickers=["A", "B"],
        target_weights={"A": 0.70, "B": 0.30},
        prices_usd={"A": 100.0, "B": 100.0},
        instruments_by_id=insts,
        brokers_by_id={"B1": broker},
        positions=[
            BrokerPosition(instrument_id="A", broker_id="B1", quantity=70.8),
            BrokerPosition(instrument_id="B", broker_id="B1", quantity=29.2),
        ],
        rebalancing_config=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=25.0),
        constraints_config=ConstraintsConfig(hhi_limit=0.582),
    )
    ach_hhi = sum(w * w for w in reb_hhi.achievable_weights.values())
    assert ach_hhi == pytest.approx(0.708**2 + 0.292**2, abs=1e-9)  # 0.586528
    assert reb_hhi.feasible is False
    assert reb_hhi.status == "infeasible_hard_limits"
    assert any("HHI" in c for c in reb_hhi.hard_limit_conflicts)

    # 2. Sector hard limit violated post-rebalance
    reb_sec = compute_rebalancing_plan(
        strategy_name="min_variance",
        tickers=["A", "B"],
        target_weights={"A": 0.70, "B": 0.30},
        prices_usd={"A": 100.0, "B": 100.0},
        instruments_by_id=insts,
        brokers_by_id={"B1": broker},
        positions=[
            BrokerPosition(instrument_id="A", broker_id="B1", quantity=70.8),
            BrokerPosition(instrument_id="B", broker_id="B1", quantity=29.2),
        ],
        rebalancing_config=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=25.0),
        constraints_config=ConstraintsConfig(max_sector_weight=0.70),
        sector_by_ticker={"A": "Technology", "B": "Healthcare"},
    )
    assert reb_sec.feasible is False
    assert reb_sec.status == "infeasible_hard_limits"
    assert any("sector" in c.lower() for c in reb_sec.hard_limit_conflicts)

    # 3. Buy initially eligible (>=$25) scaled down by cash to ~$9.90 < $25 MUST be omitted!
    reb_micro = compute_rebalancing_plan(
        strategy_name="min_variance",
        tickers=["A", "B"],
        target_weights={"A": 0.50, "B": 0.50},
        prices_usd={"A": 100.0, "B": 100.0},
        instruments_by_id=insts,
        brokers_by_id={"B1": broker},
        positions=[BrokerPosition(instrument_id="A", broker_id="B1", quantity=10.0)],
        rebalancing_config=RebalancingConfig(available_cash_usd=10.0, min_weight_diff=0.01, min_notional_usd=25.0),
    )
    # Wait: here A sells $500; let's construct a clean case where A is at target or below threshold so ONLY $10 cash exists for B!
    insts3 = {
        "A": InstrumentConfig(instrument_id="A", ticker="A", default_buy_broker_id="B1"),
        "B": InstrumentConfig(instrument_id="B", ticker="B", default_buy_broker_id="B1"),
    }
    reb_micro2 = compute_rebalancing_plan(
        strategy_name="min_variance",
        tickers=["A", "B"],
        target_weights={"A": 0.96, "B": 0.04},  # deployable = 1010 -> A desired 969.6 (diff=30.4, w_diff=0.0301 -> sells? Use 0.991 / 0.009 no -> use high min_weight_diff on A? Use 2 brokers!)
        prices_usd={"A": 100.0, "B": 100.0},
        instruments_by_id={
            "A": InstrumentConfig(instrument_id="A", ticker="A", default_buy_broker_id="B_HOLD"),
            "B": InstrumentConfig(instrument_id="B", ticker="B", default_buy_broker_id="B1"),
        },
        brokers_by_id={
            "B_HOLD": BrokerConfig(broker_id="B_HOLD", fee_structure="percentage", sell_percentage_fee=0.0),
            "B1": broker,
        },
        positions=[BrokerPosition(instrument_id="A", broker_id="B_HOLD", quantity=10.0)],
        rebalancing_config=RebalancingConfig(
            available_cash_by_broker={"B_HOLD": 0.0, "B1": 10.0},
            min_weight_diff=0.01,
            min_notional_usd=25.0,
        ),
    )
    # B desired notional = 0.04 * 1010 = $40.40 >= $25, but B1 only has $10 -> scaled gross ~$9.90 < $25 -> OMITTED!
    assert all(o.instrument_id != "B" for o in reb_micro2.orders)
    assert any(o.instrument_id == "B" and "scaled_notional" in ";".join(o.reasons) for o in reb_micro2.omitted_operations)

    # 4. Omitted SELL attributes actual holding broker (B_HOLD), NOT default_buy_broker_id (B1)!
    reb_omit_sell = compute_rebalancing_plan(
        strategy_name="min_variance",
        tickers=["A", "B"],
        target_weights={"A": 0.495, "B": 0.505},
        prices_usd={"A": 100.0, "B": 100.0},
        instruments_by_id={
            "A": InstrumentConfig(instrument_id="A", ticker="A", default_buy_broker_id="B1"),
            "B": InstrumentConfig(instrument_id="B", ticker="B", default_buy_broker_id="B1"),
        },
        brokers_by_id={
            "B_HOLD": BrokerConfig(broker_id="B_HOLD", fee_structure="percentage"),
            "B1": broker,
        },
        positions=[
            BrokerPosition(instrument_id="A", broker_id="B_HOLD", quantity=50.0),
            BrokerPosition(instrument_id="B", broker_id="B1", quantity=50.0),
        ],
        rebalancing_config=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=25.0),
    )
    omitted_a = [o for o in reb_omit_sell.omitted_operations if o.instrument_id == "A" and o.side == "SELL"][0]
    assert omitted_a.broker_id == "B_HOLD"


# =============================================================================
# AUD-04: Incomplete W-FRI week exclusion vs Friday holiday & cutoff timing
# =============================================================================

def test_aud04_incomplete_w_fri_week_excluded_and_good_friday_preserved():
    """AUD-04: Data ending Wed 2025-07-02 with cutoff 2025-07-02 23:59:59 -> last bin 2025-06-27 (never 2025-07-04)."""
    # Daily data from 2025-01-06 through Wednesday 2025-07-02, excluding Good Friday 2025-04-18
    bdays = pd.bdate_range("2025-01-06", "2025-07-02")
    eq_days = bdays[bdays != pd.Timestamp("2025-04-18")]
    rng = np.random.default_rng(104)
    df = pd.DataFrame({
        "EQ": pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(eq_days)))), index=eq_days),
        "BTC": pd.Series(50000.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, len(bdays)))), index=bdays),
    })

    bundle = compute_weekly_returns_w_fri(
        df,
        risk_config=RiskModelConfig(min_weekly_observations=20),
        as_of_cutoff=pd.Timestamp("2025-07-02 23:59:59"),
    )
    assert bundle.weekly_returns.index.max() == pd.Timestamp("2025-06-27")
    assert pd.Timestamp("2025-07-04") not in bundle.weekly_returns.index
    assert "2025-07-04" in bundle.excluded_incomplete_week_bins
    assert pd.Timestamp("2025-04-18") in bundle.weekly_returns.index  # Good Friday closed week included!

    # Without explicit as_of_cutoff, conservative max-observation reference also excludes 2025-07-04
    bundle_no_cutoff = compute_weekly_returns_w_fri(df, risk_config=RiskModelConfig(min_weekly_observations=20))
    assert bundle_no_cutoff.weekly_returns.index.max() == pd.Timestamp("2025-06-27")

    # Good Friday 2025-04-18 with Saturday 2025-04-19 cutoff -> 2025-04-18 INCLUDED (closed week)
    df_gf = df.loc[df.index <= pd.Timestamp("2025-04-17")]
    bundle_gf = compute_weekly_returns_w_fri(
        df_gf,
        risk_config=RiskModelConfig(min_weekly_observations=10),
        as_of_cutoff=pd.Timestamp("2025-04-19 10:00:00"),
    )
    assert bundle_gf.weekly_returns.index.max() == pd.Timestamp("2025-04-18")

    # Intraday Friday before/at/after 17:00 America/New_York + Saturday crypto print
    fri_times = pd.DatetimeIndex([
        pd.Timestamp("2025-01-10 16:00:00", tz="America/New_York"),
        pd.Timestamp("2025-01-17 16:00:00", tz="America/New_York"),
        pd.Timestamp("2025-01-24 16:00:00", tz="America/New_York"),
        pd.Timestamp("2025-01-31 16:00:00", tz="America/New_York"),
        pd.Timestamp("2025-02-07 16:00:00", tz="America/New_York"),
        pd.Timestamp("2025-02-14 16:00:00", tz="America/New_York"),
        pd.Timestamp("2025-02-14 19:00:00", tz="America/New_York"),
        pd.Timestamp("2025-02-15 12:00:00", tz="America/New_York"),
    ])
    df_intra = pd.DataFrame({"BTC": [100, 105, 110, 115, 120, 125, 999, 888]}, index=fri_times)
    b_before = compute_weekly_returns_w_fri(
        df_intra,
        risk_config=RiskModelConfig(min_weekly_observations=4, weekly_cutoff_hour=17),
        as_of_cutoff=pd.Timestamp("2025-02-14 16:30:00", tz="America/New_York"),
    )
    assert b_before.weekly_returns.index.max() == pd.Timestamp("2025-02-07")

    b_at = compute_weekly_returns_w_fri(
        df_intra,
        risk_config=RiskModelConfig(min_weekly_observations=5, weekly_cutoff_hour=17),
        as_of_cutoff=pd.Timestamp("2025-02-14 17:00:00", tz="America/New_York"),
    )
    assert b_at.weekly_returns.index.max() == pd.Timestamp("2025-02-14")
    assert b_at.weekly_prices_usd.loc[pd.Timestamp("2025-02-14"), "BTC"] == pytest.approx(125.0)

    b_sat = compute_weekly_returns_w_fri(
        df_intra,
        risk_config=RiskModelConfig(min_weekly_observations=5, weekly_cutoff_hour=17),
        as_of_cutoff=pd.Timestamp("2025-02-15 15:00:00", tz="America/New_York"),
    )
    assert b_sat.weekly_returns.index.max() == pd.Timestamp("2025-02-14")
    assert "2025-02-21" in b_sat.excluded_incomplete_week_bins


# =============================================================================
# AUD-05 & AUD-07: Strict convergence checks, Max Sharpe singular & non-positive
# excess return guards, and effective config parameter wiring
# =============================================================================

def test_aud05_non_convergence_not_masked_and_max_sharpe_singular_and_negative_excess(monkeypatch):
    """AUD-05: SLSQP success=False never reports optimal/converged; singular zero-var & excess<=0 handled cleanly."""
    cov = pd.DataFrame([[0.04, 0.005], [0.005, 0.09]], index=["A", "B"], columns=["A", "B"])
    mu = pd.Series([0.10, 0.12], index=["A", "B"])
    rets = pd.DataFrame({"A": [-0.04, -0.02, 0.01, 0.02, 0.03, 0.01], "B": [-0.06, -0.01, 0.02, 0.03, 0.01, 0.02]})

    # 1. Singular covariance with feasible zero-variance portfolio -> NEVER status='optimal' with Sharpe=NaN
    cov_sing = pd.DataFrame([[0.04, -0.04], [-0.04, 0.04]], index=["A", "B"], columns=["A", "B"])
    res_cons = validate_constraints_feasibility(["A", "B"])
    ms_sing = optimize_max_sharpe(pd.Series([0.10, 0.08], index=["A", "B"]), cov_sing, 0.04, res_cons)
    assert ms_sing.converged is False
    assert ms_sing.status == "singular_zero_variance"
    assert ms_sing.status != "optimal"

    # 2. All excess returns negative -> blocked by default; explicit opt-in warns without status='optimal'
    cov_diag = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    mu_neg = pd.Series([0.02, 0.02], index=["A", "B"])
    ms_neg_block = optimize_max_sharpe(mu_neg, cov_diag, 0.04, res_cons, non_positive_excess_return_policy="block")
    assert ms_neg_block.converged is False
    assert ms_neg_block.status == "blocked_non_positive_excess"

    ms_neg_optin = optimize_max_sharpe(mu_neg, cov_diag, 0.04, res_cons, non_positive_excess_return_policy="allow_with_warning")
    assert ms_neg_optin.status == "suboptimal_non_positive_excess"
    assert any("negative sharpe" in w.lower() or "higher volatility" in w.lower() for w in ms_neg_optin.warnings)

    # 3. Asset A has mu > rf (0.08 > 0.04), but constraints cap A <= 0.20 so max feasible excess = -0.016 <= 0!
    res_cap = validate_constraints_feasibility(
        ["A", "B"],
        ConstraintsConfig(asset_bounds={"A": (0.0, 0.20), "B": (0.80, 1.0)}),
    )
    ms_cap_neg = optimize_max_sharpe(pd.Series([0.08, 0.01], index=["A", "B"]), cov_diag, 0.04, res_cap)
    assert ms_cap_neg.converged is False
    assert ms_cap_neg.status == "blocked_non_positive_excess"

    # 4. Mock SLSQP returning success=False with feasible candidate across ALL constrained solver routes
    res_hhi = validate_constraints_feasibility(["A", "B"], ConstraintsConfig(hhi_limit=0.55))
    res_erc_con = validate_constraints_feasibility(["A", "B"], ConstraintsConfig(asset_bounds={"A": (0.0, 0.20), "B": (0.80, 1.0)}))
    real_minimize = optimization.minimize

    def fake_minimize(fun, x0, *args, **kwargs):
        method = kwargs.get("method", "SLSQP")
        if method == "SLSQP":
            return SimpleNamespace(success=False, x=np.asarray(x0, dtype=float), message="Mocked SLSQP failure", nit=10)
        return real_minimize(fun, x0, *args, **kwargs)

    monkeypatch.setattr(optimization, "minimize", fake_minimize)

    r_ms = optimize_max_sharpe(mu, cov, 0.04, res_cons)
    assert r_ms.converged is False and r_ms.status == "non_converged"

    r_mv = optimize_min_variance(cov, res_cons)
    assert r_mv.converged is False and r_mv.status == "non_converged"

    r_cvar = optimize_min_cvar(rets, cov, res_hhi)
    assert r_cvar.converged is False and r_cvar.status == "non_converged"

    r_erc = optimize_erc(cov, res_erc_con)
    assert r_erc.converged is False and r_erc.status == "non_converged"

    r_hrp = optimize_hrp(cov, res_erc_con)
    assert r_hrp.converged is False and r_hrp.status == "non_converged"


def test_aud07_effective_config_params_horizon_solver_tolerance_retired_fields_and_coupon_freq(monkeypatch):
    """AUD-07: default_dcf_horizon_years, solver_tolerance, retired simulation fields, coupon_frequency role."""
    rng = np.random.default_rng(77)
    dates = pd.date_range("2022-01-07", periods=80, freq="W-FRI")
    prices = pd.DataFrame({
        "A": 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.02, len(dates)))),
        "B": 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.015, len(dates)))),
    }, index=dates)

    # 1. DCF view without individual horizon inherits global default_dcf_horizon_years (3y vs 10y); explicit 5y overrides
    def _run_horizon(global_h: float, indiv_h=None) -> float:
        cfg = PortfolioAnalysisConfig(
            variable_income=(
                InstrumentConfig(
                    instrument_id="A", ticker="A",
                    dcf_view=DCFViewConfig(valuation_basis="terminal_target", total_upside=0.50, horizon_years=indiv_h, confidence=0.5),
                ),
                InstrumentConfig(instrument_id="B", ticker="B"),
            ),
            black_litterman=BlackLittermanConfig(rf_rate=0.04, default_dcf_horizon_years=global_h),
            risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
            optimization=OptimizationConfig(strategies_to_run=("min_variance",)),
        )
        res = analyze_portfolio(cfg, prices_usd_override=prices)
        return res.dcf_views["A"].annualized_expected_return

    assert _run_horizon(3.0, None) == pytest.approx((1.5 ** (1.0 / 3.0)) - 1.0, abs=1e-10)
    assert _run_horizon(10.0, None) == pytest.approx((1.5 ** (1.0 / 10.0)) - 1.0, abs=1e-10)
    assert _run_horizon(3.0, 5.0) == pytest.approx((1.5 ** (1.0 / 5.0)) - 1.0, abs=1e-10)

    # 2. solver_tolerance reaches real scipy.optimize.minimize options
    captured_tols: list[float] = []
    real_min = optimization.minimize

    def spy_min(*args, **kwargs):
        opts = kwargs.get("options") or {}
        if "ftol" in opts:
            captured_tols.append(float(opts["ftol"]))
        return real_min(*args, **kwargs)

    monkeypatch.setattr(optimization, "minimize", spy_min)
    cov = pd.DataFrame([[0.04, 0.005], [0.005, 0.09]], index=["A", "B"], columns=["A", "B"])
    res_tol = validate_constraints_feasibility(["A", "B"], ConstraintsConfig(solver_tolerance=3.5e-7))
    optimize_min_variance(cov, res_tol)
    optimize_max_sharpe(pd.Series([0.10, 0.12], index=["A", "B"]), cov, 0.04, res_tol)
    optimize_erc(cov, res_tol)
    assert captured_tols and all(math.isclose(t, 3.5e-7, rel_tol=1e-12) for t in captured_tols)

    # 3. Retired decorative fields simulation_samples / random_seed rejected cleanly
    assert not hasattr(OptimizationConfig(), "simulation_samples")
    assert not hasattr(OptimizationConfig(), "random_seed")
    with pytest.raises(ConfigurationError, match="Removed decorative"):
        OptimizationConfig.from_dict({"simulation_samples": 5000})
    with pytest.raises(ConfigurationError, match="Removed decorative"):
        OptimizationConfig.from_dict({"random_seed": 42})

    # 4. FixedIncomeTerms.coupon_frequency validates schedule spacing & provides periodic fallback without double-dividing
    # Semiannual schedule (6m): coupon_frequency=2 passes and produces exact 4.0/period (8% * 0.5)
    semi_terms = FixedIncomeTerms(
        cashflow_currency="USD", reference_nominal=100.0,
        settlement_date=date(2025, 1, 1), maturity_date=date(2026, 1, 1),
        day_count_convention="30/360", coupon_frequency=2,
        cashflows=(
            CashFlowEvent(payment_date=date(2025, 7, 1), coupon_rate=0.08, period_start_date=date(2025, 1, 1)),
            CashFlowEvent(payment_date=date(2026, 1, 1), coupon_rate=0.08, amortization_amount=100.0),
        ),
    )
    m_semi = evaluate_fixed_income_instrument("SEMI", semi_terms, quoted_price=100.0)
    assert m_semi.status == "ok" and m_semi.total_future_coupons == pytest.approx(8.0, abs=1e-9)

    # Mismatched coupon_frequency=1 on 6-month schedule raises FixedIncomeValidationError!
    bad_freq = FixedIncomeTerms(
        cashflow_currency="USD", reference_nominal=100.0,
        settlement_date=date(2025, 1, 1), maturity_date=date(2026, 1, 1),
        day_count_convention="30/360", coupon_frequency=1,
        cashflows=semi_terms.cashflows,
    )
    with pytest.raises(FixedIncomeValidationError, match="inconsistent with declared coupon_frequency=1"):
        evaluate_fixed_income_instrument("BADF", bad_freq, quoted_price=100.0)

    # Without period_start_date, coupon_frequency=2 computes periodic rate = 0.08 / 2 = 4.0 cleanly
    no_start_terms = FixedIncomeTerms(
        cashflow_currency="USD", reference_nominal=100.0,
        settlement_date=date(2025, 1, 1), maturity_date=date(2025, 7, 1),
        coupon_frequency=2,
        cashflows=(CashFlowEvent(payment_date=date(2025, 7, 1), coupon_rate=0.08, amortization_amount=100.0),),
    )
    m_ns = evaluate_fixed_income_instrument("NS", no_start_terms, quoted_price=100.0)
    assert m_ns.total_future_coupons == pytest.approx(4.0, abs=1e-9)


# =============================================================================
# AUD-06 & AUD-15(session): IOL contemporaneity, nominal units, issuer check,
# header collision, and session closing
# =============================================================================

def test_aud06_and_aud15_iol_mep_timestamps_units_issuer_headers_and_session():
    """AUD-06 + AUD-15(session): market_timestamp skew, 100x unit guard, issuer check, header collision, session close."""
    obs = "2026-11-26T20:00:00+00:00"

    # 1. AL30 (2026-11-26 16:55) vs AL30D (2026-11-10 11:00) -> rejected!
    snap_stale = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at=obs, column_headers=(),
        rows_by_ticker={
            "AL30": IOLPanelRow(ticker="AL30", name="Bono Rep Arg 2030", last_price=120000.0, market_timestamp="2026-11-26 16:55"),
            "AL30D": IOLPanelRow(ticker="AL30D", name="Bono Rep Arg 2030", last_price=100.0, market_timestamp="2026-11-10 11:00"),
        },
    )
    with pytest.raises(IOLQuoteError, match="diverge by"):
        compute_implicit_mep_al30(snap_stale)

    # Contemporaneous Argentine timestamps -> accepted with simultaneity_verified=True
    snap_ok = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at=obs, column_headers=(),
        rows_by_ticker={
            "AL30": IOLPanelRow(ticker="AL30", name="Bono Rep Arg 2030", last_price=120000.0, market_timestamp="26/11/2026 16:55:00"),
            "AL30D": IOLPanelRow(ticker="AL30D", name="Bono Rep Arg 2030", last_price=100.0, market_timestamp="26/11/2026 16:50:00"),
            "GD35C": IOLPanelRow(ticker="GD35C", name="Bono Global 2035", last_price=55.0, market_timestamp="26/11/2026 16:50:00"),
        },
    )
    mep_ok = compute_implicit_mep_al30(snap_ok)
    assert mep_ok["mep_rate_ars_per_usd"] == pytest.approx(1200.0, abs=1e-12)
    assert mep_ok["simultaneity_verified"] is True

    # 2. 100x nominal unit mismatch without verified metadata -> NEVER publish 120000 or 12!
    for p_ars, p_usd in ((120000.0, 1.0), (1200.0, 100.0)):
        snap_bad_u = IOLPanelSnapshot(
            panel_source="iol_bonos", url="mock", observed_at=obs, column_headers=(),
            rows_by_ticker={
                "AL30": IOLPanelRow(ticker="AL30", name="Bono 2030", last_price=p_ars, market_timestamp="26/11/2026 16:55"),
                "AL30D": IOLPanelRow(ticker="AL30D", name="Bono 2030", last_price=p_usd, market_timestamp="26/11/2026 16:55"),
            },
        )
        with pytest.raises(IOLQuoteError):
            compute_implicit_mep_al30(snap_bad_u)

    # With VERIFIED explicit quote_unit metadata -> safely normalized to MEP=1200.0
    snap_norm = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at=obs, column_headers=(),
        rows_by_ticker={
            "AL30": IOLPanelRow(ticker="AL30", name="Bono 2030", last_price=120000.0, market_timestamp="26/11/2026 16:55", quote_unit="per_100_original_nominal"),
            "AL30D": IOLPanelRow(ticker="AL30D", name="Bono 2030", last_price=1.0, market_timestamp="26/11/2026 16:55", quote_unit="per_1_original_nominal"),
        },
    )
    assert compute_implicit_mep_al30(snap_norm)["mep_rate_ars_per_usd"] == pytest.approx(1200.0, abs=1e-12)

    # Unknown unit -> rejected
    snap_unk = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at=obs, column_headers=(),
        rows_by_ticker={
            "AL30": IOLPanelRow(ticker="AL30", name="Bono 2030", last_price=120000.0, market_timestamp="26/11/2026 16:55", quote_unit="unknown"),
            "AL30D": IOLPanelRow(ticker="AL30D", name="Bono 2030", last_price=100.0, market_timestamp="26/11/2026 16:55", quote_unit="unknown"),
        },
    )
    with pytest.raises(IOLQuoteError, match="unknown"):
        compute_implicit_mep_al30(snap_unk)

    # Valid species C remains unaffected even when AL30/AL30D MEP is broken!
    snap_broken_mep = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at=obs, column_headers=(),
        rows_by_ticker={
            "AL30": IOLPanelRow(ticker="AL30", name="Bono 2030", last_price=None),
            "GD35C": IOLPanelRow(ticker="GD35C", name="Bono Global 2035", last_price=55.0),
        },
    )
    q_c = resolve_iol_bond_quote_usd(instrument_id="GD35", ticker="GD35", panel_snapshot=snap_broken_mep)
    assert q_c.conversion_method == "C" and q_c.price_usd == pytest.approx(55.0)

    # 3. Cross-issuer collision on D/C suffix: BOND1D (Emisor Beta) must NOT fall back to BOND1 (Emisor Alfa)!
    snap_issuer = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at=obs, column_headers=(),
        rows_by_ticker={
            "AL30": IOLPanelRow(ticker="AL30", name="Bono 2030", last_price=120000.0, market_timestamp="26/11/2026 16:55"),
            "AL30D": IOLPanelRow(ticker="AL30D", name="Bono 2030", last_price=100.0, market_timestamp="26/11/2026 16:55"),
            "BOND1D": IOLPanelRow(ticker="BOND1D", name="Emisor Beta Corp", last_price=None),
            "BOND1": IOLPanelRow(ticker="BOND1", name="Emisor Alfa SA", last_price=80000.0),
        },
    )
    with pytest.raises(IOLQuoteError):
        resolve_iol_bond_quote_usd(instrument_id="BOND1D", ticker="BOND1D", panel_snapshot=snap_issuer)

    # 4. Header "Último Horario" BEFORE "Último" must NOT hijack last_price!
    html_hdr = """
    <table>
      <tr><th>Símbolo</th><th>Último Horario</th><th>Último</th><th>Variación</th><th>T. I. R.</th><th>Duración</th></tr>
      <tr><td>AL30D Bono 2030</td><td>26/11/2026 16:55</td><td>49,16</td><td>0,50%</td><td>14,0%</td><td>2,4</td></tr>
    </table>
    """
    parsed = parse_iol_fixed_income_table(html_hdr, panel_source="iol_bonos", observed_at=obs)
    assert parsed.rows_by_ticker["AL30D"].last_price == pytest.approx(49.16)
    assert parsed.rows_by_ticker["AL30D"].market_timestamp == "26/11/2026 16:55"

    # 5. AUD-15: fetch_iol_panel closes self-created Session on success & error, NEVER closes caller-provided session
    caller_sess = MagicMock()
    caller_sess.get.return_value = MagicMock(status_code=200, content=html_hdr.encode("utf-8"))
    fetch_iol_panel("iol_bonos", session=caller_sess)
    caller_sess.close.assert_not_called()

    created_sessions: list[MagicMock] = []
    real_sess_cls = iol.requests.Session

    def make_sess_ok():
        s = MagicMock()
        s.get.return_value = MagicMock(status_code=200, content=html_hdr.encode("utf-8"))
        created_sessions.append(s)
        return s

    iol.requests.Session = make_sess_ok  # type: ignore[assignment]
    try:
        fetch_iol_panel("iol_bonos")
        assert created_sessions[-1].close.call_count == 1
    finally:
        iol.requests.Session = real_sess_cls  # type: ignore[assignment]

    def make_sess_err():
        s = MagicMock()
        s.get.side_effect = iol.requests.Timeout("timed out")
        created_sessions.append(s)
        return s

    iol.requests.Session = make_sess_err  # type: ignore[assignment]
    try:
        with pytest.raises(IOLQuoteError):
            fetch_iol_panel("iol_bonos")
        assert created_sessions[-1].close.call_count == 1
    finally:
        iol.requests.Session = real_sess_cls  # type: ignore[assignment]


# =============================================================================
# AUD-08: Black-Litterman diagonal equilibration & duplicate view ID rejection
# =============================================================================

def test_aud08_black_litterman_extreme_mixed_confidence_and_case_duplicate_rejection():
    """AUD-08: EQ1=(0.14, 1.0) + GLD=(0.06, 1e-13) preserves EQ1 ~ 0.14 (<1e-6 error) and rejects eq1/EQ1."""
    tickers = ["EQ1", "EQ2", "GLD", "BND"]
    cov = pd.DataFrame(
        [
            [0.040, 0.020, 0.002, 0.001],
            [0.020, 0.050, 0.003, 0.001],
            [0.002, 0.003, 0.025, 0.002],
            [0.001, 0.001, 0.002, 0.006],
        ],
        index=tickers,
        columns=tickers,
    )
    w_ref = pd.Series([0.35, 0.35, 0.15, 0.15], index=tickers)

    bl = compute_black_litterman(
        cov,
        w_ref,
        rf_rate=0.04,
        risk_aversion=2.5,
        tau=0.05,
        absolute_views_total={"EQ1": (0.14, 1.0), "GLD": (0.06, 1e-13)},
    )
    assert abs(bl.posterior_total_returns["EQ1"] - 0.14) < 1e-10
    assert bl.diagnostics["jitter_applied"] == 0.0
    assert np.all(np.isfinite(bl.posterior_total_returns.values))

    # Duplicate normalized keys (case / whitespace) -> ConfigurationError
    with pytest.raises(ConfigurationError, match="Duplicate Black-Litterman view"):
        compute_black_litterman(
            cov, w_ref, rf_rate=0.04,
            absolute_views_total={"eq1": (0.12, 0.5), "EQ1": (0.14, 0.8)},
        )
    with pytest.raises(ConfigurationError, match="Duplicate Black-Litterman view"):
        compute_black_litterman(
            cov, w_ref, rf_rate=0.04,
            absolute_views_total={" EQ1 ": (0.12, 0.5), "EQ1": (0.14, 0.8)},
        )


# =============================================================================
# AUD-09: Fixed-income YTM from historical price (override / yfinance), cutoff,
# adjusted-series guard, and standalone metrics on insufficient history
# =============================================================================

def test_aud09_ytm_from_historical_price_override_yfinance_cutoff_and_adjusted_guard(monkeypatch):
    """AUD-09: T-bill without manual_price_usd computes exact YTM from override/yfinance, respects cutoff & adjusted guard."""
    dates = pd.date_range("2024-01-05", periods=70, freq="W-FRI")
    tbill_terms = FixedIncomeTerms(
        cashflow_currency="USD",
        reference_nominal=100.0,
        settlement_date=date(2025, 5, 2),
        maturity_date=date(2026, 5, 2),  # Exact 365 days under ACT/365 -> t = 1.0
        price_convention="per_100_original_nominal",
        price_type="dirty",
        coupon_structure="zero_coupon",
        day_count_convention="ACT/365",
        cashflows=(CashFlowEvent(payment_date=date(2026, 5, 2), coupon_amount=0.0, amortization_amount=100.0),),
    )

    # 1. prices_usd_override ending at 95.0 -> YTM = 100/95 - 1 = 5.263158%, then 80.0 -> 25%
    for last_px, exp_ytm in ((95.0, 100.0 / 95.0 - 1.0), (80.0, 0.25)):
        p_tb = np.linspace(90.0, last_px, len(dates))
        prices = pd.DataFrame({"TBILL": p_tb}, index=dates)
        cfg = PortfolioAnalysisConfig(
            fixed_income=(
                InstrumentConfig(
                    instrument_id="TBILL", ticker="TBILL", section="fixed_income", asset_subtype="tbill",
                    quote_source="yfinance", manual_price_usd=None, fixed_income_terms=tbill_terms,
                ),
            ),
            black_litterman=BlackLittermanConfig(rf_rate=0.04),
            risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
            optimization=OptimizationConfig(strategies_to_run=("min_variance",)),
        )
        res = analyze_portfolio(cfg, prices_usd_override=prices)
        fi = res.fixed_income_metrics["TBILL"]
        assert fi.status == "ok"
        assert fi.ytm_annual == pytest.approx(exp_ytm, abs=1e-10)
        assert fi.quote_source_used == "prices_usd_override"

    # 2. Cutoff prevents using a future price after as_of_date
    future_dates = dates.append(pd.DatetimeIndex([pd.Timestamp("2025-05-09")]))
    prices_fut = pd.DataFrame({"TBILL": list(np.linspace(90.0, 95.0, len(dates))) + [50.0]}, index=future_dates)
    cfg_cut = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="TBILL", ticker="TBILL", section="fixed_income", asset_subtype="tbill",
                quote_source="yfinance", fixed_income_terms=tbill_terms,
            ),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",)),
        as_of_date=date(2025, 5, 2),
    )
    res_cut = analyze_portfolio(cfg_cut, prices_usd_override=prices_fut)
    assert res_cut.fixed_income_metrics["TBILL"].ytm_annual == pytest.approx(100.0 / 95.0 - 1.0, abs=1e-10)

    # 3. Series marked adjusted_total_return is NOT treated as tradable clean/dirty price for YTM
    cfg_adj = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="TBILL", ticker="TBILL", section="fixed_income", asset_subtype="tbill",
                quote_source="yfinance", historical_price_series_type="adjusted_total_return",
                fixed_income_terms=tbill_terms,
            ),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",)),
    )
    res_adj = analyze_portfolio(cfg_adj, prices_usd_override=pd.DataFrame({"TBILL": np.linspace(90.0, 95.0, len(dates))}, index=dates))
    assert res_adj.fixed_income_metrics["TBILL"].status == "not_evaluable_under_available_method"
    assert res_adj.fixed_income_metrics["TBILL"].ytm_annual is None

    # 4. Simulated yfinance download resolves YTM cleanly
    def fake_ticker(sym):
        t = MagicMock()
        t.info = {"currency": "USD"}
        t.history.return_value = pd.DataFrame({"Close": np.linspace(90.0, 80.0, len(dates))}, index=dates)
        return t

    monkeypatch.setattr(market_data.yf, "Ticker", fake_ticker)
    cfg_yf = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="TBILL", ticker="TBILL", section="fixed_income", asset_subtype="tbill",
                quote_source="yfinance", fixed_income_terms=tbill_terms,
            ),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",)),
    )
    res_yf = analyze_portfolio(cfg_yf)
    assert res_yf.fixed_income_metrics["TBILL"].status == "ok"
    assert res_yf.fixed_income_metrics["TBILL"].ytm_annual == pytest.approx(0.25, abs=1e-10)
    assert res_yf.fixed_income_metrics["TBILL"].quote_source_used == "yfinance"


# =============================================================================
# AUD-12: ContextVar concurrent isolation for suppress_display (threads + asyncio)
# =============================================================================

def test_aud12_display_contextvar_isolation_threads_asyncio_and_nesting():
    """AUD-12: suppress_display isolates threads, asyncio tasks, nested blocks, and exceptions."""
    assert is_display_enabled() is True

    # 1. Nested + exception restoration
    try:
        with suppress_display():
            assert is_display_enabled() is False
            with suppress_display():
                assert is_display_enabled() is False
                raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert is_display_enabled() is True

    # 2. Two concurrent threads synchronized with Barrier
    barrier = threading.Barrier(2)
    observed: dict[str, bool] = {}

    def worker_suppress():
        with suppress_display():
            barrier.wait(timeout=5)
            observed["suppressed_thread"] = is_display_enabled()
            barrier.wait(timeout=5)

    def worker_normal():
        barrier.wait(timeout=5)
        observed["normal_thread"] = is_display_enabled()
        barrier.wait(timeout=5)

    t1 = threading.Thread(target=worker_suppress)
    t2 = threading.Thread(target=worker_normal)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert observed == {"suppressed_thread": False, "normal_thread": True}

    # 3. Two concurrent asyncio tasks synchronized with Events
    async def run_async_check():
        ev_in = asyncio.Event()
        ev_done = asyncio.Event()
        async_obs: dict[str, bool] = {}

        async def task_suppress():
            with suppress_display():
                async_obs["suppressed_task"] = is_display_enabled()
                ev_in.set()
                await ev_done.wait()

        async def task_normal():
            await ev_in.wait()
            async_obs["normal_task"] = is_display_enabled()
            ev_done.set()

        await asyncio.gather(task_suppress(), task_normal())
        return async_obs

    assert asyncio.run(run_async_check()) == {"suppressed_task": False, "normal_task": True}


# =============================================================================
# AUD-13: Deterministic HRP tie-breaking across all permutations
# =============================================================================

def test_aud13_hrp_deterministic_tie_breaking_all_permutations():
    """AUD-13: Equicorrelated 3 assets (rho=0.5) with distinct vols yield identical weights across all 6 permutations."""
    base_ids = ["ALPHA", "BETA", "GAMMA"]
    vols = {"ALPHA": 0.10, "BETA": 0.20, "GAMMA": 0.30}
    rho = 0.5

    def make_cov(order: list[str]) -> pd.DataFrame:
        n = len(order)
        mat = np.zeros((n, n), dtype=float)
        for i, a in enumerate(order):
            for j, b in enumerate(order):
                mat[i, j] = vols[a] * vols[b] * (1.0 if i == j else rho)
        return pd.DataFrame(mat, index=order, columns=order)

    ref_w, _ = compute_pure_hrp_weights(make_cov(base_ids))
    ref_map = dict(zip(base_ids, ref_w))

    for perm in itertools.permutations(base_ids):
        order = list(perm)
        w, _ = compute_pure_hrp_weights(make_cov(order))
        perm_map = dict(zip(order, w))
        for k in base_ids:
            assert perm_map[k] == pytest.approx(ref_map[k], abs=1e-12)


# =============================================================================
# AUD-14: Strictly causal, gap-aware EWMA weekly covariance
# =============================================================================

def test_aud14_ewma_strict_causality_gap_decay_and_annualization():
    """AUD-14: EWMA prefix invariance (strict causality), 3-week gap lambda^3 decay, and x52 annualization."""
    rng = np.random.default_rng(1414)
    weeks = pd.date_range("2024-01-05", periods=40, freq="W-FRI")
    df_full = pd.DataFrame(rng.normal(0.002, 0.02, size=(40, 2)), index=weeks, columns=["A", "B"])

    # 1. Strict causality: adding future rows after t=20 must NOT alter estimate at t=20
    cov_20 = ewma_covariance_weekly(df_full.iloc[:20], half_life_weeks=10.0, mean_mode="zero")
    cov_20_rec = ewma_covariance_weekly(df_full.iloc[:20], half_life_weeks=10.0, mean_mode="causal_recursive")

    # Independent oracle for zero-mean RiskMetrics EWMA
    lam = 0.5 ** (1.0 / 10.0)
    arr20 = df_full.iloc[:20].to_numpy()
    s_ref = arr20[0:1].T @ arr20[0:1]
    w_ref = 1.0
    for i in range(1, 20):
        s_ref = lam * s_ref + (arr20[i : i + 1].T @ arr20[i : i + 1])
        w_ref = lam * w_ref + 1.0
    oracle_20 = (s_ref / w_ref) * 52.0
    np.testing.assert_allclose(cov_20.values, oracle_20, rtol=1e-12, atol=1e-14)
    assert np.all(np.linalg.eigvalsh(cov_20_rec.values) >= -1e-12)

    # 2. 3-week gap test: drop weeks 10 and 11 so gap between idx 9 and idx 12 is 3 weeks -> decay is lambda^3
    df_gap = df_full.iloc[:15].drop(index=[weeks[10], weeks[11]])
    cov_gap = ewma_covariance_weekly(df_gap, half_life_weeks=10.0, mean_mode="zero", gap_policy="elapsed_weeks_decay")

    arr_g = df_gap.to_numpy()
    idx_g = df_gap.index
    s_g = arr_g[0:1].T @ arr_g[0:1]
    w_g = 1.0
    for i in range(1, len(arr_g)):
        dt = round((idx_g[i] - idx_g[i - 1]).days / 7.0)
        s_g = (lam ** dt) * s_g + (arr_g[i : i + 1].T @ arr_g[i : i + 1])
        w_g = (lam ** dt) * w_g + 1.0
    np.testing.assert_allclose(cov_gap.values, (s_g / w_g) * 52.0, rtol=1e-12, atol=1e-14)

    # Under gap_policy="error_on_gap", the 3-week gap is explicitly rejected
    with pytest.raises(DataQualityError, match="missing week gap"):
        ewma_covariance_weekly(df_gap, half_life_weeks=10.0, gap_policy="error_on_gap")


# =============================================================================
# AUD-15: HTML escaping in notebook_ui & dynamic FRED_API_KEY rotation
# =============================================================================

def test_aud15_html_escape_and_dynamic_fred_api_key(monkeypatch):
    """AUD-15: XSS payloads escaped in mostrar_rebalanceo; FRED_API_KEY reflects live env changes dynamically."""
    # 1. Dynamic FRED_API_KEY rotation without import-time caching
    monkeypatch.setenv("FRED_API_KEY", "FICTITIOUS_KEY_A")
    assert get_fred_api_key() == "FICTITIOUS_KEY_A"
    assert config.FRED_API_KEY == "FICTITIOUS_KEY_A"

    monkeypatch.setenv("FRED_API_KEY", "FICTITIOUS_KEY_B")
    assert get_fred_api_key() == "FICTITIOUS_KEY_B"
    assert config.FRED_API_KEY == "FICTITIOUS_KEY_B"

    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert get_fred_api_key() == ""
    assert config.FRED_API_KEY == ""

    # 2. HTML escaping in mostrar_rebalanceo
    captured_html: list[str] = []
    monkeypatch.setattr(notebook_ui, "display", lambda obj: captured_html.append(str(obj)))
    evil_strat = '<script>alert("strat")</script>'
    reb = RebalancingResult(
        strategy_name=evil_strat,
        status='<img src=x onerror=alert(1)>',
        feasible=True,
        orders=[
            ProposedOrder(
                broker_id='<b onmouseover="evil()">BRK</b>',
                instrument_id='<script>alert("id")</script>',
                ticker="X",
                side="BUY",
                quantity=1.0,
                price_usd=100.0,
                gross_amount_usd=100.0,
                commission_usd=0.0,
                spread_cost_usd=0.0,
                net_amount_usd=-100.0,
                pre_weight=0.0,
                target_weight=1.0,
                achievable_weight=1.0,
            )
        ],
    )
    fake_res = SimpleNamespace(selected_strategy=evil_strat, rebalancing={evil_strat: reb})
    mostrar_rebalanceo(fake_res)  # type: ignore[arg-type]
    rendered = "\n".join(captured_html)
    assert "<script>" not in rendered
    assert "<img " not in rendered
    assert "&lt;script&gt;alert(&quot;id&quot;)&lt;/script&gt;" in rendered


# =============================================================================
# Full Multi-Asset Synthetic End-to-End Verification (Prompt Section 4)
# =============================================================================

def test_e2e_synthetic_multiasset_pipeline_and_degradation_variants(monkeypatch):
    """E2E: US equity + .L GBp + individual bond per 100 VN + crypto + DCF + BL + VIX + 2 segregated brokers."""
    rng = np.random.default_rng(20261126)
    dates = pd.date_range("2023-01-06", periods=120, freq="W-FRI")

    # 2 regimes: first 90 weeks calm (VIX=14), last 30 weeks crisis (VIX=34)
    vix_vals = np.concatenate([np.full(90, 14.0), np.full(30, 34.0)])
    vix_series = pd.Series(vix_vals, index=dates, name="^VIX")

    # Correlated returns in USD + AZN.L in GBp with GBPUSD=X=1.25
    aapl_usd = 160.0 * np.exp(np.cumsum(rng.normal(0.0025, 0.025, len(dates))))
    azn_gbp = 8000.0 * np.exp(np.cumsum(rng.normal(0.0015, 0.020, len(dates))))
    azn_gbp[-1] = 7801.33  # Exact oracle final print -> 97.516625 USD
    gbpusd = pd.Series(np.full(len(dates), 1.25), index=dates)
    al30_px = np.linspace(42.0, 49.16, len(dates)) + rng.normal(0.0, 0.35, len(dates))
    al30_px[-1] = 49.16
    btc_usd = 30000.0 * np.exp(np.cumsum(rng.normal(0.003, 0.045, len(dates))))

    def fake_ticker(sym):
        t = MagicMock()
        if sym == "AAPL":
            t.info = {"currency": "USD"}
            t.history.return_value = pd.DataFrame({"Close": aapl_usd}, index=dates)
        elif sym == "AZN.L":
            t.info = {"currency": "GBp"}
            t.history.return_value = pd.DataFrame({"Close": azn_gbp}, index=dates)
        elif sym == "BTC-USD":
            t.info = {"currency": "USD"}
            t.history.return_value = pd.DataFrame({"Close": btc_usd}, index=dates)
        else:
            t.info = {"currency": "USD"}
            t.history.return_value = pd.DataFrame({"Close": al30_px}, index=dates)
        return t

    def fake_yf_dl(fx_sym, period="10y", auto_adjust=True, progress=False):
        if fx_sym == "GBPUSD=X":
            return pd.DataFrame({"Close": gbpusd.values}, index=dates)
        return pd.DataFrame()

    monkeypatch.setattr(market_data.yf, "Ticker", fake_ticker)
    monkeypatch.setattr(market_data.yf, "download", fake_yf_dl)

    al30_terms = FixedIncomeTerms(
        cashflow_currency="USD",
        reference_nominal=100.0,
        settlement_date=date(2025, 4, 18),
        maturity_date=date(2026, 4, 18),  # Exact 1 year ACT/365
        price_convention="per_100_original_nominal",
        price_type="dirty",
        coupon_structure="fixed",
        day_count_convention="ACT/365",
        coupon_frequency=1,
        cashflows=(CashFlowEvent(payment_date=date(2026, 4, 18), coupon_amount=8.0, amortization_amount=100.0),),
    )

    cfg = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(
                instrument_id="AAPL", ticker="AAPL", asset_subtype="equity",
                default_buy_broker_id="IBKR",
                dcf_view=DCFViewConfig(valuation_basis="terminal_target", total_upside=0.50, confidence=0.60),
            ),
            InstrumentConfig(
                instrument_id="AZN.L", ticker="AZN.L", asset_subtype="equity",
                default_buy_broker_id="IBKR",
            ),
            InstrumentConfig(
                instrument_id="BTC", ticker="BTC-USD", asset_subtype="crypto",
                default_buy_broker_id="IBKR",
            ),
        ),
        fixed_income=(
            InstrumentConfig(
                instrument_id="AL30", ticker="AL30", section="fixed_income", asset_subtype="sovereign_bond",
                quote_source="yfinance", fractional_allowed=False, lot_size=1.0, min_nominal=100.0,
                default_buy_broker_id="IOL", fixed_income_terms=al30_terms,
            ),
        ),
        brokers=(
            BrokerConfig(broker_id="IBKR", fee_structure="fixed", buy_fixed_fee=1.0, sell_fixed_fee=1.0),
            BrokerConfig(broker_id="IOL", fee_structure="percentage", buy_percentage_fee=0.005, sell_percentage_fee=0.005),
        ),
        positions=(
            BrokerPosition(instrument_id="AAPL", broker_id="IBKR", quantity=5.0),
            BrokerPosition(instrument_id="AZN.L", broker_id="IBKR", quantity=10.0),
            BrokerPosition(instrument_id="AL30", broker_id="IOL", quantity=1000.0),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04, default_dcf_horizon_years=4.0),
        risk=RiskModelConfig(
            covariance_estimator="vix_regime",
            min_weekly_observations=52,
            vix_regime=VixRegimeConfig(min_calm_weeks=26, min_crisis_weeks=12),
        ),
        constraints=ConstraintsConfig(class_bounds={"crypto": (0.0, 0.15)}),
        rebalancing=RebalancingConfig(
            available_cash_by_broker={"IBKR": 500.0, "IOL": 200.0},
            cash_flow_by_broker={"IBKR": 0.0, "IOL": 0.0},
            min_weight_diff=0.01,
            min_notional_usd=25.0,
        ),
        as_of_date=date(2025, 4, 18),
    )

    res = analyze_portfolio(
        cfg,
        vix_history_override=vix_series,
        current_vix_override=(34.0, dates[-1], False),
    )
    res_master = ejecutar_analisis_completo(
        cfg,
        render_ui=False,
        vix_history_override=vix_series,
        current_vix_override=(34.0, dates[-1], False),
    )

    # 1. Exact USD prices, nominal execution price, YTM, and DCF inheritance
    assert res.quotes_and_data_quality["latest_prices_usd"]["AZN.L"] == pytest.approx(97.516625, rel=1e-12)
    assert res.quotes_and_data_quality["execution_prices_usd_per_unit"]["AL30"] == pytest.approx(0.4916, rel=1e-12)
    assert res.fixed_income_metrics["AL30"].status == "ok"
    assert res.fixed_income_metrics["AL30"].ytm_annual == pytest.approx((108.0 / 49.16) - 1.0, abs=1e-8)
    assert res.dcf_views["AAPL"].horizon_years == pytest.approx(4.0)
    assert res.dcf_views["AAPL"].annualized_expected_return == pytest.approx((1.5 ** 0.25) - 1.0, abs=1e-10)

    # 2. VIX regime stress weight (VIX=34 > 30 -> p=0.60) and all 5 strategies converged & feasible
    assert res.risk_model.stress_weight_p == pytest.approx(0.60)
    for strat_name in ("max_sharpe", "min_variance", "min_cvar", "erc", "hrp"):
        st = res.strategies[strat_name]
        assert st.converged is True and st.feasible is True
        assert st.weights_by_asset["BTC"] <= 0.15 + 1e-6
        assert sum(st.weights_by_asset.values()) == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(st.weights, res_master.strategies[strat_name].weights, atol=1e-12)

    # 3. Broker ledger tie-out (no negative broker cash, AL30 valued at 1000 * 0.4916 = $491.60)
    reb = res.rebalancing["max_sharpe"]
    assert reb.broker_ledgers["IOL"]["initial_holdings_value_usd"] == pytest.approx(491.60, abs=1e-6)
    for bid, led in reb.broker_ledgers.items():
        assert led["ending_cash_usd"] >= -1e-9
        assert led["ending_cash_usd"] == pytest.approx(
            led["initial_cash_usd"] + led["cash_flow_usd"] + led["sell_net_proceeds_usd"] - led["buy_total_outflow_usd"],
            abs=1e-9,
        )

    # 4. ERC and HRP invariance to mu changes with fixed Sigma
    resolved = validate_constraints_feasibility(
        res.tickers, cfg.constraints,
        risk_class_by_ticker={i.instrument_id: str(i.risk_class) for i in cfg.instruments},
    )
    mu_alt = res.expected_returns * 3.0 + 0.15
    erc_1 = optimize_erc(res.cov_matrix, resolved, mu=res.expected_returns, rf=0.04)
    erc_2 = optimize_erc(res.cov_matrix, resolved, mu=mu_alt, rf=0.04)
    hrp_1 = optimize_hrp(res.cov_matrix, resolved, mu=res.expected_returns, rf=0.04)
    hrp_2 = optimize_hrp(res.cov_matrix, resolved, mu=mu_alt, rf=0.04)
    np.testing.assert_allclose(erc_1.weights, erc_2.weights, atol=1e-12)
    np.testing.assert_allclose(hrp_1.weights, hrp_2.weights, atol=1e-12)


def test_e2e_degradation_variants_fx_partial_week_iol_constraints_and_unfunded(monkeypatch):
    """E2E variants: missing FX, partial week, invalid IOL price, infeasible constraints, and unfunded broker."""
    rng = np.random.default_rng(404)
    dates = pd.date_range("2023-01-06", periods=90, freq="W-FRI")
    # Append Wednesday 2024-09-25 observation (incomplete week ending 2024-09-27)
    wed = pd.Timestamp("2024-09-25")
    dates_plus_wed = dates.append(pd.DatetimeIndex([wed]))
    aapl_px = 150.0 * np.exp(np.cumsum(rng.normal(0.002, 0.02, len(dates_plus_wed))))
    azn_px = 7500.0 * np.exp(np.cumsum(rng.normal(0.001, 0.015, len(dates_plus_wed))))

    def fake_ticker(sym):
        t = MagicMock()
        if sym == "AZN.L":
            t.info = {"currency": "GBp"}
            t.history.return_value = pd.DataFrame({"Close": azn_px}, index=dates_plus_wed)
        else:
            t.info = {"currency": "USD"}
            t.history.return_value = pd.DataFrame({"Close": aapl_px}, index=dates_plus_wed)
        return t

    monkeypatch.setattr(market_data.yf, "Ticker", fake_ticker)

    # 1. Missing FX series -> raises FXConversionError directly
    monkeypatch.setattr(market_data.yf, "download", lambda *a, **kw: pd.DataFrame())
    cfg_fx = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(instrument_id="AAPL", ticker="AAPL"),
            InstrumentConfig(instrument_id="AZN.L", ticker="AZN.L"),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
    )
    with pytest.raises(FXConversionError) as exc_fx:
        analyze_portfolio(cfg_fx)
    assert exc_fx.value.asset_id == "AZN.L"

    # Restore valid GBPUSD=X for remaining variants
    monkeypatch.setattr(
        market_data.yf,
        "download",
        lambda sym, **kw: pd.DataFrame({"Close": np.full(len(dates_plus_wed), 1.25)}, index=dates_plus_wed),
    )

    # 2. Partial week (Wednesday 2024-09-25) excluded cleanly with diagnostic
    cfg_partial = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(instrument_id="AAPL", ticker="AAPL"),
            InstrumentConfig(instrument_id="AZN.L", ticker="AZN.L"),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        as_of_date=date(2024, 9, 25),
    )
    res_part = analyze_portfolio(cfg_partial)
    assert res_part.risk_model.last_closed_friday == "2024-09-20"
    assert "2024-09-27" in res_part.risk_model.excluded_incomplete_week_bins
    assert res_part.risk_model.weekly_returns.index.max() == pd.Timestamp("2024-09-20")

    # 3. Invalid IOL price (missing last_price on all species) -> raises IOLQuoteError
    bad_snap = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at="2026-11-26T15:00:00+00:00", column_headers=(),
        rows_by_ticker={"AL30D": IOLPanelRow(ticker="AL30D", name="Bono 2030", last_price=None)},
    )
    cfg_iol_bad = PortfolioAnalysisConfig(
        variable_income=(InstrumentConfig(instrument_id="AAPL", ticker="AAPL"),),
        fixed_income=(
            InstrumentConfig(
                instrument_id="AL30", ticker="AL30", section="fixed_income",
                asset_subtype="sovereign_bond", quote_source="iol_bonos",
            ),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
    )
    with pytest.raises(IOLQuoteError):
        analyze_portfolio(cfg_iol_bad, iol_snapshots_override={"iol_bonos": bad_snap})

    # 4. Infeasible constraints -> raises InfeasibleConstraintsError
    cfg_inf = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(instrument_id="AAPL", ticker="AAPL", asset_subtype="equity"),
            InstrumentConfig(instrument_id="BTC", ticker="BTC-USD", asset_subtype="crypto"),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        constraints=ConstraintsConfig(class_bounds={"equity": (0.70, 1.0), "crypto": (0.50, 1.0)}),
    )
    with pytest.raises(InfeasibleConstraintsError):
        analyze_portfolio(cfg_inf)

    # 5. Unfunded broker in E2E rebalancing -> explicit partially_funded degradation, zero unfunded buys
    cfg_unfunded = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(instrument_id="AAPL", ticker="AAPL", default_buy_broker_id="B_SELL"),
            InstrumentConfig(instrument_id="AZN.L", ticker="AZN.L", default_buy_broker_id="B_BUY"),
        ),
        brokers=(
            BrokerConfig(broker_id="B_SELL", fee_structure="percentage", sell_percentage_fee=0.005),
            BrokerConfig(broker_id="B_BUY", fee_structure="percentage", buy_percentage_fee=0.005),
        ),
        positions=(BrokerPosition(instrument_id="AAPL", broker_id="B_SELL", quantity=10.0),),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
        rebalancing=RebalancingConfig(
            available_cash_by_broker={"B_SELL": 0.0, "B_BUY": 0.0},
            min_weight_diff=0.01,
            min_notional_usd=25.0,
        ),
        as_of_date=date(2024, 9, 20),
    )
    res_unf = analyze_portfolio(cfg_unfunded)
    reb_unf = res_unf.rebalancing["min_variance"]
    assert reb_unf.status == "partially_funded"
    assert reb_unf.feasible is False
    assert all(o.broker_id != "B_BUY" for o in reb_unf.orders)
    assert reb_unf.ending_cash_by_broker["B_BUY"] == pytest.approx(0.0, abs=1e-9)
    assert reb_unf.ending_cash_by_broker["B_SELL"] > 0.0

