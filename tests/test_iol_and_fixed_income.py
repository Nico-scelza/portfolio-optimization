"""Deterministic tests for IOL Argentine Bonds/ONs parser and manual fixed-income valuation."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization import (
    CashFlowEvent,
    FixedIncomeTerms,
    FixedIncomeValidationError,
    InstrumentConfig,
    InsufficientHistoryError,
    IOLConfig,
    IOLQuoteError,
    PortfolioAnalysisConfig,
    RiskModelConfig,
    analyze_portfolio,
    build_fixed_income_total_return_index,
    compute_implicit_mep_al30,
    evaluate_fixed_income_instrument,
    fetch_iol_panel,
    fetch_iol_quotes_for_instruments,
    parse_argentine_number,
    parse_argentine_percentage,
    parse_iol_fixed_income_table,
    resolve_iol_bond_quote_usd,
)

BONOS_HTML_FIXTURE = """
<html><body>
<table>
  <tr><th>S&iacute;mbolo</th><th>&Uacute;ltimo</th><th>Variaci&oacute;n</th><th>Apertura</th><th>Monto</th></tr>
  <tr><td>GGAL</td><td>5.200,00</td><td>2,10%</td><td>5.100,00</td><td>1.000.000</td></tr>
</table>
<table id="panel-bonos">
  <thead>
    <tr>
      <th>S&iacute;mbolo</th>
      <th>&Uacute;ltimo<br/>Operado</th>
      <th>Variaci&oacute;n<br/>Diaria</th>
      <th>Cantidad Compra</th>
      <th>Precio Compra</th>
      <th>Precio Venta</th>
      <th>Cantidad Venta</th>
      <th>M&aacute;ximo</th>
      <th>M&iacute;nimo</th>
      <th>&Uacute;ltimo Cierre</th>
      <th>Monto Operado</th>
      <th>T. I. R.</th>
      <th>Duraci&oacute;n (MD)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="#"><b>AL30</b></a><span>&nbsp;Bono Rep. Arg. USD 2030</span></td>
      <td>75.000,00</td><td>1,50%</td><td>10.000</td><td>74.900,00</td><td>75.100,00</td><td>5.000</td>
      <td>75.500,00</td><td>74.000,00</td><td>73.890,00</td><td>1.250.000.000,50</td><td>14,20%</td><td>2,45</td>
    </tr>
    <tr>
      <td><b>AL30D</b> Bono Rep. Arg. USD 2030</td>
      <td>62,50</td><td>0,80%</td><td>2.000</td><td>62,40</td><td>62,60</td><td>1.500</td>
      <td>63,00</td><td>62,00</td><td>62,00</td><td>8.500.000,00</td><td>14,10%</td><td>2,45</td>
    </tr>
    <tr>
      <td><b>AL30C</b> Bono Rep. Arg. USD 2030</td>
      <td>61,90</td><td>0,50%</td><td>500</td><td>61,50</td><td>62,20</td><td>500</td>
      <td>62,20</td><td>61,50</td><td>61,60</td><td>120.000,00</td><td>14,40%</td><td>2,45</td>
    </tr>
    <tr>
      <td><b>GD35D</b> Bono Global USD 2035</td>
      <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
      <td>-</td><td>-</td><td>54,00</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td><b>GD35C</b> Bono Global USD 2035</td>
      <td>55,20</td><td>-0,40%</td><td>100</td><td>54,00</td><td>57,50</td><td>100</td>
      <td>55,50</td><td>54,80</td><td>55,40</td><td>400,00</td><td>12,50%</td><td>5,10</td>
    </tr>
    <tr>
      <td><b>AE38</b> Bono Rep. Arg. USD 2038</td>
      <td>60.000,00</td><td>1,00%</td><td>1.000</td><td>59.800,00</td><td>60.200,00</td><td>1.000</td>
      <td>60.500,00</td><td>59.000,00</td><td>59.400,00</td><td>50.000.000</td><td>13,00%</td><td>4,80</td>
    </tr>
  </tbody>
</table>
</body></html>
"""

ONS_HTML_FIXTURE = """
<html><body>
<table id="panel-ons">
  <thead>
    <tr>
      <th>S&iacute;mbolo</th><th>&Uacute;ltimo Operado</th><th>Variaci&oacute;n Diaria</th>
      <th>Cantidad Compra</th><th>Precio Compra</th><th>Precio Venta</th><th>Cantidad Venta</th>
      <th>M&aacute;ximo</th><th>M&iacute;nimo</th><th>&Uacute;ltimo Cierre</th><th>Monto Operado</th>
      <th>TIR</th><th>Duraci&oacute;n (MD)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MTCGO Mastellone Cl. G</td>
      <td>120.000,00</td><td>0,25%</td><td>1.000</td><td>119.500,00</td><td>120.500,00</td><td>1.000</td>
      <td>120.500,00</td><td>119.000,00</td><td>119.700,00</td><td>25.000.000</td><td>7,50%</td><td>1,80</td>
    </tr>
    <tr>
      <td>MTCGD Mastellone Cl. G</td>
      <td>100,70</td><td>0,10%</td><td>500</td><td>100,50</td><td>100,90</td><td>500</td>
      <td>101,00</td><td>100,20</td><td>100,60</td><td>350.000</td><td>7,40%</td><td>1,80</td>
    </tr>
    <tr>
      <td>YMCQO YPF Clase XXIV</td>
      <td>126.000,00</td><td>0,50%</td><td>1.000</td><td>125.000,00</td><td>127.000,00</td><td>1.000</td>
      <td>126.500,00</td><td>125.000,00</td><td>125.400,00</td><td>15.000.000</td><td>8,10%</td><td>2,20</td>
    </tr>
  </tbody>
</table>
</body></html>
"""


def test_argentine_number_parser_resolves_decimal_ambiguity_and_dashes():
    assert parse_argentine_number("1.234.567,89") == pytest.approx(1234567.89)
    assert parse_argentine_number("100,70") == pytest.approx(100.70)
    assert parse_argentine_number("154.000") == pytest.approx(154000.0)
    assert parse_argentine_number("US$ 62,50") == pytest.approx(62.50)
    assert parse_argentine_number("ARS 75.000,00") == pytest.approx(75000.0)
    assert parse_argentine_number("100.70") == pytest.approx(100.70)
    assert parse_argentine_number("0.125") == pytest.approx(0.125)
    assert parse_argentine_number("1,234.56") == pytest.approx(1234.56)
    for raw in ("-", "--", "—", "–", "", "   ", "&nbsp;", "N/A", "1.23.45", "abc"):
        assert parse_argentine_number(raw) is None
    assert parse_argentine_percentage("5,25%") == pytest.approx(0.0525)
    assert parse_argentine_percentage("-1,20%") == pytest.approx(-0.012)
    assert parse_argentine_percentage("-") is None


def test_iol_table_selection_species_priority_and_mep_fallback():
    obs_ts = "2026-11-26T15:00:00+00:00"
    bonos = parse_iol_fixed_income_table(BONOS_HTML_FIXTURE, panel_source="iol_bonos", observed_at=obs_ts)
    ons = parse_iol_fixed_income_table(ONS_HTML_FIXTURE, panel_source="iol_ons", observed_at=obs_ts)

    assert "GGAL" not in bonos.rows_by_ticker
    assert "AL30" in bonos.rows_by_ticker and "AL30D" in bonos.rows_by_ticker
    assert bonos.rows_by_ticker["GD35D"].last_price is None

    q_al30 = resolve_iol_bond_quote_usd(instrument_id="AL30", ticker="AL30", panel_snapshot=bonos)
    assert q_al30.species_used == "AL30D"
    assert q_al30.conversion_method == "D"
    assert q_al30.usd_settlement_type == "USD_LOCAL_D"
    assert q_al30.price_usd == pytest.approx(62.50)
    assert q_al30.observed_at == obs_ts
    assert q_al30.market_timestamp is None

    q_gd35 = resolve_iol_bond_quote_usd(instrument_id="GD35", ticker="GD35", panel_snapshot=bonos)
    assert q_gd35.species_used == "GD35C"
    assert q_gd35.conversion_method == "C"
    assert q_gd35.usd_settlement_type == "USD_CABLE_C"
    assert q_gd35.price_usd == pytest.approx(55.20)
    assert any("cable" in w.lower() for w in q_gd35.warnings)
    assert any("liquidity" in w.lower() for w in q_gd35.warnings)
    assert any("spread" in w.lower() for w in q_gd35.warnings)

    # Default conservative policy blocks ARS/MEP when market timestamps are absent (simultaneity unverified)
    with pytest.raises(IOLQuoteError, match="simultaneity cannot be verified"):
        resolve_iol_bond_quote_usd(instrument_id="AE38", ticker="AE38", panel_snapshot=bonos)

    # Explicit opt-in accepts unverified simultaneity with explicit diagnostic flag and warning
    opt_in_cfg = IOLConfig(allow_mep_without_market_timestamp=True)
    q_ae38 = resolve_iol_bond_quote_usd(instrument_id="AE38", ticker="AE38", panel_snapshot=bonos, iol_config=opt_in_cfg)
    assert q_ae38.species_used == "AE38"
    assert q_ae38.conversion_method == "ARS/MEP"
    assert q_ae38.usd_settlement_type == "USD_MEP_IMPLIED"
    assert q_ae38.mep_reference["simultaneity_verified"] is False
    assert q_ae38.mep_reference["mep_rate_ars_per_usd"] == pytest.approx(1200.0)
    assert q_ae38.price_usd == pytest.approx(50.0)

    q_mtcg = resolve_iol_bond_quote_usd(instrument_id="MTCGO", ticker="MTCGO", panel_snapshot=ons, bonos_snapshot=bonos)
    assert q_mtcg.species_used == "MTCGD"
    assert q_mtcg.conversion_method == "D"
    assert q_mtcg.price_usd == pytest.approx(100.70)

    q_ymcq = resolve_iol_bond_quote_usd(
        instrument_id="YMCQO", ticker="YMCQO", panel_snapshot=ons, bonos_snapshot=bonos, iol_config=opt_in_cfg
    )
    assert q_ymcq.species_used == "YMCQO"
    assert q_ymcq.conversion_method == "ARS/MEP"
    assert q_ymcq.price_usd == pytest.approx(105.0)

    q_explicit = resolve_iol_bond_quote_usd(
        instrument_id="CUSTOM_ON",
        ticker="ABCD",
        panel_snapshot=ons,
        bonos_snapshot=bonos,
        explicit_species_map={"D": "MTCGD", "ARS": "MTCGO"},
    )
    assert q_explicit.species_used == "MTCGD"
    assert q_explicit.price_usd == pytest.approx(100.70)


def test_iol_error_cases_http_missing_table_and_invalid_mep():
    with pytest.raises(IOLQuoteError, match="Expected IOL fixed-income table"):
        parse_iol_fixed_income_table("<html><table><tr><th>Foo</th><th>Bar</th></tr><tr><td>1</td><td>2</td></tr></table></html>", panel_source="iol_bonos")

    bad_session = MagicMock()
    bad_session.get.return_value = MagicMock(status_code=503, content=b"")
    with pytest.raises(IOLQuoteError, match="HTTP 503"):
        fetch_iol_panel("iol_bonos", session=bad_session)

    bonos = parse_iol_fixed_income_table(BONOS_HTML_FIXTURE, panel_source="iol_bonos", observed_at="2026-11-26T15:00:00+00:00")
    with pytest.raises(IOLQuoteError, match="not contemporaneous"):
        compute_implicit_mep_al30(
            bonos,
            reference_observed_at="2026-11-26T16:30:00+00:00",
            max_time_skew_seconds=300.0,
            allow_missing_market_timestamp=True,
        )

    bad_mep_html = BONOS_HTML_FIXTURE.replace("75.000,00", "50,00")
    bad_bonos = parse_iol_fixed_income_table(bad_mep_html, panel_source="iol_bonos")
    with pytest.raises(IOLQuoteError, match="non-comparable"):
        compute_implicit_mep_al30(bad_bonos, allow_missing_market_timestamp=True)

    ambig_html = ONS_HTML_FIXTURE.replace("MTCGO Mastellone Cl. G", "ABCD Random Issuer")
    ambig_ons = parse_iol_fixed_income_table(ambig_html, panel_source="iol_ons")
    with pytest.raises(IOLQuoteError, match="Ambiguous ticker"):
        resolve_iol_bond_quote_usd(instrument_id="ABCD", ticker="ABCD", panel_snapshot=ambig_ons, bonos_snapshot=bonos)

    ons_resp = MagicMock(status_code=200, content=ONS_HTML_FIXTURE.encode("utf-8"))
    bonos_resp = MagicMock(status_code=200, content=BONOS_HTML_FIXTURE.encode("utf-8"))
    mock_sess = MagicMock()
    mock_sess.get.side_effect = lambda url, **kwargs: bonos_resp if "bonos" in url else ons_resp
    fetched = fetch_iol_quotes_for_instruments(
        [InstrumentConfig(instrument_id="YMCQO", ticker="YMCQO", section="fixed_income", asset_subtype="corporate_bond_on", quote_source="iol_ons")],
        iol_config=IOLConfig(allow_mep_without_market_timestamp=True),
        session=mock_sess,
    )
    assert fetched["YMCQO"].conversion_method == "ARS/MEP"
    assert fetched["YMCQO"].price_usd == pytest.approx(105.0)
    assert mock_sess.get.call_count == 2


def test_fixed_income_bullet_zero_coupon_partial_amortization_and_clean_dirty():
    bullet_terms = FixedIncomeTerms(
        cashflow_currency="USD",
        reference_nominal=100.0,
        settlement_date=date(2025, 1, 1),
        maturity_date=date(2027, 1, 1),
        price_convention="per_100_original_nominal",
        price_type="dirty",
        day_count_convention="ACT/365",
        cashflows=(
            CashFlowEvent(payment_date=date(2026, 1, 1), coupon_amount=10.0, amortization_amount=0.0),
            CashFlowEvent(payment_date=date(2027, 1, 1), coupon_amount=10.0, amortization_amount=100.0),
        ),
    )
    res_par = evaluate_fixed_income_instrument("BULLET10", bullet_terms, quoted_price=100.0)
    assert res_par.status == "ok"
    assert res_par.ytm_annual == pytest.approx(0.10, abs=1e-6)
    assert res_par.weighted_average_life_years == pytest.approx(2.0, abs=1e-6)

    res_disc = evaluate_fixed_income_instrument("BULLET10", bullet_terms, quoted_price=90.0)
    assert res_disc.status == "ok"
    assert res_disc.ytm_annual > 0.15

    tbill_terms = FixedIncomeTerms(
        cashflow_currency="USD",
        reference_nominal=100.0,
        settlement_date=date(2025, 1, 1),
        maturity_date=date(2026, 1, 1),
        coupon_structure="zero_coupon",
        price_convention="per_100_original_nominal",
        price_type="clean",
        day_count_convention="ACT/365",
        cashflows=(CashFlowEvent(payment_date=date(2026, 1, 1), amortization_amount=100.0),),
    )
    res_tbill = evaluate_fixed_income_instrument("TBILL1Y", tbill_terms, quoted_price=95.0)
    assert res_tbill.status == "ok"
    assert res_tbill.ytm_annual == pytest.approx(100.0 / 95.0 - 1.0, abs=1e-8)

    amort_terms = FixedIncomeTerms(
        cashflow_currency="USD",
        reference_nominal=100.0,
        residual_nominal=60.0,
        amortization_factor=0.60,
        settlement_date=date(2025, 1, 1),
        maturity_date=date(2027, 1, 1),
        price_convention="per_100_residual_nominal",
        price_type="clean",
        accrued_interest=2.0,
        day_count_convention="ACT/365",
        coupon_frequency=1,
        cashflows=(
            CashFlowEvent(payment_date=date(2026, 1, 1), coupon_rate=0.10, amortization_amount=30.0, period_start_date=date(2025, 1, 1)),
            CashFlowEvent(payment_date=date(2027, 1, 1), coupon_rate=0.10, amortization_amount=30.0, period_start_date=date(2026, 1, 1)),
        ),
    )
    res_amort = evaluate_fixed_income_instrument("AMORT60", amort_terms, quoted_price=98.0)
    assert res_amort.status == "ok"
    assert res_amort.dirty_price_per_reference == pytest.approx(60.0)
    assert res_amort.total_future_coupons == pytest.approx(9.0)
    assert res_amort.total_future_amortizations == pytest.approx(60.0)
    assert res_amort.ytm_annual == pytest.approx(0.10, abs=1e-6)


def test_fixed_income_validation_floating_and_no_fabricated_history():
    terms_no_acc = FixedIncomeTerms(
        cashflow_currency="USD",
        reference_nominal=100.0,
        settlement_date=date(2025, 1, 1),
        maturity_date=date(2026, 1, 1),
        price_type="clean",
        accrued_interest=None,
        cashflows=(CashFlowEvent(payment_date=date(2026, 1, 1), coupon_amount=5.0, amortization_amount=100.0),),
    )
    with pytest.raises(FixedIncomeValidationError, match="accrued_interest"):
        evaluate_fixed_income_instrument("NOACC", terms_no_acc, quoted_price=98.0)

    terms_bad_amort = FixedIncomeTerms(
        cashflow_currency="USD",
        reference_nominal=100.0,
        residual_nominal=100.0,
        settlement_date=date(2025, 1, 1),
        maturity_date=date(2026, 1, 1),
        cashflows=(CashFlowEvent(payment_date=date(2026, 1, 1), coupon_amount=5.0, amortization_amount=80.0),),
    )
    with pytest.raises(FixedIncomeValidationError, match="does not reconcile"):
        evaluate_fixed_income_instrument("BADAMORT", terms_bad_amort, quoted_price=98.0)

    terms_float = FixedIncomeTerms(
        cashflow_currency="ARS",
        reference_nominal=100.0,
        settlement_date=date(2025, 1, 1),
        maturity_date=date(2026, 1, 1),
        coupon_structure="indexed",
        cashflows=(CashFlowEvent(payment_date=date(2026, 1, 1), amortization_amount=100.0),),
    )
    res_float = evaluate_fixed_income_instrument("CER26", terms_float, quoted_price=100.0)
    assert res_float.status == "not_evaluable_under_available_method"
    assert res_float.ytm_annual is None

    from portfolio_optimization import solve_xirr_with_root_check
    ytm_mr, root_cnt, status_mr = solve_xirr_with_root_check(100.0, [(1.0, 230.0), (2.0, -132.0)])
    assert status_mr == "multiple_roots"
    assert root_cnt == 2
    assert ytm_mr is None

    idx = pd.to_datetime(["2025-01-03", "2025-01-10", "2025-01-17"])
    dirty_px = pd.Series([100.0, 96.0, 96.0], index=idx, name="BOND")
    paid_cf = pd.Series([5.0], index=pd.to_datetime(["2025-01-09"]))
    tr_idx = build_fixed_income_total_return_index(dirty_px, paid_cf)
    assert tr_idx.iloc[1] == pytest.approx(101.0)

    bonos = parse_iol_fixed_income_table(BONOS_HTML_FIXTURE, panel_source="iol_bonos")
    cfg_no_hist = PortfolioAnalysisConfig(
        variable_income=(InstrumentConfig(instrument_id="SPY", ticker="SPY", section="variable_income", asset_subtype="equity_etf"),),
        fixed_income=(
            InstrumentConfig(
                instrument_id="AL30",
                ticker="AL30",
                section="fixed_income",
                asset_subtype="sovereign_bond",
                quote_source="iol_bonos",
            ),
        ),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=20),
    )
    spy_prices = pd.DataFrame(
        {"SPY": np.linspace(400.0, 500.0, 150)},
        index=pd.bdate_range("2024-01-02", periods=150),
    )
    with pytest.raises(InsufficientHistoryError, match="AL30"):
        analyze_portfolio(
            cfg_no_hist,
            prices_usd_override=spy_prices,
            iol_snapshots_override={"iol_bonos": bonos},
            iol_history_override={},
        )
