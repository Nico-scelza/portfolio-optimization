"""Deterministic offline test suite for IOL DatosHistoricos fixed-income history & pipeline integration."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
import requests

from portfolio_optimization import (
    BlackLittermanConfig,
    BrokerConfig,
    BrokerPosition,
    CashFlowEvent,
    ConfigurationError,
    FixedIncomeTerms,
    InstrumentConfig,
    InsufficientHistoryError,
    IOLConfig,
    IOLHistoricalBar,
    IOLHistoricalSeries,
    IOLHistoryError,
    IOLPanelRow,
    IOLPanelSnapshot,
    OptimizationConfig,
    PortfolioAnalysisConfig,
    RiskModelConfig,
    analyze_portfolio,
    build_iol_historical_segments,
    compute_weekly_returns_w_fri,
    ejecutar_analisis_completo,
    fetch_iol_historical_prices,
    fetch_iol_historical_prices_for_instruments,
    parse_iol_historical_page_metadata,
    parse_iol_historical_table,
    resolve_period_date_range,
    run_portfolio_analysis,
)

_AR_TZ = ZoneInfo("America/Argentina/Buenos_Aires")


def _make_iol_get_html(
    symbol: str,
    id_titulo: str,
    rows: List[Tuple[str, str, str, str, str, str, str, str]],
) -> str:
    """Build a realistic IOL /Titulo/DatosHistoricos GET HTML page fixture."""
    tr_html = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in r) + "</tr>"
        for r in rows
    )
    return f"""
    <html><body>
      <h2 class="colorIOL">Cotizacion hist&oacute;rica de {symbol}</h2>
      <form action="/Titulo/ImportarHistoriaAExcel" method="post">
        <input type="text" name="DesdeHasta" id="DesdeHasta" value="" />
        <select name="IdFrecuencias" id="IdFrecuencias">
          <option value="1">Diaria</option>
          <option value="2">Semanal</option>
          <option value="3">Mensual</option>
        </select>
        <input type="hidden" name="Simbolo" id="Simbolo" value="{symbol}" />
        <input type="hidden" name="IdTitulo" id="IdTitulo" value="{id_titulo}" />
      </form>
      <div id="cotizacionestab">
        <table id="tbcotizaciones" class="table table-striped">
          <thead>
            <tr>
              <th>Fecha Cotizaci&oacute;n</th>
              <th>Apertura</th>
              <th>M&aacute;ximo</th>
              <th>M&iacute;nimo</th>
              <th>Cierre</th>
              <th>Cierre ajustado</th>
              <th>Volumen Monto</th>
              <th>Volumen Nominal</th>
            </tr>
          </thead>
          <tbody>
            {tr_html}
          </tbody>
        </table>
      </div>
    </body></html>
    """


def _make_iol_post_html(rows: List[Tuple[str, str, str, str, str, str, str, str]]) -> str:
    """Build a realistic IOL /Titulo/DatosHistoricos POST AJAX HTML fragment fixture."""
    tr_html = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in r) + "</tr>"
        for r in rows
    )
    return f"""
    <table id="tbcotizaciones" class="table table-striped">
      <thead>
        <tr>
          <th>Fecha Cotizaci&oacute;n</th>
          <th>Apertura</th>
          <th>M&aacute;ximo</th>
          <th>M&iacute;nimo</th>
          <th>Cierre</th>
          <th>Cierre ajustado</th>
          <th>Volumen Monto</th>
          <th>Volumen Nominal</th>
        </tr>
      </thead>
      <tbody>
        {tr_html}
      </tbody>
    </table>
    """


def _mock_response(status_code: int, text: str, url: str = "https://iol.invertironline.com/Titulo/DatosHistoricos", headers: Dict[str, str] | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = text.encode("utf-8")
    resp.text = text
    resp.url = url
    resp.headers = headers or {}
    return resp


# =============================================================================
# 1. Query construction, parameters, exclusive upper bound, and caching
# =============================================================================

def test_query_construction_exact_symbols_ranges_market_and_request_cache():
    rows_aec3d = [
        ("11/26/2026", "99.70", "99.90", "98.75", "99.40", "99.40", "40,558.76", "40,876.00"),
        ("11/25/2026", "99.80", "99.85", "98.53", "99.85", "99.85", "43,945.66", "44,219.00"),
    ]
    get_html = _make_iol_get_html("AEC3D", "163723", rows_aec3d)
    post_html = _make_iol_post_html(rows_aec3d)

    session = MagicMock()
    session.get.return_value = _mock_response(
        200, get_html, url="https://iol.invertironline.com/Titulo/DatosHistoricos?simbolo=AEC3D&mercado=BCBA"
    )
    session.post.return_value = _mock_response(200, post_html)

    req_cache: Dict[Tuple[Any, ...], IOLHistoricalSeries] = {}
    fixed_now = lambda tz=None: datetime(2026, 11, 27, 18, 0, tzinfo=tz)

    res1 = fetch_iol_historical_prices(
        "AEC3D",
        start_date=date(2026, 8, 1),
        end_date=date(2026, 11, 26),
        market="BCBA",
        frequency="daily",
        session=session,
        cache=req_cache,
        sleep_fn=lambda _: None,
        now_fn=fixed_now,
    )

    assert session.get.call_count == 1
    get_kwargs = session.get.call_args.kwargs
    assert get_kwargs["params"] == {"simbolo": "AEC3D", "mercado": "BCBA"}
    assert get_kwargs["verify"] is True
    assert get_kwargs["timeout"] == pytest.approx(15.0)

    assert session.post.call_count == 1
    post_kwargs = session.post.call_args.kwargs
    # Upper bound adds +1 day because IOL treats right boundary of desdehasta as exclusive
    assert post_kwargs["data"] == {
        "desdehasta": "01/08/2026 - 27/11/2026",
        "idtitulo": "163723",
        "idfrecuencias": "1",
    }
    assert post_kwargs["headers"]["X-Requested-With"] == "XMLHttpRequest"
    assert post_kwargs["verify"] is True

    assert res1.symbol == "AEC3D"
    assert res1.market == "BCBA"
    assert res1.id_titulo == "163723"
    assert res1.valid_row_count == 2
    assert list(res1.price_series.index) == [pd.Timestamp("2026-11-25"), pd.Timestamp("2026-11-26")]
    assert list(res1.price_series.values) == pytest.approx([99.85, 99.40])

    # Second identical call hits request-scoped cache with zero extra HTTP requests
    res2 = fetch_iol_historical_prices(
        "AEC3D",
        start_date=date(2026, 8, 1),
        end_date=date(2026, 11, 26),
        market="BCBA",
        frequency="daily",
        session=session,
        cache=req_cache,
        sleep_fn=lambda _: None,
        now_fn=fixed_now,
    )
    assert res2 is res1
    assert session.get.call_count == 1
    assert session.post.call_count == 1

    # Distinct symbol, custom market, and weekly frequency build matching parameters
    bpa_get = _make_iol_get_html("BPA8D", "154636", [("11/26/2026", "83.20", "84.53", "81.59", "83.19", "83.19", "8,767.72", "10,502.00")])
    bpa_post = _make_iol_post_html([("11/26/2026", "83.20", "84.53", "81.59", "83.19", "83.19", "8,767.72", "10,502.00")])
    session.get.return_value = _mock_response(200, bpa_get)
    session.post.return_value = _mock_response(200, bpa_post)

    res_bpa = fetch_iol_historical_prices(
        "BPA8D",
        start_date=date(2025, 10, 1),
        end_date=date(2026, 11, 26),
        market="BCBA",
        frequency="weekly",
        session=session,
        sleep_fn=lambda _: None,
        now_fn=fixed_now,
    )
    assert session.get.call_args.kwargs["params"] == {"simbolo": "BPA8D", "mercado": "BCBA"}
    assert session.post.call_args.kwargs["data"]["idfrecuencias"] == "2"
    assert session.post.call_args.kwargs["data"]["idtitulo"] == "154636"
    assert res_bpa.symbol == "BPA8D"

    # Period resolution helper handles leap-year & standard lookbacks deterministically
    s_10y, e_10y = resolve_period_date_range("10y", as_of_date=date(2024, 2, 29))
    assert (s_10y, e_10y) == (date(2014, 2, 28), date(2024, 2, 29))
    s_6m, e_6m = resolve_period_date_range("6mo", as_of_date=date(2026, 8, 31))
    assert (s_6m, e_6m) == (date(2026, 2, 28), date(2026, 8, 31))
    with pytest.raises(ConfigurationError):
        resolve_period_date_range("invalid_period", as_of_date=date(2026, 11, 26))


# =============================================================================
# 2. Date & numeric parsing, Cierre vs Cierre ajustado, ordering, duplicates
# =============================================================================

def test_parsing_locales_dates_close_vs_adjusted_ordering_and_duplicate_conflict():
    # Live IOL en-US numeric + MM/DD/YYYY date format (including 3-decimal price 100.125!)
    us_html = _make_iol_post_html([
        ("11/26/2026", "99.70", "100.125", "98.75", "100.125", "99.500", "40,558.76", "40,876.00"),
        ("11/24/2026", "98.81", "99.70", "97.00", "99.70", "99.70", "45,151.25", "45,804.00"),
    ])
    bars_us, _ = parse_iol_historical_table(us_html, symbol="AEC3D")
    assert len(bars_us) == 2
    assert bars_us[0].trade_date == date(2026, 11, 26)
    assert bars_us[0].close == pytest.approx(100.125)
    assert bars_us[0].adjusted_close == pytest.approx(99.50)
    assert bars_us[0].volume_amount == pytest.approx(40558.76)
    assert bars_us[1].trade_date == date(2026, 11, 24)

    # Argentine es-AR numeric + DD/MM/YYYY date format support
    ar_html = _make_iol_post_html([
        ("26/11/2026", "81.700,00", "82.730,00", "80.810,00", "82.150,00", "82.150,00", "208.383.715.118,10", "254.411.972,00"),
        ("25/11/2026", "80.000,00", "81.500,00", "79.500,00", "81.200,50", "81.200,50", "150.000.000,00", "185.000.000,00"),
    ])
    bars_ar, _ = parse_iol_historical_table(ar_html, symbol="AL30")
    assert bars_ar[0].trade_date == date(2026, 11, 26)
    assert bars_ar[0].close == pytest.approx(82150.0)
    assert bars_ar[1].close == pytest.approx(81200.50)

    # Contradictory mixed date orders in same table rejected
    bad_dates_html = _make_iol_post_html([
        ("26/11/2026", "99.00", "99.50", "98.50", "99.10", "99.10", "1,000.00", "1,000.00"),
        ("11/25/2026", "99.00", "99.50", "98.50", "99.20", "99.20", "1,000.00", "1,000.00"),
    ])
    with pytest.raises(IOLHistoryError) as exc_dt:
        parse_iol_historical_table(bad_dates_html, symbol="BAD")
    assert exc_dt.value.code == "iol_schema_changed"

    # Identical duplicate dates collapse deterministically; adjusted_close tracked separately from close
    dup_ok_rows = [
        ("11/26/2026", "99.70", "99.90", "98.75", "99.40", "99.00", "10,000.00", "10,000.00"),
        ("11/26/2026", "99.70", "99.90", "98.75", "99.40", "99.00", "40,558.76", "40,876.00"),
        ("11/24/2026", "98.81", "99.70", "97.00", "99.70", "99.70", "45,151.25", "45,804.00"),
    ]
    sess = MagicMock()
    sess.get.return_value = _mock_response(200, _make_iol_get_html("TESTD", "999", dup_ok_rows))
    sess.post.return_value = _mock_response(200, _make_iol_post_html(dup_ok_rows))

    res_close = fetch_iol_historical_prices(
        "TESTD",
        start_date=date(2026, 11, 20),
        end_date=date(2026, 11, 26),
        price_field="close",
        session=sess,
        sleep_fn=lambda _: None,
        now_fn=lambda tz=None: datetime(2026, 11, 27, 18, 0, tzinfo=tz),
    )
    assert res_close.duplicate_dates_resolved == 1
    assert res_close.adjusted_close_differs_count == 1
    assert res_close.is_total_return is False
    assert res_close.series_type == "market_price"
    assert res_close.price_series.loc[pd.Timestamp("2026-11-26")] == pytest.approx(99.40)
    assert res_close.bars.loc[pd.Timestamp("2026-11-26"), "volume_amount"] == pytest.approx(40558.76)
    assert res_close.price_series.index.is_monotonic_increasing

    # Explicit adjusted_close opt-in selects adjusted_close and emits explicit fixed-income caveat
    res_adj = fetch_iol_historical_prices(
        "TESTD",
        start_date=date(2026, 11, 20),
        end_date=date(2026, 11, 26),
        price_field="adjusted_close",
        session=sess,
        sleep_fn=lambda _: None,
        now_fn=lambda tz=None: datetime(2026, 11, 27, 18, 0, tzinfo=tz),
    )
    assert res_adj.price_series.loc[pd.Timestamp("2026-11-26")] == pytest.approx(99.00)
    assert res_adj.is_total_return is False
    assert any("Cierre ajustado" in w for w in res_adj.warnings)

    # Conflicting duplicate close prices on same date raise explicit IOLHistoryError
    conflict_rows = [
        ("11/26/2026", "99.70", "99.90", "98.75", "99.40", "99.40", "40,558.76", "40,876.00"),
        ("11/26/2026", "99.70", "99.90", "98.75", "98.10", "98.10", "40,558.76", "40,876.00"),
    ]
    sess.post.return_value = _mock_response(200, _make_iol_post_html(conflict_rows))
    with pytest.raises(IOLHistoryError) as exc_dup:
        fetch_iol_historical_prices(
            "TESTD",
            start_date=date(2026, 11, 20),
            end_date=date(2026, 11, 26),
            session=sess,
            sleep_fn=lambda _: None,
            now_fn=lambda tz=None: datetime(2026, 11, 27, 18, 0, tzinfo=tz),
        )
    assert exc_dup.value.code == "iol_duplicate_conflict"


# =============================================================================
# 3. Multi-segment extended ranges, partial coverage, no-trade rows
# =============================================================================

def test_multi_segment_range_partial_coverage_and_no_trade_filtering():
    segments = build_iol_historical_segments(date(2020, 1, 1), date(2026, 11, 26), chunk_days=1000)
    assert len(segments) == 3
    assert segments[0][0] == date(2020, 1, 1)
    assert segments[-1][1] == date(2026, 11, 26)
    for (s1, e1), (s2, e2) in zip(segments, segments[1:]):
        assert s2 == e1 + timedelta(days=1)

    # Simulate 3 segments where segment 1 is pre-issuance (empty table), segment 2 has listing zero row + trades, segment 3 has overlapping boundary duplicate + trades
    seg1_html = _make_iol_post_html([])
    seg2_html = _make_iol_post_html([
        ("06/20/2025", "85.00", "86.00", "84.50", "85.50", "85.50", "100,000.00", "117,000.00"),
        ("09/30/2025", "0.00", "0.00", "0.00", "0.00", "0.00", "0.00", "0.00"),  # no-trade listing day
    ])
    seg3_html = _make_iol_post_html([
        ("11/26/2026", "83.20", "84.53", "81.59", "83.19", "83.19", "8,767.72", "10,502.00"),
        ("06/20/2025", "85.00", "86.00", "84.50", "85.50", "85.50", "100,000.00", "117,000.00"),  # overlap duplicate
    ])

    sess = MagicMock()
    sess.get.return_value = _mock_response(200, _make_iol_get_html("BPA8D", "154636", []))
    sess.post.side_effect = [
        _mock_response(200, seg1_html),
        _mock_response(200, seg2_html),
        _mock_response(200, seg3_html),
    ]
    sleeps: List[float] = []

    res = fetch_iol_historical_prices(
        "BPA8D",
        start_date=date(2020, 1, 1),
        end_date=date(2026, 11, 26),
        iol_config=IOLConfig(historical_chunk_days=1000, historical_request_delay_seconds=0.05),
        session=sess,
        sleep_fn=sleeps.append,
        now_fn=lambda tz=None: datetime(2026, 11, 27, 18, 0, tzinfo=tz),
    )

    assert sess.post.call_count == 3
    assert len(sleeps) == 2  # polite pacing between segments 2 & 3
    assert res.segments_fetched == 3
    assert res.valid_row_count == 2
    assert res.zero_or_no_trade_rows_dropped == 1
    assert res.duplicate_dates_resolved == 1
    assert res.partial_coverage is True
    assert res.effective_start_date == date(2025, 6, 20)
    assert res.effective_end_date == date(2026, 11, 26)
    assert any("Partial historical coverage" in w for w in res.warnings)
    assert any("zero-price/no-trade" in w for w in res.warnings)

    # All-zero rows in range -> distinct iol_no_valid_trades error (not download failure)
    sess_zero = MagicMock()
    sess_zero.get.return_value = _mock_response(200, _make_iol_get_html("BPA8D", "154636", []))
    sess_zero.post.return_value = _mock_response(
        200, _make_iol_post_html([("09/30/2025", "0.00", "0.00", "0.00", "0.00", "0.00", "0.00", "0.00")])
    )
    with pytest.raises(IOLHistoryError) as exc_zero:
        fetch_iol_historical_prices(
            "BPA8D",
            start_date=date(2025, 9, 1),
            end_date=date(2025, 9, 30),
            session=sess_zero,
            sleep_fn=lambda _: None,
            now_fn=lambda tz=None: datetime(2026, 11, 27, 18, 0, tzinfo=tz),
        )
    assert exc_zero.value.code == "iol_no_valid_trades"


# =============================================================================
# 4. Error handling: symbol not found, empty, network/timeout, 429, 403, schema
# =============================================================================

def test_error_handling_not_found_empty_retries_rate_limit_access_denied_and_schema():
    cfg = IOLConfig(historical_max_retries=2, historical_retry_backoff_seconds=0.25, historical_request_delay_seconds=0.0)
    fixed_now = lambda tz=None: datetime(2026, 11, 27, 18, 0, tzinfo=tz)

    # 4a. Non-existent symbol (missing IdTitulo, like ZZZZZZ999 on real IOL) -> immediate iol_symbol_not_found
    not_found_html = """
    <html><body>
      <h2 class="colorIOL">Cotizacion hist&oacute;rica de</h2>
      <input type="text" name="DesdeHasta" id="DesdeHasta" value="" />
    </body></html>
    """
    sess_nf = MagicMock()
    sess_nf.get.return_value = _mock_response(200, not_found_html)
    with pytest.raises(IOLHistoryError) as exc_nf:
        fetch_iol_historical_prices("ZZZZZZ999", start_date=date(2026, 1, 1), end_date=date(2026, 11, 26), iol_config=cfg, session=sess_nf, sleep_fn=lambda _: None)
    assert exc_nf.value.code == "iol_symbol_not_found"
    assert sess_nf.get.call_count == 1  # never retries non-existent symbol

    # 4b. Empty rows across requested range -> iol_empty_history
    sess_empty = MagicMock()
    sess_empty.get.return_value = _mock_response(200, _make_iol_get_html("AEC3D", "163723", []))
    sess_empty.post.return_value = _mock_response(200, _make_iol_post_html([]))
    with pytest.raises(IOLHistoryError) as exc_emp:
        fetch_iol_historical_prices("AEC3D", start_date=date(2020, 1, 1), end_date=date(2020, 12, 31), iol_config=cfg, session=sess_empty, sleep_fn=lambda _: None)
    assert exc_emp.value.code == "iol_empty_history"

    # 4c. Transient timeout recovers on retry with exponential backoff
    good_rows = [("11/26/2026", "99.70", "99.90", "98.75", "99.40", "99.40", "40,558.76", "40,876.00")]
    sess_retry = MagicMock()
    sess_retry.get.side_effect = [
        requests.Timeout("read timed out"),
        _mock_response(200, _make_iol_get_html("AEC3D", "163723", good_rows)),
    ]
    sess_retry.post.return_value = _mock_response(200, _make_iol_post_html(good_rows))
    sleeps: List[float] = []
    res_rec = fetch_iol_historical_prices(
        "AEC3D", start_date=date(2026, 11, 1), end_date=date(2026, 11, 26),
        iol_config=cfg, session=sess_retry, sleep_fn=sleeps.append, now_fn=fixed_now,
    )
    assert res_rec.valid_row_count == 1
    assert sleeps == [0.25]

    # 4d. Exhausted timeout -> iol_timeout
    sess_to = MagicMock()
    sess_to.get.side_effect = requests.Timeout("timed out")
    sleeps_to: List[float] = []
    with pytest.raises(IOLHistoryError) as exc_to:
        fetch_iol_historical_prices("AEC3D", start_date=date(2026, 11, 1), end_date=date(2026, 11, 26), iol_config=cfg, session=sess_to, sleep_fn=sleeps_to.append)
    assert exc_to.value.code == "iol_timeout"
    assert sess_to.get.call_count == 3
    assert sleeps_to == [0.25, 0.5]

    # 4e. Rate limit HTTP 429 honors Retry-After and raises iol_rate_limited when exhausted
    sess_429 = MagicMock()
    sess_429.get.return_value = _mock_response(429, "Too Many Requests", headers={"Retry-After": "1.5"})
    sleeps_429: List[float] = []
    with pytest.raises(IOLHistoryError) as exc_429:
        fetch_iol_historical_prices("AEC3D", start_date=date(2026, 11, 1), end_date=date(2026, 11, 26), iol_config=cfg, session=sess_429, sleep_fn=sleeps_429.append)
    assert exc_429.value.code == "iol_rate_limited"
    assert sleeps_429 == [1.5, 1.5]

    # 4f. Access denied HTTP 403 -> immediate iol_access_denied with zero retries
    sess_403 = MagicMock()
    sess_403.get.return_value = _mock_response(403, "Forbidden")
    with pytest.raises(IOLHistoryError) as exc_403:
        fetch_iol_historical_prices("AEC3D", start_date=date(2026, 11, 1), end_date=date(2026, 11, 26), iol_config=cfg, session=sess_403, sleep_fn=lambda _: None)
    assert exc_403.value.code == "iol_access_denied"
    assert sess_403.get.call_count == 1

    # 4g. Schema change (missing #tbcotizaciones / expected headers) -> iol_schema_changed
    sess_schema = MagicMock()
    sess_schema.get.return_value = _mock_response(200, _make_iol_get_html("AEC3D", "163723", good_rows))
    sess_schema.post.return_value = _mock_response(200, "<table><tr><th>Foo</th><th>Bar</th></tr><tr><td>1</td><td>2</td></tr></table>")
    with pytest.raises(IOLHistoryError) as exc_sc:
        fetch_iol_historical_prices("AEC3D", start_date=date(2026, 11, 1), end_date=date(2026, 11, 26), iol_config=cfg, session=sess_schema, sleep_fn=lambda _: None)
    assert exc_sc.value.code == "iol_schema_changed"

    # 4h. Symbol mismatch on GET page -> iol_symbol_mismatch
    with pytest.raises(IOLHistoryError) as exc_mm:
        parse_iol_historical_page_metadata(_make_iol_get_html("OTHERD", "12345", good_rows), requested_symbol="AEC3D")
    assert exc_mm.value.code == "iol_symbol_mismatch"


# =============================================================================
# 5. Routing preservation (bonds/ONs -> IOL, stocks ending in D -> yfinance)
# =============================================================================

def test_routing_bonds_ons_to_iol_and_preserving_yfinance_including_trailing_d_equities():
    dates = pd.bdate_range("2025-01-03", periods=220)
    bpa_prices = pd.Series(np.linspace(75.0, 84.0, len(dates)), index=dates, name="BPA8D")
    amd_prices = pd.Series(np.linspace(120.0, 160.0, len(dates)), index=dates, name="AMD")

    bpa_snap = IOLPanelSnapshot(
        panel_source="iol_bonos",
        url="mock",
        observed_at="2025-11-15T15:00:00+00:00",
        column_headers=(),
        rows_by_ticker={
            "BPA8D": IOLPanelRow(ticker="BPA8D", name="BOPREAL 2028 USD", last_price=84.0, market_timestamp="14/11/2025 16:50"),
        },
    )

    iol_called_symbols: List[Tuple[str, str]] = []

    def fake_iol_fetcher(**kwargs: Any) -> IOLHistoricalSeries:
        iol_called_symbols.append((kwargs["symbol"], kwargs["market"]))
        return IOLHistoricalSeries(
            symbol=kwargs["symbol"],
            market=kwargs["market"],
            id_titulo="154636",
            requested_start_date=date(2020, 1, 1),
            requested_end_date=date(2025, 11, 14),
            effective_start_date=dates[0].date(),
            effective_end_date=dates[-1].date(),
            frequency="daily",
            price_field="close",
            series_type="market_price",
            currency="USD",
            usd_settlement_type="USD_LOCAL_D",
            quote_unit="per_100_original_nominal",
            observed_at="2025-11-15T15:00:00+00:00",
            source_url="https://iol.invertironline.com/Titulo/DatosHistoricos?simbolo=BPA8D&mercado=BCBA",
            bars=pd.DataFrame({"close": bpa_prices.values}, index=dates),
            price_series=bpa_prices,
            raw_row_count=len(dates),
            valid_row_count=len(dates),
            is_total_return=False,
        )

    yf_called_tickers: List[str] = []

    def fake_download_prices(tickers: List[str], **kwargs: Any) -> Tuple[pd.DataFrame, List[str]]:
        yf_called_tickers.extend(tickers)
        return pd.DataFrame({"AMD": amd_prices}), []

    cfg = PortfolioAnalysisConfig(
        variable_income=(
            # Ticker ends in 'D' ("AMD"), must NEVER be misrouted to IOL!
            InstrumentConfig(instrument_id="AMD", ticker="AMD", section="variable_income", asset_subtype="equity", quote_source="yfinance"),
        ),
        fixed_income=(
            # Custom internal instrument_id mapped explicitly via iol_symbol="BPA8D"
            InstrumentConfig(
                instrument_id="BOPREAL_28",
                ticker="BPA8D",
                iol_symbol="BPA8D",
                iol_market="BCBA",
                section="fixed_income",
                asset_subtype="sovereign_bond",
                quote_source="iol_bonos",
            ),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=20),
        optimization=OptimizationConfig(strategies_to_run=("min_variance", "erc"), selected_strategy="erc"),
        as_of_date=date(2025, 11, 14),
    )

    with patch("portfolio_optimization.optimization._download_prices", side_effect=fake_download_prices):
        res = run_portfolio_analysis(
            cfg,
            iol_snapshots_override={"iol_bonos": bpa_snap},
            iol_history_fetcher=fake_iol_fetcher,
            render_ui=False,
        )

    assert yf_called_tickers == ["AMD"]
    assert iol_called_symbols == [("BPA8D", "BCBA")]
    assert "BOPREAL_28" in res.quotes_and_data_quality["iol_historical_metadata"]
    assert res.quotes_and_data_quality["iol_historical_metadata"]["BOPREAL_28"]["symbol"] == "BPA8D"
    assert res.quotes_and_data_quality["iol_historical_metadata"]["BOPREAL_28"]["is_total_return"] is False


# =============================================================================
# 6. Currency/species guards, no look-ahead, intraday exclusion, W-FRI scale
# =============================================================================

def test_currency_species_guards_no_lookahead_intraday_and_w_fri_compatibility():
    # 6a. ARS-only bond (USD_MEP_IMPLIED) MUST be blocked from USD portfolio history
    bonos_ars_only = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at="2026-11-26T15:00:00+00:00", column_headers=(),
        rows_by_ticker={
            "AL30": IOLPanelRow(ticker="AL30", name="Bono 2030", last_price=75000.0, market_timestamp="26/11/2026 15:00"),
            "AL30D": IOLPanelRow(ticker="AL30D", name="Bono 2030", last_price=62.50, market_timestamp="26/11/2026 15:00"),
            "AE38": IOLPanelRow(ticker="AE38", name="Bono 2038", last_price=60000.0, market_timestamp="26/11/2026 15:00"),
        },
    )
    cfg_ars = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(instrument_id="AE38", ticker="AE38", section="fixed_income", asset_subtype="sovereign_bond", quote_source="iol_bonos"),
        ),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=20),
    )
    with pytest.raises(IOLHistoryError) as exc_ars:
        analyze_portfolio(cfg_ars, iol_snapshots_override={"iol_bonos": bonos_ars_only})
    assert exc_ars.value.code == "iol_currency_incompatible"

    # 6b. USD Cable (C) species blocked by default, permitted only with explicit allow_usd_cable_history=True
    bonos_c_only = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at="2026-11-26T15:00:00+00:00", column_headers=(),
        rows_by_ticker={
            "GD35C": IOLPanelRow(ticker="GD35C", name="Global 2035", last_price=55.20, market_timestamp="26/11/2026 15:00"),
        },
    )
    cfg_c_blocked = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(instrument_id="GD35", ticker="GD35", section="fixed_income", asset_subtype="sovereign_bond", quote_source="iol_bonos"),
        ),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=20),
    )
    with pytest.raises(IOLHistoryError) as exc_c:
        analyze_portfolio(cfg_c_blocked, iol_snapshots_override={"iol_bonos": bonos_c_only})
    assert exc_c.value.code == "iol_currency_incompatible"

    # With explicit opt-in, USD_CABLE_C proceeds with explicit warning and metadata tag
    dates = pd.bdate_range("2025-01-03", periods=160)
    c_series = pd.Series(np.linspace(50.0, 55.2, len(dates)), index=dates, name="GD35C")
    cfg_c_opt = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(instrument_id="GD35", ticker="GD35", section="fixed_income", asset_subtype="sovereign_bond", quote_source="iol_bonos"),
        ),
        iol=IOLConfig(allow_usd_cable_history=True),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=20),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
    )
    res_c = analyze_portfolio(
        cfg_c_opt,
        iol_snapshots_override={"iol_bonos": bonos_c_only},
        iol_history_fetcher=lambda **kw: IOLHistoricalSeries(
            symbol=kw["symbol"], market=kw["market"], id_titulo="99",
            requested_start_date=dates[0].date(), requested_end_date=dates[-1].date(),
            effective_start_date=dates[0].date(), effective_end_date=dates[-1].date(),
            frequency="daily", price_field="close", series_type="market_price",
            currency=kw["currency"], usd_settlement_type=kw["usd_settlement_type"],
            quote_unit=kw["quote_unit"], observed_at="2026-11-26T15:00:00+00:00",
            source_url="mock", bars=pd.DataFrame({"close": c_series.values}, index=dates),
            price_series=c_series, raw_row_count=len(dates), valid_row_count=len(dates),
        ),
    )
    assert res_c.quotes_and_data_quality["iol_historical_metadata"]["GD35"]["usd_settlement_type"] == "USD_CABLE_C"
    assert any("USD Cable (C)" in w for w in res_c.warnings)

    # 6c. Requesting adjusted_total_return from IOL historical fails closed
    cfg_tr_fail = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="AL30", ticker="AL30", section="fixed_income", asset_subtype="sovereign_bond",
                quote_source="iol_bonos", historical_price_series_type="adjusted_total_return",
            ),
        ),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=20),
    )
    with pytest.raises(IOLHistoryError) as exc_tr:
        analyze_portfolio(cfg_tr_fail, iol_snapshots_override={"iol_bonos": bonos_ars_only})
    assert exc_tr.value.code == "iol_total_return_unavailable"

    # 6d. No look-ahead (as_of_date) and intraday exclusion prior to 17:00 ART
    rows_cutoff = [
        ("11/27/2026", "99.50", "99.80", "99.10", "99.60", "99.60", "5,000.00", "5,000.00"),  # intraday / future
        ("11/26/2026", "99.70", "99.90", "98.75", "99.40", "99.40", "40,558.76", "40,876.00"),
        ("11/25/2026", "99.80", "99.85", "98.53", "99.85", "99.85", "43,945.66", "44,219.00"),
    ]
    sess_cut = MagicMock()
    sess_cut.get.return_value = _mock_response(200, _make_iol_get_html("AEC3D", "163723", rows_cutoff))
    sess_cut.post.return_value = _mock_response(200, _make_iol_post_html(rows_cutoff))

    # During open BCBA session (13:00 ART on 2026-11-27), 11/27/2026 is dropped as intraday
    res_intraday = fetch_iol_historical_prices(
        "AEC3D",
        start_date=date(2026, 11, 20),
        end_date=date(2026, 11, 27),
        session=sess_cut,
        sleep_fn=lambda _: None,
        now_fn=lambda tz=None: datetime(2026, 11, 27, 13, 0, tzinfo=tz),
    )
    assert res_intraday.intraday_rows_dropped == 1
    assert res_intraday.effective_end_date == date(2026, 11, 26)
    assert pd.Timestamp("2026-11-27") not in res_intraday.price_series.index

    # Strict as_of_date=2026-11-25 excludes 11/26 and 11/27 (zero look-ahead)
    res_asof = fetch_iol_historical_prices(
        "AEC3D",
        start_date=date(2026, 11, 20),
        end_date=date(2026, 11, 27),
        as_of_date=date(2026, 11, 25),
        session=sess_cut,
        sleep_fn=lambda _: None,
    )
    assert res_asof.effective_end_date == date(2026, 11, 25)
    assert res_asof.valid_row_count == 1

    # 6e. Scale consistency: per_100_original_nominal returns == per_1_original_nominal returns identically
    idx = pd.bdate_range("2025-01-03", periods=160)
    px_per_100 = pd.Series(80.0 + np.sin(np.linspace(0, 8, len(idx))) * 5.0, index=idx, name="BPA8D")
    px_per_1 = px_per_100 / 100.0
    b100 = compute_weekly_returns_w_fri(pd.DataFrame({"BPA8D": px_per_100}), risk_config=RiskModelConfig(min_weekly_observations=20))
    b1 = compute_weekly_returns_w_fri(pd.DataFrame({"BPA8D": px_per_1}), risk_config=RiskModelConfig(min_weekly_observations=20))
    np.testing.assert_allclose(b100.weekly_returns["BPA8D"].to_numpy(), b1.weekly_returns["BPA8D"].to_numpy(), rtol=0.0, atol=1e-14)


# =============================================================================
# 7. End-to-end mixed portfolio integration test (IOL bond + IOL ON + US ETF)
# =============================================================================

def test_end_to_end_mixed_portfolio_iol_fixed_income_and_equity_returns_oracle():
    """Full pipeline from mocked IOL HTML responses -> W-FRI weekly returns -> optimization -> rebalancing."""
    # Construct 30 controlled closed Friday-ending business weeks (2025-01-03 through 2025-07-25)
    bdays = pd.bdate_range("2025-01-03", "2025-07-25")
    # Drop a few mid-week Argentine holidays to verify zero fabrication / clean W-FRI grouping
    ar_holidays = {pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-04"), pd.Timestamp("2025-05-01")}
    bond_days = [d for d in bdays if d not in ar_holidays]

    rng = np.random.default_rng(20261127)
    spy_vals = 500.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.009, len(bdays))))
    bpa_vals = 80.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.006, len(bond_days))))
    aec_vals = 99.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.003, len(bond_days))))

    spy_series = pd.Series(spy_vals, index=bdays, name="SPY")

    def _rows_from_series(dts: List[pd.Timestamp], vals: np.ndarray) -> List[Tuple[str, str, str, str, str, str, str, str]]:
        # Descending order matching live IOL, MM/DD/YYYY dates, en-US numbers
        out = []
        for dt, v in zip(reversed(dts), reversed(vals)):
            ds = dt.strftime("%m/%d/%Y")
            p = f"{float(v):.4f}"
            out.append((ds, p, p, p, p, p, "50,000.00", "50,000.00"))
        # Add one listing-day zero row before first valid trade to verify clean filtering
        out.append(("01/02/2025", "0.00", "0.00", "0.00", "0.00", "0.00", "0.00", "0.00"))
        return out

    bpa_rows = _rows_from_series(bond_days, bpa_vals)
    aec_rows = _rows_from_series(bond_days, aec_vals)

    html_by_symbol = {
        "BPA8D": (_make_iol_get_html("BPA8D", "154636", bpa_rows[:50]), _make_iol_post_html(bpa_rows)),
        "AEC3D": (_make_iol_get_html("AEC3D", "163723", aec_rows[:50]), _make_iol_post_html(aec_rows)),
    }

    mock_session = MagicMock()

    def _mock_get(url: str, **kwargs: Any) -> MagicMock:
        sym = kwargs["params"]["simbolo"]
        return _mock_response(200, html_by_symbol[sym][0], url=f"{url}?simbolo={sym}&mercado=BCBA")

    def _mock_post(url: str, **kwargs: Any) -> MagicMock:
        id_t = kwargs["data"]["idtitulo"]
        sym = "BPA8D" if id_t == "154636" else "AEC3D"
        return _mock_response(200, html_by_symbol[sym][1], url=url)

    mock_session.get.side_effect = _mock_get
    mock_session.post.side_effect = _mock_post

    bonos_snap = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at="2025-07-25T20:00:00+00:00", column_headers=(),
        rows_by_ticker={"BPA8D": IOLPanelRow(ticker="BPA8D", name="BOPREAL 2028", last_price=float(bpa_vals[-1]), market_timestamp="25/07/2025 16:55")},
    )
    ons_snap = IOLPanelSnapshot(
        panel_source="iol_ons", url="mock", observed_at="2025-07-25T20:00:00+00:00", column_headers=(),
        rows_by_ticker={"AEC3D": IOLPanelRow(ticker="AEC3D", name="AES Arg Cl 3", last_price=float(aec_vals[-1]), market_timestamp="25/07/2025 16:55")},
    )

    cfg = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(
                instrument_id="SPY", ticker="SPY", section="variable_income", asset_subtype="equity_etf",
                quote_source="yfinance", default_buy_broker_id="IBKR",
            ),
        ),
        fixed_income=(
            InstrumentConfig(
                instrument_id="BPA8D", ticker="BPA8D", section="fixed_income", asset_subtype="sovereign_bond",
                quote_source="iol_bonos", market="BCBA", default_buy_broker_id="IOL",
                fractional_allowed=False, lot_size=1.0, min_nominal=10.0,
            ),
            InstrumentConfig(
                instrument_id="AEC3D", ticker="AEC3D", section="fixed_income", asset_subtype="corporate_bond_on",
                quote_source="iol_ons", market="BCBA", default_buy_broker_id="IOL",
                fractional_allowed=False, lot_size=1.0, min_nominal=10.0,
            ),
        ),
        brokers=(
            BrokerConfig(broker_id="IBKR", buy_percentage_fee=0.001),
            BrokerConfig(broker_id="IOL", buy_percentage_fee=0.005),
        ),
        positions=(
            BrokerPosition(instrument_id="SPY", broker_id="IBKR", quantity=2.0),
            BrokerPosition(instrument_id="BPA8D", broker_id="IOL", quantity=500.0),
            BrokerPosition(instrument_id="AEC3D", broker_id="IOL", quantity=500.0),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(period="1y", covariance_estimator="static_shrinkage", min_weekly_observations=20),
        optimization=OptimizationConfig(
            strategies_to_run=("max_sharpe", "min_variance", "min_cvar", "erc", "hrp"),
            selected_strategy="erc",
        ),
        iol=IOLConfig(historical_request_delay_seconds=0.0),
        as_of_date=date(2025, 7, 25),
    )

    with patch("portfolio_optimization.iol.requests.Session", return_value=mock_session):
        res = ejecutar_analisis_completo(
            cfg,
            render_ui=False,
            prices_usd_override=pd.DataFrame({"SPY": spy_series}),
            iol_snapshots_override={"iol_bonos": bonos_snap, "iol_ons": ons_snap},
        )

    assert res.tickers == ["SPY", "BPA8D", "AEC3D"]
    assert res.risk_model.total_weekly_samples >= 25

    # Independent mathematical oracle: recompute W-FRI weekly simple returns directly from raw fixture series
    raw_df = pd.DataFrame({
        "SPY": spy_series,
        "BPA8D": pd.Series(np.round(bpa_vals, 4), index=pd.DatetimeIndex(bond_days)),
        "AEC3D": pd.Series(np.round(aec_vals, 4), index=pd.DatetimeIndex(bond_days)),
    })
    expected_weekly_prices = raw_df.resample("W-FRI").last()
    expected_weekly_returns = expected_weekly_prices.pct_change().dropna(how="any")

    actual_ret = res.risk_model.weekly_returns[["SPY", "BPA8D", "AEC3D"]]
    assert list(actual_ret.index) == list(expected_weekly_returns.index)
    np.testing.assert_allclose(actual_ret.to_numpy(), expected_weekly_returns.to_numpy(), rtol=0.0, atol=1e-12)

    # Verify optimization strategies converged and sum to 1.0
    for sname in ("min_variance", "min_cvar", "erc", "hrp"):
        strat = res.strategies[sname]
        assert strat.feasible is True
        assert sum(strat.weights_by_asset.values()) == pytest.approx(1.0, abs=1e-6)

    # Verify provenance metadata, per-100-VN execution scaling, and JSON serializability
    hist_meta = res.quotes_and_data_quality["iol_historical_metadata"]
    assert set(hist_meta.keys()) == {"BPA8D", "AEC3D"}
    assert hist_meta["BPA8D"]["zero_or_no_trade_rows_dropped"] == 1
    assert hist_meta["AEC3D"]["zero_or_no_trade_rows_dropped"] == 1
    assert hist_meta["BPA8D"]["is_total_return"] is False
    assert hist_meta["AEC3D"]["is_total_return"] is False
    assert res.quotes_and_data_quality["execution_prices_usd_per_unit"]["BPA8D"] == pytest.approx(float(bpa_vals[-1]) / 100.0)
    assert res.quotes_and_data_quality["execution_prices_usd_per_unit"]["AEC3D"] == pytest.approx(float(aec_vals[-1]) / 100.0)
    assert any("price return only, NOT total return" in w for w in res.warnings)

    serialized = json.dumps(res.to_dict())
    assert "iol_historical_metadata" in serialized
    assert "BPA8D" in serialized and "AEC3D" in serialized


# =============================================================================
# 8. Reviewer remediation regressions (HIGH-01..03, MED-01..03, LOW-01..02)
# =============================================================================

def test_remediation_en_us_comma_thousands_locale_detection_and_short_range_dates():
    from portfolio_optimization import parse_argentine_number

    # HIGH-01: en-US single-comma thousands must NOT be parsed as decimal (1000x guard)
    assert parse_argentine_number("75,000", locale_mode="en-US") == pytest.approx(75000.0)
    assert parse_argentine_number("40,876", locale_mode="en-US") == pytest.approx(40876.0)
    assert parse_argentine_number("0,125", locale_mode="en-US") == pytest.approx(0.125)
    assert parse_argentine_number("100,70", locale_mode="es-AR") == pytest.approx(100.70)
    assert parse_argentine_number("75,000", locale_mode="es-AR") == pytest.approx(75.0)

    # HIGH-01: Multi-separator integer es-AR table ("1.250.000", "75.000") detected as es-AR
    es_int_html = _make_iol_post_html([
        ("05/09/2026", "75.000", "76.000", "74.000", "75.000", "75.000", "1.250.000", "1.000.000"),
        ("04/09/2026", "74.000", "75.000", "73.000", "74.000", "74.000", "1.100.000", "1.000.000"),
    ])
    bars_es, _ = parse_iol_historical_table(es_int_html, symbol="AL30")
    assert bars_es[0].close == pytest.approx(75000.0)
    assert bars_es[0].volume_amount == pytest.approx(1250000.0)
    # MED-01: Short-range es-AR table with day <= 12 parses DD/MM/YYYY (Sept 5, Sept 4), NOT May 9 / Apr 9!
    assert bars_es[0].trade_date == date(2026, 9, 5)
    assert bars_es[1].trade_date == date(2026, 9, 4)

    # MED-01: Expected range disambiguates short-range all-<=12 slash dates even without decimal clues
    ambig_html = _make_iol_post_html([
        ("05/09/2026", "100", "101", "99", "100", "100", "1000", "1000"),
        ("04/09/2026", "99", "100", "98", "99", "99", "1000", "1000"),
    ])
    bars_rng, _ = parse_iol_historical_table(
        ambig_html, symbol="TESTD", expected_range=(date(2026, 9, 1), date(2026, 9, 10))
    )
    assert [b.trade_date for b in bars_rng] == [date(2026, 9, 5), date(2026, 9, 4)]

    # MED-02: price_field="adjusted_close" never silently falls back to Cierre when column missing or "-"
    no_adj_col_html = """
    <table id="tbcotizaciones">
      <tr><th>Fecha Cotizaci&oacute;n</th><th>Cierre</th></tr>
      <tr><td>11/26/2026</td><td>99.40</td></tr>
    </table>
    """
    with pytest.raises(IOLHistoryError) as exc_col:
        parse_iol_historical_table(no_adj_col_html, symbol="AEC3D", require_adjusted_close=True)
    assert exc_col.value.code == "iol_schema_changed"

    dash_adj_rows = [("11/26/2026", "99.70", "99.90", "98.75", "99.40", "-", "40,558.76", "40,876.00")]
    sess_dash = MagicMock()
    sess_dash.get.return_value = _mock_response(200, _make_iol_get_html("AEC3D", "163723", dash_adj_rows))
    sess_dash.post.return_value = _mock_response(200, _make_iol_post_html(dash_adj_rows))
    with pytest.raises(IOLHistoryError) as exc_dash:
        fetch_iol_historical_prices(
            "AEC3D", start_date=date(2026, 11, 1), end_date=date(2026, 11, 26),
            price_field="adjusted_close", session=sess_dash, sleep_fn=lambda _: None,
            now_fn=lambda tz=None: datetime(2026, 11, 27, 18, 0, tzinfo=tz),
        )
    assert exc_dash.value.code == "iol_no_valid_trades"


def test_remediation_iol_symbol_quote_alignment_cache_key_immutability_and_lookahead_guards():
    # HIGH-02: Custom instrument_id != ticker != iol_symbol works end-to-end in analyze_portfolio
    dates = pd.bdate_range("2025-01-03", periods=160)
    aec_vals = np.linspace(98.0, 100.0, len(dates))
    aec_series = pd.Series(aec_vals, index=dates, name="AEC3D")
    spy_series = pd.Series(np.linspace(500.0, 540.0, len(dates)), index=dates, name="SPY")

    ons_snap = IOLPanelSnapshot(
        panel_source="iol_ons", url="mock", observed_at="2025-08-14T20:00:00+00:00", column_headers=(),
        rows_by_ticker={"AEC3D": IOLPanelRow(ticker="AEC3D", name="AES Arg Cl 3", last_price=100.0, market_timestamp="14/08/2025 16:50")},
    )
    cfg_mapped = PortfolioAnalysisConfig(
        variable_income=(InstrumentConfig(instrument_id="SPY", ticker="SPY", section="variable_income", asset_subtype="equity_etf"),),
        fixed_income=(
            InstrumentConfig(
                instrument_id="MY_AES_ON",
                ticker="AES_INTERNAL_TICKER",
                iol_symbol="AEC3D",
                section="fixed_income",
                asset_subtype="corporate_bond_on",
                quote_source="iol_ons",
            ),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=20),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
        as_of_date=date(2025, 8, 14),
    )
    res_mapped = analyze_portfolio(
        cfg_mapped,
        prices_usd_override=pd.DataFrame({"SPY": spy_series}),
        iol_snapshots_override={"iol_ons": ons_snap},
        iol_history_fetcher=lambda **kw: IOLHistoricalSeries(
            symbol=kw["symbol"], market=kw["market"], id_titulo="163723",
            requested_start_date=dates[0].date(), requested_end_date=dates[-1].date(),
            effective_start_date=dates[0].date(), effective_end_date=dates[-1].date(),
            frequency="daily", price_field="close", series_type="market_price",
            currency=kw["currency"], usd_settlement_type=kw["usd_settlement_type"],
            quote_unit=kw["quote_unit"], observed_at="2025-08-14T20:00:00+00:00",
            source_url="mock", bars=pd.DataFrame({"close": aec_vals}, index=dates),
            price_series=aec_series, raw_row_count=len(dates), valid_row_count=len(dates),
        ),
    )
    assert res_mapped.quotes_and_data_quality["iol_quotes"]["MY_AES_ON"]["species_used"] == "AEC3D"
    assert res_mapped.quotes_and_data_quality["iol_historical_metadata"]["MY_AES_ON"]["symbol"] == "AEC3D"

    # HIGH-02: Explicit iol_symbol="AL30C" resolves quote & history consistently on AL30C (not AL30D quote + AL30C history)
    bonos_both = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at="2025-08-14T20:00:00+00:00", column_headers=(),
        rows_by_ticker={
            "AL30D": IOLPanelRow(ticker="AL30D", name="Bono 2030", last_price=62.50, market_timestamp="14/08/2025 16:50"),
            "AL30C": IOLPanelRow(ticker="AL30C", name="Bono 2030", last_price=60.10, market_timestamp="14/08/2025 16:50"),
        },
    )
    cfg_cable = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="AL30_C", ticker="AL30", iol_symbol="AL30C",
                iol_species_map={"C": "AL30C", "D": "AL30D"},
                section="fixed_income", asset_subtype="sovereign_bond", quote_source="iol_bonos",
            ),
        ),
        iol=IOLConfig(allow_usd_cable_history=True),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=20),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
        as_of_date=date(2025, 8, 14),
    )
    res_cable = analyze_portfolio(
        cfg_cable,
        iol_snapshots_override={"iol_bonos": bonos_both},
        iol_history_fetcher=lambda **kw: IOLHistoricalSeries(
            symbol=kw["symbol"], market=kw["market"], id_titulo="99927",
            requested_start_date=dates[0].date(), requested_end_date=dates[-1].date(),
            effective_start_date=dates[0].date(), effective_end_date=dates[-1].date(),
            frequency="daily", price_field="close", series_type="market_price",
            currency=kw["currency"], usd_settlement_type=kw["usd_settlement_type"],
            quote_unit=kw["quote_unit"], observed_at="2025-08-14T20:00:00+00:00",
            source_url="mock", bars=pd.DataFrame({"close": aec_vals}, index=dates),
            price_series=pd.Series(np.linspace(55.0, 60.10, len(dates)), index=dates, name="AL30C"),
            raw_row_count=len(dates), valid_row_count=len(dates),
        ),
    )
    assert res_cable.quotes_and_data_quality["iol_quotes"]["AL30_C"]["species_used"] == "AL30C"
    assert res_cable.quotes_and_data_quality["iol_quotes"]["AL30_C"]["usd_settlement_type"] == "USD_CABLE_C"
    assert res_cable.quotes_and_data_quality["iol_historical_metadata"]["AL30_C"]["symbol"] == "AL30C"

    # HIGH-03: Future live IOL snapshot (2026-11-26) MUST NOT leak into historical as_of_date (2025-07-25) YTM/valuation/rebalancing
    terms = FixedIncomeTerms(
        cashflow_currency="USD", reference_nominal=100.0,
        settlement_date=date(2025, 7, 25), maturity_date=date(2026, 7, 25),
        coupon_structure="zero_coupon", price_convention="per_100_original_nominal", price_type="dirty",
        cashflows=(CashFlowEvent(payment_date=date(2026, 7, 25), amortization_amount=100.0),),
    )
    hist_dates = pd.bdate_range("2025-01-03", "2025-07-25")
    hist_vals = np.linspace(80.0, 90.0, len(hist_dates))  # 2025-07-25 close == 90.0
    future_snap = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at="2026-11-26T18:00:00+00:00", column_headers=(),
        rows_by_ticker={"BPA8D": IOLPanelRow(ticker="BPA8D", name="BOPREAL 2028", last_price=99.50, market_timestamp="26/11/2026 16:50")},
    )
    cfg_past = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="BPA8D", ticker="BPA8D", section="fixed_income", asset_subtype="sovereign_bond",
                quote_source="iol_bonos", fixed_income_terms=terms, default_buy_broker_id="IOL",
            ),
        ),
        brokers=(BrokerConfig(broker_id="IOL"),),
        positions=(BrokerPosition(instrument_id="BPA8D", broker_id="IOL", quantity=1000.0),),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=20),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
        as_of_date=date(2025, 7, 25),
    )
    res_past = analyze_portfolio(
        cfg_past,
        iol_snapshots_override={"iol_bonos": future_snap},
        iol_history_fetcher=lambda **kw: IOLHistoricalSeries(
            symbol="BPA8D", market="BCBA", id_titulo="154636",
            requested_start_date=hist_dates[0].date(), requested_end_date=date(2025, 7, 25),
            effective_start_date=hist_dates[0].date(), effective_end_date=date(2025, 7, 25),
            frequency="daily", price_field="close", series_type="market_price",
            currency="USD", usd_settlement_type="USD_LOCAL_D", quote_unit="per_100_original_nominal",
            observed_at="2026-11-26T18:00:00+00:00", source_url="mock",
            bars=pd.DataFrame({"close": hist_vals}, index=hist_dates),
            price_series=pd.Series(hist_vals, index=hist_dates, name="BPA8D"),
            raw_row_count=len(hist_dates), valid_row_count=len(hist_dates),
        ),
    )
    # Must use 2025-07-25 historical close (90.0), NOT future 2026-11-26 live snapshot (99.50)!
    assert res_past.quotes_and_data_quality["latest_prices_usd"]["BPA8D"] == pytest.approx(90.0)
    assert res_past.quotes_and_data_quality["execution_prices_usd_per_unit"]["BPA8D"] == pytest.approx(0.90)
    assert res_past.quotes_and_data_quality["price_resolution"]["BPA8D"]["source"] == "iol_historical"
    assert res_past.fixed_income_metrics["BPA8D"].quoted_price == pytest.approx(90.0)
    assert res_past.fixed_income_metrics["BPA8D"].ytm_annual == pytest.approx(100.0 / 90.0 - 1.0, abs=1e-6)
    assert any("strictly after as_of_cutoff" in w for w in res_past.warnings)

    # MED-03: Preloaded history warnings are NOT mutated in place; cache_key distinguishes duplicate_conflict_policy
    pre_obj = IOLHistoricalSeries(
        symbol="GD35C", market="BCBA", id_titulo="99",
        requested_start_date=dates[0].date(), requested_end_date=dates[-1].date(),
        effective_start_date=dates[0].date(), effective_end_date=dates[-1].date(),
        frequency="daily", price_field="close", series_type="market_price",
        currency="USD_CABLE", usd_settlement_type="USD_CABLE_C", quote_unit="per_100_original_nominal",
        observed_at="2025-08-14T20:00:00+00:00", source_url="mock",
        bars=pd.DataFrame({"close": aec_vals}, index=dates),
        price_series=aec_series, raw_row_count=len(dates), valid_row_count=len(dates),
        warnings=["orig"],
    )
    out_map = fetch_iol_historical_prices_for_instruments(
        [InstrumentConfig(instrument_id="GD35C", ticker="GD35C", section="fixed_income", asset_subtype="sovereign_bond", quote_source="iol_bonos", currency="USD_CABLE")],
        iol_config=IOLConfig(allow_usd_cable_history=True),
        preloaded_history={"GD35C": pre_obj},
    )
    assert pre_obj.warnings == ["orig"]
    assert len(out_map["GD35C"].warnings) == 2

    # LOW-01 & LOW-02: Out-of-range duplicate conflict does NOT fail in-range fetch, AND as_of_date == today_art before 17:00 ART still drops intraday bar
    rows_lo = [
        ("11/28/2026", "99.00", "99.00", "99.00", "99.00", "99.00", "1,000.00", "1,000.00"),  # out of range dup 1
        ("11/28/2026", "95.00", "95.00", "95.00", "95.00", "95.00", "1,000.00", "1,000.00"),  # out of range dup 2 (conflict!)
        ("11/27/2026", "99.50", "99.80", "99.10", "99.60", "99.60", "5,000.00", "5,000.00"),  # today intraday (13:00 ART)
        ("11/26/2026", "99.70", "99.90", "98.75", "99.40", "99.40", "40,558.76", "40,876.00"),
    ]
    sess_lo = MagicMock()
    sess_lo.get.return_value = _mock_response(200, _make_iol_get_html("AEC3D", "163723", rows_lo))
    sess_lo.post.return_value = _mock_response(200, _make_iol_post_html(rows_lo))
    res_lo = fetch_iol_historical_prices(
        "AEC3D",
        start_date=date(2026, 11, 20),
        end_date=date(2026, 11, 27),
        as_of_date=date(2026, 11, 27),
        session=sess_lo,
        sleep_fn=lambda _: None,
        now_fn=lambda tz=None: datetime(2026, 11, 27, 13, 0, tzinfo=tz),
    )
    assert res_lo.out_of_range_rows_dropped == 2
    assert res_lo.intraday_rows_dropped == 1
    assert res_lo.duplicate_dates_resolved == 0
    assert res_lo.effective_end_date == date(2026, 11, 26)
    assert res_lo.valid_row_count == 1


# =============================================================================
# 9. Audit remediation regressions (AUD-IOL-01 .. AUD-IOL-10)
# =============================================================================

def test_aud_iol_01_zero_volume_forward_fill_filtering_and_preserved_valid_cases():
    fixed_now = lambda tz=None: datetime(2026, 11, 27, 18, 0, tzinfo=tz)

    # Real IOL pattern: stale carried-forward rows with vol_amount=0.00 & vol_nominal=0.00
    # plus a genuine trade where vol_amount>0 & vol_nominal==0 (e.g. AL30D 2023-09-19).
    rows = [
        ("11/26/2026", "15.50", "16.00", "15.50", "15.85", "15.85", "0.00", "0.00"),      # stale forward-fill -> drop
        ("11/25/2026", "15.50", "16.00", "15.50", "15.85", "15.85", "0.00", "0.00"),      # stale forward-fill -> drop
        ("11/24/2026", "15.50", "16.00", "15.50", "15.85", "15.85", "12,450.00", "0.00"), # positive amount -> KEEP
        ("11/20/2026", "0.00", "0.00", "0.00", "0.00", "0.00", "0.00", "0.00"),           # zero price -> drop
        ("11/19/2026", "15.10", "15.60", "15.00", "15.50", "15.50", "9,300.00", "6,000.00"),
    ]
    sess = MagicMock()
    sess.get.return_value = _mock_response(200, _make_iol_get_html("YMCHD", "102835", rows))
    sess.post.return_value = _mock_response(200, _make_iol_post_html(rows))

    res = fetch_iol_historical_prices(
        "YMCHD",
        start_date=date(2026, 11, 18),
        end_date=date(2026, 11, 26),
        session=sess,
        sleep_fn=lambda _: None,
        now_fn=fixed_now,
    )
    assert res.valid_row_count == 2
    assert res.zero_or_no_trade_rows_dropped == 3
    assert res.zero_volume_no_trade_rows_dropped == 2
    assert list(res.price_series.index) == [pd.Timestamp("2026-11-19"), pd.Timestamp("2026-11-24")]
    assert any("zero-volume carried-forward" in w for w in res.warnings)
    assert res.to_metadata_dict()["zero_volume_no_trade_rows_dropped"] == 2

    # Tables without volume columns keep valid positive closes cleanly
    no_vol_html = """
    <table id="tbcotizaciones">
      <tr><th>Fecha Cotizaci&oacute;n</th><th>Cierre</th></tr>
      <tr><td>11/26/2026</td><td>99.40</td></tr>
      <tr><td>11/25/2026</td><td>99.85</td></tr>
    </table>
    """
    sess_nv = MagicMock()
    sess_nv.get.return_value = _mock_response(200, _make_iol_get_html("AEC3D", "163723", []))
    sess_nv.post.return_value = _mock_response(200, no_vol_html)
    res_nv = fetch_iol_historical_prices(
        "AEC3D", start_date=date(2026, 11, 20), end_date=date(2026, 11, 26),
        session=sess_nv, sleep_fn=lambda _: None, now_fn=fixed_now,
    )
    assert res_nv.valid_row_count == 2
    assert res_nv.zero_volume_no_trade_rows_dropped == 0


def test_aud_iol_02_corrupt_negative_nonfinite_prices_and_leading_zero_thousands():
    from portfolio_optimization import parse_argentine_number

    # Leading-zero thousands groupings MUST return None (not 123.45)
    assert parse_argentine_number("0.123,45", locale_mode="es-AR") is None
    assert parse_argentine_number("0,123.45", locale_mode="en-US") is None
    assert parse_argentine_number("0.000.000", locale_mode="es-AR") is None
    assert parse_argentine_number("0,000,000", locale_mode="en-US") is None

    # Negative close price fails closed with iol_invalid_price_data
    neg_html = _make_iol_post_html([
        ("11/26/2026", "99.70", "99.90", "98.75", "-49.40", "-49.40", "40,558.76", "40,876.00"),
    ])
    with pytest.raises(IOLHistoryError) as exc_neg:
        parse_iol_historical_table(neg_html, symbol="AL30D")
    assert exc_neg.value.code == "iol_invalid_price_data"

    # Non-finite / garbage price cell fails closed (never silently dropped as no-trade)
    for bad_cell in ("NaN", "Infinity", "#VALUE!", "12.34.56", "ERROR"):
        bad_html = _make_iol_post_html([
            ("11/26/2026", "99.70", "99.90", "98.75", bad_cell, "99.40", "40,558.76", "40,876.00"),
        ])
        with pytest.raises(IOLHistoryError) as exc_bad:
            parse_iol_historical_table(bad_html, symbol="AL30D")
        assert exc_bad.value.code == "iol_invalid_price_data"

    # Negative volume fails closed
    neg_vol_html = _make_iol_post_html([
        ("11/26/2026", "99.70", "99.90", "98.75", "99.40", "99.40", "-100.00", "40,876.00"),
    ])
    with pytest.raises(IOLHistoryError) as exc_nv:
        parse_iol_historical_table(neg_vol_html, symbol="AL30D")
    assert exc_nv.value.code == "iol_invalid_price_data"


def test_aud_iol_03_fail_closed_unparseable_snapshot_timestamp_and_backtest_no_leak():
    hist_dates = pd.bdate_range("2025-01-03", "2025-07-25")
    hist_vals = np.linspace(80.0, 90.0, len(hist_dates))

    # Time-only market_timestamp ("16:50") fails _parse_iol_market_timestamp -> MUST fall back to
    # observed_at (2026-11-26 > 2025-07-25) and exclude the future snapshot!
    snap_time_only = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at="2026-11-26T18:00:00+00:00", column_headers=(),
        rows_by_ticker={"BPA8D": IOLPanelRow(ticker="BPA8D", name="BOPREAL 2028", last_price=99.50, market_timestamp="16:50")},
    )
    # Completely unparseable market_timestamp AND unparseable observed_at -> MUST fail closed!
    snap_unverifiable = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at="not-a-timestamp", column_headers=(),
        rows_by_ticker={"BPA8D": IOLPanelRow(ticker="BPA8D", name="BOPREAL 2028", last_price=99.50, market_timestamp="invalid-ts")},
    )
    cfg_past = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="BPA8D", ticker="BPA8D", section="fixed_income", asset_subtype="sovereign_bond",
                quote_source="iol_bonos",
            ),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=20),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
        as_of_date=date(2025, 7, 25),
    )

    def _hist_fetcher(**kw: Any) -> IOLHistoricalSeries:
        return IOLHistoricalSeries(
            symbol="BPA8D", market="BCBA", id_titulo="154636",
            requested_start_date=hist_dates[0].date(), requested_end_date=date(2025, 7, 25),
            effective_start_date=hist_dates[0].date(), effective_end_date=date(2025, 7, 25),
            frequency="daily", price_field="close", series_type="market_price",
            currency="USD", usd_settlement_type="USD_LOCAL_D", quote_unit="per_100_original_nominal",
            observed_at="2026-11-26T18:00:00+00:00", source_url="mock",
            bars=pd.DataFrame({"close": hist_vals}, index=hist_dates),
            price_series=pd.Series(hist_vals, index=hist_dates, name="BPA8D"),
            raw_row_count=len(hist_dates), valid_row_count=len(hist_dates),
        )

    for bad_snap in (snap_time_only, snap_unverifiable):
        res = analyze_portfolio(cfg_past, iol_snapshots_override={"iol_bonos": bad_snap}, iol_history_fetcher=_hist_fetcher)
        assert res.quotes_and_data_quality["latest_prices_usd"]["BPA8D"] == pytest.approx(90.0)
        assert res.quotes_and_data_quality["price_resolution"]["BPA8D"]["source"] == "iol_historical"


def test_aud_iol_04_preloaded_history_clipping_currency_guard_and_matured_backtest():
    # Preloaded history MUST enforce as_of_date clipping and reject unverified currency=None
    full_dates = pd.bdate_range("2025-01-03", "2026-06-30")
    full_vals = np.linspace(80.0, 95.0, len(full_dates))
    pre_unverified = IOLHistoricalSeries(
        symbol="BPA8D", market="BCBA", id_titulo="154636",
        requested_start_date=full_dates[0].date(), requested_end_date=full_dates[-1].date(),
        effective_start_date=full_dates[0].date(), effective_end_date=full_dates[-1].date(),
        frequency="daily", price_field="close", series_type="market_price",
        currency=None, usd_settlement_type=None, quote_unit="per_100_original_nominal",
        observed_at="2026-11-26T18:00:00+00:00", source_url="mock",
        bars=pd.DataFrame({"close": full_vals}, index=full_dates),
        price_series=pd.Series(full_vals, index=full_dates, name="BPA8D"),
        raw_row_count=len(full_dates), valid_row_count=len(full_dates),
    )
    inst_usd = InstrumentConfig(
        instrument_id="YMCHD", ticker="YMCHD", section="fixed_income", asset_subtype="corporate_bond_on",
        quote_source="iol_ons", currency="USD",
    )
    with pytest.raises(IOLHistoryError) as exc_none_ccy:
        fetch_iol_historical_prices_for_instruments(
            [inst_usd], as_of_date=date(2025, 12, 31), preloaded_history={"YMCHD": pre_unverified},
        )
    assert exc_none_ccy.value.code == "iol_currency_incompatible"

    # Valid USD preloaded history is strictly clipped <= as_of_date without mutating caller's object
    pre_usd = IOLHistoricalSeries(
        symbol="YMCHD", market="BCBA", id_titulo="102835",
        requested_start_date=full_dates[0].date(), requested_end_date=full_dates[-1].date(),
        effective_start_date=full_dates[0].date(), effective_end_date=full_dates[-1].date(),
        frequency="daily", price_field="close", series_type="market_price",
        currency="USD", usd_settlement_type="USD_LOCAL_D", quote_unit="per_100_original_nominal",
        observed_at="2026-11-26T18:00:00+00:00", source_url="mock",
        bars=pd.DataFrame({"close": full_vals}, index=full_dates),
        price_series=pd.Series(full_vals, index=full_dates, name="YMCHD"),
        raw_row_count=len(full_dates), valid_row_count=len(full_dates),
    )
    # Strictly historical as_of_date with explicit currency="USD" skips live future panel HTTP fetch!
    cfg_matured = PortfolioAnalysisConfig(
        fixed_income=(inst_usd,),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=20),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
        as_of_date=date(2025, 12, 31),
    )
    with patch("portfolio_optimization.iol.fetch_iol_panel", side_effect=AssertionError("Should not fetch live panel")):
        res_mat = analyze_portfolio(cfg_matured, iol_history_override={"YMCHD": pre_usd})

    assert res_mat.quotes_and_data_quality["iol_historical_metadata"]["YMCHD"]["effective_end_date"] <= "2025-12-31"
    assert res_mat.prices_hist.index.max() <= pd.Timestamp("2025-12-31")
    assert len(pre_usd.price_series) == len(full_dates)  # caller object untouched
    assert any("price return only" in w.lower() for w in res_mat.warnings)
    assert any("Skipped live IOL panel snapshot fetch" in w for w in res_mat.warnings)


def test_aud_iol_05_truncation_detection_recovery_and_trailing_partial_coverage():
    fixed_now = lambda tz=None: datetime(2026, 11, 27, 18, 0, tzinfo=tz)

    # Simulate a server capping POST at 500 rows for a 3-year window, then returning the remaining 100 rows on probe
    all_bdays = pd.bdate_range("2023-01-02", periods=600)
    recent_500 = [
        (d.strftime("%m/%d/%Y"), "50.00", "51.00", "49.00", "50.00", "50.00", "10,000.00", "20,000.00")
        for d in reversed(all_bdays[100:])
    ]
    older_100 = [
        (d.strftime("%m/%d/%Y"), "45.00", "46.00", "44.00", "45.00", "45.00", "10,000.00", "22,000.00")
        for d in reversed(all_bdays[:100])
    ]
    assert len(recent_500) == 500

    sess = MagicMock()
    sess.get.return_value = _mock_response(200, _make_iol_get_html("AL30D", "99926", recent_500[:10]))
    sess.post.side_effect = [
        _mock_response(200, _make_iol_post_html(recent_500)),
        _mock_response(200, _make_iol_post_html(older_100)),
    ]
    res_rec = fetch_iol_historical_prices(
        "AL30D",
        start_date=date(2023, 1, 2),
        end_date=all_bdays[-1].date(),
        iol_config=IOLConfig(historical_request_delay_seconds=0.0),
        session=sess,
        sleep_fn=lambda _: None,
        now_fn=fixed_now,
    )
    assert sess.post.call_count == 2
    assert res_rec.truncation_splits_performed == 1
    assert res_rec.valid_row_count == 600
    assert res_rec.effective_start_date == date(2023, 1, 2)
    assert any("truncation" in w.lower() for w in res_rec.warnings)

    # Persistent truncation exceeding max splits raises typed iol_truncated_response
    sess_stuck = MagicMock()
    sess_stuck.get.return_value = _mock_response(200, _make_iol_get_html("AL30D", "99926", recent_500[:10]))
    sess_stuck.post.return_value = _mock_response(200, _make_iol_post_html(recent_500))
    with pytest.raises(IOLHistoryError) as exc_tr:
        fetch_iol_historical_prices(
            "AL30D",
            start_date=date(2020, 1, 1),
            end_date=all_bdays[-1].date(),
            iol_config=IOLConfig(historical_chunk_days=3000, historical_request_delay_seconds=0.0),
            session=sess_stuck,
            sleep_fn=lambda _: None,
            now_fn=fixed_now,
        )
    assert exc_tr.value.code == "iol_truncated_response"

    # Trailing matured/stale gap sets partial_coverage=True even when start_date is fully covered
    mat_rows = [
        ("05/08/2026", "15.50", "16.00", "15.50", "15.85", "15.85", "5,000.00", "30,000.00"),
        ("01/02/2026", "14.50", "15.00", "14.50", "14.80", "14.80", "5,000.00", "30,000.00"),
    ]
    sess_mat = MagicMock()
    sess_mat.get.return_value = _mock_response(200, _make_iol_get_html("YMCHD", "102835", mat_rows))
    sess_mat.post.return_value = _mock_response(200, _make_iol_post_html(mat_rows))
    res_tail = fetch_iol_historical_prices(
        "YMCHD", start_date=date(2026, 1, 1), end_date=date(2026, 11, 26),
        session=sess_mat, sleep_fn=lambda _: None, now_fn=fixed_now,
    )
    assert res_tail.partial_coverage is True
    assert any("trailing gap" in w.lower() for w in res_tail.warnings)


def test_aud_iol_06_ars_sibling_bypass_blocked_and_species_pinning():
    from portfolio_optimization import IOLQuoteError, fetch_iol_quotes_for_instruments

    bonos_snap = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at="2026-11-26T15:00:00+00:00", column_headers=(),
        rows_by_ticker={
            "AL30": IOLPanelRow(ticker="AL30", name="Bono 2030", last_price=82000.0, market_timestamp="26/11/2026 15:00"),
            "AL30D": IOLPanelRow(ticker="AL30D", name="Bono 2030", last_price=49.40, market_timestamp="26/11/2026 15:00"),
            "AL30C": IOLPanelRow(ticker="AL30C", name="Bono 2030", last_price=47.00, market_timestamp="26/11/2026 15:00"),
        },
    )
    # Explicitly pinning iol_symbol="AL30" (ARS sibling of AL30D) with currency="USD" MUST be rejected!
    inst_ars_bypass = InstrumentConfig(
        instrument_id="AL30_BAD", ticker="AL30", iol_symbol="AL30", currency="USD",
        section="fixed_income", asset_subtype="sovereign_bond", quote_source="iol_bonos",
    )
    q_map = fetch_iol_quotes_for_instruments([inst_ars_bypass], preloaded_snapshots={"iol_bonos": bonos_snap})
    with pytest.raises(IOLHistoryError) as exc_ars:
        fetch_iol_historical_prices_for_instruments([inst_ars_bypass], iol_quotes=q_map)
    assert exc_ars.value.code == "iol_currency_incompatible"

    # Explicit iol_symbol="AL30D" absent from panel MUST NOT silently fall back to AL30C/ARS
    bonos_no_d = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at="2026-11-26T15:00:00+00:00", column_headers=(),
        rows_by_ticker={
            "AL30": IOLPanelRow(ticker="AL30", name="Bono 2030", last_price=82000.0, market_timestamp="26/11/2026 15:00"),
            "AL30C": IOLPanelRow(ticker="AL30C", name="Bono 2030", last_price=47.00, market_timestamp="26/11/2026 15:00"),
        },
    )
    inst_pin_d = InstrumentConfig(
        instrument_id="MY_AL30D", ticker="AL30", iol_symbol="AL30D",
        section="fixed_income", asset_subtype="sovereign_bond", quote_source="iol_bonos",
    )
    with pytest.raises(IOLQuoteError):
        fetch_iol_quotes_for_instruments([inst_pin_d], preloaded_snapshots={"iol_bonos": bonos_no_d})

    # Currency conflict between declared USD and species C is rejected even when allow_usd_cable_history=True
    inst_conflict = InstrumentConfig(
        instrument_id="AL30_CF", ticker="AL30", iol_symbol="AL30C", iol_species_map={"C": "AL30C"}, currency="USD",
        section="fixed_income", asset_subtype="sovereign_bond", quote_source="iol_bonos",
    )
    q_cf = fetch_iol_quotes_for_instruments([inst_conflict], preloaded_snapshots={"iol_bonos": bonos_snap})
    with pytest.raises(IOLHistoryError) as exc_cf:
        fetch_iol_historical_prices_for_instruments(
            [inst_conflict], iol_quotes=q_cf, iol_config=IOLConfig(allow_usd_cable_history=True),
        )
    assert exc_cf.value.code == "iol_currency_incompatible"


def test_aud_iol_07_adjusted_close_does_not_leak_into_tradable_price_ytm_or_rebalancing():
    dates = pd.bdate_range("2025-01-03", "2025-07-25")
    unadj_close = np.linspace(80.0, 90.0, len(dates))
    adj_close = unadj_close * 0.85  # synthetic divergence: 76.50 vs 90.00 on final date

    terms = FixedIncomeTerms(
        cashflow_currency="USD", reference_nominal=100.0,
        settlement_date=date(2025, 7, 25), maturity_date=date(2026, 7, 25),
        coupon_structure="zero_coupon", price_convention="per_100_original_nominal", price_type="dirty",
        cashflows=(CashFlowEvent(payment_date=date(2026, 7, 25), amortization_amount=100.0),),
    )
    future_snap = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at="2026-11-26T18:00:00+00:00", column_headers=(),
        rows_by_ticker={"BPA8D": IOLPanelRow(ticker="BPA8D", name="BOPREAL 2028", last_price=99.50, market_timestamp="26/11/2026 16:50")},
    )
    cfg = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="BPA8D", ticker="BPA8D", section="fixed_income", asset_subtype="sovereign_bond",
                quote_source="iol_bonos", fixed_income_terms=terms, default_buy_broker_id="IOL",
            ),
        ),
        brokers=(BrokerConfig(broker_id="IOL"),),
        positions=(BrokerPosition(instrument_id="BPA8D", broker_id="IOL", quantity=1000.0),),
        iol=IOLConfig(historical_price_field="adjusted_close"),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=20),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
        as_of_date=date(2025, 7, 25),
    )
    res = analyze_portfolio(
        cfg,
        iol_snapshots_override={"iol_bonos": future_snap},
        iol_history_fetcher=lambda **kw: IOLHistoricalSeries(
            symbol="BPA8D", market="BCBA", id_titulo="154636",
            requested_start_date=dates[0].date(), requested_end_date=date(2025, 7, 25),
            effective_start_date=dates[0].date(), effective_end_date=date(2025, 7, 25),
            frequency="daily", price_field="adjusted_close", series_type="market_price",
            currency="USD", usd_settlement_type="USD_LOCAL_D", quote_unit="per_100_original_nominal",
            observed_at="2026-11-26T18:00:00+00:00", source_url="mock",
            bars=pd.DataFrame({"close": unadj_close, "adjusted_close": adj_close}, index=dates),
            price_series=pd.Series(adj_close, index=dates, name="BPA8D"),
            raw_row_count=len(dates), valid_row_count=len(dates), adjusted_close_differs_count=len(dates),
        ),
    )
    # Tradable price, YTM, and execution price MUST use unadjusted close (90.0), NOT adjusted_close (76.5)!
    assert res.quotes_and_data_quality["latest_prices_usd"]["BPA8D"] == pytest.approx(90.0)
    assert res.quotes_and_data_quality["execution_prices_usd_per_unit"]["BPA8D"] == pytest.approx(0.90)
    assert res.quotes_and_data_quality["price_resolution"]["BPA8D"]["usd_settlement_type"] == "USD_LOCAL_D"
    assert res.fixed_income_metrics["BPA8D"].quoted_price == pytest.approx(90.0)
    assert res.fixed_income_metrics["BPA8D"].ytm_annual == pytest.approx(100.0 / 90.0 - 1.0, abs=1e-6)


def test_aud_iol_08_09_10_url_security_http_date_retry_after_bounded_backoff_dates_and_dedup():
    # AUD-IOL-08: Non-HTTPS, credentialed, and whitespace URLs rejected in IOLConfig
    for bad_url in (
        "http://iol.invertironline.com/Titulo/DatosHistoricos",
        "file:///etc/passwd",
        "https://user:pass@iol.invertironline.com/Titulo/DatosHistoricos",
        "https://iol.invertironline.com/Titulo/DatosHistoricos\r\nX-Injected: 1",
    ):
        with pytest.raises(ConfigurationError):
            IOLConfig(historical_url=bad_url)

    # AUD-IOL-08: HTTP redirect downgrade rejected fail-closed
    sess_redir = MagicMock()
    sess_redir.get.return_value = _mock_response(200, "<html></html>", url="http://insecure.example/downgrade")
    with pytest.raises(IOLHistoryError) as exc_rd:
        fetch_iol_historical_prices("AEC3D", start_date=date(2026, 11, 1), end_date=date(2026, 11, 26), session=sess_redir, sleep_fn=lambda _: None)
    assert exc_rd.value.code == "iol_insecure_redirect"

    # AUD-IOL-09: HTTP-date Retry-After parsed accurately + exponential backoff capped at 10s
    fixed_utc = datetime(2026, 11, 27, 18, 0, 0, tzinfo=ZoneInfo("UTC"))
    cfg_cap = IOLConfig(historical_max_retries=4, historical_retry_backoff_seconds=4.0, historical_request_delay_seconds=0.0)
    sess_ra = MagicMock()
    sess_ra.get.side_effect = [
        _mock_response(429, "Too Many Requests", headers={"Retry-After": "Fri, 27 Nov 2026 18:00:07 GMT"}),
        requests.Timeout("slow"),
        requests.Timeout("slow"),
        requests.Timeout("slow"),
        requests.Timeout("slow"),
    ]
    sleeps: List[float] = []
    with pytest.raises(IOLHistoryError) as exc_to:
        fetch_iol_historical_prices(
            "AEC3D", start_date=date(2026, 11, 1), end_date=date(2026, 11, 26),
            iol_config=cfg_cap, session=sess_ra, sleep_fn=sleeps.append,
            now_fn=lambda tz=None: fixed_utc.astimezone(tz) if tz else fixed_utc,
        )
    assert exc_to.value.code == "iol_timeout"
    assert sleeps == pytest.approx([7.0, 8.0, 10.0, 10.0])

    # AUD-IOL-10: Mixed ISO + slash dates rejected; duplicate tie-break uses volume_nominal; cache avoids wasted sleeps
    mixed_dt_html = _make_iol_post_html([
        ("2026-11-26", "99.00", "99.50", "98.50", "99.40", "99.40", "10,000.00", "10,000.00"),
        ("11/25/2026", "99.00", "99.50", "98.50", "99.50", "99.50", "10,000.00", "10,000.00"),
    ])
    with pytest.raises(IOLHistoryError) as exc_mix:
        parse_iol_historical_table(mixed_dt_html, symbol="AEC3D")
    assert exc_mix.value.code == "iol_schema_changed"

    # Secondary tie-breaker on volume_nominal (matching real IOL GGAL 05/17/2022 behavior)
    tie_rows = [
        ("11/26/2026", "99.00", "99.50", "98.50", "99.40", "99.40", "403,358,702.55", "1,892.00"),
        ("11/26/2026", "99.00", "99.50", "98.50", "99.40", "99.40", "403,358,702.55", "2,018,090.00"),
    ]
    sess_tie = MagicMock()
    sess_tie.get.return_value = _mock_response(200, _make_iol_get_html("TESTD", "999", tie_rows))
    sess_tie.post.return_value = _mock_response(200, _make_iol_post_html(tie_rows))
    res_tie = fetch_iol_historical_prices(
        "TESTD", start_date=date(2026, 11, 20), end_date=date(2026, 11, 26),
        session=sess_tie, sleep_fn=lambda _: None,
        now_fn=lambda tz=None: fixed_utc.astimezone(tz) if tz else fixed_utc,
    )
    assert res_tie.bars.loc[pd.Timestamp("2026-11-26"), "volume_nominal"] == pytest.approx(2018090.0)
    assert res_tie.observed_at == fixed_utc.isoformat()

    # Recent issuance with < min_weekly_observations fails cleanly with InsufficientHistoryError (not provider failure)
    short_dates = pd.bdate_range("2026-08-07", "2026-11-20")
    short_vals = np.linspace(98.0, 100.0, len(short_dates))
    cfg_short = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="AEC3D", ticker="AEC3D", section="fixed_income", asset_subtype="corporate_bond_on",
                quote_source="iol_ons", currency="USD",
            ),
        ),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        as_of_date=date(2026, 11, 20),
    )
    pre_short = IOLHistoricalSeries(
        symbol="AEC3D", market="BCBA", id_titulo="163723",
        requested_start_date=date(2021, 11, 20), requested_end_date=date(2026, 11, 20),
        effective_start_date=short_dates[0].date(), effective_end_date=short_dates[-1].date(),
        frequency="daily", price_field="close", series_type="market_price",
        currency="USD", usd_settlement_type="USD_LOCAL_D", quote_unit="per_100_original_nominal",
        observed_at="2026-11-20T21:00:00+00:00", source_url="mock",
        bars=pd.DataFrame({"close": short_vals}, index=short_dates),
        price_series=pd.Series(short_vals, index=short_dates, name="AEC3D"),
        raw_row_count=len(short_dates), valid_row_count=len(short_dates), partial_coverage=True,
    )
    with pytest.raises(InsufficientHistoryError, match="Insufficient aligned weekly returns"):
        analyze_portfolio(cfg_short, iol_history_override={"AEC3D": pre_short})


def test_aud_iol_concurrency_isolation_and_non_fixed_income_portfolio_preservation():
    """Verify concurrent analyze_portfolio runs stay stateless/isolated and pure-equity portfolios never call IOL."""
    from concurrent.futures import ThreadPoolExecutor

    dates = pd.bdate_range("2024-01-02", periods=320)
    rng = np.random.default_rng(42)
    spy_px = pd.Series(500.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.008, len(dates)))), index=dates, name="SPY")
    qqq_px = pd.Series(400.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.011, len(dates)))), index=dates, name="QQQ")
    amd_px = pd.Series(150.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.018, len(dates)))), index=dates, name="AMD")

    # 1. Pure equity portfolio (including trailing-'D' ticker AMD) MUST NEVER invoke any IOL endpoint
    cfg_eq = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(instrument_id="SPY", ticker="SPY", section="variable_income", asset_subtype="equity_etf"),
            InstrumentConfig(instrument_id="QQQ", ticker="QQQ", section="variable_income", asset_subtype="equity_etf"),
            InstrumentConfig(instrument_id="AMD", ticker="AMD", section="variable_income", asset_subtype="equity"),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        optimization=OptimizationConfig(strategies_to_run=("min_variance", "erc", "hrp"), selected_strategy="erc"),
        as_of_date=dates[-1].date(),
    )
    with patch("portfolio_optimization.iol.fetch_iol_panel", side_effect=AssertionError("Must not touch IOL panel")):
        with patch("portfolio_optimization.iol.fetch_iol_historical_prices", side_effect=AssertionError("Must not touch IOL history")):
            res_eq = analyze_portfolio(
                cfg_eq,
                prices_usd_override=pd.DataFrame({"SPY": spy_px, "QQQ": qqq_px, "AMD": amd_px}),
            )
    assert res_eq.quotes_and_data_quality["iol_quotes"] == {}
    assert res_eq.quotes_and_data_quality["iol_historical_metadata"] == {}
    assert sum(res_eq.strategies["erc"].weights_by_asset.values()) == pytest.approx(1.0, abs=1e-6)

    # 2. Concurrent multi-threaded execution of IOL historical fetches + analyze_portfolio stays isolated
    bpa_px_a = pd.Series(80.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.005, len(dates)))), index=dates, name="BPA8D")
    bpa_px_b = pd.Series(60.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.009, len(dates)))), index=dates, name="AL30D")

    def _run_worker(bond_id: str, bond_s: pd.Series) -> Tuple[str, Dict[str, float]]:
        cfg = PortfolioAnalysisConfig(
            variable_income=(InstrumentConfig(instrument_id="SPY", ticker="SPY", section="variable_income", asset_subtype="equity_etf"),),
            fixed_income=(
                InstrumentConfig(
                    instrument_id=bond_id, ticker=bond_id, section="fixed_income", asset_subtype="sovereign_bond",
                    quote_source="iol_bonos", currency="USD",
                ),
            ),
            black_litterman=BlackLittermanConfig(rf_rate=0.04),
            risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
            optimization=OptimizationConfig(strategies_to_run=("erc",), selected_strategy="erc"),
            as_of_date=dates[-1].date(),
        )
        pre = IOLHistoricalSeries(
            symbol=bond_id, market="BCBA", id_titulo="1",
            requested_start_date=dates[0].date(), requested_end_date=dates[-1].date(),
            effective_start_date=dates[0].date(), effective_end_date=dates[-1].date(),
            frequency="daily", price_field="close", series_type="market_price",
            currency="USD", usd_settlement_type="USD_LOCAL_D", quote_unit="per_100_original_nominal",
            observed_at="2026-11-26T18:00:00+00:00", source_url="mock",
            bars=pd.DataFrame({"close": bond_s.values}, index=dates),
            price_series=bond_s, raw_row_count=len(dates), valid_row_count=len(dates),
        )
        res = analyze_portfolio(cfg, prices_usd_override=pd.DataFrame({"SPY": spy_px}), iol_history_override={bond_id: pre})
        return bond_id, res.strategies["erc"].weights_by_asset

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [
            pool.submit(_run_worker, "BPA8D", bpa_px_a),
            pool.submit(_run_worker, "AL30D", bpa_px_b),
            pool.submit(_run_worker, "BPA8D", bpa_px_a),
            pool.submit(_run_worker, "AL30D", bpa_px_b),
        ]
        outs = [f.result() for f in futs]

    assert outs[0][1] == pytest.approx(outs[2][1], abs=1e-12)
    assert outs[1][1] == pytest.approx(outs[3][1], abs=1e-12)
    assert outs[0][1]["BPA8D"] != pytest.approx(outs[1][1]["AL30D"], abs=1e-3)


# =============================================================================
# 10. Independent Review Round-1 Remediation Regressions (REV-01 .. REV-05)
# =============================================================================

def test_rev_01_no_ars_as_usd_or_cable_bypass_on_past_as_of_or_mismatched_species():
    from portfolio_optimization import IOLQuoteError, fetch_iol_quotes_for_instruments

    dates = pd.bdate_range("2025-01-03", "2025-11-20")
    al30d_vals = np.linspace(45.0, 50.0, len(dates))
    fetched_symbols: List[str] = []

    def _recording_fetcher(**kw: Any) -> IOLHistoricalSeries:
        fetched_symbols.append(kw["symbol"])
        return IOLHistoricalSeries(
            symbol=kw["symbol"], market=kw["market"], id_titulo="99926",
            requested_start_date=dates[0].date(), requested_end_date=dates[-1].date(),
            effective_start_date=dates[0].date(), effective_end_date=dates[-1].date(),
            frequency="daily", price_field="close", series_type="market_price",
            currency=kw["currency"], usd_settlement_type=kw["usd_settlement_type"],
            quote_unit=kw["quote_unit"], observed_at="2025-11-20T21:00:00+00:00",
            source_url="mock", bars=pd.DataFrame({"close": al30d_vals}, index=dates),
            price_series=pd.Series(al30d_vals, index=dates, name=kw["symbol"]),
            raw_row_count=len(dates), valid_row_count=len(dates),
        )

    # 1a. Root ticker AL30 + currency="USD" + iol_species_map={"D":"AL30D","ARS":"AL30"} on past as_of_date
    #     MUST resolve AL30D (USD_LOCAL_D), NEVER ARS AL30!
    cfg_mapped = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="AL30", ticker="AL30", currency="USD",
                iol_species_map={"D": "AL30D", "ARS": "AL30"},
                section="fixed_income", asset_subtype="sovereign_bond", quote_source="iol_bonos",
            ),
        ),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=20),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
        as_of_date=date(2025, 11, 20),
    )
    with patch("portfolio_optimization.iol.fetch_iol_panel", side_effect=AssertionError("Should skip live panel")):
        res_m = analyze_portfolio(cfg_mapped, iol_history_fetcher=_recording_fetcher)
    assert fetched_symbols == ["AL30D"]
    assert res_m.quotes_and_data_quality["iol_historical_metadata"]["AL30"]["symbol"] == "AL30D"
    assert res_m.quotes_and_data_quality["iol_historical_metadata"]["AL30"]["usd_settlement_type"] == "USD_LOCAL_D"

    # 1b. Root ARS tickers (AL30, MTCGO) with currency="USD" and NO species map / NO panel quote -> FAIL CLOSED!
    for ars_root, src, sub in [("AL30", "iol_bonos", "sovereign_bond"), ("MTCGO", "iol_ons", "corporate_bond_on")]:
        inst_root = InstrumentConfig(
            instrument_id=ars_root, ticker=ars_root, currency="USD",
            section="fixed_income", asset_subtype=sub, quote_source=src,
        )
        with pytest.raises(IOLHistoryError) as exc_root:
            fetch_iol_historical_prices_for_instruments([inst_root], iol_quotes=None)
        assert exc_root.value.code == "iol_currency_incompatible"

    # 1c. Cable ticker AL30C with currency="USD" (with OR without iol_species_map={"C": "AL30C"}) and iol_quotes=None -> FAIL CLOSED!
    for sp_map in ({"C": "AL30C"}, None):
        inst_c_bad = InstrumentConfig(
            instrument_id="AL30_C", ticker="AL30C", currency="USD", iol_species_map=sp_map,
            section="fixed_income", asset_subtype="sovereign_bond", quote_source="iol_bonos",
        )
        with pytest.raises(IOLHistoryError) as exc_cb:
            fetch_iol_historical_prices_for_instruments([inst_c_bad], iol_quotes=None)
        assert exc_cb.value.code == "iol_currency_incompatible"

    # 1d. Mismatched iol_symbol="AL30C" vs iol_species_map={"D": "AL30D"} -> FAIL CLOSED in BOTH quotes and history!
    bonos_snap = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at="2026-11-26T15:00:00+00:00", column_headers=(),
        rows_by_ticker={
            "AL30D": IOLPanelRow(ticker="AL30D", name="Bono 2030", last_price=49.40, market_timestamp="26/11/2026 15:00"),
            "AL30C": IOLPanelRow(ticker="AL30C", name="Bono 2030", last_price=47.00, market_timestamp="26/11/2026 15:00"),
        },
    )
    inst_mismatch = InstrumentConfig(
        instrument_id="X_MIS", ticker="AL30", iol_symbol="AL30C", iol_species_map={"D": "AL30D"}, currency="USD",
        section="fixed_income", asset_subtype="sovereign_bond", quote_source="iol_bonos",
    )
    with pytest.raises(IOLQuoteError):
        fetch_iol_quotes_for_instruments([inst_mismatch], preloaded_snapshots={"iol_bonos": bonos_snap})
    with pytest.raises(IOLHistoryError) as exc_mm:
        fetch_iol_historical_prices_for_instruments([inst_mismatch], iol_quotes=None)
    assert exc_mm.value.code == "iol_currency_incompatible"

    # 1e. Unrelated ARS root iol_symbol="GD30" with currency="USD" when quote resolved AL30D -> FAIL CLOSED!
    inst_gd30_ars = InstrumentConfig(
        instrument_id="AL30_GD", ticker="AL30", iol_symbol="GD30", currency="USD",
        section="fixed_income", asset_subtype="sovereign_bond", quote_source="iol_bonos",
    )
    q_al30d = fetch_iol_quotes_for_instruments(
        [InstrumentConfig(instrument_id="AL30_GD", ticker="AL30", section="fixed_income", asset_subtype="sovereign_bond", quote_source="iol_bonos")],
        preloaded_snapshots={"iol_bonos": bonos_snap},
    )
    with pytest.raises(IOLHistoryError) as exc_gd:
        fetch_iol_historical_prices_for_instruments([inst_gd30_ars], iol_quotes=q_al30d)
    assert exc_gd.value.code == "iol_currency_incompatible"


def test_rev_02_excluded_future_iol_snapshot_never_leaks_quote_unit_100x():
    dates = pd.bdate_range("2025-01-03", "2025-07-25")
    hist_vals = np.linspace(0.55, 0.60, len(dates))

    future_snap = IOLPanelSnapshot(
        panel_source="iol_bonos", url="mock", observed_at="2026-11-26T18:00:00+00:00", column_headers=(),
        rows_by_ticker={"BPA8D": IOLPanelRow(ticker="BPA8D", name="BOPREAL 2028", last_price=83.19, market_timestamp="26/11/2026 16:50")},
    )
    cfg = PortfolioAnalysisConfig(
        fixed_income=(
            InstrumentConfig(
                instrument_id="BPA8D", ticker="BPA8D", section="fixed_income", asset_subtype="sovereign_bond",
                quote_source="iol_bonos", manual_price_usd=0.60, quote_unit="per_unit", default_buy_broker_id="IOL",
            ),
        ),
        brokers=(BrokerConfig(broker_id="IOL"),),
        positions=(BrokerPosition(instrument_id="BPA8D", broker_id="IOL", quantity=1000.0),),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=20),
        optimization=OptimizationConfig(strategies_to_run=("min_variance",), selected_strategy="min_variance"),
        as_of_date=date(2025, 7, 25),
    )
    res = analyze_portfolio(
        cfg,
        prices_usd_override=pd.DataFrame({"BPA8D": hist_vals}, index=dates),
        iol_snapshots_override={"iol_bonos": future_snap},
    )
    # Must preserve manual_price_usd=0.60 with quote_unit="per_unit" (0.60 USD/VN, NOT 0.006 from excluded snapshot!)
    assert res.quotes_and_data_quality["price_resolution"]["BPA8D"]["source"] == "manual_price_usd"
    assert res.quotes_and_data_quality["price_resolution"]["BPA8D"]["quote_unit"] == "per_unit"
    assert res.quotes_and_data_quality["execution_prices_usd_per_unit"]["BPA8D"] == pytest.approx(0.60)


def test_rev_03_pre_hop_https_redirect_validation_blocks_http_downgrade_and_intermediate_hops():
    from portfolio_optimization import IOLQuoteError, fetch_iol_panel

    # 3a. 302 Location: http://... blocked BEFORE sending the second hop (allow_redirects=False!)
    sess_302 = MagicMock()
    sess_302.get.return_value = _mock_response(
        302, "", url="https://iol.invertironline.com/Titulo/DatosHistoricos",
        headers={"Location": "http://downgrade.example/mid"},
    )
    with pytest.raises(IOLHistoryError) as exc_302:
        fetch_iol_historical_prices("AEC3D", start_date=date(2026, 11, 1), end_date=date(2026, 11, 26), session=sess_302, sleep_fn=lambda _: None)
    assert exc_302.value.code == "iol_insecure_redirect"
    assert sess_302.get.call_count == 1
    assert sess_302.get.call_args.kwargs["allow_redirects"] is False

    # 3b. Mocked response.history with intermediate HTTP hop (HTTPS -> HTTP -> HTTPS) rejected fail-closed
    sess_hist = MagicMock()
    resp_h = _mock_response(200, "<html></html>", url="https://iol.invertironline.com/Titulo/DatosHistoricos")
    resp_h.history = [MagicMock(url="http://intermediate.insecure.example/hop")]
    sess_hist.get.return_value = resp_h
    with pytest.raises(IOLHistoryError) as exc_hist:
        fetch_iol_historical_prices("AEC3D", start_date=date(2026, 11, 1), end_date=date(2026, 11, 26), session=sess_hist, sleep_fn=lambda _: None)
    assert exc_hist.value.code == "iol_insecure_redirect"

    # 3c. Valid HTTPS -> HTTPS redirect succeeds cleanly without automatic requests redirects
    good_rows = [("11/26/2026", "99.70", "99.90", "98.75", "99.40", "99.40", "40,558.76", "40,876.00")]
    sess_ok = MagicMock()
    sess_ok.get.side_effect = [
        _mock_response(302, "", url="https://iol.invertironline.com/Titulo/DatosHistoricos", headers={"Location": "/Titulo/DatosHistoricos?simbolo=AEC3D&mercado=BCBA"}),
        _mock_response(200, _make_iol_get_html("AEC3D", "163723", good_rows), url="https://iol.invertironline.com/Titulo/DatosHistoricos?simbolo=AEC3D&mercado=BCBA"),
    ]
    sess_ok.post.return_value = _mock_response(200, _make_iol_post_html(good_rows))
    res_ok = fetch_iol_historical_prices(
        "AEC3D", start_date=date(2026, 11, 20), end_date=date(2026, 11, 26),
        session=sess_ok, sleep_fn=lambda _: None,
        now_fn=lambda tz=None: datetime(2026, 11, 27, 18, 0, tzinfo=tz),
    )
    assert sess_ok.get.call_count == 2
    assert res_ok.valid_row_count == 1

    # 3d. fetch_iol_panel also rejects HTTP redirect pre-hop
    sess_panel = MagicMock()
    sess_panel.get.return_value = _mock_response(
        301, "", url="https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos",
        headers={"Location": "http://insecure.example/bonos"},
    )
    with pytest.raises(IOLQuoteError, match="non-HTTPS"):
        fetch_iol_panel("iol_bonos", session=sess_panel)
    assert sess_panel.get.call_count == 1


def test_rev_04_and_rev_05_truncation_blind_spots_and_preloaded_defensive_checks():
    fixed_now = lambda tz=None: datetime(2026, 11, 27, 18, 0, tzinfo=tz)

    # 4a. Ignored date filter on initial non-probe segment (500 rows completely out of range) -> iol_truncated_response
    future_500 = [
        (d.strftime("%m/%d/%Y"), "50.00", "51.00", "49.00", "50.00", "50.00", "10,000.00", "20,000.00")
        for d in reversed(pd.bdate_range("2024-01-02", periods=500))
    ]
    sess_ign = MagicMock()
    sess_ign.get.return_value = _mock_response(200, _make_iol_get_html("AL30D", "99926", future_500[:5]))
    sess_ign.post.return_value = _mock_response(200, _make_iol_post_html(future_500))
    with pytest.raises(IOLHistoryError) as exc_ign:
        fetch_iol_historical_prices(
            "AL30D", start_date=date(2021, 1, 4), end_date=date(2022, 12, 30),
            iol_config=IOLConfig(historical_request_delay_seconds=0.0),
            session=sess_ign, sleep_fn=lambda _: None, now_fn=fixed_now,
        )
    assert exc_ign.value.code == "iol_truncated_response"
    assert sess_ign.post.call_count == 1

    # 4b. 4-day leading blind spot (<7 days) on exact 500-row response -> probed & recovered!
    bdays_504 = pd.bdate_range("2023-01-02", periods=504)
    lead_4 = [
        (d.strftime("%m/%d/%Y"), "48.00", "49.00", "47.00", "48.00", "48.00", "10,000.00", "20,000.00")
        for d in reversed(bdays_504[:4])
    ]
    tail_500 = [
        (d.strftime("%m/%d/%Y"), "50.00", "51.00", "49.00", "50.00", "50.00", "10,000.00", "20,000.00")
        for d in reversed(bdays_504[4:])
    ]
    sess_4d = MagicMock()
    sess_4d.get.return_value = _mock_response(200, _make_iol_get_html("AL30D", "99926", tail_500[:5]))
    sess_4d.post.side_effect = [
        _mock_response(200, _make_iol_post_html(tail_500)),
        _mock_response(200, _make_iol_post_html(lead_4)),
    ]
    res_4d = fetch_iol_historical_prices(
        "AL30D", start_date=bdays_504[0].date(), end_date=bdays_504[-1].date(),
        iol_config=IOLConfig(historical_request_delay_seconds=0.0),
        session=sess_4d, sleep_fn=lambda _: None, now_fn=fixed_now,
    )
    assert res_4d.valid_row_count == 504
    assert res_4d.truncation_splits_performed == 1

    # 4c. Oldest-first 500-row cap leaving trailing gap -> probed & recovered!
    oldest_500 = [
        (d.strftime("%m/%d/%Y"), "48.00", "49.00", "47.00", "48.00", "48.00", "10,000.00", "20,000.00")
        for d in reversed(bdays_504[:500])
    ]
    newest_4 = [
        (d.strftime("%m/%d/%Y"), "52.00", "53.00", "51.00", "52.00", "52.00", "10,000.00", "20,000.00")
        for d in reversed(bdays_504[500:])
    ]
    sess_old = MagicMock()
    sess_old.get.return_value = _mock_response(200, _make_iol_get_html("AL30D", "99926", newest_4))
    sess_old.post.side_effect = [
        _mock_response(200, _make_iol_post_html(oldest_500)),
        _mock_response(200, _make_iol_post_html(newest_4)),
    ]
    res_old = fetch_iol_historical_prices(
        "AL30D", start_date=bdays_504[0].date(), end_date=bdays_504[-1].date(),
        iol_config=IOLConfig(historical_request_delay_seconds=0.0),
        session=sess_old, sleep_fn=lambda _: None, now_fn=fixed_now,
    )
    assert res_old.valid_row_count == 504
    assert res_old.effective_end_date == bdays_504[-1].date()
    assert res_old.truncation_splits_performed == 1

    # 5a. Preloaded history settlement-type mismatch (currency="USD" paired with USD_CABLE_C) rejected!
    dup_idx = pd.DatetimeIndex([pd.Timestamp("2025-07-24"), pd.Timestamp("2025-07-24"), pd.Timestamp("2025-08-01")])
    bad_settle_pre = IOLHistoricalSeries(
        symbol="BPA8D", market="BCBA", id_titulo="154636",
        requested_start_date=date(2025, 7, 24), requested_end_date=date(2025, 8, 1),
        effective_start_date=date(2025, 7, 24), effective_end_date=date(2025, 8, 1),
        frequency="daily", price_field="close", series_type="market_price",
        currency="USD", usd_settlement_type="USD_CABLE_C", quote_unit="per_100_original_nominal",
        observed_at="2026-11-26T18:00:00+00:00", source_url="mock",
        bars=pd.DataFrame({"close": [80.0, 80.0, 81.0]}, index=dup_idx),
        price_series=pd.Series([80.0, 80.0, 81.0], index=dup_idx, name="BPA8D"),
        raw_row_count=3, valid_row_count=3,
    )
    inst_bpa = InstrumentConfig(
        instrument_id="BPA8D", ticker="BPA8D", currency="USD",
        section="fixed_income", asset_subtype="sovereign_bond", quote_source="iol_bonos",
    )
    with pytest.raises(IOLHistoryError) as exc_st:
        fetch_iol_historical_prices_for_instruments(
            [inst_bpa], iol_config=IOLConfig(allow_usd_cable_history=True),
            preloaded_history={"BPA8D": bad_settle_pre},
        )
    assert exc_st.value.code == "iol_currency_incompatible"

    # 5b. Preloaded series with duplicate timestamps clips cleanly without pandas index-alignment errors
    dup_ok_pre = IOLHistoricalSeries(
        symbol="BPA8D", market="BCBA", id_titulo="154636",
        requested_start_date=date(2025, 7, 24), requested_end_date=date(2025, 8, 1),
        effective_start_date=date(2025, 7, 24), effective_end_date=date(2025, 8, 1),
        frequency="daily", price_field="close", series_type="market_price",
        currency="USD", usd_settlement_type="USD_LOCAL_D", quote_unit="per_100_original_nominal",
        observed_at="2026-11-26T18:00:00+00:00", source_url="mock",
        bars=pd.DataFrame({"close": [80.0, 80.0, 81.0]}, index=dup_idx),
        price_series=pd.Series([80.0, 80.0, 81.0], index=dup_idx, name="BPA8D"),
        raw_row_count=3, valid_row_count=3,
    )
    clipped = fetch_iol_historical_prices_for_instruments(
        [inst_bpa], as_of_date=date(2025, 7, 25), preloaded_history={"BPA8D": dup_ok_pre},
    )
    assert clipped["BPA8D"].valid_row_count == 2
    assert clipped["BPA8D"].effective_end_date == date(2025, 7, 24)
