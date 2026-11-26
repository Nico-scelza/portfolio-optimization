"""Thin optional Jupyter/Colab notebook adapter (stateless, no ML, no country risk).

All financial calculations live exclusively in the core package. No process-wide
mutable user portfolio state is persisted; interactive helpers operate on explicit
in-memory configuration/result objects passed by the caller.
"""

from __future__ import annotations

import html as html_lib
from typing import Any, Dict, Optional

import yfinance as yf

from ._display import HTML, display
from .pipeline import ejecutar_analisis_completo
from .schemas import AnalysisResult, PortfolioAnalysisConfig


def obtener_info_activo(ticker: str) -> Dict[str, Any]:
    """Retrieve basic descriptive metadata and latest price via yfinance (no country-risk logic)."""
    try:
        tkr = ticker.strip().upper()
        activo = yf.Ticker(tkr)
        info = activo.info or {}
        hist = activo.history(period="5d", auto_adjust=True)
        precio_actual = float(hist["Close"].dropna().iloc[-1]) if hist is not None and not hist.empty else None
        return {
            "ticker": tkr,
            "nombre": info.get("longName") or info.get("shortName") or tkr,
            "currency": info.get("currency"),
            "market": info.get("exchange", "US"),
            "country": info.get("country"),  # Descriptive/calendar metadata only
            "precio_actual": precio_actual,
            "valido": True,
        }
    except Exception:
        return {"valido": False}


def ejecutar_interfaz_analisis(
    config: PortfolioAnalysisConfig,
    *,
    render_ui: bool = True,
) -> AnalysisResult:
    """Run canonical portfolio analysis from a validated ``PortfolioAnalysisConfig``."""
    return ejecutar_analisis_completo(config, render_ui=render_ui)


def mostrar_rebalanceo(
    result: AnalysisResult,
    strategy_name: Optional[str] = None,
) -> None:
    """Display indicative multi-broker rebalancing orders from an ``AnalysisResult``."""
    strat_key = str(strategy_name or result.selected_strategy)
    safe_strat = html_lib.escape(strat_key, quote=True)
    reb = result.rebalancing.get(strat_key)
    if reb is None:
        display(HTML(f"<p>No rebalancing result available for strategy <b>{safe_strat}</b>.</p>"))
        return

    safe_status = html_lib.escape(str(reb.status), quote=True)
    rows_html = ""
    for order in reb.orders:
        safe_broker = html_lib.escape(str(order.broker_id), quote=True)
        safe_inst = html_lib.escape(str(order.instrument_id), quote=True)
        safe_side = html_lib.escape(str(order.side), quote=True)
        rows_html += (
            f"<tr>"
            f"<td>{safe_broker}</td>"
            f"<td>{safe_inst}</td>"
            f"<td>{safe_side}</td>"
            f"<td>{order.quantity:.6f}</td>"
            f"<td>${order.gross_amount_usd:,.2f}</td>"
            f"<td>${order.commission_usd:,.2f}</td>"
            f"<td>${order.spread_cost_usd:,.2f}</td>"
            f"<td>${order.net_amount_usd:,.2f}</td>"
            f"</tr>"
        )

    html = f"""
    <div>
        <h3>Indicative Rebalancing Proposal ({safe_strat}) — Status: {safe_status}</h3>
        <table border="1" cellpadding="6" cellspacing="0">
            <thead>
                <tr>
                    <th>Broker</th><th>Instrument</th><th>Side</th><th>Quantity</th>
                    <th>Gross (USD)</th><th>Commission (USD)</th><th>Spread (USD)</th><th>Net Cash Impact (USD)</th>
                </tr>
            </thead>
            <tbody>{rows_html or '<tr><td colspan="8">No trades proposed.</td></tr>'}</tbody>
        </table>
        <p>One-way turnover: {reb.one_way_turnover:.2%} | Total transaction cost: ${reb.total_transaction_cost_usd:,.2f} | Ending cash: ${reb.ending_cash_usd:,.2f}</p>
    </div>
    """
    display(HTML(html))