"""Unified stateless execution pipeline for backend and notebook consumers.

``ejecutar_analisis_completo`` and ``run_portfolio_analysis`` consume the exact
same validated ``PortfolioAnalysisConfig`` contract and delegate directly to
``analyze_portfolio`` with zero divergent code paths, zero global state, zero ML,
and zero country-risk spreads.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd

from ._display import HTML, Markdown, display, suppress_display
from .errors import ConfigurationError
from .iol import IOLHistoricalSeries, IOLPanelSnapshot
from .optimization import analyze_portfolio
from .schemas import REMOVED_LEGACY_KEYS, AnalysisResult, PortfolioAnalysisConfig


def _render_analysis_summary_html(result: AnalysisResult) -> None:
    """Render optional notebook HTML tables strictly from ``AnalysisResult``."""
    display(HTML("""
    <div style="background: linear-gradient(135deg, #1d3557 0%, #457b9d 100%);
                border-radius: 12px; padding: 20px; margin: 16px 0; text-align: center;">
        <h2 style="color: white; margin: 0;">Multi-Asset Portfolio Analysis</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0;">
            Annualized DCF -> Black-Litterman | Weekly (W-FRI) VIX-Regime Covariance | Stateless Engine
        </p>
    </div>
    """))

    rows = []
    for strat_name, strat in result.strategies.items():
        rows.append({
            "Strategy": strat_name,
            "Status": strat.status,
            "Expected Return": strat.expected_return,
            "Volatility": strat.volatility,
            "Sharpe": strat.sharpe_ratio,
            "Weekly CVaR 95%": strat.cvar_95,
            "HHI": strat.hhi,
        })
    if rows:
        df_summary = pd.DataFrame(rows)
        display(Markdown("### Strategy Comparison"))
        display(df_summary)

    sel = result.strategies.get(result.selected_strategy)
    if sel is not None:
        df_w = pd.DataFrame([
            {
                "Instrument": t,
                "Capital Weight": sel.weights_by_asset[t],
                "Risk Share": float(sel.risk_contributions_rel.get(t, 0.0)),
            }
            for t in sel.tickers
        ]).sort_values("Capital Weight", ascending=False)
        display(Markdown(f"### Selected Strategy Allocation: `{result.selected_strategy}`"))
        display(df_w)


def ejecutar_analisis_completo(
    config: Union[PortfolioAnalysisConfig, Dict[str, Any]],
    *,
    render_ui: bool = True,
    prices_usd_override: Optional[pd.DataFrame] = None,
    vix_history_override: Optional[pd.Series] = None,
    current_vix_override: Optional[Tuple[Optional[float], Optional[pd.Timestamp], bool]] = None,
    iol_snapshots_override: Optional[Dict[str, IOLPanelSnapshot]] = None,
    iol_history_override: Optional[Dict[str, IOLHistoricalSeries]] = None,
    iol_history_fetcher: Optional[Callable[..., IOLHistoricalSeries]] = None,
    market_caps_override: Optional[Dict[str, float]] = None,
    sector_map_override: Optional[Dict[str, str]] = None,
    **unexpected_kwargs: Any,
) -> AnalysisResult:
    """Master stateless entrypoint sharing the exact contract of ``analyze_portfolio``."""
    if unexpected_kwargs:
        forbidden = REMOVED_LEGACY_KEYS.intersection(unexpected_kwargs.keys())
        if forbidden:
            raise ConfigurationError(
                f"Removed legacy parameter(s) are no longer supported: {sorted(forbidden)}."
            )
        raise ConfigurationError(
            f"Unsupported keyword argument(s) in ejecutar_analisis_completo: {sorted(unexpected_kwargs.keys())}."
        )

    result = analyze_portfolio(
        config,
        prices_usd_override=prices_usd_override,
        vix_history_override=vix_history_override,
        current_vix_override=current_vix_override,
        iol_snapshots_override=iol_snapshots_override,
        iol_history_override=iol_history_override,
        iol_history_fetcher=iol_history_fetcher,
        market_caps_override=market_caps_override,
        sector_map_override=sector_map_override,
    )

    if render_ui:
        _render_analysis_summary_html(result)
    return result


def run_portfolio_analysis(
    config: Union[PortfolioAnalysisConfig, Dict[str, Any]],
    *,
    render_ui: bool = False,
    prices_usd_override: Optional[pd.DataFrame] = None,
    vix_history_override: Optional[pd.Series] = None,
    current_vix_override: Optional[Tuple[Optional[float], Optional[pd.Timestamp], bool]] = None,
    iol_snapshots_override: Optional[Dict[str, IOLPanelSnapshot]] = None,
    iol_history_override: Optional[Dict[str, IOLHistoricalSeries]] = None,
    iol_history_fetcher: Optional[Callable[..., IOLHistoricalSeries]] = None,
    market_caps_override: Optional[Dict[str, float]] = None,
    sector_map_override: Optional[Dict[str, str]] = None,
    **unexpected_kwargs: Any,
) -> AnalysisResult:
    """Headless web-backend entrypoint wrapping ``ejecutar_analisis_completo``."""
    if render_ui:
        return ejecutar_analisis_completo(
            config,
            render_ui=True,
            prices_usd_override=prices_usd_override,
            vix_history_override=vix_history_override,
            current_vix_override=current_vix_override,
            iol_snapshots_override=iol_snapshots_override,
            iol_history_override=iol_history_override,
            iol_history_fetcher=iol_history_fetcher,
            market_caps_override=market_caps_override,
            sector_map_override=sector_map_override,
            **unexpected_kwargs,
        )
    with suppress_display():
        return ejecutar_analisis_completo(
            config,
            render_ui=False,
            prices_usd_override=prices_usd_override,
            vix_history_override=vix_history_override,
            current_vix_override=current_vix_override,
            iol_snapshots_override=iol_snapshots_override,
            iol_history_override=iol_history_override,
            iol_history_fetcher=iol_history_fetcher,
            market_caps_override=market_caps_override,
            sector_map_override=sector_map_override,
            **unexpected_kwargs,
        )
