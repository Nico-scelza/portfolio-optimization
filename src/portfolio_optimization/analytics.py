"""Stateless portfolio analytics, risk visualizations, and capital allocation helpers.

All functions receive ``AnalysisResult`` explicitly rather than reading mutable
process-wide global state, ensuring complete request isolation and pandas 3.x safety.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._display import display
from .errors import ConfigurationError
from .risk_model import WEEKS_PER_YEAR
from .schemas import AnalysisResult


def generar_frontera_eficiente(
    result: AnalysisResult,
    *,
    n_samples: int = 5000,
    seed: Optional[int] = 42,
    render: bool = True,
) -> Dict[str, np.ndarray]:
    """Simulate random long-only portfolios on the canonical annual weekly risk model."""
    mu = result.expected_returns
    cov = result.cov_matrix
    rf = result.black_litterman.rf_rate
    n_assets = len(result.tickers)
    if n_assets == 0:
        raise ConfigurationError("AnalysisResult contains no assets.")

    rng = np.random.default_rng(seed)
    weights_sim = rng.dirichlet(np.ones(n_assets), size=int(n_samples))
    ret_port = weights_sim @ mu.reindex(result.tickers).to_numpy(dtype=float)
    vol_port = np.sqrt(
        np.maximum(
            np.einsum("ij,jk,ik->i", weights_sim, cov.loc[result.tickers, result.tickers].to_numpy(dtype=float), weights_sim),
            0.0,
        )
    )
    sharpe_port = np.where(vol_port > 1e-12, (ret_port - rf) / vol_port, np.nan)

    if render:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sc = ax.scatter(vol_port * 100.0, ret_port * 100.0, c=sharpe_port, cmap="viridis", s=12, alpha=0.55)
        for sname, strat in result.strategies.items():
            if strat.expected_return is not None:
                ax.scatter(
                    strat.volatility * 100.0,
                    strat.expected_return * 100.0,
                    s=110,
                    edgecolors="black",
                    linewidth=1.2,
                    label=sname,
                    zorder=5,
                )
        ax.set_xlabel("Annualized Volatility (%)")
        ax.set_ylabel("Expected Annual Return (%)")
        ax.set_title("Simulated Frontier vs Optimized Strategies")
        plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        display(fig)
        plt.close(fig)

    return {
        "weights": weights_sim,
        "returns": ret_port,
        "volatilities": vol_port,
        "sharpe": sharpe_port,
    }


def analizar_rendimiento_historico(
    result: AnalysisResult,
    *,
    render: bool = True,
) -> pd.DataFrame:
    """Summarize historical weekly-aligned USD performance per instrument."""
    weekly_rets = result.risk_model.weekly_returns
    if weekly_rets.empty:
        return pd.DataFrame()

    rf = result.black_litterman.rf_rate
    n_weeks = len(weekly_rets)
    years = n_weeks / WEEKS_PER_YEAR
    cum_wealth = (1.0 + weekly_rets).cumprod()

    rows = []
    for t in weekly_rets.columns:
        total_ret = float(cum_wealth[t].iloc[-1] - 1.0)
        ann_ret = float((1.0 + total_ret) ** (1.0 / years) - 1.0) if years > 0 and (1.0 + total_ret) > 0 else float("nan")
        ann_vol = float(weekly_rets[t].std(ddof=1) * np.sqrt(WEEKS_PER_YEAR))
        sharpe = (ann_ret - rf) / ann_vol if ann_vol > 1e-12 and np.isfinite(ann_ret) else float("nan")
        rows.append({
            "instrument_id": t,
            "total_return": total_ret,
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "historical_sharpe": sharpe,
        })

    summary = pd.DataFrame(rows).set_index("instrument_id")

    if render:
        fig, ax = plt.subplots(figsize=(9, 5))
        norm = cum_wealth * 100.0
        for col in norm.columns:
            ax.plot(norm.index, norm[col], label=col, linewidth=1.4)
        ax.set_title("Aligned Weekly USD Wealth Index (Base 100)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Index")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        display(fig)
        plt.close(fig)

    return summary


def generar_heatmap_correlaciones(
    result: AnalysisResult,
    *,
    render: bool = True,
) -> pd.DataFrame:
    """Return and optionally plot the canonical correlation matrix from the risk model."""
    corr = result.risk_model.corr_matrix
    if render and not corr.empty:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        im = ax.imshow(corr.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.index)
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, label="Correlation")
        ax.set_title("Risk Model Correlation Matrix (Weekly W-FRI USD)")
        display(fig)
        plt.close(fig)
    return corr


def generar_underwater_plot(
    result: AnalysisResult,
    *,
    strategy_name: Optional[str] = None,
    render: bool = True,
) -> pd.DataFrame:
    """Compute weekly historical drawdowns for each asset and the selected strategy."""
    weekly_rets = result.risk_model.weekly_returns
    if weekly_rets.empty:
        return pd.DataFrame()

    strat_key = strategy_name or result.selected_strategy
    strat = result.strategies.get(strat_key)

    wealth = (1.0 + weekly_rets).cumprod()
    dd = wealth.div(wealth.cummax()) - 1.0

    if strat is not None:
        port_ret = weekly_rets[strat.tickers].to_numpy(dtype=float) @ strat.weights
        port_wealth = pd.Series((1.0 + port_ret).cumprod(), index=weekly_rets.index)
        dd[f"PORTFOLIO_{strat_key.upper()}"] = port_wealth / port_wealth.cummax() - 1.0

    if render:
        fig, ax = plt.subplots(figsize=(9, 5))
        for col in dd.columns:
            lw = 2.2 if col.startswith("PORTFOLIO_") else 1.2
            ax.plot(dd.index, dd[col] * 100.0, label=col, linewidth=lw)
        ax.axhline(0.0, color="red", linewidth=1.0)
        ax.set_title("Historical Drawdowns (Weekly W-FRI USD)")
        ax.set_ylabel("Drawdown (%)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.25)
        display(fig)
        plt.close(fig)

    return dd


def analizar_contribucion_marginal_riesgo(
    result: AnalysisResult,
    *,
    render: bool = True,
) -> pd.DataFrame:
    """Build side-by-side capital weight vs risk contribution table across all strategies."""
    data: Dict[str, pd.Series] = {}
    for sname, strat in result.strategies.items():
        data[f"weight_{sname}"] = pd.Series(strat.weights_by_asset)
        data[f"mcr_{sname}"] = strat.marginal_risk_contributions
        data[f"rc_rel_{sname}"] = strat.risk_contributions_rel
    df = pd.DataFrame(data, index=result.tickers)

    if render and not df.empty:
        strat_names = list(result.strategies.keys())
        fig, ax = plt.subplots(figsize=(9, 5))
        bottom = np.zeros(len(strat_names))
        for t in result.tickers:
            vals = [float(result.strategies[s].risk_contributions_rel.get(t, 0.0)) * 100.0 for s in strat_names]
            ax.bar(strat_names, vals, bottom=bottom, label=t)
            bottom += np.array(vals)
        ax.set_ylabel("Relative Risk Contribution (%)")
        ax.set_title("Risk Contribution Decomposition by Strategy")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        display(fig)
        plt.close(fig)

    return df


def generar_grafico_riesgo(
    result: AnalysisResult,
    *,
    render: bool = True,
) -> pd.DataFrame:
    """Compare Volatility, CVaR 95%, Expected Return, and Sharpe across all strategies."""
    rows = []
    for sname, strat in result.strategies.items():
        rows.append({
            "strategy": sname,
            "expected_return": strat.expected_return,
            "volatility": strat.volatility,
            "cvar_95": strat.cvar_95,
            "sharpe_ratio": strat.sharpe_ratio,
            "status": strat.status,
        })
    df = pd.DataFrame(rows).set_index("strategy") if rows else pd.DataFrame()

    if render and not df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for sname, row in df.iterrows():
            if pd.notna(row["cvar_95"]):
                ax.scatter(row["volatility"] * 100.0, row["cvar_95"] * 100.0, s=160, label=sname)
                ax.annotate(sname, (row["volatility"] * 100.0, row["cvar_95"] * 100.0))
        ax.set_xlabel("Annualized Volatility (%)")
        ax.set_ylabel("Weekly CVaR 95% Magnitude (%)")
        ax.set_title("Strategy Risk Profile: Volatility vs CVaR")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        display(fig)
        plt.close(fig)

    return df


def armar_cartera(
    result: AnalysisResult,
    capital_disponible: float,
) -> Dict[str, pd.DataFrame]:
    """Compute target USD dollar allocations for each optimized strategy without side effects."""
    capital = float(capital_disponible)
    if not np.isfinite(capital) or capital <= 0.0:
        raise ConfigurationError("'capital_disponible' must be finite and strictly positive.")

    allocations: Dict[str, pd.DataFrame] = {}
    for sname, strat in result.strategies.items():
        df = pd.DataFrame([
            {
                "Instrument": t,
                "Weight (%)": strat.weights_by_asset[t] * 100.0,
                "Amount (USD)": capital * strat.weights_by_asset[t],
                "Risk Share (%)": float(strat.risk_contributions_rel.get(t, 0.0)) * 100.0,
            }
            for t in strat.tickers
        ]).sort_values("Amount (USD)", ascending=False, ignore_index=True)
        allocations[sname] = df
    return allocations
