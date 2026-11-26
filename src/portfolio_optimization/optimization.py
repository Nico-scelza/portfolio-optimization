"""Portfolio optimization, covariance estimation, Black-Litterman, and stress testing.

Migrated faithfully from Cells 09, 10, 12, and 13 of the source notebook.
Preserves exact execution order and effective redefinition of ``analyze_portfolio``
(Cell 10 defined first as ``_analyze_portfolio_v1``, overridden by Cell 13's
``analyze_portfolio``).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf, OAS as OAS_estimator

from . import state
from .market_data import _download_prices, obtener_tasa_libre_riesgo


# =============================================================================
# Cell 09: Sharpe optimization with HHI constraint
# =============================================================================

def optimize_sharpe_hhi(mu: pd.Series, cov: pd.DataFrame, rf: float, hhi_limit: float = 0.15) -> np.ndarray:
    """
    Optimiza el portfolio para maximizar el Sharpe Ratio con restricción de diversificación HHI.

    Parámetros:
    -----------
    mu : pd.Series
        Vector de retornos esperados anualizados para cada activo.
    cov : pd.DataFrame
        Matriz de covarianza anualizada de los retornos.
    rf : float
        Tasa libre de riesgo anualizada.
    hhi_limit : float, opcional (default=0.15)
        Límite superior para el Índice Herfindahl-Hirschman (HHI).
        HHI = Σ(wi²), donde wi es el peso del activo i.
        Valores típicos: 0.15-0.20 para mayor diversificación.

    Retorna:
    --------
    np.ndarray
        Vector de pesos óptimos que maximiza el Sharpe Ratio sujeto a:
        - Suma de pesos = 1
        - Cada peso entre 0% y 50%
        - HHI ≤ hhi_limit

    Notas:
    ------
    La restricción HHI evita soluciones de esquina (concentración excesiva).
    Un HHI más bajo implica mayor diversificación efectiva.
    """

    n_assets = len(mu)

    def neg_sharpe(weights):
        ret = weights @ mu.values
        vol = np.sqrt(weights @ cov.values @ weights)
        return -(ret - rf) / vol if vol > 0 else 1e10

    def hhi_constraint(weights):
        """
        Retorna un valor positivo si se cumple la restricción.
        Para 'ineq': la función debe ser ≥ 0 para cumplir la restricción.
        Queremos: hhi_limit - Σ(wi²) ≥ 0
        """
        hhi = np.sum(weights ** 2)
        return hhi_limit - hhi

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},      # Suma de pesos = 1
        {'type': 'ineq', 'fun': hhi_constraint}              # HHI ≤ hhi_limit
    ]

    bounds = tuple((0.05, 0.20) for _ in range(n_assets))

    initial_guess = np.array([1/n_assets] * n_assets)

    result = minimize(
        neg_sharpe,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )

    if result.success:
        final_hhi = np.sum(result.x ** 2)
        print(f"✅ Optimización exitosa con HHI = {final_hhi:.4f} (límite: {hhi_limit:.4f})")
        return result.x
    else:
        print(f"⚠️ La optimización no convergió. Usando pesos iguales.")
        return initial_guess


# =============================================================================
# Cell 10: AnalysisResult, core optimizers, metrics, chart stubs, v1 pipeline
# =============================================================================

@dataclass
class AnalysisResult:
    """Resultado del análisis de portfolio"""
    tickers: List[str]
    expected_returns: pd.Series
    cov_matrix: pd.DataFrame
    weights_sharpe: np.ndarray
    weights_minvar: np.ndarray
    weights_cvar: np.ndarray
    chart_path: Path
    warnings: List[str]
    stats_sharpe: dict
    stats_minvar: dict
    stats_cvar: dict
    start_date: Optional[pd.Timestamp]
    end_date: Optional[pd.Timestamp]
    sample_days: int
    used_expected: dict
    hist_performance: dict
    mcr_sharpe: pd.Series
    mcr_minvar: pd.Series
    mcr_cvar: pd.Series
    prices_hist: pd.DataFrame
    historical_perf_chart_path: Path
    correlation_heatmap_path: Path
    mcr_chart_path: Path
    patrimonio_chart_path: Path
    total_return_chart_path: Path
    individual_returns_chart_path: Path
    is_existing_portfolio: bool
    stress_test_results: dict = field(default_factory=dict)
    mu_sources: dict = field(default_factory=dict)
    vix_level: Optional[float] = None
    cov_inflation_factor: float = 1.0


def _optimize_min_cvar(mu: pd.Series, cov: pd.DataFrame, daily_returns: pd.DataFrame, stress_bounds=None) -> np.ndarray:
    n_assets = len(mu)
    def portfolio_cvar(weights):
        portfolio_returns = daily_returns.values @ weights
        return abs(_calc_cvar(portfolio_returns, 0.95))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = stress_bounds if stress_bounds else tuple((0.05, 0.20) for _ in range(n_assets))
    initial_guess = np.array([1/n_assets] * n_assets)
    result = minimize(portfolio_cvar, initial_guess, method='SLSQP',
                     bounds=bounds, constraints=constraints, options={'maxiter': 1000})
    return result.x if result.success else initial_guess


def _optimize_min_variance(mu: pd.Series, cov: pd.DataFrame, stress_bounds=None) -> np.ndarray:
    n_assets = len(mu)
    def portfolio_variance(weights):
        return weights @ cov.values @ weights
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = stress_bounds if stress_bounds else tuple((0.05, 0.20) for _ in range(n_assets))
    initial_guess = np.array([1/n_assets] * n_assets)
    result = minimize(portfolio_variance, initial_guess, method='SLSQP',
                     bounds=bounds, constraints=constraints, options={'maxiter': 1000})
    return result.x if result.success else initial_guess


def _portfolio_metrics(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, daily_returns: pd.DataFrame, rf: float) -> dict:
    ret = weights @ mu.values
    vol = np.sqrt(weights @ cov.values @ weights)
    sharpe = (ret - rf) / vol if vol > 0 else 0
    port_returns = daily_returns.values @ weights
    cvar = abs(_calc_cvar(port_returns, 0.95))
    return {'ret': ret, 'vol': vol, 'sharpe': sharpe, 'cvar': cvar}


def _calc_mcr(weights: np.ndarray, cov: pd.DataFrame, tickers: List[str]) -> pd.Series:
    portfolio_variance = weights @ cov.values @ weights
    portfolio_vol = np.sqrt(portfolio_variance)
    if portfolio_vol == 0:
        return pd.Series(np.zeros(len(tickers)), index=tickers)
    marginal_contrib = (cov.values @ weights) / portfolio_vol
    mcr = weights * marginal_contrib
    return pd.Series(mcr, index=tickers)


def _calc_historical_performance(prices: pd.DataFrame, weights: np.ndarray, tickers: List[str]) -> pd.Series:
    prices_subset = prices[tickers].ffill().bfill()
    portfolio_prices = (prices_subset * weights).sum(axis=1)
    returns = portfolio_prices.pct_change().fillna(0)
    cumulative = (1 + returns).cumprod()
    cumulative = cumulative.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(1)
    return cumulative


def _build_static_figure(mu, port_ret, port_vol, sharpe, output_path):
    return output_path

def _build_historical_performance_plot(prices, weights_dict, tickers, output_path):
    return output_path

def _build_correlation_heatmap(prices, tickers, output_path):
    return output_path

def _build_mcr_chart(mcr_sharpe, mcr_minvar, mcr_cvar, cov, ws, wm, wc, tickers, output_path):
    return output_path

def _build_real_portfolio_performance(prices, assets, sales, output_path):
    return output_path

def _build_real_portfolio_with_sales(prices, assets, sales, output_path):
    return output_path

def _build_real_individual_returns_plot(prices, assets, sales, tickers, output_path):
    return output_path

def _build_patrimonio_plot(prices, ws, wm, wc, tickers, assets, output_path):
    return output_path

def _build_portfolio_strategies_performance_plot(prices, ws, wm, wc, tickers, output_path):
    return output_path

def _build_individual_returns_plot(prices, tickers, output_path):
    return output_path


def _is_existing_portfolio(assets_list: List[dict]) -> bool:
    for asset in assets_list:
        if asset.get("mode") == "existing":
            return True
        if asset.get("quantity", 0) > 0 and asset.get("buy_price", 0) > 0:
            return True
    return False


def _calc_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var_threshold].mean()
    return cvar if not np.isnan(cvar) else var_threshold


def _simulate_portfolios(mu: pd.Series, cov: pd.DataFrame, daily_returns: pd.DataFrame, rf: float, samples: int) -> tuple:
    n_assets = len(mu)
    weights_array = np.zeros((samples, n_assets))
    returns_array = np.zeros(samples)
    volatility_array = np.zeros(samples)
    cvar_array = np.zeros(samples)
    for i in range(samples):
        weights = np.random.random(n_assets)
        weights /= weights.sum()
        weights_array[i] = weights
        returns_array[i] = weights @ mu.values
        volatility_array[i] = np.sqrt(weights @ cov.values @ weights)
        port_returns = daily_returns.values @ weights
        cvar_array[i] = abs(_calc_cvar(port_returns, 0.95))
    return weights_array, returns_array, volatility_array, cvar_array


def analyze_portfolio(assets: Iterable[dict], rf_rate: Optional[float] = None,
                      samples: int = 2000, dcf_upsides: Optional[dict] = None,
                      ml_predictions_5y: Optional[dict] = None,
                      investment_horizon: int = 3, sales: Optional[List[dict]] = None,
                      country_risk_argentina: Optional[float] = None) -> AnalysisResult:
    """
    Pipeline institucional de optimización de portfolio.

    μ_DCF (upside usuario) es sagrado e inmodificable.
    Si no existe μ_DCF → Black-Litterman (mercado + ML views baja confianza).
    Σ: Ledoit-Wolf shrinkage + régimen (70/30) + forward-looking (VIX).
    Post-optimización: stress test con correlaciones = 1.
    """
    if rf_rate is None:
        rf_rate = obtener_tasa_libre_riesgo()

    tickers = sorted({a.get("ticker", "").upper() for a in assets if a.get("ticker")})
    if not tickers:
        raise ValueError("No hay tickers para analizar.")

    sales = sales or []
    assets_list = list(assets)
    is_existing = _is_existing_portfolio(assets_list)

    if is_existing:
        buy_dates = []
        for asset in assets_list:
            if asset.get("mode") == "existing" and asset.get("buy_date"):
                buy_dates.append(pd.Timestamp(asset.get("buy_date")))
        if buy_dates:
            earliest_date = min(buy_dates)
            start_download = earliest_date - pd.Timedelta(days=30)
            prices, dropped = _download_prices(tickers, period="max")
            if not prices.empty:
                start_download_tz = pd.Timestamp(start_download)
                if prices.index.tz is not None:
                    start_download_tz = start_download_tz.tz_localize('UTC')
                prices = prices[prices.index >= start_download_tz]
        else:
            prices, dropped = _download_prices(tickers, period="10y")
    else:
        prices, dropped = _download_prices(tickers, period="10y")

    if prices.empty:
        raise ValueError("No se pudieron descargar precios para los tickers indicados.")

    available_tickers = [t for t in prices.columns]
    start_date = prices.index.min() if not prices.empty else None
    prices_daily = prices.resample('D').last().ffill().bfill()

    prices_daily = prices.resample('D').last().ffill()
    daily = prices_daily.pct_change(fill_method=None).dropna(how='all')
    daily = daily.replace([np.inf, -np.inf], 0)
    available_tickers = list(daily.columns)

    if daily.empty or len(available_tickers) < 2:
        raise ValueError("No hay retornos suficientes para el análisis.")

    print("\n📐 Construyendo Σ: Ledoit-Wolf + régimen (70% calma / 30% estrés)...")
    cov, _, _ = regime_adjusted_covariance(daily, stress_weight=0.30, lookback_stress=63)

    vix_current = get_current_vix()
    cov, cov_inflation = forward_looking_cov_adjustment(cov, vix_current)
    if cov_inflation > 1.0:
        print(f"  📊 VIX={vix_current:.1f} → Inflación de covarianza: {cov_inflation:.2f}x")
    else:
        print(f"  📊 VIX={'N/A' if vix_current is None else f'{vix_current:.1f}'} → Sin inflación adicional")

    if country_risk_argentina is None:
        country_risk_argentina = state.prima_riesgo_mas_reciente
        if country_risk_argentina is not None and country_risk_argentina > 1.0:
            country_risk_argentina = country_risk_argentina / 100.0
        elif country_risk_argentina is None:
            country_risk_argentina = 0.0

    asset_country_risk_map = {}
    for asset in assets_list:
        ticker_a = asset.get("ticker", "").upper()
        pais_riesgo = asset.get("pais_riesgo", asset.get("pais", ""))
        if ticker_a and pais_riesgo:
            asset_country_risk_map[ticker_a] = pais_riesgo

    dcf_upsides_clean = {}
    if dcf_upsides:
        dcf_upsides_clean = {k.upper(): v for k, v in dcf_upsides.items()
                             if isinstance(v, (int, float)) and v != 0}

    print("\n📌 Calculando μ: DCF (sagrado) → Black-Litterman (resto)...")
    mu, mu_source = compute_expected_returns_pipeline(
        tickers_list=available_tickers,
        daily_returns=daily,
        cov_matrix=cov,
        dcf_upsides=dcf_upsides_clean,
        ml_predictions_5y=ml_predictions_5y,
        country_risk_map=asset_country_risk_map,
        country_risk_value=country_risk_argentina or 0.0
    )
    for t_src, src in mu_source.items():
        print(f"  {t_src}: μ={mu[t_src]*100:.2f}% → {src}")

    rf = rf_rate
    sample_days = daily.shape[0]

    weights, port_ret, port_vol, cvars = _simulate_portfolios(mu, cov, daily, rf, samples)
    sharpe = (port_ret - rf) / port_vol

    print("\n🎯 Optimizando portfolios con restricciones HHI + sectoriales + stress...")
    sector_map = get_sector_map(available_tickers)
    stress_bounds = compute_stress_weight_caps(cov, available_tickers)
    weights_sharpe = optimize_with_sector_and_stress(mu, cov, rf, available_tickers,
                                                      hhi_limit=0.35, max_sector_weight=0.40,
                                                      sector_map=sector_map)
    weights_minvar = _optimize_min_variance(mu, cov, stress_bounds=stress_bounds)
    weights_cvar = _optimize_min_cvar(mu, cov, daily, stress_bounds=stress_bounds)
    stats_sharpe = _portfolio_metrics(weights_sharpe, mu, cov, daily, rf)
    stats_minvar = _portfolio_metrics(weights_minvar, mu, cov, daily, rf)
    stats_cvar = _portfolio_metrics(weights_cvar, mu, cov, daily, rf)

    mcr_sharpe = _calc_mcr(weights_sharpe, cov, available_tickers)
    mcr_minvar = _calc_mcr(weights_minvar, cov, available_tickers)
    mcr_cvar = _calc_mcr(weights_cvar, cov, available_tickers)

    hist_performance = {
        "sharpe": _calc_historical_performance(prices, weights_sharpe, available_tickers),
        "minvar": _calc_historical_performance(prices, weights_minvar, available_tickers),
        "cvar": _calc_historical_performance(prices, weights_cvar, available_tickers),
    }

    chart_path = _build_static_figure(mu, port_ret, port_vol, sharpe,
                                       Path(tempfile.gettempdir()) / "portfolio_chart.png")
    weights_dict = {"sharpe": weights_sharpe, "minvar": weights_minvar, "cvar": weights_cvar}
    historical_perf_chart_path = _build_historical_performance_plot(
        prices, weights_dict, available_tickers, Path(tempfile.gettempdir()) / "historical_perf_chart.png")
    correlation_heatmap_path = _build_correlation_heatmap(
        prices, available_tickers, Path(tempfile.gettempdir()) / "correlation_heatmap.png")
    mcr_chart_path = _build_mcr_chart(
        mcr_sharpe, mcr_minvar, mcr_cvar, cov, weights_sharpe, weights_minvar, weights_cvar,
        available_tickers, Path(tempfile.gettempdir()) / "mcr_chart.png")

    if is_existing:
        patrimonio_chart_path = _build_real_portfolio_performance(
            prices, assets_list, sales, Path(tempfile.gettempdir()) / "patrimonio_chart.png")
        total_return_chart_path = _build_real_portfolio_with_sales(
            prices, assets_list, sales, Path(tempfile.gettempdir()) / "total_return_chart.png")
        individual_returns_chart_path = _build_real_individual_returns_plot(
            prices, assets_list, sales, available_tickers, Path(tempfile.gettempdir()) / "individual_returns_chart.png")
    else:
        patrimonio_chart_path = _build_patrimonio_plot(
            prices, weights_sharpe, weights_minvar, weights_cvar, available_tickers, assets_list,
            Path(tempfile.gettempdir()) / "patrimonio_chart.png")
        total_return_chart_path = _build_portfolio_strategies_performance_plot(
            prices, weights_sharpe, weights_minvar, weights_cvar, available_tickers,
            Path(tempfile.gettempdir()) / "total_return_chart.png")
        individual_returns_chart_path = _build_individual_returns_plot(
            prices, available_tickers, Path(tempfile.gettempdir()) / "individual_returns_chart.png")

    print("\n🔴 Stress test (correlaciones = 1)...")
    stress_sharpe = stress_test_corr_one(weights_sharpe, cov, available_tickers)
    stress_minvar = stress_test_corr_one(weights_minvar, cov, available_tickers)
    stress_cvar = stress_test_corr_one(weights_cvar, cov, available_tickers)
    stress_results = {'sharpe': stress_sharpe, 'minvar': stress_minvar, 'cvar': stress_cvar}
    for st_name, st in [('Sharpe', stress_sharpe), ('MinVar', stress_minvar), ('CVaR', stress_cvar)]:
        print(f"  {st_name}: Vol {st['normal_vol']*100:.2f}%→{st['stress_vol']*100:.2f}% "
              f"(+{st['vol_increase_pct']:.1f}%) Diversif={st['diversification_ratio']:.3f}")
        if st['false_diversification']:
            for rec in st['recommendations']:
                print(f"    ⚠️ {rec}")

    warnings_list: List[str] = []
    if dropped:
        warnings_list.append(f"Tickers sin datos: {', '.join(dropped)}")
    if daily.shape[0] < 250:
        warnings_list.append("La serie histórica es corta; los resultados pueden ser inestables.")
    for stn, std in stress_results.items():
        if std.get('false_diversification'):
            warnings_list.append(f"Stress test ({stn}): Falsa diversificación (ratio={std['diversification_ratio']:.3f})")

    return AnalysisResult(
        tickers=available_tickers,
        expected_returns=mu,
        cov_matrix=cov,
        weights_sharpe=weights_sharpe,
        weights_minvar=weights_minvar,
        weights_cvar=weights_cvar,
        chart_path=chart_path,
        warnings=warnings_list,
        stats_sharpe=stats_sharpe,
        stats_minvar=stats_minvar,
        stats_cvar=stats_cvar,
        start_date=start_date,
        end_date=end_date,  # noqa: F821 - preserved verbatim from Cell 10 before Cell 13 override
        sample_days=sample_days,
        used_expected=dcf_upsides_clean,
        hist_performance=hist_performance,
        mcr_sharpe=mcr_sharpe,
        mcr_minvar=mcr_minvar,
        mcr_cvar=mcr_cvar,
        prices_hist=prices,
        historical_perf_chart_path=historical_perf_chart_path,
        correlation_heatmap_path=correlation_heatmap_path,
        mcr_chart_path=mcr_chart_path,
        patrimonio_chart_path=patrimonio_chart_path,
        total_return_chart_path=total_return_chart_path,
        individual_returns_chart_path=individual_returns_chart_path,
        is_existing_portfolio=is_existing,
        stress_test_results=stress_results,
        mu_sources=mu_source,
        vix_level=vix_current,
        cov_inflation_factor=cov_inflation
    )


# Preserve reference to the pre-override Cell 10 definition before Cell 13 redefines it.
_analyze_portfolio_cell10 = analyze_portfolio


# =============================================================================
# Cell 12: Black-Litterman, Shrinkage, Regime Covariance, Stress Tests, Pipeline μ
# =============================================================================

def get_market_caps(tickers_list):
    caps = {}
    for t in tickers_list:
        try:
            info = yf.Ticker(t).info
            cap = info.get('marketCap', None)
            if cap and cap > 0:
                caps[t] = float(cap)
        except Exception:
            pass
    return caps


def compute_market_weights(tickers_list, market_caps=None):
    if market_caps is None:
        market_caps = get_market_caps(tickers_list)
    available = {t: market_caps[t] for t in tickers_list if t in market_caps and market_caps[t] > 0}
    if len(available) < max(2, int(len(tickers_list) * 0.5)):
        n = len(tickers_list)
        return pd.Series([1.0 / n] * n, index=tickers_list)
    median_cap = float(np.median(list(available.values())))
    all_caps = {t: available.get(t, median_cap) for t in tickers_list}
    total = sum(all_caps.values())
    return pd.Series({t: c / total for t, c in all_caps.items()})


def compute_equilibrium_returns(cov_matrix, market_weights, risk_aversion=2.5):
    w = market_weights.reindex(cov_matrix.index).fillna(1.0 / len(cov_matrix))
    pi = risk_aversion * cov_matrix.values @ w.values
    return pd.Series(pi, index=cov_matrix.index)


def black_litterman_returns(cov_matrix, market_weights, risk_aversion=2.5, tau=0.05,
                            P=None, Q=None, omega=None):
    pi = compute_equilibrium_returns(cov_matrix, market_weights, risk_aversion)
    if P is None or Q is None or len(Q) == 0:
        return pi
    tau_sigma = tau * cov_matrix.values
    if omega is None:
        omega = np.diag(np.diag(P @ tau_sigma @ P.T)) * 10
    try:
        inv_tau_sigma = np.linalg.inv(tau_sigma)
        inv_omega = np.linalg.inv(omega)
        M = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
        mu_bl = M @ (inv_tau_sigma @ pi.values + P.T @ inv_omega @ Q)
        return pd.Series(mu_bl, index=cov_matrix.index)
    except np.linalg.LinAlgError:
        return pi


def build_ml_views_for_bl(tickers_list, ml_predictions_5y, cov_matrix):
    if not ml_predictions_5y:
        return None, None, None
    available = [t for t in tickers_list if t in ml_predictions_5y]
    if len(available) < 2:
        return None, None, None

    def _get_return(t):
        try:
            return float(ml_predictions_5y[t]['scenarios']['base']['predicted_return'])
        except (KeyError, TypeError):
            return 0.0

    sorted_by_return = sorted(available, key=_get_return, reverse=True)
    views_P = []
    views_Q = []
    n_views = min(len(sorted_by_return) // 2, 3)
    for i in range(n_views):
        long_t = sorted_by_return[i]
        short_t = sorted_by_return[-(i + 1)]
        if long_t == short_t:
            continue
        spread = (_get_return(long_t) - _get_return(short_t)) / 5.0 * 0.5
        if abs(spread) < 0.001:
            continue
        p_row = np.zeros(len(tickers_list))
        p_row[tickers_list.index(long_t)] = 1.0
        p_row[tickers_list.index(short_t)] = -1.0
        views_P.append(p_row)
        views_Q.append(spread)
    if not views_P:
        return None, None, None
    P = np.array(views_P)
    Q = np.array(views_Q)
    tau_sigma = 0.05 * cov_matrix.values
    omega = np.diag(np.diag(P @ tau_sigma @ P.T)) * 20
    return P, Q, omega


def shrinkage_covariance(daily_returns, method='ledoit_wolf'):
    if method == 'oas':
        estimator = OAS_estimator().fit(daily_returns.values)
    else:
        estimator = LedoitWolf().fit(daily_returns.values)
    cov_shrunk = pd.DataFrame(
        estimator.covariance_, index=daily_returns.columns, columns=daily_returns.columns
    ) * 252
    return cov_shrunk, getattr(estimator, 'shrinkage_', None)


def regime_adjusted_covariance(daily_returns, stress_weight=0.30, lookback_stress=63,
                                shrinkage_method='ledoit_wolf'):
    cov_calm, shrink_calm = shrinkage_covariance(daily_returns, method=shrinkage_method)
    print(f"  Shrinkage ({shrinkage_method}): coef={shrink_calm:.4f}" if shrink_calm is not None else f"  Shrinkage ({shrinkage_method})")

    rolling_vol = daily_returns.rolling(21).std().mean(axis=1).dropna()
    cov_crisis = cov_calm.copy()

    if len(rolling_vol) > lookback_stress:
        vol_threshold = rolling_vol.quantile(0.75)
        stress_mask = rolling_vol > vol_threshold
        stress_returns = daily_returns.loc[stress_mask.index[stress_mask]]
        if len(stress_returns) >= 30:
            try:
                cov_crisis, shrink_crisis = shrinkage_covariance(stress_returns, method=shrinkage_method)
                print(f"  Régimen crisis: {len(stress_returns)} obs, shrinkage={shrink_crisis:.4f}" if shrink_crisis is not None else f"  Régimen crisis: {len(stress_returns)} obs")
            except Exception:
                cov_crisis = cov_calm * 1.5
                print("  Régimen crisis: fallback 1.5x")
        else:
            cov_crisis = cov_calm * 1.5
            print("  Régimen crisis: fallback 1.5x (obs insuficientes)")
    else:
        cov_crisis = cov_calm * 1.5
        print("  Régimen crisis: fallback 1.5x (serie corta)")

    p_calm = 1.0 - stress_weight
    p_crisis = stress_weight
    cov_blended = p_calm * cov_calm + p_crisis * cov_crisis
    print(f"  Σ_final = {p_calm:.0%}·Σ_calma + {p_crisis:.0%}·Σ_crisis")

    return cov_blended, cov_calm, cov_crisis


def get_current_vix():
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if not hist.empty:
            return float(hist['Close'].dropna().iloc[-1])
    except Exception:
        pass
    return None


def forward_looking_cov_adjustment(cov_matrix, vix_current=None):
    factor = 1.0
    if vix_current is not None and np.isfinite(vix_current):
        if vix_current > 35:
            factor = 1.40
        elif vix_current > 30:
            factor = 1.25
        elif vix_current > 25:
            factor = 1.15
        elif vix_current > 20:
            factor = 1.05
    return cov_matrix * factor, factor


def stress_test_corr_one(weights, cov_matrix, tickers_list):
    w = np.array(weights)
    cov_vals = cov_matrix.values if hasattr(cov_matrix, 'values') else np.array(cov_matrix)
    vols = np.sqrt(np.diag(cov_vals))
    stress_vol = float(np.sum(w * vols))
    normal_vol = float(np.sqrt(w @ cov_vals @ w))
    div_ratio = normal_vol / stress_vol if stress_vol > 0 else 1.0
    stress_contrib = (w * vols / stress_vol) if stress_vol > 0 else w
    recommendations = []
    if div_ratio > 0.90:
        recommendations.append("CRÍTICO: Diversificación casi nula bajo estrés.")
    elif div_ratio > 0.85:
        recommendations.append("ADVERTENCIA: Diversificación limitada bajo estrés.")
    for i, t in enumerate(tickers_list):
        if stress_contrib[i] > 0.30:
            recommendations.append(f"Reducir {t} ({stress_contrib[i]*100:.1f}% del riesgo estrés)")
    return {
        'normal_vol': normal_vol,
        'stress_vol': stress_vol,
        'diversification_ratio': div_ratio,
        'vol_increase_pct': ((stress_vol / normal_vol) - 1) * 100 if normal_vol > 0 else 0,
        'false_diversification': div_ratio > 0.85,
        'stress_contributions': pd.Series(stress_contrib, index=tickers_list),
        'recommendations': recommendations
    }


def stress_test_vol_shock(weights, cov_matrix, tickers_list, shock_factor=2.0):
    w = np.array(weights)
    cov_vals = cov_matrix.values if hasattr(cov_matrix, 'values') else np.array(cov_matrix)
    normal_vol = float(np.sqrt(w @ cov_vals @ w))
    shocked_cov = cov_vals * (shock_factor ** 2)
    stress_vol = float(np.sqrt(w @ shocked_cov @ w))
    recommendations = []
    if stress_vol > 0.60:
        recommendations.append(f"CRÍTICO: Vol estrés {stress_vol*100:.1f}% supera umbral 60%.")
    return {
        'normal_vol': normal_vol,
        'stress_vol': stress_vol,
        'shock_factor': shock_factor,
        'vol_increase_pct': ((stress_vol / normal_vol) - 1) * 100 if normal_vol > 0 else 0,
        'recommendations': recommendations
    }


def stress_test_systemic(weights, cov_matrix, tickers_list, corr_floor=0.80, vol_shock=1.5):
    w = np.array(weights)
    cov_vals = cov_matrix.values if hasattr(cov_matrix, 'values') else np.array(cov_matrix)
    vols = np.sqrt(np.diag(cov_vals))
    normal_vol = float(np.sqrt(w @ cov_vals @ w))
    shocked_vols = vols * vol_shock
    n = len(vols)
    stress_corr = np.full((n, n), corr_floor)
    np.fill_diagonal(stress_corr, 1.0)
    stress_cov = np.outer(shocked_vols, shocked_vols) * stress_corr
    stress_vol = float(np.sqrt(w @ stress_cov @ w))
    max_loss_1d = stress_vol / np.sqrt(252) * 2.33
    recommendations = []
    if stress_vol > normal_vol * 2.5:
        recommendations.append("CRÍTICO: Portfolio colapsa bajo crisis sistémica.")
    elif stress_vol > normal_vol * 2.0:
        recommendations.append("ADVERTENCIA: Riesgo elevado bajo crisis sistémica.")
    return {
        'normal_vol': normal_vol,
        'stress_vol': stress_vol,
        'vol_increase_pct': ((stress_vol / normal_vol) - 1) * 100 if normal_vol > 0 else 0,
        'corr_floor': corr_floor,
        'vol_shock_factor': vol_shock,
        'max_daily_loss_99': max_loss_1d,
        'recommendations': recommendations
    }


def run_all_stress_tests(weights, cov_matrix, tickers_list):
    return {
        'corr_one': stress_test_corr_one(weights, cov_matrix, tickers_list),
        'vol_shock': stress_test_vol_shock(weights, cov_matrix, tickers_list, shock_factor=2.0),
        'systemic': stress_test_systemic(weights, cov_matrix, tickers_list, corr_floor=0.80, vol_shock=1.5)
    }


def compute_expected_returns_pipeline(tickers_list, daily_returns, cov_matrix,
                                      dcf_upsides, ml_predictions_5y=None,
                                      country_risk_map=None, country_risk_value=0.0):
    n = len(tickers_list)
    mu_final = pd.Series(np.zeros(n), index=tickers_list)
    mu_source = {}
    dcf_upsides = dcf_upsides or {}
    has_any_dcf = len(dcf_upsides) > 0

    if has_any_dcf:
        print("  🔒 Pipeline A: μ_DCF detectado → BL completamente excluido")
        hist_annual = daily_returns.mean() * 252
        for t in tickers_list:
            if t in dcf_upsides and dcf_upsides[t] != 0:
                mu_final[t] = dcf_upsides[t]
                mu_source[t] = "DCF/Usuario (sagrado, sin modificar)"
            else:
                mu_final[t] = hist_annual[t]
                mu_source[t] = "Media histórica anualizada (Pipeline A, sin BL)"
    else:
        print("  Sin μ_DCF → Ejecutando Black-Litterman con prior de mercado...")
        market_caps = get_market_caps(tickers_list)
        w_mkt = compute_market_weights(tickers_list, market_caps)
        P, Q, omega = None, None, None
        if ml_predictions_5y:
            P, Q, omega = build_ml_views_for_bl(tickers_list, ml_predictions_5y, cov_matrix)
            if P is not None:
                print(f"  🤖 {len(Q)} views ML relativas (Ω inflada 20x)")
        mu_bl = black_litterman_returns(cov_matrix, w_mkt, risk_aversion=2.5, tau=0.05,
                                        P=P, Q=Q, omega=omega)
        for t in tickers_list:
            mu_final[t] = mu_bl[t]
            if P is not None:
                mu_source[t] = "Black-Litterman (mercado + ML views baja confianza)"
            else:
                mu_source[t] = "Black-Litterman (equilibrio de mercado)"

    country_risk_map = country_risk_map or {}
    non_dcf_tickers = [t for t in tickers_list if t not in dcf_upsides]

    global_country_risk_premia = state.country_risk_premia or {}

    for t in non_dcf_tickers:
        pais = country_risk_map.get(t, "")
        if pais == "Argentina" and country_risk_value > 0:
            mu_final[t] -= country_risk_value
            mu_source[t] += f" - Riesgo País AR (-{country_risk_value*100:.2f}%)"
        elif pais in global_country_risk_premia:
            risk_score = global_country_risk_premia[pais]
            excess_vol_ratio = max(0, risk_score - 1.0)
            adjustment = min(excess_vol_ratio * 0.03, 0.15)
            if adjustment > 0.001:
                mu_final[t] -= adjustment
                mu_source[t] += f" - Riesgo País {pais} (-{adjustment*100:.2f}%)"

    return mu_final, mu_source


def get_sector_map(tickers_list):
    sectors = {}
    for t in tickers_list:
        try:
            info = yf.Ticker(t).info
            sectors[t] = info.get('sector', 'Unknown')
        except Exception:
            sectors[t] = 'Unknown'
    return sectors


def compute_stress_weight_caps(cov_matrix, tickers_list, floor=0.05, ceiling=0.20):
    cov_vals = cov_matrix.values if hasattr(cov_matrix, 'values') else np.array(cov_matrix)
    vols = np.sqrt(np.diag(cov_vals))
    median_vol = np.median(vols)
    caps = []
    for i, t in enumerate(tickers_list):
        vol_ratio = vols[i] / median_vol if median_vol > 0 else 1.0
        if vol_ratio > 2.5:
            cap = 0.10
        elif vol_ratio > 2.0:
            cap = 0.15
        elif vol_ratio > 1.5:
            cap = 0.20
        else:
            cap = ceiling
        caps.append((floor, max(floor, cap)))
    return caps


def optimize_with_sector_and_stress(mu, cov_matrix, rf, tickers_list,
                                     hhi_limit=0.35, max_sector_weight=0.40,
                                     sector_map=None, stress_bounds=None):
    n = len(mu)
    if sector_map is None:
        sector_map = {}

    bounds = stress_bounds if stress_bounds else compute_stress_weight_caps(cov_matrix, tickers_list)

    def neg_sharpe(w):
        ret = w @ mu.values
        vol = np.sqrt(w @ cov_matrix.values @ w)
        return -(ret - rf) / vol if vol > 1e-10 else 1e10

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: hhi_limit - np.sum(w ** 2)}
    ]

    sector_groups = {}
    for i, t in enumerate(tickers_list):
        s = sector_map.get(t, 'Unknown')
        if s not in sector_groups:
            sector_groups[s] = []
        sector_groups[s].append(i)

    for sector, indices in sector_groups.items():
        if len(indices) >= 2 and sector != 'Unknown':
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, idx=indices: max_sector_weight - sum(w[j] for j in idx)
            })

    w0 = np.array([1.0 / n] * n)
    result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 2000, 'ftol': 1e-12})
    if result.success:
        hhi_val = np.sum(result.x ** 2)
        print(f"✅ Optimización exitosa — HHI={hhi_val:.4f} (límite: {hhi_limit})")
        return result.x
    print("⚠️ Optimización no convergió. Usando pesos iguales.")
    return w0


# =============================================================================
# Cell 13: Effective redefinition of analyze_portfolio (overrides Cell 10)
# =============================================================================

def analyze_portfolio(assets: Iterable[dict], rf_rate: Optional[float] = None,  # noqa: F811
                      samples: int = 2000, dcf_upsides: Optional[dict] = None,
                      ml_predictions_5y: Optional[dict] = None,
                      investment_horizon: int = 3, sales: Optional[List[dict]] = None,
                      country_risk_argentina: Optional[float] = None,
                      shrinkage_method: str = 'ledoit_wolf') -> AnalysisResult:
    if rf_rate is None:
        rf_rate = obtener_tasa_libre_riesgo()

    tickers = sorted({a.get("ticker", "").upper() for a in assets if a.get("ticker")})
    if not tickers:
        raise ValueError("No hay tickers para analizar.")

    sales = sales or []
    assets_list = list(assets)
    is_existing = _is_existing_portfolio(assets_list)

    if is_existing:
        buy_dates = []
        for asset in assets_list:
            if asset.get("mode") == "existing" and asset.get("buy_date"):
                buy_dates.append(pd.Timestamp(asset.get("buy_date")))
        if buy_dates:
            earliest_date = min(buy_dates)
            start_download = earliest_date - pd.Timedelta(days=30)
            prices, dropped = _download_prices(tickers, period="max")
            if not prices.empty:
                start_download_tz = pd.Timestamp(start_download)
                if prices.index.tz is not None:
                    start_download_tz = start_download_tz.tz_localize('UTC')
                prices = prices[prices.index >= start_download_tz]
        else:
            prices, dropped = _download_prices(tickers, period="10y")
    else:
        prices, dropped = _download_prices(tickers, period="10y")

    if prices.empty:
        raise ValueError("No se pudieron descargar precios para los tickers indicados.")

    available_tickers = [t for t in prices.columns]
    start_date = prices.index.min() if not prices.empty else None
    end_date = prices.index.max() if not prices.empty else None

    prices_daily = prices.resample('D').last().ffill()
    daily = prices_daily.pct_change(fill_method=None).dropna(how='all')
    daily = daily.dropna(axis=1, how='all')
    daily = daily.fillna(0)
    available_tickers = list(daily.columns)

    if daily.empty or len(available_tickers) < 2:
        raise ValueError("No hay retornos suficientes para el análisis.")

    print(f"\n📐 Σ: {shrinkage_method.upper()} shrinkage + régimen 70/30 + forward-looking VIX")
    cov, cov_calm, cov_crisis = regime_adjusted_covariance(
        daily, stress_weight=0.30, lookback_stress=63, shrinkage_method=shrinkage_method
    )

    vix_current = get_current_vix()
    cov, cov_inflation = forward_looking_cov_adjustment(cov, vix_current)
    if cov_inflation > 1.0:
        print(f"  VIX={vix_current:.1f} → Inflación Σ: {cov_inflation:.2f}x")
    else:
        print(f"  VIX={'N/A' if vix_current is None else f'{vix_current:.1f}'} → Sin inflación")

    if country_risk_argentina is None:
        country_risk_argentina = state.prima_riesgo_mas_reciente
        if country_risk_argentina is not None and country_risk_argentina > 1.0:
            country_risk_argentina = country_risk_argentina / 100.0
        elif country_risk_argentina is None:
            country_risk_argentina = 0.0

    asset_country_risk_map = {}
    for asset in assets_list:
        ticker_a = asset.get("ticker", "").upper()
        pais_riesgo = asset.get("pais_riesgo", asset.get("pais", ""))
        if ticker_a and pais_riesgo:
            asset_country_risk_map[ticker_a] = pais_riesgo

    dcf_upsides_clean = {}
    if dcf_upsides:
        dcf_upsides_clean = {k.upper(): v for k, v in dcf_upsides.items()
                             if isinstance(v, (int, float)) and v != 0}

    print("\n📌 μ pipeline...")
    mu, mu_source = compute_expected_returns_pipeline(
        tickers_list=available_tickers,
        daily_returns=daily,
        cov_matrix=cov,
        dcf_upsides=dcf_upsides_clean,
        ml_predictions_5y=ml_predictions_5y,
        country_risk_map=asset_country_risk_map,
        country_risk_value=country_risk_argentina or 0.0
    )
    for t_src, src in mu_source.items():
        print(f"  {t_src}: μ={mu[t_src]*100:.2f}% → {src}")

    rf = rf_rate
    sample_days = daily.shape[0]

    weights, port_ret, port_vol, cvars = _simulate_portfolios(mu, cov, daily, rf, samples)
    sharpe = (port_ret - rf) / port_vol

    print("\n🎯 Optimización HHI + sector + stress bounds...")
    sector_map = get_sector_map(available_tickers)
    stress_bounds = compute_stress_weight_caps(cov, available_tickers)

    for _stress_iter in range(3):
        weights_sharpe = optimize_with_sector_and_stress(mu, cov, rf, available_tickers,
                                                          hhi_limit=0.35, max_sector_weight=0.40,
                                                          sector_map=sector_map, stress_bounds=stress_bounds)
        weights_minvar = _optimize_min_variance(mu, cov, stress_bounds=stress_bounds)
        weights_cvar = _optimize_min_cvar(mu, cov, daily, stress_bounds=stress_bounds)

        _tightened = False
        for _w_test in [weights_sharpe, weights_minvar, weights_cvar]:
            _st = stress_test_corr_one(_w_test, cov, available_tickers)
            for _i, _t in enumerate(available_tickers):
                if _st['stress_contributions'][_t] > 0.20:
                    _old_floor, _old_ceil = stress_bounds[_i]
                    _new_ceil = max(_old_floor, _old_ceil * 0.70)
                    if _new_ceil < _old_ceil:
                        stress_bounds[_i] = (_old_floor, _new_ceil)
                        _tightened = True
                        print(f"  ⚡ Stress limiter: {_t} cap {_old_ceil*100:.0f}%→{_new_ceil*100:.0f}%")
        if not _tightened:
            break
    if _stress_iter > 0:
        print(f"  Stress limiter: {_stress_iter + 1} iteraciones")

    stats_sharpe = _portfolio_metrics(weights_sharpe, mu, cov, daily, rf)
    stats_minvar = _portfolio_metrics(weights_minvar, mu, cov, daily, rf)
    stats_cvar = _portfolio_metrics(weights_cvar, mu, cov, daily, rf)

    mcr_sharpe = _calc_mcr(weights_sharpe, cov, available_tickers)
    mcr_minvar = _calc_mcr(weights_minvar, cov, available_tickers)
    mcr_cvar = _calc_mcr(weights_cvar, cov, available_tickers)

    hist_performance = {
        "sharpe": _calc_historical_performance(prices, weights_sharpe, available_tickers),
        "minvar": _calc_historical_performance(prices, weights_minvar, available_tickers),
        "cvar": _calc_historical_performance(prices, weights_cvar, available_tickers),
    }

    chart_path = _build_static_figure(mu, port_ret, port_vol, sharpe,
                                       Path(tempfile.gettempdir()) / "portfolio_chart.png")
    weights_dict = {"sharpe": weights_sharpe, "minvar": weights_minvar, "cvar": weights_cvar}
    historical_perf_chart_path = _build_historical_performance_plot(
        prices, weights_dict, available_tickers, Path(tempfile.gettempdir()) / "historical_perf_chart.png")
    correlation_heatmap_path = _build_correlation_heatmap(
        prices, available_tickers, Path(tempfile.gettempdir()) / "correlation_heatmap.png")
    mcr_chart_path = _build_mcr_chart(
        mcr_sharpe, mcr_minvar, mcr_cvar, cov, weights_sharpe, weights_minvar, weights_cvar,
        available_tickers, Path(tempfile.gettempdir()) / "mcr_chart.png")

    if is_existing:
        patrimonio_chart_path = _build_real_portfolio_performance(
            prices, assets_list, sales, Path(tempfile.gettempdir()) / "patrimonio_chart.png")
        total_return_chart_path = _build_real_portfolio_with_sales(
            prices, assets_list, sales, Path(tempfile.gettempdir()) / "total_return_chart.png")
        individual_returns_chart_path = _build_real_individual_returns_plot(
            prices, assets_list, sales, available_tickers, Path(tempfile.gettempdir()) / "individual_returns_chart.png")
    else:
        patrimonio_chart_path = _build_patrimonio_plot(
            prices, weights_sharpe, weights_minvar, weights_cvar, available_tickers, assets_list,
            Path(tempfile.gettempdir()) / "patrimonio_chart.png")
        total_return_chart_path = _build_portfolio_strategies_performance_plot(
            prices, weights_sharpe, weights_minvar, weights_cvar, available_tickers,
            Path(tempfile.gettempdir()) / "total_return_chart.png")
        individual_returns_chart_path = _build_individual_returns_plot(
            prices, available_tickers, Path(tempfile.gettempdir()) / "individual_returns_chart.png")

    print("\n🔴 Stress tests: corr=1 | vol shock 2x | crisis sistémica (corr≥0.8 + vol×1.5)...")
    stress_results = {}
    for st_label, st_w in [('sharpe', weights_sharpe), ('minvar', weights_minvar), ('cvar', weights_cvar)]:
        all_st = run_all_stress_tests(st_w, cov, available_tickers)
        entry = all_st['corr_one']
        entry['vol_shock'] = all_st['vol_shock']
        entry['systemic'] = all_st['systemic']
        stress_results[st_label] = entry
    for st_name, st in [('Sharpe', stress_results['sharpe']), ('MinVar', stress_results['minvar']), ('CVaR', stress_results['cvar'])]:
        print(f"  {st_name}: Vol {st['normal_vol']*100:.2f}%→{st['stress_vol']*100:.2f}% "
              f"(+{st['vol_increase_pct']:.1f}%) Diversif={st['diversification_ratio']:.3f}")
        if st['false_diversification']:
            for rec in st['recommendations']:
                print(f"    ⚠️ {rec}")
        sys_st = st['systemic']
        print(f"    Sistémico: Vol→{sys_st['stress_vol']*100:.2f}% MaxLoss1D(99%)={sys_st['max_daily_loss_99']*100:.2f}%")

    warnings_list: List[str] = []
    if dropped:
        warnings_list.append(f"Tickers sin datos: {', '.join(dropped)}")
    if daily.shape[0] < 250:
        warnings_list.append("La serie histórica es corta; los resultados pueden ser inestables.")
    for stn, std in stress_results.items():
        if std.get('false_diversification'):
            warnings_list.append(f"Stress test ({stn}): Falsa diversificación (ratio={std['diversification_ratio']:.3f})")

    return AnalysisResult(
        tickers=available_tickers,
        expected_returns=mu,
        cov_matrix=cov,
        weights_sharpe=weights_sharpe,
        weights_minvar=weights_minvar,
        weights_cvar=weights_cvar,
        chart_path=chart_path,
        warnings=warnings_list,
        stats_sharpe=stats_sharpe,
        stats_minvar=stats_minvar,
        stats_cvar=stats_cvar,
        start_date=start_date,
        end_date=end_date,
        sample_days=sample_days,
        used_expected=dcf_upsides_clean,
        hist_performance=hist_performance,
        mcr_sharpe=mcr_sharpe,
        mcr_minvar=mcr_minvar,
        mcr_cvar=mcr_cvar,
        prices_hist=prices,
        historical_perf_chart_path=historical_perf_chart_path,
        correlation_heatmap_path=correlation_heatmap_path,
        mcr_chart_path=mcr_chart_path,
        patrimonio_chart_path=patrimonio_chart_path,
        total_return_chart_path=total_return_chart_path,
        individual_returns_chart_path=individual_returns_chart_path,
        is_existing_portfolio=is_existing,
        stress_test_results=stress_results,
        mu_sources=mu_source,
        vix_level=vix_current,
        cov_inflation_factor=cov_inflation,
    )
