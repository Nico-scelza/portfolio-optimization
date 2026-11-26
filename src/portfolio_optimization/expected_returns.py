"""Annualized DCF valuation views and multi-asset Black-Litterman expected returns.

Key invariants:
- DCF views are ALWAYS annualized over an explicit positive horizon (never passing
  3-5 year cumulative upside as a 1-year return).
- Present intrinsic value V_0 is explicitly distinguished from terminal value V_T
  and documented as a convergence hypothesis.
- Black-Litterman ALWAYS remains active for the full multi-asset universe even
  when DCF views are present (never switching non-view assets to historical means).
- Excess returns over the annual USD risk-free rate are converted ONCE into BL
  and ONCE out of BL.
- Uncertainty matrix Omega uses Idzorek-consistent calibration with a Woodbury-form
  posterior solve so confidence=1.0 is non-singular and confidence=0.0 is inactive.
- Zero country-risk/sovereign spreads and zero ML views exist in this pipeline.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .errors import ConfigurationError
from .fixed_income import solve_xirr_with_root_check
from .schemas import (
    BlackLittermanConfig,
    BlackLittermanResult,
    DCFViewConfig,
    DCFViewResult,
    FixedIncomeMetricsResult,
    InstrumentConfig,
)


def annualize_dcf_view(
    instrument_id: str,
    view: DCFViewConfig,
    *,
    current_price_usd: Optional[float] = None,
    default_confidence: float = 0.50,
    asset_subtype: str = "equity",
) -> DCFViewResult:
    """Convert a validated DCF valuation view into an annualized expected return view."""
    iid = instrument_id.strip().upper()
    warnings: List[str] = []
    assumptions: List[str] = []

    if asset_subtype != "equity":
        warnings.append(
            f"DCF view supplied for non-equity subtype '{asset_subtype}' ('{iid}'); "
            "fundamental DCF is normally applicable only to equities."
        )

    if view.currency != "USD":
        raise ConfigurationError(
            f"DCF view for '{iid}' is in '{view.currency}'; convert target/price inputs to USD before Black-Litterman integration.",
            asset_id=iid,
        )

    horizon = float(view.horizon_years)
    conf = float(default_confidence if view.confidence is None else view.confidence)
    p0 = view.current_price if view.current_price is not None else current_price_usd

    target_or_iv: Optional[float] = None

    if view.valuation_basis == "terminal_target":
        if view.target_price is not None:
            if p0 is None or p0 <= 0.0:
                raise ConfigurationError(
                    f"DCF terminal_target view for '{iid}' requires a valid positive current USD price P_0.",
                    asset_id=iid,
                )
            ratio = float(view.target_price) / float(p0)
            ann_ret = (ratio ** (1.0 / horizon)) - 1.0
            target_or_iv = float(view.target_price)
            assumptions.append(
                f"Geometric annualization of terminal target V_T={view.target_price:.4f} vs P_0={p0:.4f} over T={horizon:g}y: (V_T/P_0)^(1/T) - 1."
            )
            if view.expected_dividend_yield > 0.0:
                ann_ret += float(view.expected_dividend_yield)
                assumptions.append(
                    f"Added explicit expected_dividend_yield={view.expected_dividend_yield:.2%} (assumed excluded from terminal price V_T)."
                )
        elif view.total_upside is not None:
            ann_ret = ((1.0 + float(view.total_upside)) ** (1.0 / horizon)) - 1.0
            if p0 is not None and p0 > 0.0:
                target_or_iv = float(p0) * (1.0 + float(view.total_upside))
            assumptions.append(
                f"Geometric annualization of cumulative {horizon:g}-year upside {view.total_upside:.2%}: (1 + upside)^(1/T) - 1."
            )
            if view.expected_dividend_yield > 0.0:
                ann_ret += float(view.expected_dividend_yield)
                assumptions.append(
                    f"Added explicit expected_dividend_yield={view.expected_dividend_yield:.2%}."
                )
        else:
            raise ConfigurationError(
                f"DCF terminal_target view for '{iid}' must specify either 'target_price' or 'total_upside'.",
                asset_id=iid,
            )

    elif view.valuation_basis == "present_intrinsic_value":
        if view.present_intrinsic_value is None:
            raise ConfigurationError(
                f"DCF present_intrinsic_value view for '{iid}' requires 'present_intrinsic_value' (V_0).",
                asset_id=iid,
            )
        if p0 is None or p0 <= 0.0:
            raise ConfigurationError(
                f"DCF present_intrinsic_value view for '{iid}' requires a valid positive current USD price P_0.",
                asset_id=iid,
            )
        ratio = float(view.present_intrinsic_value) / float(p0)
        conv_ret = (ratio ** (1.0 / horizon)) - 1.0
        ann_ret = conv_ret + float(view.expected_dividend_yield)
        target_or_iv = float(view.present_intrinsic_value)
        assumptions.append(
            f"Convergence hypothesis: present discounted intrinsic value V_0={view.present_intrinsic_value:.4f} "
            f"converges from P_0={p0:.4f} over T_conv={horizon:g}y (annualized convergence return, NOT a guaranteed CAGR; no double discounting)."
        )
        if view.expected_dividend_yield > 0.0:
            assumptions.append(
                f"Included explicit expected_dividend_yield={view.expected_dividend_yield:.2%} (must not already be double-counted in V_0)."
            )

    elif view.valuation_basis == "dated_cashflows":
        if p0 is None or p0 <= 0.0:
            raise ConfigurationError(
                f"DCF dated_cashflows view for '{iid}' requires positive current price P_0.",
                asset_id=iid,
            )
        if view.target_price is None:
            raise ConfigurationError(
                f"DCF dated_cashflows view for '{iid}' requires terminal 'target_price' at horizon T={horizon:g}y.",
                asset_id=iid,
            )
        cashflows = list(view.distributions) + [(horizon, float(view.target_price))]
        irr, root_count, status = solve_xirr_with_root_check(float(p0), cashflows)
        if status != "ok" or irr is None:
            raise ConfigurationError(
                f"Could not compute unique annualized IRR for DCF dated_cashflows on '{iid}' (status={status}, roots={root_count}).",
                asset_id=iid,
            )
        ann_ret = float(irr)
        target_or_iv = float(view.target_price)
        assumptions.append(
            f"Total-return annualized IRR over T={horizon:g}y combining {len(view.distributions)} dated distribution(s) and terminal value V_T={view.target_price:.4f} vs P_0={p0:.4f} (no double counting)."
        )
    else:
        raise ConfigurationError(f"Unsupported DCF valuation_basis '{view.valuation_basis}'.", asset_id=iid)

    if not math.isfinite(ann_ret):
        raise ConfigurationError(f"Computed annualized DCF return for '{iid}' is non-finite.", asset_id=iid)

    active = conf > 0.0
    if not active:
        warnings.append(f"DCF view for '{iid}' has confidence=0.0 and is excluded from Black-Litterman views.")

    return DCFViewResult(
        instrument_id=iid,
        valuation_basis=view.valuation_basis,
        horizon_years=horizon,
        current_price_usd=None if p0 is None else float(p0),
        target_or_intrinsic_usd=target_or_iv,
        annualized_expected_return=float(ann_ret),
        confidence=conf,
        active_in_bl=active,
        currency="USD",
        assumptions=assumptions,
        warnings=warnings,
    )


def resolve_prior_reference_weights(
    tickers: List[str],
    cov_matrix: pd.DataFrame,
    bl_config: BlackLittermanConfig,
    *,
    instruments_by_id: Optional[Dict[str, InstrumentConfig]] = None,
    market_caps: Optional[Dict[str, float]] = None,
) -> Tuple[pd.Series, str, List[str]]:
    """Resolve defensible multi-asset equilibrium prior reference weights without inventing market caps."""
    n = len(tickers)
    if n == 0:
        raise ConfigurationError("Cannot resolve prior reference weights for an empty universe.")

    warnings: List[str] = []
    policy = bl_config.prior_weights_policy
    explicit_map = {k.upper(): float(v) for k, v in (bl_config.reference_weights or {}).items()}

    def _validate_explicit() -> pd.Series:
        missing = [t for t in tickers if t not in explicit_map]
        if missing:
            raise ConfigurationError(
                f"Explicit prior reference weights required for all assets; missing: {missing}.",
                details={"missing_assets": missing},
            )
        w = np.array([explicit_map[t] for t in tickers], dtype=float)
        if np.any(w < 0.0) or float(np.sum(w)) <= 0.0:
            raise ConfigurationError("Explicit prior reference weights must be non-negative with positive sum.")
        w_sum = float(np.sum(w))
        if not math.isclose(w_sum, 1.0, abs_tol=1e-4):
            warnings.append(
                f"Explicit reference weights summed to {w_sum:.6f}; normalized proportionally to 1.0 for BL equilibrium prior."
            )
        return pd.Series(w / w_sum, index=tickers, dtype=float)

    if bl_config.require_explicit_reference_weights or policy == "explicit":
        return _validate_explicit(), "explicit", warnings

    if explicit_map and all(t in explicit_map for t in tickers):
        return _validate_explicit(), "explicit", warnings

    all_equity = True
    if instruments_by_id:
        for t in tickers:
            inst = instruments_by_id.get(t)
            if inst is not None and inst.asset_subtype not in ("equity", "equity_etf"):
                all_equity = False
                break

    if policy in ("market_cap_equity_only", "explicit_or_warned_equal_weight"):
        if all_equity and market_caps:
            valid_caps = {t: float(market_caps[t]) for t in tickers if t in market_caps and market_caps[t] is not None and float(market_caps[t]) > 0.0}
            if len(valid_caps) == n:
                total = sum(valid_caps.values())
                w_mc = pd.Series({t: valid_caps[t] / total for t in tickers}, index=tickers, dtype=float)
                return w_mc, "market_cap_equity_only", warnings
        if policy == "market_cap_equity_only":
            raise ConfigurationError(
                "prior_weights_policy='market_cap_equity_only' requires a pure-equity universe with complete positive market caps; "
                "cannot fabricate market caps for non-equity or missing assets.",
            )

    if policy == "inverse_volatility":
        vols = np.sqrt(np.maximum(np.diag(cov_matrix.loc[tickers, tickers].to_numpy(dtype=float)), 1e-12))
        inv_vol = 1.0 / vols
        w_iv = pd.Series(inv_vol / np.sum(inv_vol), index=tickers, dtype=float)
        warnings.append(
            "Black-Litterman prior uses inverse-volatility reference weights (configurable via black_litterman.reference_weights)."
        )
        return w_iv, "inverse_volatility", warnings

    w_eq = pd.Series(np.full(n, 1.0 / n, dtype=float), index=tickers, dtype=float)
    if policy == "equal_weight":
        warnings.append("Black-Litterman prior uses equal reference weights (1/n) as explicitly configured.")
        return w_eq, "equal_weight", warnings

    warnings.append(
        "Multi-asset or incomplete market-cap universe without explicit prior weights: using neutral equal reference weights (1/n) "
        "without fabricating market caps for bonds, gold, or crypto. Configure 'black_litterman.reference_weights' to override."
    )
    return w_eq, "equal_weight_fallback", warnings


def compute_equilibrium_returns(
    cov_matrix: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float = 2.5,
) -> pd.Series:
    """Compute implied equilibrium EXCESS returns ``Pi = delta * Sigma * w_ref``."""
    tickers = list(cov_matrix.index)
    w = market_weights.reindex(tickers).to_numpy(dtype=float)
    if not np.all(np.isfinite(w)):
        raise ConfigurationError("Reference weights contain NaN/Inf values.")
    pi = float(risk_aversion) * (cov_matrix.to_numpy(dtype=float) @ w)
    return pd.Series(pi, index=tickers, dtype=float)


def compute_black_litterman(
    cov_matrix: pd.DataFrame,
    reference_weights: pd.Series,
    *,
    rf_rate: float,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    absolute_views_total: Optional[Dict[str, Tuple[float, float]]] = None,
    prior_policy_used: str = "explicit",
    dcf_view_results: Optional[Dict[str, DCFViewResult]] = None,
    view_source_labels: Optional[Dict[str, str]] = None,
    prior_warnings: Optional[List[str]] = None,
) -> BlackLittermanResult:
    """Run multi-asset Black-Litterman on consistent annual USD excess returns."""
    tickers = list(cov_matrix.index)
    sigma = cov_matrix.loc[tickers, tickers].to_numpy(dtype=float)
    n = len(tickers)
    warnings_list = list(prior_warnings or [])

    pi_excess = compute_equilibrium_returns(cov_matrix.loc[tickers, tickers], reference_weights, risk_aversion)
    mu_prior_total = pi_excess + float(rf_rate)

    active_assets: List[str] = []
    inactive_assets: List[str] = []
    view_confidences: Dict[str, float] = {}
    p_rows: List[np.ndarray] = []
    q_excess_list: List[float] = []
    omega_diag: List[float] = []

    tau_sigma = float(tau) * sigma
    source_labels = view_source_labels or {}

    for asset_id, (q_total, conf_raw) in (absolute_views_total or {}).items():
        aid = asset_id.strip().upper()
        if aid not in tickers:
            warnings_list.append(f"View for '{aid}' ignored because asset is not in the active covariance universe.")
            continue
        conf = float(conf_raw)
        if not math.isfinite(conf) or conf < 0.0 or conf > 1.0:
            raise ConfigurationError(f"View confidence for '{aid}' ({conf}) must be in [0.0, 1.0].", asset_id=aid)
        if not math.isfinite(float(q_total)):
            raise ConfigurationError(f"View expected return for '{aid}' must be finite.", asset_id=aid)

        view_confidences[aid] = conf
        if conf == 0.0:
            inactive_assets.append(aid)
            continue

        idx = tickers.index(aid)
        p_row = np.zeros(n, dtype=float)
        p_row[idx] = 1.0
        view_var = float(tau_sigma[idx, idx])
        if view_var <= 0.0:
            warnings_list.append(f"Asset '{aid}' has non-positive prior variance; excluding view to avoid degeneracy.")
            inactive_assets.append(aid)
            continue

        omega_k = ((1.0 - conf) / conf) * view_var
        p_rows.append(p_row)
        q_excess_list.append(float(q_total) - float(rf_rate))
        omega_diag.append(omega_k)
        active_assets.append(aid)

    mu_sources: Dict[str, str] = {}
    diagnostics: Dict[str, Any] = {
        "return_basis": "excess_return_over_rf_converted_once",
        "omega_calibration": "idzorek_absolute_view",
        "posterior_solver": "woodbury_no_omega_inversion",
        "active_view_count": len(active_assets),
        "inactive_view_count": len(inactive_assets),
    }

    if not active_assets:
        diagnostics["status"] = "no_active_views_prior_returned"
        warnings_list.append(
            "No active views available for Black-Litterman; returning equilibrium prior (rf + Pi) without switching methods."
        )
        for t in tickers:
            mu_sources[t] = f"Black-Litterman prior ({prior_policy_used}, no active views)"
        return BlackLittermanResult(
            tickers=tickers,
            rf_rate=float(rf_rate),
            risk_aversion=float(risk_aversion),
            tau=float(tau),
            prior_policy_used=prior_policy_used,
            reference_weights=reference_weights.reindex(tickers),
            prior_excess_returns=pi_excess,
            prior_total_returns=mu_prior_total,
            posterior_excess_returns=pi_excess.copy(),
            posterior_total_returns=mu_prior_total.copy(),
            mu_sources=mu_sources,
            active_view_assets=[],
            inactive_view_assets=inactive_assets,
            P=None,
            Q_excess=None,
            omega=None,
            view_confidences=view_confidences,
            dcf_view_results=dict(dcf_view_results or {}),
            diagnostics=diagnostics,
            warnings=warnings_list,
        )

    P = np.vstack(p_rows)
    Q = np.asarray(q_excess_list, dtype=float)
    Omega = np.diag(np.asarray(omega_diag, dtype=float))

    S = P @ tau_sigma @ P.T + Omega
    S = 0.5 * (S + S.T)
    innovation = Q - (P @ pi_excess.to_numpy(dtype=float))

    jitter_applied = 0.0
    try:
        cond = float(np.linalg.cond(S))
        if not math.isfinite(cond) or cond > 1e12:
            raise np.linalg.LinAlgError("Ill-conditioned S matrix")
        gain_rhs = np.linalg.solve(S, innovation)
    except np.linalg.LinAlgError:
        jitter_applied = 1e-10 * max(float(np.mean(np.diag(S))), 1.0)
        S_reg = S + np.eye(S.shape[0]) * jitter_applied
        gain_rhs = np.linalg.solve(S_reg, innovation)
        warnings_list.append(
            f"Applied diagonal jitter ({jitter_applied:.2e}) to Black-Litterman view system matrix for numerical stability."
        )

    pi_post_arr = pi_excess.to_numpy(dtype=float) + (tau_sigma @ P.T @ gain_rhs)
    pi_post = pd.Series(pi_post_arr, index=tickers, dtype=float)
    mu_post_total = pi_post + float(rf_rate)

    diagnostics["status"] = "posterior_computed"
    diagnostics["jitter_applied"] = jitter_applied

    for t in tickers:
        if t in active_assets:
            lbl = source_labels.get(t, "DCF view")
            mu_sources[t] = f"Black-Litterman posterior ({lbl}, confidence={view_confidences[t]:.0%})"
        else:
            mu_sources[t] = f"Black-Litterman posterior (prior {prior_policy_used} + cross-asset correlation update)"

    return BlackLittermanResult(
        tickers=tickers,
        rf_rate=float(rf_rate),
        risk_aversion=float(risk_aversion),
        tau=float(tau),
        prior_policy_used=prior_policy_used,
        reference_weights=reference_weights.reindex(tickers),
        prior_excess_returns=pi_excess,
        prior_total_returns=mu_prior_total,
        posterior_excess_returns=pi_post,
        posterior_total_returns=mu_post_total,
        mu_sources=mu_sources,
        active_view_assets=active_assets,
        inactive_view_assets=inactive_assets,
        P=P,
        Q_excess=Q,
        omega=Omega,
        view_confidences=view_confidences,
        dcf_view_results=dict(dcf_view_results or {}),
        diagnostics=diagnostics,
        warnings=warnings_list,
    )


def compute_expected_returns_pipeline(
    *,
    tickers: List[str],
    cov_matrix: pd.DataFrame,
    rf_rate: float,
    bl_config: Optional[BlackLittermanConfig] = None,
    instruments_by_id: Optional[Dict[str, InstrumentConfig]] = None,
    latest_prices_usd: Optional[Dict[str, float]] = None,
    fixed_income_metrics: Optional[Dict[str, FixedIncomeMetricsResult]] = None,
    market_caps: Optional[Dict[str, float]] = None,
) -> BlackLittermanResult:
    """Unified expected returns pipeline: Annualized DCF (+ optional FI YTM view) -> Black-Litterman."""
    cfg = bl_config or BlackLittermanConfig()
    inst_map = instruments_by_id or {}
    prices_map = latest_prices_usd or {}
    fi_map = fixed_income_metrics or {}

    ref_weights, prior_policy, prior_warnings = resolve_prior_reference_weights(
        tickers,
        cov_matrix,
        cfg,
        instruments_by_id=inst_map,
        market_caps=market_caps,
    )

    dcf_results: Dict[str, DCFViewResult] = {}
    views_total: Dict[str, Tuple[float, float]] = {}
    view_labels: Dict[str, str] = {}
    extra_warnings: List[str] = list(prior_warnings)

    for iid in tickers:
        inst = inst_map.get(iid)
        if inst is None:
            continue

        if inst.dcf_view is not None:
            dcf_res = annualize_dcf_view(
                iid,
                inst.dcf_view,
                current_price_usd=prices_map.get(iid),
                default_confidence=cfg.default_dcf_confidence,
                asset_subtype=inst.asset_subtype,
            )
            dcf_results[iid] = dcf_res
            extra_warnings.extend(dcf_res.warnings)
            views_total[iid] = (dcf_res.annualized_expected_return, dcf_res.confidence)
            view_labels[iid] = f"Annualized DCF ({dcf_res.horizon_years:g}y {dcf_res.valuation_basis})"
        elif (
            cfg.fixed_income_bl_view_mode == "explicit_ytm_view"
            and iid in fi_map
            and fi_map[iid].status == "ok"
            and fi_map[iid].ytm_annual is not None
        ):
            fi_m = fi_map[iid]
            if fi_m.cashflow_currency == "USD":
                views_total[iid] = (float(fi_m.ytm_annual), float(cfg.fixed_income_view_confidence))
                view_labels[iid] = "Explicit fixed-income contractual YTM view (hold-to-maturity/reinvestment hypothesis)"
                extra_warnings.append(
                    f"Fixed-income contractual YTM ({fi_m.ytm_annual:.2%}) integrated into BL for '{iid}' under explicit hold-to-maturity & reinvestment hypothesis (confidence={cfg.fixed_income_view_confidence:.0%})."
                )
            else:
                extra_warnings.append(
                    f"Fixed-income YTM for '{iid}' is in {fi_m.cashflow_currency} (not USD); kept as standalone metric rather than USD BL view."
                )

    return compute_black_litterman(
        cov_matrix=cov_matrix,
        reference_weights=ref_weights,
        rf_rate=rf_rate,
        risk_aversion=cfg.risk_aversion,
        tau=cfg.tau,
        absolute_views_total=views_total,
        prior_policy_used=prior_policy,
        dcf_view_results=dcf_results,
        view_source_labels=view_labels,
        prior_warnings=extra_warnings,
    )
