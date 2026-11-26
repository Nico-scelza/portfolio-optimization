"""Multi-asset portfolio constraints, optimizers (Sharpe, MinVar, CVaR, ERC, HRP),
stress testing, and canonical stateless ``analyze_portfolio`` orchestration.

Key invariants:
- Zero hardcoded ``(0.05, 0.20)`` bounds and zero silent ``1/n`` fallbacks.
- Joint LP feasibility validation runs BEFORE optimization and post-solution
  feasibility verification runs AFTER every optimizer.
- ERC and HRP operate purely on covariance/correlation without requiring ``mu``.
- HRP never inverts Sigma and explicitly separates pure HRP from constraint-adjusted HRP.
- Zero global mutable state, zero ML, zero country-risk spreads, zero dead chart stubs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform

from .errors import (
    ConfigurationError,
    DataQualityError,
    FXConversionError,
    InfeasibleConstraintsError,
    InsufficientHistoryError,
    OptimizationConvergenceError,
)
from .expected_returns import compute_expected_returns_pipeline
from .fixed_income import (
    evaluate_fixed_income_instrument,
    normalize_execution_price_to_usd_per_unit,
)
from .iol import IOLPanelSnapshot, fetch_iol_quotes_for_instruments
from .market_data import (
    FXConversionDiagnostics,
    _download_prices,
    calcular_cvar_historico,
    compute_weekly_returns_w_fri,
    fetch_vix_history,
    get_current_vix_observation,
    get_risk_free_rate,
    sanitize_price_series,
)
from .rebalancing import compute_rebalancing_plan
from .risk_model import WEEKS_PER_YEAR, covariance_to_correlation, ensure_psd_covariance, estimate_risk_model
from .schemas import (
    INDIVIDUAL_DEBT_SUBTYPES,
    REMOVED_LEGACY_KEYS,
    AnalysisResult,
    ConstraintsConfig,
    FixedIncomeMetricsResult,
    InstrumentConfig,
    IOLQuoteResult,
    PortfolioAnalysisConfig,
    RebalancingResult,
    RiskModelResult,
    StrategyAllocationResult,
)


@dataclass(frozen=True)
class ResolvedConstraints:
    """Normalized numerical constraints bound to a specific ordered asset list."""

    tickers: Tuple[str, ...]
    bounds: Tuple[Tuple[float, float], ...]
    class_indices: Dict[str, Tuple[int, ...]]
    class_bounds: Dict[str, Tuple[float, float]]
    sector_indices: Dict[str, Tuple[int, ...]]
    max_sector_weight: Optional[float]
    hhi_limit: Optional[float]
    budget_sum: float
    solver_tolerance: float
    feasibility_tolerance: float
    feasible_starting_point: np.ndarray


def _build_linear_ineq_matrices(
    n: int,
    class_indices: Dict[str, Tuple[int, ...]],
    class_bounds: Dict[str, Tuple[float, float]],
    sector_indices: Dict[str, Tuple[int, ...]],
    max_sector_weight: Optional[float],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    rows: List[np.ndarray] = []
    rhs: List[float] = []

    for rc, (lo, hi) in class_bounds.items():
        idxs = class_indices.get(rc, ())
        if lo > 0.0:
            row = np.zeros(n, dtype=float)
            for idx in idxs:
                row[idx] = -1.0
            rows.append(row)
            rhs.append(-lo)
        if hi < 1.0:
            row = np.zeros(n, dtype=float)
            for idx in idxs:
                row[idx] = 1.0
            rows.append(row)
            rhs.append(hi)

    if max_sector_weight is not None and max_sector_weight < 1.0:
        for sec, idxs in sector_indices.items():
            if sec and sec != "Unknown" and len(idxs) >= 1:
                row = np.zeros(n, dtype=float)
                for idx in idxs:
                    row[idx] = 1.0
                rows.append(row)
                rhs.append(max_sector_weight)

    if not rows:
        return None, None
    return np.vstack(rows), np.asarray(rhs, dtype=float)


def validate_constraints_feasibility(
    tickers: Sequence[str],
    constraints: Optional[ConstraintsConfig] = None,
    *,
    risk_class_by_ticker: Optional[Dict[str, str]] = None,
    sector_by_ticker: Optional[Dict[str, str]] = None,
) -> ResolvedConstraints:
    """Validate individual, class, sector, budget, and HHI constraints jointly via exact LP."""
    cfg = constraints or ConstraintsConfig()
    ordered = tuple(t.strip().upper() for t in tickers)
    n = len(ordered)
    if n == 0:
        raise InfeasibleConstraintsError("Cannot optimize an empty asset universe.")

    rc_map = {k.strip().upper(): v.strip().lower() for k, v in (risk_class_by_ticker or {}).items()}
    sec_map = {k.strip().upper(): v for k, v in (sector_by_ticker or {}).items()}

    bounds: List[Tuple[float, float]] = []
    for t in ordered:
        lo, hi = cfg.asset_bounds.get(t, (0.0, 1.0))
        if lo < 0.0 or hi > 1.0 or lo > hi:
            raise InfeasibleConstraintsError(
                f"Invalid bounds ({lo}, {hi}) for '{t}'.",
                asset_id=t,
            )
        bounds.append((float(lo), float(hi)))

    class_indices_mut: Dict[str, List[int]] = {}
    for idx, t in enumerate(ordered):
        rc = rc_map.get(t, "other")
        class_indices_mut.setdefault(rc, []).append(idx)
    class_indices = {rc: tuple(idxs) for rc, idxs in class_indices_mut.items()}

    sector_indices_mut: Dict[str, List[int]] = {}
    for idx, t in enumerate(ordered):
        sec = sec_map.get(t, "Unknown")
        sector_indices_mut.setdefault(sec, []).append(idx)
    sector_indices = {s: tuple(idxs) for s, idxs in sector_indices_mut.items()}

    conflicts: List[str] = []
    suggestions: List[str] = []

    sum_lo = sum(lo for lo, _ in bounds)
    sum_hi = sum(hi for _, hi in bounds)
    if sum_lo > cfg.budget_sum + cfg.feasibility_tolerance:
        conflicts.append(
            f"Sum of individual minimum weights ({sum_lo:.4%}) exceeds budget ({cfg.budget_sum:.4%})."
        )
        suggestions.append("Reduce individual minimum asset weights so their sum is <= 100%.")
    if sum_hi < cfg.budget_sum - cfg.feasibility_tolerance:
        conflicts.append(
            f"Sum of individual maximum weights ({sum_hi:.4%}) is below budget ({cfg.budget_sum:.4%})."
        )
        suggestions.append("Increase individual maximum asset weights or add more assets so upper bounds sum to >= 100%.")

    for rc, (clo, chi) in cfg.class_bounds.items():
        idxs = class_indices.get(rc, ())
        if len(idxs) == 0:
            if clo > cfg.feasibility_tolerance:
                conflicts.append(
                    f"Risk class '{rc}' requires minimum exposure {clo:.2%}, but zero instruments belong to class '{rc}'."
                )
                suggestions.append(f"Add at least one '{rc}' instrument or set its class minimum to 0%.")
            continue
        cls_sum_lo = sum(bounds[i][0] for i in idxs)
        cls_sum_hi = sum(bounds[i][1] for i in idxs)
        if cls_sum_hi < clo - cfg.feasibility_tolerance:
            conflicts.append(
                f"Risk class '{rc}' minimum ({clo:.2%}) exceeds sum of its assets' upper bounds ({cls_sum_hi:.2%})."
            )
            suggestions.append(f"Lower '{rc}' class minimum or raise upper bounds of its member assets.")
        if cls_sum_lo > chi + cfg.feasibility_tolerance:
            conflicts.append(
                f"Risk class '{rc}' maximum ({chi:.2%}) is below sum of its assets' lower bounds ({cls_sum_lo:.2%})."
            )
            suggestions.append(f"Raise '{rc}' class maximum or lower minimum bounds of its member assets.")

    if conflicts:
        raise InfeasibleConstraintsError(
            "Portfolio constraints are jointly infeasible: " + " | ".join(conflicts),
            details={"conflicts": conflicts, "suggestions": suggestions},
        )

    A_ub, b_ub = _build_linear_ineq_matrices(
        n, class_indices, cfg.class_bounds, sector_indices, cfg.max_sector_weight
    )
    A_eq = np.ones((1, n), dtype=float)
    b_eq = np.array([cfg.budget_sum], dtype=float)

    lp_res = linprog(
        c=np.zeros(n, dtype=float),
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={
            "primal_feasibility_tolerance": min(float(cfg.feasibility_tolerance), 1e-7),
            "dual_feasibility_tolerance": min(float(cfg.solver_tolerance), 1e-7),
        },
    )
    if not lp_res.success or lp_res.x is None:
        raise InfeasibleConstraintsError(
            "Portfolio constraints are jointly infeasible (exact LP verification failed). "
            "Individual bounds, class bounds, sector caps, and the 100% budget constraint have no common solution.",
            details={
                "lp_status": int(lp_res.status),
                "lp_message": str(lp_res.message),
                "asset_bounds": {t: b for t, b in zip(ordered, bounds)},
                "class_bounds": dict(cfg.class_bounds),
                "suggestions": [
                    "Check that sum of class maximums across represented classes is >= 100% and sum of class minimums is <= 100%.",
                    "Check overlaps between asset bounds, class bounds, and optional sector caps.",
                ],
            },
        )

    w_lp = np.asarray(lp_res.x, dtype=float)

    lin_slsqp_cons = _build_slsqp_constraints(
        n, class_indices, cfg.class_bounds, sector_indices, cfg.max_sector_weight, cfg.budget_sum, hhi_limit=None
    )
    min_hhi_res = minimize(
        lambda w: float(np.sum(w * w)),
        w_lp,
        jac=lambda w: 2.0 * w,
        method="SLSQP",
        bounds=bounds,
        constraints=lin_slsqp_cons,
        options={"maxiter": 1000, "ftol": cfg.solver_tolerance},
    )
    if min_hhi_res.success and _check_linear_feasibility(
        min_hhi_res.x, bounds, class_indices, cfg.class_bounds, sector_indices, cfg.max_sector_weight, cfg.budget_sum, cfg.feasibility_tolerance
    ):
        w_center = np.asarray(min_hhi_res.x, dtype=float)
    else:
        w_center = w_lp

    min_achievable_hhi = float(np.sum(w_center * w_center))
    if cfg.hhi_limit is not None and min_achievable_hhi > cfg.hhi_limit + cfg.feasibility_tolerance:
        raise InfeasibleConstraintsError(
            f"Requested HHI limit ({cfg.hhi_limit:.4f}) is jointly infeasible with asset/class bounds "
            f"(minimum achievable HHI is {min_achievable_hhi:.4f} for {n} asset(s)).",
            details={
                "requested_hhi_limit": cfg.hhi_limit,
                "min_achievable_hhi": min_achievable_hhi,
                "n_assets": n,
                "suggestions": [
                    f"Increase hhi_limit to at least {min_achievable_hhi:.4f} (or set hhi_limit=None), or add more diversified assets."
                ],
            },
        )

    return ResolvedConstraints(
        tickers=ordered,
        bounds=tuple(bounds),
        class_indices=class_indices,
        class_bounds=dict(cfg.class_bounds),
        sector_indices=sector_indices,
        max_sector_weight=cfg.max_sector_weight,
        hhi_limit=cfg.hhi_limit,
        budget_sum=cfg.budget_sum,
        solver_tolerance=cfg.solver_tolerance,
        feasibility_tolerance=cfg.feasibility_tolerance,
        feasible_starting_point=w_center,
    )


def _build_slsqp_constraints(
    n: int,
    class_indices: Dict[str, Tuple[int, ...]],
    class_bounds: Dict[str, Tuple[float, float]],
    sector_indices: Dict[str, Tuple[int, ...]],
    max_sector_weight: Optional[float],
    budget_sum: float,
    hhi_limit: Optional[float],
) -> List[Dict[str, Any]]:
    cons: List[Dict[str, Any]] = [
        {
            "type": "eq",
            "fun": lambda w, bs=budget_sum: float(np.sum(w) - bs),
            "jac": lambda w: np.ones(n, dtype=float),
        }
    ]
    for rc, (lo, hi) in class_bounds.items():
        idxs = class_indices.get(rc, ())
        if not idxs:
            continue
        vec = np.zeros(n, dtype=float)
        vec[list(idxs)] = 1.0
        if lo > 0.0:
            cons.append({
                "type": "ineq",
                "fun": lambda w, v=vec, l=lo: float(v @ w - l),
                "jac": lambda w, v=vec: v,
            })
        if hi < 1.0:
            cons.append({
                "type": "ineq",
                "fun": lambda w, v=vec, u=hi: float(u - v @ w),
                "jac": lambda w, v=vec: -v,
            })

    if max_sector_weight is not None and max_sector_weight < 1.0:
        for sec, idxs in sector_indices.items():
            if sec and sec != "Unknown" and len(idxs) >= 1:
                vec = np.zeros(n, dtype=float)
                vec[list(idxs)] = 1.0
                cons.append({
                    "type": "ineq",
                    "fun": lambda w, v=vec, u=max_sector_weight: float(u - v @ w),
                    "jac": lambda w, v=vec: -v,
                })

    if hhi_limit is not None:
        cons.append({
            "type": "ineq",
            "fun": lambda w, hl=hhi_limit: float(hl - np.sum(w * w)),
            "jac": lambda w: -2.0 * w,
        })

    return cons


def _check_linear_feasibility(
    w: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    class_indices: Dict[str, Tuple[int, ...]],
    class_bounds: Dict[str, Tuple[float, float]],
    sector_indices: Dict[str, Tuple[int, ...]],
    max_sector_weight: Optional[float],
    budget_sum: float,
    tol: float,
) -> bool:
    if w is None or not np.all(np.isfinite(w)):
        return False
    if abs(float(np.sum(w)) - budget_sum) > tol:
        return False
    for i, (lo, hi) in enumerate(bounds):
        if w[i] < lo - tol or w[i] > hi + tol:
            return False
    for rc, (lo, hi) in class_bounds.items():
        idxs = class_indices.get(rc, ())
        cw = float(sum(w[i] for i in idxs))
        if cw < lo - tol or cw > hi + tol:
            return False
    if max_sector_weight is not None:
        for sec, idxs in sector_indices.items():
            if sec and sec != "Unknown" and len(idxs) >= 1:
                if float(sum(w[i] for i in idxs)) > max_sector_weight + tol:
                    return False
    return True


def verify_solution_feasibility(
    weights: np.ndarray,
    resolved: ResolvedConstraints,
) -> Tuple[bool, List[str]]:
    """Verify a candidate weight vector satisfies EVERY constraint within tolerance."""
    tol = resolved.feasibility_tolerance
    violations: List[str] = []
    w = np.asarray(weights, dtype=float)

    if w.ndim != 1 or len(w) != len(resolved.tickers):
        return False, ["Weight vector dimension mismatch."]
    if not np.all(np.isfinite(w)):
        return False, ["Weight vector contains NaN or Inf values."]

    w_sum = float(np.sum(w))
    if abs(w_sum - resolved.budget_sum) > tol:
        violations.append(f"Budget violation: sum(w)={w_sum:.8f} != {resolved.budget_sum:.8f}.")

    for i, (t, (lo, hi)) in enumerate(zip(resolved.tickers, resolved.bounds)):
        if w[i] < lo - tol:
            violations.append(f"Asset '{t}' weight {w[i]:.6f} < min {lo:.6f}.")
        if w[i] > hi + tol:
            violations.append(f"Asset '{t}' weight {w[i]:.6f} > max {hi:.6f}.")

    for rc, (lo, hi) in resolved.class_bounds.items():
        idxs = resolved.class_indices.get(rc, ())
        cw = float(sum(w[i] for i in idxs))
        if cw < lo - tol:
            violations.append(f"Class '{rc}' weight {cw:.6f} < min {lo:.6f}.")
        if cw > hi + tol:
            violations.append(f"Class '{rc}' weight {cw:.6f} > max {hi:.6f}.")

    if resolved.max_sector_weight is not None:
        for sec, idxs in resolved.sector_indices.items():
            if sec and sec != "Unknown" and len(idxs) >= 1:
                sw = float(sum(w[i] for i in idxs))
                if sw > resolved.max_sector_weight + tol:
                    violations.append(f"Sector '{sec}' weight {sw:.6f} > max {resolved.max_sector_weight:.6f}.")

    if resolved.hhi_limit is not None:
        hhi = float(np.sum(w * w))
        if hhi > resolved.hhi_limit + tol:
            violations.append(f"HHI {hhi:.6f} > limit {resolved.hhi_limit:.6f}.")

    return len(violations) == 0, violations


def _project_weights_onto_feasible(
    target_w: np.ndarray,
    resolved: ResolvedConstraints,
) -> Tuple[np.ndarray, bool, bool]:
    """Minimum-distance Euclidean projection of ``target_w`` onto the feasible set (zero Sigma inversion).

    Returns ``(projected_weights, converged, feasible)``.
    """
    n = len(resolved.tickers)
    t_arr = np.asarray(target_w, dtype=float)
    cons = _build_slsqp_constraints(
        n,
        resolved.class_indices,
        resolved.class_bounds,
        resolved.sector_indices,
        resolved.max_sector_weight,
        resolved.budget_sum,
        resolved.hhi_limit,
    )
    res = minimize(
        lambda w: 0.5 * float(np.sum((w - t_arr) ** 2)),
        resolved.feasible_starting_point.copy(),
        jac=lambda w: w - t_arr,
        method="SLSQP",
        bounds=resolved.bounds,
        constraints=cons,
        options={"maxiter": 2000, "ftol": resolved.solver_tolerance},
    )
    w_proj = np.asarray(res.x, dtype=float)
    ok, _ = verify_solution_feasibility(w_proj, resolved)
    return w_proj, bool(res.success), bool(ok)


def _calc_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return float("nan")
    var_threshold = float(np.percentile(arr, (1.0 - confidence) * 100.0))
    tail = arr[arr <= var_threshold]
    return float(tail.mean()) if tail.size > 0 else var_threshold


def compute_risk_contributions(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame,
) -> Tuple[float, pd.Series, pd.Series, pd.Series]:
    """Compute portfolio volatility, MCR, absolute RC, and relative RC share."""
    tickers = list(cov_matrix.index)
    w = np.asarray(weights, dtype=float)
    sigma = cov_matrix.to_numpy(dtype=float)
    sw = sigma @ w
    var_p = float(w @ sw)
    vol_p = float(np.sqrt(max(var_p, 0.0)))
    if vol_p <= 1e-14:
        zeros = pd.Series(np.zeros(len(tickers), dtype=float), index=tickers)
        return 0.0, zeros.copy(), zeros.copy(), zeros.copy()
    mcr = sw / vol_p
    rc_abs = w * mcr
    rc_rel = rc_abs / vol_p
    return (
        vol_p,
        pd.Series(mcr, index=tickers, dtype=float),
        pd.Series(rc_abs, index=tickers, dtype=float),
        pd.Series(rc_rel, index=tickers, dtype=float),
    )


def _calc_mcr(weights: np.ndarray, cov: pd.DataFrame, tickers: List[str]) -> pd.Series:
    """Return absolute risk contributions ``w_i * (Sigma w)_i / sigma_p``."""
    _, _, rc_abs, _ = compute_risk_contributions(weights, cov.loc[tickers, tickers])
    return rc_abs


def _assemble_strategy_result(
    *,
    strategy_name: str,
    weights: np.ndarray,
    resolved: ResolvedConstraints,
    cov_matrix: pd.DataFrame,
    mu: Optional[pd.Series] = None,
    rf: float = 0.0,
    weekly_returns: Optional[pd.DataFrame] = None,
    cvar_confidence: float = 0.95,
    converged: bool = True,
    status: str = "optimal",
    diagnostics: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
) -> StrategyAllocationResult:
    tickers = list(resolved.tickers)
    cov_ordered = cov_matrix.loc[tickers, tickers]
    w = np.asarray(weights, dtype=float)
    feasible, violations = verify_solution_feasibility(w, resolved)
    warns = list(warnings or [])
    if not feasible:
        warns.extend(violations)
        status = "failed"

    vol, mcr, rc_abs, rc_rel = compute_risk_contributions(w, cov_ordered)

    exp_ret: Optional[float] = None
    sharpe: Optional[float] = None
    if mu is not None:
        mu_aligned = mu.reindex(tickers).to_numpy(dtype=float)
        if np.all(np.isfinite(mu_aligned)):
            exp_ret = float(w @ mu_aligned)
            sharpe = (exp_ret - float(rf)) / vol if vol > 1e-12 else float("nan")

    cvar_val: Optional[float] = None
    if weekly_returns is not None and not weekly_returns.empty:
        port_ret = weekly_returns[tickers].to_numpy(dtype=float) @ w
        cvar_val = float(abs(_calc_cvar(port_ret, cvar_confidence)))

    class_exp: Dict[str, float] = {}
    class_risk_exp: Dict[str, float] = {}
    for rc, idxs in resolved.class_indices.items():
        class_exp[rc] = float(sum(w[i] for i in idxs))
        class_risk_exp[rc] = float(sum(float(rc_rel.iloc[i]) for i in idxs))

    hhi = float(np.sum(w * w))
    w_dict = {t: float(val) for t, val in zip(tickers, w)}

    return StrategyAllocationResult(
        strategy_name=strategy_name,
        tickers=tickers,
        weights=w,
        weights_by_asset=w_dict,
        converged=bool(converged),
        feasible=feasible,
        status=status,
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        cvar_95=cvar_val,
        marginal_risk_contributions=mcr,
        risk_contributions_abs=rc_abs,
        risk_contributions_rel=rc_rel,
        class_exposure=class_exp,
        class_risk_exposure=class_risk_exp,
        hhi=hhi,
        diagnostics=dict(diagnostics or {}),
        warnings=warns,
    )


def _compute_max_feasible_excess_return(
    mu_arr: np.ndarray,
    rf: float,
    resolved: ResolvedConstraints,
) -> float:
    """Compute exact maximum achievable excess return ``max_{w in F} (w' mu - rf)`` under ALL constraints."""
    n = len(resolved.tickers)
    excess_vec = mu_arr - float(rf)
    A_ub, b_ub = _build_linear_ineq_matrices(
        n,
        resolved.class_indices,
        resolved.class_bounds,
        resolved.sector_indices,
        resolved.max_sector_weight,
    )
    A_eq = np.ones((1, n), dtype=float)
    b_eq = np.array([resolved.budget_sum], dtype=float)
    lp_res = linprog(
        c=-excess_vec,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=resolved.bounds,
        method="highs",
        options={
            "primal_feasibility_tolerance": min(float(resolved.feasibility_tolerance), 1e-7),
            "dual_feasibility_tolerance": min(float(resolved.solver_tolerance), 1e-7),
        },
    )
    if not lp_res.success or lp_res.x is None:
        return float(resolved.feasible_starting_point @ excess_vec)

    w_lp = np.asarray(lp_res.x, dtype=float)
    if resolved.hhi_limit is None or float(np.sum(w_lp * w_lp)) <= resolved.hhi_limit + resolved.feasibility_tolerance:
        return float(w_lp @ excess_vec)

    cons = _build_slsqp_constraints(
        n,
        resolved.class_indices,
        resolved.class_bounds,
        resolved.sector_indices,
        resolved.max_sector_weight,
        resolved.budget_sum,
        resolved.hhi_limit,
    )
    res_hhi = minimize(
        lambda w: -float(w @ excess_vec),
        resolved.feasible_starting_point.copy(),
        jac=lambda w: -excess_vec,
        method="SLSQP",
        bounds=resolved.bounds,
        constraints=cons,
        options={"maxiter": 2000, "ftol": resolved.solver_tolerance},
    )
    ok_hhi, _ = verify_solution_feasibility(res_hhi.x, resolved)
    if res_hhi.success and ok_hhi:
        return float(np.asarray(res_hhi.x, dtype=float) @ excess_vec)
    return float(resolved.feasible_starting_point @ excess_vec)


def optimize_max_sharpe(
    mu: pd.Series,
    cov_matrix: pd.DataFrame,
    rf: float,
    resolved: ResolvedConstraints,
    *,
    weekly_returns: Optional[pd.DataFrame] = None,
    cvar_confidence: float = 0.95,
    non_positive_excess_return_policy: str = "block",
) -> StrategyAllocationResult:
    """Maximize Sharpe ratio over the exact feasible set with strict convergence & edge-case guards."""
    tickers = list(resolved.tickers)
    n = len(tickers)
    mu_arr = mu.reindex(tickers).to_numpy(dtype=float)
    sigma = cov_matrix.loc[tickers, tickers].to_numpy(dtype=float)
    if not np.all(np.isfinite(mu_arr)) or not np.all(np.isfinite(sigma)):
        raise OptimizationConvergenceError(
            "Max Sharpe inputs contain non-finite expected returns or covariance entries.",
            details={"strategy": "max_sharpe"},
        )

    if non_positive_excess_return_policy not in ("block", "allow_with_warning"):
        raise ConfigurationError(
            f"Invalid non_positive_excess_return_policy '{non_positive_excess_return_policy}'."
        )

    vol_zero_tol = 1e-8

    if n == 1:
        w1 = np.array([1.0], dtype=float)
        vol1 = math.sqrt(max(float(sigma[0, 0]), 0.0))
        excess1 = float(mu_arr[0]) - float(rf)
        if vol1 <= vol_zero_tol:
            return _assemble_strategy_result(
                strategy_name="max_sharpe",
                weights=w1,
                resolved=resolved,
                cov_matrix=cov_matrix,
                mu=mu,
                rf=rf,
                weekly_returns=weekly_returns,
                cvar_confidence=cvar_confidence,
                converged=False,
                status="singular_zero_variance",
                diagnostics={"n_assets": 1, "zero_variance_detected": True, "is_valid_target": False},
                warnings=["Single-asset variance is zero; Sharpe ratio denominator is undefined."],
            )
        if excess1 <= 1e-12 and non_positive_excess_return_policy == "block":
            return _assemble_strategy_result(
                strategy_name="max_sharpe",
                weights=w1,
                resolved=resolved,
                cov_matrix=cov_matrix,
                mu=mu,
                rf=rf,
                weekly_returns=weekly_returns,
                cvar_confidence=cvar_confidence,
                converged=False,
                status="blocked_non_positive_excess",
                diagnostics={"n_assets": 1, "max_feasible_excess_return": excess1, "is_valid_target": False},
                warnings=["Single-asset expected excess return is <= 0; Max Sharpe blocked under policy='block'."],
            )
        return _assemble_strategy_result(
            strategy_name="max_sharpe",
            weights=w1,
            resolved=resolved,
            cov_matrix=cov_matrix,
            mu=mu,
            rf=rf,
            weekly_returns=weekly_returns,
            cvar_confidence=cvar_confidence,
            diagnostics={"n_assets": 1, "note": "Single-asset unique feasible allocation.", "is_valid_target": True},
        )

    # 1. Exact check of maximum feasible excess return under ALL active constraints (AUD-05)
    max_feasible_excess = _compute_max_feasible_excess_return(mu_arr, rf, resolved)
    non_positive_excess = max_feasible_excess <= 1e-12
    warnings_list: List[str] = []

    if non_positive_excess:
        if non_positive_excess_return_policy == "block":
            return _assemble_strategy_result(
                strategy_name="max_sharpe",
                weights=resolved.feasible_starting_point.copy(),
                resolved=resolved,
                cov_matrix=cov_matrix,
                mu=mu,
                rf=rf,
                weekly_returns=weekly_returns,
                cvar_confidence=cvar_confidence,
                converged=False,
                status="blocked_non_positive_excess",
                diagnostics={
                    "solver": "exact_excess_return_gate",
                    "max_feasible_excess_return": max_feasible_excess,
                    "policy": "block",
                    "is_valid_target": False,
                    "solver_tolerance": resolved.solver_tolerance,
                },
                warnings=[
                    f"Maximum feasible excess return under active constraints is {max_feasible_excess:.6%} (<= 0%). "
                    "Max Sharpe is blocked to prevent degenerate concentration in maximum volatility; "
                    "use min_variance/erc/hrp or opt in via non_positive_excess_return_policy='allow_with_warning'."
                ],
            )
        warnings_list.append(
            f"Maximum feasible excess return under active constraints is {max_feasible_excess:.6%} (<= 0%); "
            "maximizing a negative Sharpe ratio favors higher volatility (explicitly permitted via policy='allow_with_warning')."
        )

    cons = _build_slsqp_constraints(
        n,
        resolved.class_indices,
        resolved.class_bounds,
        resolved.sector_indices,
        resolved.max_sector_weight,
        resolved.budget_sum,
        resolved.hhi_limit,
    )

    # 2. Detect singular / zero-variance feasible portfolio before running Sharpe optimization (AUD-05)
    eigvals = np.linalg.eigvalsh(0.5 * (sigma + sigma.T))
    if float(np.min(eigvals)) <= 1e-10:
        mv_probe = minimize(
            lambda w: float(w @ sigma @ w),
            resolved.feasible_starting_point.copy(),
            jac=lambda w: 2.0 * (sigma @ w),
            method="SLSQP",
            bounds=resolved.bounds,
            constraints=cons,
            options={"maxiter": 2000, "ftol": resolved.solver_tolerance},
        )
        ok_mv, _ = verify_solution_feasibility(mv_probe.x, resolved)
        if ok_mv:
            w_mv = np.asarray(mv_probe.x, dtype=float)
            vol_mv = math.sqrt(max(float(w_mv @ sigma @ w_mv), 0.0))
            if vol_mv <= vol_zero_tol:
                ex_mv = float(w_mv @ mu_arr) - float(rf)
                if ex_mv > 1e-12 or abs(ex_mv) <= 1e-12:
                    return _assemble_strategy_result(
                        strategy_name="max_sharpe",
                        weights=w_mv,
                        resolved=resolved,
                        cov_matrix=cov_matrix,
                        mu=mu,
                        rf=rf,
                        weekly_returns=weekly_returns,
                        cvar_confidence=cvar_confidence,
                        converged=False,
                        status="singular_zero_variance",
                        diagnostics={
                            "solver": "zero_variance_singularity_guard",
                            "min_feasible_volatility": vol_mv,
                            "excess_at_zero_vol": ex_mv,
                            "zero_variance_detected": True,
                            "is_valid_target": False,
                            "solver_tolerance": resolved.solver_tolerance,
                        },
                        warnings=warnings_list + [
                            f"Singular covariance admits a feasible zero-volatility portfolio (vol={vol_mv:.3e}, excess={ex_mv:.4%}); "
                            "Sharpe ratio denominator is zero (undefined / theoretical arbitrage). Refusing spurious optimal status."
                        ],
                    )

    # 3. Objective without artificial volatility floor rewarding vol -> 0
    def neg_sharpe_and_grad(w: np.ndarray) -> Tuple[float, np.ndarray]:
        sw = sigma @ w
        var = float(w @ sw)
        if var <= vol_zero_tol * vol_zero_tol:
            return 1e12, np.zeros(n, dtype=float)
        vol = math.sqrt(var)
        excess = float(w @ mu_arr) - float(rf)
        obj = -excess / vol
        grad = -(mu_arr * vol - excess * (sw / vol)) / var
        return obj, grad

    starts = [resolved.feasible_starting_point.copy()]
    indiv_vols = np.sqrt(np.maximum(np.diag(sigma), 1e-12))
    indiv_sharpes = (mu_arr - float(rf)) / indiv_vols
    best_idx = int(np.argmax(indiv_sharpes))
    e_best = np.zeros(n, dtype=float)
    e_best[best_idx] = 1.0
    proj_best, conv_proj, ok_proj = _project_weights_onto_feasible(e_best, resolved)
    if conv_proj and ok_proj:
        starts.append(proj_best)

    best_conv_w: Optional[np.ndarray] = None
    best_conv_obj = float("inf")
    best_conv_nit = 0

    best_nonconv_w: Optional[np.ndarray] = None
    best_nonconv_obj = float("inf")
    best_nonconv_msg = ""

    for w0 in starts:
        res = minimize(
            lambda w: neg_sharpe_and_grad(w)[0],
            w0,
            jac=lambda w: neg_sharpe_and_grad(w)[1],
            method="SLSQP",
            bounds=resolved.bounds,
            constraints=cons,
            options={"maxiter": 2000, "ftol": resolved.solver_tolerance},
        )
        w_cand = np.asarray(res.x, dtype=float)
        ok, _ = verify_solution_feasibility(w_cand, resolved)
        if not ok:
            continue
        var_c = float(w_cand @ sigma @ w_cand)
        vol_c = math.sqrt(max(var_c, 0.0))
        if vol_c <= vol_zero_tol:
            return _assemble_strategy_result(
                strategy_name="max_sharpe",
                weights=w_cand,
                resolved=resolved,
                cov_matrix=cov_matrix,
                mu=mu,
                rf=rf,
                weekly_returns=weekly_returns,
                cvar_confidence=cvar_confidence,
                converged=False,
                status="singular_zero_variance",
                diagnostics={
                    "solver": "SLSQP",
                    "zero_variance_detected": True,
                    "is_valid_target": False,
                    "solver_tolerance": resolved.solver_tolerance,
                },
                warnings=warnings_list + [
                    f"Max Sharpe reached a zero-volatility portfolio (vol={vol_c:.3e}); Sharpe ratio is undefined."
                ],
            )
        val = neg_sharpe_and_grad(w_cand)[0]
        if not math.isfinite(val):
            continue
        if bool(res.success):
            if val < best_conv_obj:
                best_conv_obj = val
                best_conv_w = w_cand
                best_conv_nit = int(getattr(res, "nit", 0))
        else:
            if val < best_nonconv_obj:
                best_nonconv_obj = val
                best_nonconv_w = w_cand
                best_nonconv_msg = str(getattr(res, "message", "SLSQP failed to converge"))

    if best_conv_w is None:
        if best_nonconv_w is not None:
            return _assemble_strategy_result(
                strategy_name="max_sharpe",
                weights=best_nonconv_w,
                resolved=resolved,
                cov_matrix=cov_matrix,
                mu=mu,
                rf=rf,
                weekly_returns=weekly_returns,
                cvar_confidence=cvar_confidence,
                converged=False,
                status="non_converged",
                diagnostics={
                    "solver": "SLSQP",
                    "solver_message": best_nonconv_msg,
                    "solver_tolerance": resolved.solver_tolerance,
                    "is_valid_target": False,
                },
                warnings=warnings_list + [
                    f"Max Sharpe SLSQP did not converge ({best_nonconv_msg}); feasible candidate retained for diagnostics only and is NOT optimal."
                ],
            )
        raise OptimizationConvergenceError(
            "Max Sharpe optimizer failed to converge to a feasible solution within tolerance.",
            details={"strategy": "max_sharpe", "n_assets": n},
        )

    final_status = "suboptimal_non_positive_excess" if non_positive_excess else "optimal"
    return _assemble_strategy_result(
        strategy_name="max_sharpe",
        weights=best_conv_w,
        resolved=resolved,
        cov_matrix=cov_matrix,
        mu=mu,
        rf=rf,
        weekly_returns=weekly_returns,
        cvar_confidence=cvar_confidence,
        converged=True,
        status=final_status,
        diagnostics={
            "solver": "SLSQP",
            "iterations": best_conv_nit,
            "max_feasible_excess_return": max_feasible_excess,
            "solver_tolerance": resolved.solver_tolerance,
            "is_valid_target": not non_positive_excess,
        },
        warnings=warnings_list,
    )


def optimize_min_variance(
    cov_matrix: pd.DataFrame,
    resolved: ResolvedConstraints,
    *,
    mu: Optional[pd.Series] = None,
    rf: float = 0.0,
    weekly_returns: Optional[pd.DataFrame] = None,
    cvar_confidence: float = 0.95,
) -> StrategyAllocationResult:
    """Minimize portfolio variance ``w' Sigma w`` over the exact feasible set."""
    tickers = list(resolved.tickers)
    n = len(tickers)
    sigma = cov_matrix.loc[tickers, tickers].to_numpy(dtype=float)

    if n == 1:
        return _assemble_strategy_result(
            strategy_name="min_variance",
            weights=np.array([1.0], dtype=float),
            resolved=resolved,
            cov_matrix=cov_matrix,
            mu=mu,
            rf=rf,
            weekly_returns=weekly_returns,
            cvar_confidence=cvar_confidence,
            diagnostics={"n_assets": 1, "is_valid_target": True},
        )

    cons = _build_slsqp_constraints(
        n,
        resolved.class_indices,
        resolved.class_bounds,
        resolved.sector_indices,
        resolved.max_sector_weight,
        resolved.budget_sum,
        resolved.hhi_limit,
    )
    res = minimize(
        lambda w: float(w @ sigma @ w),
        resolved.feasible_starting_point.copy(),
        jac=lambda w: 2.0 * (sigma @ w),
        method="SLSQP",
        bounds=resolved.bounds,
        constraints=cons,
        options={"maxiter": 2000, "ftol": resolved.solver_tolerance},
    )
    w_sol = np.asarray(res.x, dtype=float)
    ok, violations = verify_solution_feasibility(w_sol, resolved)
    if not ok:
        raise OptimizationConvergenceError(
            "Min Variance optimizer failed to produce a feasible solution: " + "; ".join(violations),
            details={"strategy": "min_variance", "solver_message": str(getattr(res, "message", ""))},
        )

    converged = bool(res.success)
    status = "optimal" if converged else "non_converged"
    warnings_list: List[str] = []
    if not converged:
        warnings_list.append(
            f"Min Variance SLSQP did not converge ({getattr(res, 'message', '')}); feasible candidate retained for diagnostics only and is NOT optimal."
        )

    return _assemble_strategy_result(
        strategy_name="min_variance",
        weights=w_sol,
        resolved=resolved,
        cov_matrix=cov_matrix,
        mu=mu,
        rf=rf,
        weekly_returns=weekly_returns,
        cvar_confidence=cvar_confidence,
        converged=converged,
        status=status,
        diagnostics={
            "solver": "SLSQP",
            "iterations": int(getattr(res, "nit", 0)),
            "solver_tolerance": resolved.solver_tolerance,
            "solver_message": str(getattr(res, "message", "")),
            "is_valid_target": converged,
        },
        warnings=warnings_list,
    )


def optimize_min_cvar(
    weekly_returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    resolved: ResolvedConstraints,
    *,
    mu: Optional[pd.Series] = None,
    rf: float = 0.0,
    cvar_confidence: float = 0.95,
) -> StrategyAllocationResult:
    """Minimize historical weekly CVaR via exact Rockafellar-Uryasev LP (plus optional HHI refinement)."""
    tickers = list(resolved.tickers)
    n = len(tickers)

    if n == 1:
        return _assemble_strategy_result(
            strategy_name="min_cvar",
            weights=np.array([1.0], dtype=float),
            resolved=resolved,
            cov_matrix=cov_matrix,
            mu=mu,
            rf=rf,
            weekly_returns=weekly_returns,
            cvar_confidence=cvar_confidence,
            diagnostics={"n_assets": 1, "is_valid_target": True},
        )

    R = weekly_returns[tickers].to_numpy(dtype=float)
    T = R.shape[0]
    alpha = float(cvar_confidence)
    denom = (1.0 - alpha) * T

    c = np.zeros(n + 1 + T, dtype=float)
    c[n] = 1.0
    c[n + 1 :] = 1.0 / denom

    A_loss = np.zeros((T, n + 1 + T), dtype=float)
    A_loss[:, :n] = -R
    A_loss[:, n] = -1.0
    A_loss[:, n + 1 :] = -np.eye(T, dtype=float)
    b_loss = np.zeros(T, dtype=float)

    A_lin, b_lin = _build_linear_ineq_matrices(
        n,
        resolved.class_indices,
        resolved.class_bounds,
        resolved.sector_indices,
        resolved.max_sector_weight,
    )
    if A_lin is not None and b_lin is not None:
        A_extra = np.zeros((A_lin.shape[0], n + 1 + T), dtype=float)
        A_extra[:, :n] = A_lin
        A_ub = np.vstack([A_loss, A_extra])
        b_ub = np.concatenate([b_loss, b_lin])
    else:
        A_ub = A_loss
        b_ub = b_loss

    A_eq = np.zeros((1, n + 1 + T), dtype=float)
    A_eq[0, :n] = 1.0
    b_eq = np.array([resolved.budget_sum], dtype=float)

    lp_bounds: List[Tuple[Optional[float], Optional[float]]] = list(resolved.bounds)
    lp_bounds.append((None, None))
    lp_bounds.extend([(0.0, None)] * T)

    lp_res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=lp_bounds,
        method="highs",
        options={
            "primal_feasibility_tolerance": min(float(resolved.feasibility_tolerance), 1e-7),
            "dual_feasibility_tolerance": min(float(resolved.solver_tolerance), 1e-7),
        },
    )
    if not lp_res.success or lp_res.x is None:
        raise OptimizationConvergenceError(
            f"Min CVaR LP solver failed: {lp_res.message}",
            details={"strategy": "min_cvar"},
        )
    w_sol = np.asarray(lp_res.x[:n], dtype=float)

    solver_used = "rockafellar_uryasev_lp"
    converged = True
    warnings_list: List[str] = []
    solver_msg = str(getattr(lp_res, "message", ""))

    if resolved.hhi_limit is not None and float(np.sum(w_sol * w_sol)) > resolved.hhi_limit + resolved.feasibility_tolerance:
        solver_used = "rockafellar_uryasev_slsqp_hhi"
        w0 = resolved.feasible_starting_point.copy()
        zeta0 = float(np.percentile(-(R @ w0), alpha * 100.0))
        x0 = np.concatenate([w0, [zeta0]])

        def ru_obj(x: np.ndarray) -> float:
            w = x[:n]
            zeta = x[n]
            excess = np.maximum(-(R @ w) - zeta, 0.0)
            return float(zeta + np.sum(excess) / denom)

        base_cons = _build_slsqp_constraints(
            n,
            resolved.class_indices,
            resolved.class_bounds,
            resolved.sector_indices,
            resolved.max_sector_weight,
            resolved.budget_sum,
            resolved.hhi_limit,
        )
        ext_cons = []
        for con in base_cons:
            ext_cons.append({
                "type": con["type"],
                "fun": lambda x, f=con["fun"]: f(x[:n]),
                "jac": lambda x, j=con["jac"]: np.concatenate([j(x[:n]), [0.0]]),
            })
        ext_bounds = list(resolved.bounds) + [(None, None)]
        res = minimize(
            ru_obj,
            x0,
            method="SLSQP",
            bounds=ext_bounds,
            constraints=ext_cons,
            options={"maxiter": 2000, "ftol": resolved.solver_tolerance},
        )
        w_sol = np.asarray(res.x[:n], dtype=float)
        converged = bool(res.success)
        solver_msg = str(getattr(res, "message", ""))
        if not converged:
            warnings_list.append(
                f"Min CVaR HHI SLSQP did not converge ({solver_msg}); feasible candidate retained for diagnostics only and is NOT optimal."
            )

    ok, violations = verify_solution_feasibility(w_sol, resolved)
    if not ok:
        raise OptimizationConvergenceError(
            "Min CVaR solution failed feasibility verification: " + "; ".join(violations),
            details={"strategy": "min_cvar", "solver_message": solver_msg},
        )

    status = "optimal" if converged else "non_converged"
    return _assemble_strategy_result(
        strategy_name="min_cvar",
        weights=w_sol,
        resolved=resolved,
        cov_matrix=cov_matrix,
        mu=mu,
        rf=rf,
        weekly_returns=weekly_returns,
        cvar_confidence=cvar_confidence,
        converged=converged,
        status=status,
        diagnostics={
            "solver": solver_used,
            "confidence": alpha,
            "solver_tolerance": resolved.solver_tolerance,
            "solver_message": solver_msg,
            "is_valid_target": converged,
        },
        warnings=warnings_list,
    )


def _solve_unconstrained_erc_spinu(
    sigma: np.ndarray,
    *,
    solver_tolerance: float = 1e-8,
) -> Tuple[np.ndarray, bool]:
    """Solve unconstrained long-only ERC via Spinu (2013) strictly convex log-barrier formulation."""
    n = sigma.shape[0]
    vols = np.sqrt(np.maximum(np.diag(sigma), 1e-12))
    y0 = (1.0 / vols) / np.sum(1.0 / vols)
    z0 = np.log(np.maximum(y0, 1e-12))
    inv_n = 1.0 / n

    def obj_and_grad(z: np.ndarray) -> Tuple[float, np.ndarray]:
        z_clipped = np.clip(z, -40.0, 40.0)
        y = np.exp(z_clipped)
        sy = sigma @ y
        val = 0.5 * float(y @ sy) - inv_n * float(np.sum(z_clipped))
        grad = y * sy - inv_n
        return val, grad

    res = minimize(
        lambda z: obj_and_grad(z)[0],
        z0,
        jac=lambda z: obj_and_grad(z)[1],
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": solver_tolerance, "gtol": solver_tolerance},
    )
    y_opt = np.exp(np.clip(res.x, -40.0, 40.0))
    w_opt = y_opt / np.sum(y_opt)
    return w_opt, bool(res.success and np.all(np.isfinite(w_opt)))


def optimize_erc(
    cov_matrix: pd.DataFrame,
    resolved: ResolvedConstraints,
    *,
    mu: Optional[pd.Series] = None,
    rf: float = 0.0,
    weekly_returns: Optional[pd.DataFrame] = None,
    cvar_confidence: float = 0.95,
    exact_parity_tolerance: float = 1e-4,
) -> StrategyAllocationResult:
    """Compute Equal Risk Contribution (ERC / Risk Parity) portfolio without requiring ``mu``."""
    tickers = list(resolved.tickers)
    n = len(tickers)
    cov_ordered = cov_matrix.loc[tickers, tickers]
    cov_psd, psd_adj, min_b, min_a = ensure_psd_covariance(cov_ordered)
    sigma = cov_psd.to_numpy(dtype=float)

    warnings_list: List[str] = []
    if psd_adj:
        warnings_list.append(
            f"ERC input covariance required PSD adjustment (min eig {min_b:.3e} -> {min_a:.3e})."
        )

    if n == 1:
        return _assemble_strategy_result(
            strategy_name="erc",
            weights=np.array([1.0], dtype=float),
            resolved=resolved,
            cov_matrix=cov_psd,
            mu=mu,
            rf=rf,
            weekly_returns=weekly_returns,
            cvar_confidence=cvar_confidence,
            diagnostics={
                "solver": "single_asset",
                "target_risk_share": 1.0,
                "max_risk_share_deviation": 0.0,
                "risk_share_std": 0.0,
                "has_negative_risk_contributions": False,
                "is_constrained_erc": False,
                "is_valid_target": True,
            },
            warnings=warnings_list,
        )

    target_share = 1.0 / n

    w_spinu, spinu_ok = _solve_unconstrained_erc_spinu(sigma, solver_tolerance=resolved.solver_tolerance)
    spinu_feasible, _ = verify_solution_feasibility(w_spinu, resolved) if spinu_ok else (False, ["spinu_failed"])

    solver_used = "spinu_convex_exact"
    best_w: Optional[np.ndarray] = None
    converged = False
    solver_msg = ""

    if spinu_ok and spinu_feasible:
        vol_s, _, _, rc_rel_s = compute_risk_contributions(w_spinu, cov_psd)
        if vol_s > 1e-12 and float(np.max(np.abs(rc_rel_s.to_numpy() - target_share))) <= exact_parity_tolerance:
            best_w = w_spinu
            converged = True

    if best_w is None:
        solver_used = "slsqp_constrained_risk_dispersion"

        def dispersion_obj_and_grad(w: np.ndarray) -> Tuple[float, np.ndarray]:
            sw = sigma @ w
            var = float(w @ sw)
            if var <= 1e-18:
                return 1e6, np.zeros(n, dtype=float)
            rc_rel = (w * sw) / var
            diff = rc_rel - target_share
            obj = float(np.sum(diff * diff))
            term1 = (diff * sw) / var
            term2 = (sigma @ (diff * w)) / var
            term3 = -2.0 * float(np.sum(diff * rc_rel)) * sw / var
            grad = 2.0 * (term1 + term2 + term3)
            return obj, grad

        cons = _build_slsqp_constraints(
            n,
            resolved.class_indices,
            resolved.class_bounds,
            resolved.sector_indices,
            resolved.max_sector_weight,
            resolved.budget_sum,
            resolved.hhi_limit,
        )

        starts: List[np.ndarray] = [resolved.feasible_starting_point.copy()]
        if spinu_ok:
            w_proj_s, conv_s, ok_s = _project_weights_onto_feasible(w_spinu, resolved)
            if conv_s and ok_s:
                starts.insert(0, w_proj_s)
        vols = np.sqrt(np.maximum(np.diag(sigma), 1e-12))
        w_iv = (1.0 / vols) / np.sum(1.0 / vols)
        w_proj_iv, conv_iv, ok_iv = _project_weights_onto_feasible(w_iv, resolved)
        if conv_iv and ok_iv:
            starts.append(w_proj_iv)

        best_conv_w: Optional[np.ndarray] = None
        best_conv_val = float("inf")
        best_nonconv_w: Optional[np.ndarray] = None
        best_nonconv_val = float("inf")

        for w0 in starts:
            res = minimize(
                lambda w: dispersion_obj_and_grad(w)[0],
                w0,
                jac=lambda w: dispersion_obj_and_grad(w)[1],
                method="SLSQP",
                bounds=resolved.bounds,
                constraints=cons,
                options={"maxiter": 2000, "ftol": resolved.solver_tolerance},
            )
            w_cand = np.asarray(res.x, dtype=float)
            ok, _ = verify_solution_feasibility(w_cand, resolved)
            if not ok:
                continue
            val = dispersion_obj_and_grad(w_cand)[0]
            if not math.isfinite(val):
                continue
            if bool(res.success):
                if val < best_conv_val:
                    best_conv_val = val
                    best_conv_w = w_cand
            else:
                if val < best_nonconv_val:
                    best_nonconv_val = val
                    best_nonconv_w = w_cand
                    solver_msg = str(getattr(res, "message", "SLSQP failed to converge"))

        if best_conv_w is not None:
            best_w = best_conv_w
            converged = True
        elif best_nonconv_w is not None:
            best_w = best_nonconv_w
            converged = False
            warnings_list.append(
                f"Constrained ERC SLSQP did not converge ({solver_msg}); feasible candidate retained for diagnostics only and is NOT optimal."
            )

    if best_w is None:
        raise OptimizationConvergenceError(
            "ERC optimizer failed to converge to a feasible solution within tolerance.",
            details={"strategy": "erc", "n_assets": n},
        )

    vol, _, _, rc_rel = compute_risk_contributions(best_w, cov_psd)
    if vol <= 1e-12:
        raise OptimizationConvergenceError(
            "ERC produced zero portfolio volatility (degenerate covariance).",
            details={"strategy": "erc"},
        )

    rc_arr = rc_rel.to_numpy(dtype=float)
    max_dev = float(np.max(np.abs(rc_arr - target_share)))
    std_rc = float(np.std(rc_arr))
    has_neg_rc = bool(np.any(rc_arr < -1e-9))

    if has_neg_rc:
        neg_assets = [tickers[i] for i in range(n) if rc_arr[i] < -1e-9]
        warnings_list.append(
            f"Negative risk contribution(s) detected in ERC solution for {neg_assets} (hedge correlation structure)."
        )

    is_constrained = max_dev > exact_parity_tolerance
    if not converged:
        status = "non_converged"
    else:
        status = "constrained_approximate" if is_constrained else "optimal"
        if is_constrained:
            warnings_list.append(
                f"Active asset/class constraints prevented exact 1/n ({target_share:.2%}) risk parity; "
                f"returning constrained approximate ERC (max risk-share deviation={max_dev:.4%}, std={std_rc:.4%})."
            )

    return _assemble_strategy_result(
        strategy_name="erc",
        weights=best_w,
        resolved=resolved,
        cov_matrix=cov_psd,
        mu=mu,
        rf=rf,
        weekly_returns=weekly_returns,
        cvar_confidence=cvar_confidence,
        converged=converged,
        status=status,
        diagnostics={
            "solver": solver_used,
            "solver_tolerance": resolved.solver_tolerance,
            "solver_message": solver_msg,
            "target_risk_share": target_share,
            "max_risk_share_deviation": max_dev,
            "risk_share_std": std_rc,
            "has_negative_risk_contributions": has_neg_rc,
            "is_constrained_erc": is_constrained,
            "is_valid_target": converged,
        },
        warnings=warnings_list,
    )


def _cluster_ivp_variance(sigma: np.ndarray, indices: List[int]) -> float:
    """Compute cluster variance using inverse-variance weights (diagonal variances ONLY, zero Sigma inversion)."""
    sub_cov = sigma[np.ix_(indices, indices)]
    diag_var = np.maximum(np.diag(sub_cov), 1e-16)
    ivp = (1.0 / diag_var) / np.sum(1.0 / diag_var)
    return float(ivp @ sub_cov @ ivp)


def compute_pure_hrp_weights(
    cov_matrix: pd.DataFrame,
    *,
    linkage_method: str = "single",
) -> Tuple[np.ndarray, List[str]]:
    """Compute pure López de Prado (2016) HRP weights without inverting Sigma.

    Deterministic tie-breaking policy (AUD-13):
    - Validates unique asset labels and exact index/column alignment.
    - Sorts asset IDs canonically in alphabetical order BEFORE constructing the
      correlation distance matrix and SciPy hierarchical clustering linkage tree
      (``linkage_method``), ensuring exact permutation invariance even when pairs
      share identical correlations (ties), then remaps weights back to the caller's
      original ticker order.
    """
    orig_tickers = [str(t) for t in cov_matrix.index]
    orig_cols = [str(c) for c in cov_matrix.columns]
    if len(set(orig_tickers)) != len(orig_tickers):
        raise DataQualityError("HRP covariance matrix contains duplicate asset labels.")
    if set(orig_tickers) != set(orig_cols) or len(orig_tickers) != len(orig_cols):
        raise DataQualityError("HRP covariance matrix index and columns must contain the same unique asset IDs.")

    n = len(orig_tickers)
    if n == 1:
        return np.array([1.0], dtype=float), orig_tickers

    # Canonical alphabetical ordering guarantees deterministic tie-breaking invariant to input permutation
    canon_tickers = sorted(orig_tickers)
    cov_canon = cov_matrix.loc[canon_tickers, canon_tickers]

    corr_df, _ = covariance_to_correlation(cov_canon)
    corr = corr_df.to_numpy(dtype=float)
    sigma = cov_canon.to_numpy(dtype=float)

    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)
    sort_ix = [int(i) for i in leaves_list(Z)]
    ordered_tickers = [canon_tickers[i] for i in sort_ix]

    weights_canon = np.ones(n, dtype=float)
    clusters: List[List[int]] = [sort_ix]

    while clusters:
        next_clusters: List[List[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]

            var_left = _cluster_ivp_variance(sigma, left)
            var_right = _cluster_ivp_variance(sigma, right)
            denom = var_left + var_right
            alpha = 0.5 if denom <= 1e-18 else (1.0 - var_left / denom)

            weights_canon[left] *= alpha
            weights_canon[right] *= 1.0 - alpha

            if len(left) > 1:
                next_clusters.append(left)
            if len(right) > 1:
                next_clusters.append(right)
        clusters = next_clusters

    w_sum = float(np.sum(weights_canon))
    if w_sum <= 0.0 or not np.all(np.isfinite(weights_canon)):
        raise OptimizationConvergenceError("Pure HRP produced invalid weights.")
    weights_canon /= w_sum

    w_by_ticker = {t: float(w) for t, w in zip(canon_tickers, weights_canon)}
    weights_orig = np.array([w_by_ticker[t] for t in orig_tickers], dtype=float)
    return weights_orig, ordered_tickers


def optimize_hrp(
    cov_matrix: pd.DataFrame,
    resolved: ResolvedConstraints,
    *,
    linkage_method: str = "single",
    mu: Optional[pd.Series] = None,
    rf: float = 0.0,
    weekly_returns: Optional[pd.DataFrame] = None,
    cvar_confidence: float = 0.95,
) -> StrategyAllocationResult:
    """Compute HRP weights (never inverting Sigma) and apply explicit feasible projection if needed."""
    tickers = list(resolved.tickers)
    cov_ordered = cov_matrix.loc[tickers, tickers]
    cov_psd, psd_adj, min_b, min_a = ensure_psd_covariance(cov_ordered)

    warnings_list: List[str] = []
    if psd_adj:
        warnings_list.append(
            f"HRP input covariance required PSD adjustment (min eig {min_b:.3e} -> {min_a:.3e})."
        )

    w_pure, quasi_order = compute_pure_hrp_weights(cov_psd, linkage_method=linkage_method)
    pure_feasible, pure_violations = verify_solution_feasibility(w_pure, resolved)

    if pure_feasible:
        w_final = w_pure
        converged = True
        status = "optimal"
        is_adjusted = False
        l1_adj = 0.0
        max_adj = 0.0
    else:
        w_proj, conv_proj, ok_proj = _project_weights_onto_feasible(w_pure, resolved)
        if not ok_proj:
            raise OptimizationConvergenceError(
                "Constrained HRP projection onto feasible constraint polytope failed to find a feasible point.",
                details={"strategy": "hrp", "pure_violations": pure_violations},
            )
        w_final = w_proj
        converged = conv_proj
        is_adjusted = True
        l1_adj = float(np.sum(np.abs(w_final - w_pure)))
        max_adj = float(np.max(np.abs(w_final - w_pure)))
        if conv_proj:
            status = "adjusted_to_constraints"
            warnings_list.append(
                f"Pure HRP weights violated active asset/class/HHI constraints ({'; '.join(pure_violations[:3])}); "
                f"applied minimum-L2 feasible projection (L1 adjustment={l1_adj:.4f}, max weight shift={max_adj:.4f}). "
                "Result is constraint-adjusted HRP, not unmodified standard HRP."
            )
        else:
            status = "non_converged"
            warnings_list.append(
                "Constrained HRP projection solver did not converge (res.success=False); "
                "feasible projected candidate retained for diagnostics only and is NOT optimal."
            )

    return _assemble_strategy_result(
        strategy_name="hrp",
        weights=w_final,
        resolved=resolved,
        cov_matrix=cov_psd,
        mu=mu,
        rf=rf,
        weekly_returns=weekly_returns,
        cvar_confidence=cvar_confidence,
        converged=converged,
        status=status,
        diagnostics={
            "linkage_method": linkage_method,
            "tie_break_policy": "canonical_alphabetical_instrument_id_ordering",
            "quasi_diagonal_order": quasi_order,
            "uses_sigma_inversion": False,
            "is_constrained_adjusted": is_adjusted,
            "pure_hrp_weights": {t: float(v) for t, v in zip(tickers, w_pure)},
            "l1_adjustment": l1_adj,
            "max_weight_adjustment": max_adj,
            "pure_hrp_violations": pure_violations,
            "solver_tolerance": resolved.solver_tolerance,
            "is_valid_target": converged,
        },
        warnings=warnings_list,
    )


def stress_test_corr_one(weights: np.ndarray, cov_matrix: pd.DataFrame, tickers_list: List[str]) -> Dict[str, Any]:
    w = np.asarray(weights, dtype=float)
    cov_vals = cov_matrix.loc[tickers_list, tickers_list].to_numpy(dtype=float) if hasattr(cov_matrix, "loc") else np.asarray(cov_matrix, dtype=float)
    vols = np.sqrt(np.maximum(np.diag(cov_vals), 0.0))
    stress_vol = float(np.sum(w * vols))
    normal_vol = float(np.sqrt(max(float(w @ cov_vals @ w), 0.0)))
    div_ratio = normal_vol / stress_vol if stress_vol > 1e-12 else 1.0
    stress_contrib = (w * vols / stress_vol) if stress_vol > 1e-12 else w
    recommendations: List[str] = []
    if div_ratio > 0.90:
        recommendations.append("CRITICAL: Almost no diversification benefit under correlation=1 stress.")
    elif div_ratio > 0.85:
        recommendations.append("WARNING: Limited diversification under correlation=1 stress.")
    for i, t in enumerate(tickers_list):
        if stress_contrib[i] > 0.30:
            recommendations.append(f"High stress concentration in {t} ({stress_contrib[i] * 100:.1f}% of corr=1 stress risk).")
    return {
        "normal_vol": normal_vol,
        "stress_vol": stress_vol,
        "diversification_ratio": div_ratio,
        "vol_increase_pct": ((stress_vol / normal_vol) - 1.0) * 100.0 if normal_vol > 1e-12 else 0.0,
        "false_diversification": bool(div_ratio > 0.85),
        "stress_contributions": pd.Series(stress_contrib, index=tickers_list, dtype=float),
        "recommendations": recommendations,
    }


def stress_test_vol_shock(
    weights: np.ndarray, cov_matrix: pd.DataFrame, tickers_list: List[str], shock_factor: float = 2.0
) -> Dict[str, Any]:
    w = np.asarray(weights, dtype=float)
    cov_vals = cov_matrix.loc[tickers_list, tickers_list].to_numpy(dtype=float) if hasattr(cov_matrix, "loc") else np.asarray(cov_matrix, dtype=float)
    normal_vol = float(np.sqrt(max(float(w @ cov_vals @ w), 0.0)))
    stress_vol = normal_vol * float(shock_factor)
    recommendations: List[str] = []
    if stress_vol > 0.60:
        recommendations.append(f"CRITICAL: Volatility under {shock_factor:g}x shock reaches {stress_vol * 100:.1f}% (>60%).")
    return {
        "normal_vol": normal_vol,
        "stress_vol": stress_vol,
        "shock_factor": float(shock_factor),
        "vol_increase_pct": ((stress_vol / normal_vol) - 1.0) * 100.0 if normal_vol > 1e-12 else 0.0,
        "recommendations": recommendations,
    }


def stress_test_systemic(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame,
    tickers_list: List[str],
    corr_floor: float = 0.80,
    vol_shock: float = 1.5,
) -> Dict[str, Any]:
    w = np.asarray(weights, dtype=float)
    cov_vals = cov_matrix.loc[tickers_list, tickers_list].to_numpy(dtype=float) if hasattr(cov_matrix, "loc") else np.asarray(cov_matrix, dtype=float)
    vols = np.sqrt(np.maximum(np.diag(cov_vals), 0.0))
    normal_vol = float(np.sqrt(max(float(w @ cov_vals @ w), 0.0)))
    shocked_vols = vols * float(vol_shock)
    n = len(vols)
    stress_corr = np.full((n, n), float(corr_floor), dtype=float)
    np.fill_diagonal(stress_corr, 1.0)
    stress_cov = np.outer(shocked_vols, shocked_vols) * stress_corr
    stress_vol = float(np.sqrt(max(float(w @ stress_cov @ w), 0.0)))
    max_weekly_loss_99 = stress_vol / math.sqrt(WEEKS_PER_YEAR) * 2.3263
    recommendations: List[str] = []
    if normal_vol > 1e-12 and stress_vol > normal_vol * 2.5:
        recommendations.append("CRITICAL: Portfolio risk expands >2.5x under systemic stress.")
    elif normal_vol > 1e-12 and stress_vol > normal_vol * 2.0:
        recommendations.append("WARNING: Elevated risk under systemic stress (>2.0x).")
    return {
        "normal_vol": normal_vol,
        "stress_vol": stress_vol,
        "vol_increase_pct": ((stress_vol / normal_vol) - 1.0) * 100.0 if normal_vol > 1e-12 else 0.0,
        "corr_floor": float(corr_floor),
        "vol_shock_factor": float(vol_shock),
        "max_weekly_loss_99": max_weekly_loss_99,
        "recommendations": recommendations,
    }


def run_all_stress_tests(weights: np.ndarray, cov_matrix: pd.DataFrame, tickers_list: List[str]) -> Dict[str, Any]:
    return {
        "corr_one": stress_test_corr_one(weights, cov_matrix, tickers_list),
        "vol_shock": stress_test_vol_shock(weights, cov_matrix, tickers_list, shock_factor=2.0),
        "systemic": stress_test_systemic(weights, cov_matrix, tickers_list, corr_floor=0.80, vol_shock=1.5),
    }


def get_market_caps(tickers_list: Sequence[str]) -> Dict[str, float]:
    caps: Dict[str, float] = {}
    for t in tickers_list:
        try:
            info = yf.Ticker(t).info or {}
            cap = info.get("marketCap")
            if cap is not None and float(cap) > 0.0:
                caps[t.upper()] = float(cap)
        except Exception:
            continue
    return caps


def get_sector_map(tickers_list: Sequence[str]) -> Dict[str, str]:
    sectors: Dict[str, str] = {}
    for t in tickers_list:
        try:
            info = yf.Ticker(t).info or {}
            sectors[t.upper()] = str(info.get("sector") or "Unknown")
        except Exception:
            sectors[t.upper()] = "Unknown"
    return sectors


def analyze_portfolio(
    config: Union[PortfolioAnalysisConfig, Dict[str, Any]],
    *,
    prices_usd_override: Optional[pd.DataFrame] = None,
    vix_history_override: Optional[pd.Series] = None,
    current_vix_override: Optional[Tuple[Optional[float], Optional[pd.Timestamp], bool]] = None,
    iol_snapshots_override: Optional[Dict[str, IOLPanelSnapshot]] = None,
    market_caps_override: Optional[Dict[str, float]] = None,
    sector_map_override: Optional[Dict[str, str]] = None,
) -> AnalysisResult:
    """Execute the complete stateless multi-asset portfolio optimization pipeline."""
    if isinstance(config, dict):
        forbidden = REMOVED_LEGACY_KEYS.intersection(config.keys())
        if forbidden:
            raise ConfigurationError(
                f"Removed legacy parameter(s) are not supported: {sorted(forbidden)}."
            )
        cfg = PortfolioAnalysisConfig.from_dict(config)
    elif isinstance(config, PortfolioAnalysisConfig):
        cfg = config
    else:
        raise ConfigurationError(
            f"'config' must be PortfolioAnalysisConfig or dict (got {type(config).__name__})."
        )

    instruments = list(cfg.instruments)
    inst_map = cfg.instrument_map
    ordered_ids = [inst.instrument_id for inst in instruments]
    warnings_list: List[str] = []

    as_of_cutoff: Optional[pd.Timestamp] = None
    if cfg.as_of_date is not None:
        as_of_cutoff = pd.Timestamp(cfg.as_of_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    iol_quotes: Dict[str, IOLQuoteResult] = fetch_iol_quotes_for_instruments(
        instruments,
        iol_config=cfg.iol,
        preloaded_snapshots=iol_snapshots_override,
    )
    for q in iol_quotes.values():
        warnings_list.extend(q.warnings)

    prices_cols: Dict[str, pd.Series] = {}
    direct_price_sources: Dict[str, str] = {}
    proxies_used: Dict[str, str] = {}
    missing_history_assets: List[str] = []

    override_df = prices_usd_override.copy() if prices_usd_override is not None else pd.DataFrame()
    if not override_df.empty:
        override_df.columns = [str(c).strip().upper() for c in override_df.columns]

    yfinance_download_map: Dict[str, str] = {}
    yfinance_direct_ids: set[str] = set()
    currency_overrides: Dict[str, str] = {}

    for inst in instruments:
        iid = inst.instrument_id
        if not override_df.empty and iid in override_df.columns:
            prices_cols[iid] = override_df[iid]
            direct_price_sources[iid] = "prices_usd_override"
            continue
        if not override_df.empty and inst.ticker.upper() in override_df.columns:
            prices_cols[iid] = override_df[inst.ticker.upper()]
            direct_price_sources[iid] = "prices_usd_override"
            continue

        if inst.quote_source == "yfinance":
            yfinance_download_map[iid] = inst.ticker
            yfinance_direct_ids.add(iid)
            # AUD-01: ONLY forward explicit user-declared currency overrides (default is None = auto-detect)
            if inst.currency is not None:
                currency_overrides[inst.ticker.upper()] = inst.currency
        else:
            if inst.historical_proxy_ticker and cfg.risk.allow_explicit_proxies:
                proxy_tkr = inst.historical_proxy_ticker.strip().upper()
                if not override_df.empty and proxy_tkr in override_df.columns:
                    prices_cols[iid] = override_df[proxy_tkr]
                else:
                    yfinance_download_map[iid] = inst.historical_proxy_ticker
                proxies_used[iid] = inst.historical_proxy_ticker
                warnings_list.append(
                    f"Instrument '{iid}' lacks direct historical USD prices; using user-approved explicit proxy "
                    f"'{inst.historical_proxy_ticker}' for weekly risk estimation. Proxy tracking error is not captured."
                )
            else:
                missing_history_assets.append(iid)

    fx_diagnostics: Dict[str, FXConversionDiagnostics] = {}
    unadjusted_closes: Dict[str, pd.Series] = {}
    if yfinance_download_map:
        unique_dl_tickers = list(dict.fromkeys(yfinance_download_map.values()))
        # AUD-10: allow_partial=False propagates FXConversionError / DataQualityError directly with exact asset_id & cause
        downloaded_df, _ = _download_prices(
            unique_dl_tickers,
            period=cfg.risk.period,
            fx_max_staleness_days=cfg.risk.fx_max_staleness_days,
            min_fx_coverage=cfg.risk.min_fx_coverage,
            duplicate_date_policy=cfg.risk.duplicate_date_policy,
            currency_overrides=currency_overrides,
            as_of_cutoff=as_of_cutoff,
            allow_partial=False,
            fx_diagnostics_out=fx_diagnostics,
            unadjusted_close_out=unadjusted_closes,
        )
        for diag in fx_diagnostics.values():
            warnings_list.extend(diag.warnings)
        for iid, dl_tkr in yfinance_download_map.items():
            key = dl_tkr.upper()
            if downloaded_df.empty or key not in downloaded_df.columns:
                missing_history_assets.append(iid)
            else:
                prices_cols[iid] = downloaded_df[key]
                if iid in yfinance_direct_ids:
                    direct_price_sources[iid] = "yfinance"

    # AUD-09: Evaluate fixed-income contractual metrics AFTER resolving direct market prices (including
    # prices_usd_override and yfinance) and BEFORE blocking on insufficient covariance history,
    # so standalone YTM remains available and respects source precedence & non-adjusted price checks.
    fixed_income_metrics: Dict[str, FixedIncomeMetricsResult] = {}
    for inst in cfg.fixed_income:
        if inst.fixed_income_terms is None:
            continue
        iid = inst.instrument_id
        terms = inst.fixed_income_terms
        quoted_px: Optional[float] = None
        quoted_ccy: Optional[str] = None
        src_used: Optional[str] = None
        val_date_str: Optional[str] = None
        fi_extra_warns: List[str] = []
        not_eval_reason: Optional[str] = None

        if iid in iol_quotes:
            iq = iol_quotes[iid]
            quoted_px = iq.price_usd if terms.cashflow_currency == "USD" else iq.price_original
            quoted_ccy = "USD" if terms.cashflow_currency == "USD" else iq.original_currency
            src_used = iq.panel_source
            val_date_str = iq.market_timestamp or iq.observed_at
        elif inst.manual_price_usd is not None and terms.cashflow_currency == "USD":
            quoted_px = float(inst.manual_price_usd)
            quoted_ccy = "USD"
            src_used = "manual_price_usd"
            val_date_str = terms.settlement_date.isoformat()
        elif iid in direct_price_sources and terms.cashflow_currency == "USD":
            hist_src = direct_price_sources[iid]
            if inst.historical_price_series_type == "adjusted_total_return":
                src_used = hist_src
                not_eval_reason = (
                    f"Historical series from '{hist_src}' for '{iid}' is marked 'adjusted_total_return' "
                    "and is not a tradable clean/dirty market price suitable for contractual YTM."
                )
            else:
                dl_key = inst.ticker.upper()
                candidate_series = (
                    unadjusted_closes[dl_key]
                    if (hist_src == "yfinance" and dl_key in unadjusted_closes and not unadjusted_closes[dl_key].empty)
                    else prices_cols[iid]
                )
                val_cutoff = as_of_cutoff
                san_fi = sanitize_price_series(
                    candidate_series,
                    asset_id=iid,
                    duplicate_date_policy=cfg.risk.duplicate_date_policy,
                    target_tz=cfg.risk.weekly_cutoff_tz,
                    as_of_cutoff=val_cutoff,
                    require_positive=True,
                )
                if not san_fi.series.empty:
                    quoted_px = float(san_fi.series.iloc[-1])
                    quoted_ccy = "USD"
                    src_used = hist_src
                    val_date_str = pd.Timestamp(san_fi.series.index[-1]).date().isoformat()
                    if hist_src == "yfinance" and dl_key not in unadjusted_closes:
                        fi_extra_warns.append(
                            f"Using yfinance Close series for '{iid}' YTM valuation (unadjusted Close unavailable); verify series represents tradable {terms.price_type} price."
                        )

        fi_res = evaluate_fixed_income_instrument(
            iid,
            terms,
            quoted_price=quoted_px,
            quoted_price_currency=quoted_ccy,
            quote_source_used=src_used,
            valuation_price_date=val_date_str,
            extra_warnings=fi_extra_warns,
            not_evaluable_reason=not_eval_reason,
        )
        fixed_income_metrics[iid] = fi_res
        warnings_list.extend(fi_res.warnings)

    if missing_history_assets:
        raise InsufficientHistoryError(
            f"Optimization blocked: instrument(s) {missing_history_assets} lack valid historical USD price series "
            "(IOL quotes are point-in-time snapshots only; historical series are never fabricated from coupons or snapshots, "
            "and instruments are never silently dropped). Supply historical total-return prices, or configure an explicit "
            "proxy with risk.allow_explicit_proxies=True. Independent IOL quotes and fixed-income YTM calculations remain available.",
            details={
                "missing_history_assets": missing_history_assets,
                "iol_quotes_available": list(iol_quotes.keys()),
                "fixed_income_ytm_available": {
                    k: v.ytm_annual for k, v in fixed_income_metrics.items() if v.status == "ok"
                },
            },
        )

    prices_usd = pd.DataFrame({iid: prices_cols[iid] for iid in ordered_ids}).sort_index(ascending=True)

    weekly_bundle = compute_weekly_returns_w_fri(
        prices_usd,
        risk_config=cfg.risk,
        as_of_cutoff=as_of_cutoff,
    )

    vix_hist: Optional[pd.Series] = vix_history_override
    curr_vix: Optional[float] = None
    curr_vix_date: Optional[pd.Timestamp] = None
    curr_vix_stale: bool = False

    if cfg.risk.covariance_estimator == "vix_regime":
        if vix_hist is None and cfg.risk.vix_regime.regime_classification_method == "vix_history":
            vix_hist = fetch_vix_history(period=cfg.risk.period, as_of_cutoff=as_of_cutoff)

        if current_vix_override is not None:
            curr_vix, curr_vix_date, curr_vix_stale = current_vix_override
        else:
            curr_vix, curr_vix_date, curr_vix_stale = get_current_vix_observation(
                vix_history=vix_hist,
                as_of_date=cfg.as_of_date,
                max_staleness_days=cfg.risk.vix_regime.max_vix_staleness_days,
            )

    risk_result: RiskModelResult = estimate_risk_model(
        weekly_bundle,
        risk_config=cfg.risk,
        vix_history=vix_hist,
        current_vix=curr_vix,
        current_vix_date=curr_vix_date,
        current_vix_is_stale=curr_vix_stale,
        proxies_used=proxies_used,
    )
    warnings_list.extend(risk_result.warnings)

    # Preserve published/latest market USD quote in latest_prices_usd AND compute canonical dirty
    # execution USD price per 1 quantity unit (AUD-02) via shared helper.
    latest_prices_usd: Dict[str, float] = {}
    quote_units_by_id: Dict[str, str] = {}
    execution_prices_usd_per_unit: Dict[str, float] = {}
    price_unit_metadata: Dict[str, Dict[str, Any]] = {}

    for iid in ordered_ids:
        inst = inst_map[iid]
        if iid in iol_quotes:
            latest_prices_usd[iid] = float(iol_quotes[iid].price_usd)
            quote_units_by_id[iid] = iol_quotes[iid].quote_unit
        elif inst.manual_price_usd is not None:
            latest_prices_usd[iid] = float(inst.manual_price_usd)
        else:
            san_latest = sanitize_price_series(
                prices_usd[iid],
                asset_id=iid,
                duplicate_date_policy=cfg.risk.duplicate_date_policy,
                target_tz=cfg.risk.weekly_cutoff_tz,
                as_of_cutoff=as_of_cutoff,
                require_positive=True,
            )
            if not san_latest.series.empty:
                latest_prices_usd[iid] = float(san_latest.series.iloc[-1])

        if iid in latest_prices_usd:
            unit_px, unit_meta = normalize_execution_price_to_usd_per_unit(
                inst,
                latest_prices_usd[iid],
                quote_unit_override=quote_units_by_id.get(iid),
            )
            execution_prices_usd_per_unit[iid] = float(unit_px)
            price_unit_metadata[iid] = unit_meta
            warnings_list.extend(unit_meta.get("warnings", []))

    rf_rate = (
        float(cfg.black_litterman.rf_rate)
        if cfg.black_litterman.rf_rate is not None
        else get_risk_free_rate(default=cfg.black_litterman.default_rf_rate)
    )

    m_caps = market_caps_override
    if m_caps is None and cfg.black_litterman.prior_weights_policy in ("market_cap_equity_only", "explicit_or_warned_equal_weight"):
        all_equity = all(inst.asset_subtype in ("equity", "equity_etf") for inst in instruments)
        has_explicit_all = all(iid in cfg.black_litterman.reference_weights for iid in ordered_ids)
        if all_equity and not has_explicit_all:
            raw_caps = get_market_caps([inst.ticker for inst in instruments])
            m_caps = {inst.instrument_id: raw_caps[inst.ticker.upper()] for inst in instruments if inst.ticker.upper() in raw_caps}

    bl_result = compute_expected_returns_pipeline(
        tickers=ordered_ids,
        cov_matrix=risk_result.cov_matrix,
        rf_rate=rf_rate,
        bl_config=cfg.black_litterman,
        instruments_by_id=inst_map,
        latest_prices_usd=latest_prices_usd,
        fixed_income_metrics=fixed_income_metrics,
        market_caps=m_caps,
    )
    warnings_list.extend(bl_result.warnings)

    risk_class_by_ticker = {inst.instrument_id: str(inst.risk_class) for inst in instruments}
    sector_by_ticker = sector_map_override
    if sector_by_ticker is None and cfg.constraints.max_sector_weight is not None:
        raw_sec = get_sector_map([inst.ticker for inst in instruments])
        sector_by_ticker = {inst.instrument_id: raw_sec.get(inst.ticker.upper(), "Unknown") for inst in instruments}

    resolved_constraints = validate_constraints_feasibility(
        ordered_ids,
        cfg.constraints,
        risk_class_by_ticker=risk_class_by_ticker,
        sector_by_ticker=sector_by_ticker,
    )

    strategies: Dict[str, StrategyAllocationResult] = {}
    mu_post = bl_result.posterior_total_returns

    for strat_name in cfg.optimization.strategies_to_run:
        if strat_name == "max_sharpe":
            strategies[strat_name] = optimize_max_sharpe(
                mu_post,
                risk_result.cov_matrix,
                rf_rate,
                resolved_constraints,
                weekly_returns=risk_result.weekly_returns,
                cvar_confidence=cfg.optimization.cvar_confidence,
                non_positive_excess_return_policy=cfg.optimization.non_positive_excess_return_policy,
            )
        elif strat_name == "min_variance":
            strategies[strat_name] = optimize_min_variance(
                risk_result.cov_matrix,
                resolved_constraints,
                mu=mu_post,
                rf=rf_rate,
                weekly_returns=risk_result.weekly_returns,
                cvar_confidence=cfg.optimization.cvar_confidence,
            )
        elif strat_name == "min_cvar":
            strategies[strat_name] = optimize_min_cvar(
                risk_result.weekly_returns,
                risk_result.cov_matrix,
                resolved_constraints,
                mu=mu_post,
                rf=rf_rate,
                cvar_confidence=cfg.optimization.cvar_confidence,
            )
        elif strat_name == "erc":
            strategies[strat_name] = optimize_erc(
                risk_result.cov_matrix,
                resolved_constraints,
                mu=mu_post,
                rf=rf_rate,
                weekly_returns=risk_result.weekly_returns,
                cvar_confidence=cfg.optimization.cvar_confidence,
            )
        elif strat_name == "hrp":
            strategies[strat_name] = optimize_hrp(
                risk_result.cov_matrix,
                resolved_constraints,
                linkage_method=cfg.optimization.hrp_linkage_method,
                mu=mu_post,
                rf=rf_rate,
                weekly_returns=risk_result.weekly_returns,
                cvar_confidence=cfg.optimization.cvar_confidence,
            )
        else:
            raise ConfigurationError(f"Unsupported optimization strategy '{strat_name}'.")
        warnings_list.extend(strategies[strat_name].warnings)

    stress_results: Dict[str, Dict[str, Any]] = {}
    if cfg.optimization.include_stress_tests:
        for strat_name, strat_res in strategies.items():
            all_st = run_all_stress_tests(strat_res.weights, risk_result.cov_matrix, ordered_ids)
            entry = dict(all_st["corr_one"])
            entry["vol_shock"] = all_st["vol_shock"]
            entry["systemic"] = all_st["systemic"]
            stress_results[strat_name] = entry

    rebalancing_results: Dict[str, Any] = {}
    invalid_target_statuses = {"non_converged", "failed", "singular_zero_variance", "blocked_non_positive_excess"}
    for strat_name, strat_res in strategies.items():
        if not strat_res.converged or strat_res.status in invalid_target_statuses:
            rebalancing_results[strat_name] = RebalancingResult(
                strategy_name=strat_name,
                status="skipped_invalid_strategy_target",
                feasible=False,
                theoretical_target_weights=dict(strat_res.weights_by_asset),
                warnings=[
                    f"Rebalancing skipped for strategy '{strat_name}' because optimizer did not produce a valid converged target "
                    f"(status='{strat_res.status}', converged={strat_res.converged})."
                ],
            )
            continue
        reb = compute_rebalancing_plan(
            strategy_name=strat_name,
            tickers=ordered_ids,
            target_weights=strat_res.weights_by_asset,
            prices_usd=latest_prices_usd,
            instruments_by_id=inst_map,
            brokers_by_id=cfg.broker_map,
            positions=cfg.positions,
            rebalancing_config=cfg.rebalancing,
            constraints_config=cfg.constraints,
            sector_by_ticker=sector_by_ticker,
            quote_units_by_id=quote_units_by_id,
            prices_are_dirty_per_unit=False,
        )
        rebalancing_results[strat_name] = reb

    unique_warnings = list(dict.fromkeys(warnings_list))

    quotes_and_dq = {
        "iol_quotes": {k: v.to_dict() for k, v in iol_quotes.items()},
        "latest_prices_usd": latest_prices_usd,
        "execution_prices_usd_per_unit": execution_prices_usd_per_unit,
        "price_unit_metadata": price_unit_metadata,
        "fx_diagnostics": {
            k: {
                "original_currency": d.original_currency,
                "economic_currency": d.economic_currency,
                "fx_symbol": d.fx_symbol,
                "direction": d.direction,
                "unit_scale": d.unit_scale,
                "coverage_ratio": d.coverage_ratio,
                "dropped_stale_or_missing_fx": d.dropped_stale_or_missing_fx,
            }
            for k, d in fx_diagnostics.items()
        },
        "weekly_alignment": {
            "frequency": cfg.risk.return_frequency,
            "cutoff_tz": cfg.risk.weekly_cutoff_tz,
            "cutoff_hour": cfg.risk.weekly_cutoff_hour,
            "last_closed_friday": risk_result.last_closed_friday,
            "excluded_incomplete_week_bins": risk_result.excluded_incomplete_week_bins,
            "total_grid_weeks": weekly_bundle.total_grid_weeks,
            "aligned_weeks": weekly_bundle.aligned_weeks,
            "excluded_dates_count": weekly_bundle.excluded_dates_count,
            "missing_weeks_by_asset": weekly_bundle.missing_weeks_by_asset,
            "coverage_by_asset": weekly_bundle.coverage_by_asset,
            "duplicate_dates_removed": weekly_bundle.duplicate_dates_removed,
        },
        "proxies_used": proxies_used,
    }

    constraints_summary = {
        "long_only": cfg.constraints.long_only,
        "budget_sum": cfg.constraints.budget_sum,
        "asset_bounds": {t: list(b) for t, b in zip(resolved_constraints.tickers, resolved_constraints.bounds)},
        "class_bounds": {k: list(v) for k, v in resolved_constraints.class_bounds.items()},
        "hhi_limit": resolved_constraints.hhi_limit,
        "max_sector_weight": resolved_constraints.max_sector_weight,
        "feasible": True,
    }

    return AnalysisResult(
        tickers=ordered_ids,
        selected_strategy=cfg.optimization.selected_strategy,
        config_snapshot=cfg.to_dict(),
        quotes_and_data_quality=quotes_and_dq,
        dcf_views=bl_result.dcf_view_results,
        fixed_income_metrics=fixed_income_metrics,
        black_litterman=bl_result,
        risk_model=risk_result,
        constraints_summary=constraints_summary,
        strategies=strategies,
        stress_test_results=stress_results,
        rebalancing=rebalancing_results,
        prices_hist=prices_usd,
        warnings=unique_warnings,
        errors=[],
    )
