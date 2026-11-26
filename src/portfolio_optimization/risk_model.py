"""Dynamic VIX-regime weekly covariance estimation and PSD diagnostics.

Key invariants:
- NO global scalar stress multiplier on Sigma (old forward_looking_cov_adjustment removed).
- Dynamic regime blend: ``Sigma(p) = (1 - p) * Sigma_calm + p * Sigma_crisis`` where
  both matrices are estimated from aligned weekly (W-FRI) returns and annualized ONCE (* 52).
- Historical regime classification (calm vs crisis weeks) is separated from the
  current-VIX stress weight function ``p(VIX)``.
- Default ``p(VIX)`` schedule (fully editable):
    * VIX < 15.0             -> p = 0.15
    * 15.0 <= VIX <= 20.0    -> linear interpolation 0.15 -> 0.35
    * 20.0 < VIX <= 30.0     -> p = 0.35
    * VIX > 30.0             -> p = 0.60 (deliberate discontinuity above 30)
- Missing/stale VIX is NEVER silently replaced by 0; requires explicit manual ``p``
  or an explicit non-VIX estimator, otherwise raises ``MissingVixDataError``.
- Insufficient crisis/calm sample sizes NEVER fabricate ``1.5 * Sigma_calm``.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS as OAS_estimator

from .errors import DataQualityError, InsufficientRegimeSampleError, MissingVixDataError
from .market_data import WeeklyReturnsBundle, _assign_weekly_friday_bin, sanitize_price_series
from .schemas import RiskModelConfig, RiskModelResult, VixRegimeConfig

WEEKS_PER_YEAR = 52.0


def compute_vix_stress_weight(
    vix_value: float,
    vix_config: Optional[VixRegimeConfig] = None,
) -> Tuple[float, str]:
    """Compute dynamic crisis-regime mixing weight ``p in [0, 1]`` from current VIX."""
    cfg = vix_config or VixRegimeConfig()
    if vix_value is None or not isinstance(vix_value, (int, float)) or not math.isfinite(float(vix_value)) or float(vix_value) <= 0.0:
        raise MissingVixDataError(
            f"Invalid VIX observation ({vix_value}); VIX must be finite and strictly positive.",
            details={"vix_value": vix_value},
        )
    v = float(vix_value)

    if v < cfg.vix_low:
        p = cfg.p_low
        note = f"VIX ({v:.2f}) < {cfg.vix_low:g} -> calm baseline p={p:.4f}."
    elif v <= cfg.vix_mid_low:
        frac = (v - cfg.vix_low) / (cfg.vix_mid_low - cfg.vix_low)
        p = cfg.p_low + frac * (cfg.p_mid - cfg.p_low)
        note = (
            f"VIX ({v:.2f}) in [{cfg.vix_low:g}, {cfg.vix_mid_low:g}] -> "
            f"linearly interpolated p={p:.4f}."
        )
    elif v <= cfg.vix_high:
        p = cfg.p_mid
        note = (
            f"VIX ({v:.2f}) in ({cfg.vix_mid_low:g}, {cfg.vix_high:g}] -> "
            f"elevated plateau p={p:.4f}."
        )
    else:
        p = cfg.p_high
        note = (
            f"VIX ({v:.2f}) > {cfg.vix_high:g} -> deliberate crisis step discontinuity to p={p:.4f}."
        )
    return float(p), note


def ensure_psd_covariance(
    cov_df: pd.DataFrame,
    *,
    min_eigenvalue: float = 1e-10,
) -> Tuple[pd.DataFrame, bool, float, float]:
    """Enforce exact symmetry, finiteness, and positive semi-definiteness via spectral clipping."""
    raw = cov_df.to_numpy(dtype=float)
    if not np.all(np.isfinite(raw)):
        raise DataQualityError("Covariance matrix contains non-finite (NaN/Inf) entries.")

    sym = 0.5 * (raw + raw.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    min_before = float(np.min(eigvals))
    adjusted = False

    if min_before < min_eigenvalue:
        eigvals_clipped = np.maximum(eigvals, min_eigenvalue)
        reconstructed = (eigvecs * eigvals_clipped) @ eigvecs.T
        sym = 0.5 * (reconstructed + reconstructed.T)
        adjusted = True
        min_after = float(np.min(np.linalg.eigvalsh(sym)))
    else:
        min_after = min_before

    out = pd.DataFrame(sym, index=cov_df.index, columns=cov_df.columns, dtype=float)
    return out, adjusted, min_before, min_after


def covariance_to_correlation(cov_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract annualized volatilities and a clean correlation matrix in [-1, 1]."""
    mat = cov_df.to_numpy(dtype=float)
    diag = np.diag(mat)
    if np.any(diag <= 0.0):
        raise DataQualityError(
            "Covariance matrix contains non-positive asset variance(s); cannot construct valid correlation matrix.",
            details={"non_positive_assets": [str(cov_df.index[i]) for i in range(len(diag)) if diag[i] <= 0.0]},
        )
    vols = np.sqrt(diag)
    outer = np.outer(vols, vols)
    corr = np.clip(mat / outer, -1.0, 1.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return (
        pd.DataFrame(corr, index=cov_df.index, columns=cov_df.columns, dtype=float),
        pd.Series(vols, index=cov_df.index, dtype=float),
    )


def shrinkage_covariance_weekly(
    weekly_returns: pd.DataFrame,
    *,
    method: str = "ledoit_wolf",
    annualization_factor: float = WEEKS_PER_YEAR,
) -> Tuple[pd.DataFrame, Optional[float]]:
    """Estimate annualized covariance matrix from weekly returns (annualized ONCE)."""
    if weekly_returns.empty or len(weekly_returns) < 2:
        raise DataQualityError("At least 2 weekly return observations are required to estimate covariance.")

    arr = weekly_returns.to_numpy(dtype=float)
    if not np.all(np.isfinite(arr)):
        raise DataQualityError("Weekly return matrix contains non-finite values.")

    if arr.shape[1] == 1:
        var = float(np.var(arr[:, 0], ddof=1)) * float(annualization_factor)
        return pd.DataFrame([[var]], index=weekly_returns.columns, columns=weekly_returns.columns, dtype=float), 0.0

    if method == "sample":
        cov_w = np.cov(arr, rowvar=False, ddof=1)
        shrink_coef: Optional[float] = 0.0
    elif method == "oas":
        est = OAS_estimator().fit(arr)
        cov_w = est.covariance_
        shrink_coef = float(getattr(est, "shrinkage_", 0.0))
    else:
        est = LedoitWolf().fit(arr)
        cov_w = est.covariance_
        shrink_coef = float(getattr(est, "shrinkage_", 0.0))

    cov_ann = pd.DataFrame(
        cov_w * float(annualization_factor),
        index=weekly_returns.columns,
        columns=weekly_returns.columns,
        dtype=float,
    )
    return cov_ann, shrink_coef


def ewma_covariance_weekly(
    weekly_returns: pd.DataFrame,
    *,
    half_life_weeks: float = 26.0,
    annualization_factor: float = WEEKS_PER_YEAR,
    mean_mode: str = "zero",
    gap_policy: str = "elapsed_weeks_decay",
) -> pd.DataFrame:
    """Optional strictly causal, gap-aware EWMA weekly covariance estimator (annualized ONCE).

    Mathematical properties (AUD-14):
    - Strictly causal: NEVER uses a full-sample mean. Supports ``mean_mode="zero"``
      (RiskMetrics standard, default) and ``mean_mode="causal_recursive"`` (innovations
      centered against prior causal EWMA mean). Appending observations after date ``t``
      leaves the estimated state at ``t`` completely invariant.
    - Gap-aware: validates strictly increasing W-FRI timestamps and scales prior state
      by ``lambda ** delta_weeks`` when intermediate weeks are absent (under
      ``gap_policy="elapsed_weeks_decay"``) or raises ``DataQualityError`` (under
      ``gap_policy="error_on_gap"``), never treating multi-week returns as single-week
      or inventing zero returns.
    - Exact finite-sample weight normalization guarantees symmetry and positive
      semi-definiteness before annualizing once by ``annualization_factor`` (52).
    """
    if weekly_returns.empty or len(weekly_returns) < 2:
        raise DataQualityError("At least 2 weekly observations are required for EWMA covariance.")
    if not math.isfinite(float(half_life_weeks)) or float(half_life_weeks) <= 0.0:
        raise DataQualityError(f"'half_life_weeks' must be finite and > 0 (got {half_life_weeks}).")
    if mean_mode not in ("zero", "causal_recursive"):
        raise DataQualityError(f"Unsupported EWMA mean_mode '{mean_mode}'. Expected 'zero' or 'causal_recursive'.")
    if gap_policy not in ("elapsed_weeks_decay", "error_on_gap"):
        raise DataQualityError(f"Unsupported EWMA gap_policy '{gap_policy}'. Expected 'elapsed_weeks_decay' or 'error_on_gap'.")

    if isinstance(weekly_returns.index, pd.DatetimeIndex):
        if not weekly_returns.index.is_monotonic_increasing:
            raise DataQualityError("EWMA weekly_returns index must be strictly increasing chronologically.")
        if weekly_returns.index.duplicated().any():
            raise DataQualityError("EWMA weekly_returns index contains duplicate timestamps.")
        non_friday = weekly_returns.index[weekly_returns.index.day_of_week != 4]
        if len(non_friday) > 0:
            raise DataQualityError(
                f"EWMA weekly_returns requires W-FRI (Friday) timestamps; found non-Friday date(s): "
                f"{[str(d.date()) for d in non_friday[:3]]}."
            )
        norm_idx = weekly_returns.index.normalize()
        dt_weeks = ((norm_idx[1:] - norm_idx[:-1]) / pd.Timedelta(days=7)).to_numpy(dtype=float)
        if np.any(dt_weeks < 1.0 - 1e-6) or np.any(np.abs(dt_weeks - np.round(dt_weeks)) > 1e-3):
            raise DataQualityError("EWMA weekly_returns timestamps must be positive integer multiples of 7 days.")
        dt_weeks = np.round(dt_weeks)
        if gap_policy == "error_on_gap" and np.any(dt_weeks > 1.0):
            raise DataQualityError(
                f"EWMA weekly_returns contains missing week gap(s) (max delta={int(np.max(dt_weeks))} weeks) "
                "under gap_policy='error_on_gap'."
            )
    else:
        dt_weeks = np.ones(len(weekly_returns) - 1, dtype=float)

    df = weekly_returns
    arr = df.to_numpy(dtype=float)
    if not np.all(np.isfinite(arr)):
        raise DataQualityError("Weekly returns for EWMA contain non-finite values.")

    n_obs, n_assets = arr.shape
    lam = 0.5 ** (1.0 / float(half_life_weeks))

    if mean_mode == "zero":
        r0 = arr[0:1].T
        s_mat = r0 @ r0.T
        w_sum = 1.0
        for i in range(1, n_obs):
            decay = lam ** float(dt_weeks[i - 1])
            ri = arr[i : i + 1].T
            s_mat = decay * s_mat + (ri @ ri.T)
            w_sum = decay * w_sum + 1.0
        cov_t = s_mat / w_sum
    else:
        # Strictly causal recursive mean: innovation at step i uses mean estimated from 0..i-1 only
        m_vec = arr[0].copy()
        w_m = 1.0
        s_mat = np.zeros((n_assets, n_assets), dtype=float)
        w_c = 0.0
        for i in range(1, n_obs):
            decay = lam ** float(dt_weeks[i - 1])
            mu_prev = m_vec / w_m
            innov = (arr[i] - mu_prev).reshape(n_assets, 1)
            s_mat = decay * s_mat + (innov @ innov.T)
            w_c = decay * w_c + 1.0
            m_vec = decay * m_vec + arr[i]
            w_m = decay * w_m + 1.0
        cov_t = s_mat / w_c

    cov_t = 0.5 * (cov_t + cov_t.T)
    cov_ann = cov_t * float(annualization_factor)
    return pd.DataFrame(cov_ann, index=df.columns, columns=df.columns, dtype=float)


def _align_vix_weekly(
    vix_history: pd.Series,
    target_weeks: pd.DatetimeIndex,
    *,
    cutoff_tz: str = "America/New_York",
    cutoff_hour: int = 17,
) -> pd.Series:
    """Align historical VIX causally onto target W-FRI weekly return dates (no future leakage)."""
    san = sanitize_price_series(
        vix_history,
        asset_id="^VIX",
        duplicate_date_policy="last",
        target_tz=cutoff_tz,
        as_of_cutoff=target_weeks.max() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
        require_positive=True,
    )
    if san.series.empty:
        return pd.Series(dtype=float, index=target_weeks)
    bins = _assign_weekly_friday_bin(san.series.index, cutoff_hour=cutoff_hour)
    vix_weekly = san.series.groupby(bins, sort=True).last()
    target_df = pd.DataFrame({"ts": target_weeks.as_unit("ns")})
    vix_df = pd.DataFrame({"ts": vix_weekly.index.as_unit("ns"), "vix": vix_weekly.to_numpy(dtype=float)})
    merged = pd.merge_asof(target_df, vix_df, on="ts", direction="backward", tolerance=pd.Timedelta(days=7))
    return pd.Series(merged["vix"].to_numpy(dtype=float), index=target_weeks, name="^VIX")


def estimate_risk_model(
    weekly_bundle: WeeklyReturnsBundle,
    *,
    risk_config: Optional[RiskModelConfig] = None,
    vix_history: Optional[pd.Series] = None,
    current_vix: Optional[float] = None,
    current_vix_date: Optional[pd.Timestamp] = None,
    current_vix_is_stale: bool = False,
    proxies_used: Optional[Dict[str, str]] = None,
) -> RiskModelResult:
    """Estimate the canonical annualized multi-asset covariance matrix and diagnostics."""
    cfg = risk_config or RiskModelConfig()
    vcfg = cfg.vix_regime
    returns = weekly_bundle.weekly_returns
    tickers = list(returns.columns)
    warnings_list = list(weekly_bundle.warnings)

    last_closed_str = (
        weekly_bundle.last_closed_friday.date().isoformat()
        if weekly_bundle.last_closed_friday is not None
        else None
    )
    incomplete_bins = list(weekly_bundle.excluded_incomplete_week_bins)

    if cfg.covariance_estimator == "static_shrinkage":
        cov_raw, shrink = shrinkage_covariance_weekly(returns, method=cfg.shrinkage_method)
        cov_psd, psd_adj, min_b, min_a = ensure_psd_covariance(cov_raw)
        if psd_adj:
            warnings_list.append(
                f"PSD spectral adjustment applied to covariance matrix (min eigenvalue {min_b:.3e} -> {min_a:.3e})."
            )
        corr_df, vols = covariance_to_correlation(cov_psd)
        return RiskModelResult(
            tickers=tickers,
            cov_matrix=cov_psd,
            corr_matrix=corr_df,
            annualized_volatilities=vols,
            weekly_returns=returns,
            estimator_name="static_shrinkage",
            shrinkage_method=cfg.shrinkage_method,
            shrinkage_calm=shrink,
            total_weekly_samples=len(returns),
            start_date=weekly_bundle.start_date,
            end_date=weekly_bundle.end_date,
            psd_adjusted=psd_adj,
            min_eigenvalue_before=min_b,
            min_eigenvalue_after=min_a,
            excluded_dates_count=weekly_bundle.excluded_dates_count,
            excluded_incomplete_week_bins=incomplete_bins,
            last_closed_friday=last_closed_str,
            missing_weeks_by_asset=dict(weekly_bundle.missing_weeks_by_asset),
            duplicate_dates_removed=dict(weekly_bundle.duplicate_dates_removed),
            proxies_used=dict(proxies_used or {}),
            p_source="not_applicable",
            warnings=warnings_list,
        )

    if cfg.covariance_estimator == "ewma":
        cov_raw = ewma_covariance_weekly(
            returns,
            half_life_weeks=cfg.ewma_half_life_weeks,
            mean_mode=cfg.ewma_mean_mode,
            gap_policy=cfg.ewma_gap_policy,
        )
        cov_psd, psd_adj, min_b, min_a = ensure_psd_covariance(cov_raw)
        if psd_adj:
            warnings_list.append(
                f"PSD spectral adjustment applied to EWMA covariance matrix (min eigenvalue {min_b:.3e} -> {min_a:.3e})."
            )
        corr_df, vols = covariance_to_correlation(cov_psd)
        return RiskModelResult(
            tickers=tickers,
            cov_matrix=cov_psd,
            corr_matrix=corr_df,
            annualized_volatilities=vols,
            weekly_returns=returns,
            estimator_name=f"ewma(half_life={cfg.ewma_half_life_weeks:g}w,mean={cfg.ewma_mean_mode})",
            shrinkage_method="none",
            total_weekly_samples=len(returns),
            start_date=weekly_bundle.start_date,
            end_date=weekly_bundle.end_date,
            psd_adjusted=psd_adj,
            min_eigenvalue_before=min_b,
            min_eigenvalue_after=min_a,
            excluded_dates_count=weekly_bundle.excluded_dates_count,
            excluded_incomplete_week_bins=incomplete_bins,
            last_closed_friday=last_closed_str,
            missing_weeks_by_asset=dict(weekly_bundle.missing_weeks_by_asset),
            duplicate_dates_removed=dict(weekly_bundle.duplicate_dates_removed),
            proxies_used=dict(proxies_used or {}),
            p_source="not_applicable",
            warnings=warnings_list,
        )

    p_value: float
    p_source: str
    discontinuity_note = (
        f"Configured VIX schedule: <{vcfg.vix_low:g}->p={vcfg.p_low:g}; "
        f"[{vcfg.vix_low:g},{vcfg.vix_mid_low:g}] linear to {vcfg.p_mid:g}; "
        f"({vcfg.vix_mid_low:g},{vcfg.vix_high:g}]->p={vcfg.p_mid:g}; "
        f">{vcfg.vix_high:g}->p={vcfg.p_high:g} (deliberate upward step discontinuity above {vcfg.vix_high:g})."
    )

    if vcfg.manual_stress_weight is not None:
        p_value = float(vcfg.manual_stress_weight)
        p_source = "manual_override"
        if current_vix is None or current_vix_is_stale:
            warnings_list.append(
                f"Current VIX is missing or stale (value={current_vix}, stale={current_vix_is_stale}); using explicitly configured manual_stress_weight p={p_value:.4f}."
            )
    else:
        if current_vix is None or not math.isfinite(float(current_vix)) or float(current_vix) <= 0.0:
            raise MissingVixDataError(
                "Current VIX observation is missing or invalid and no 'manual_stress_weight' was configured. "
                "Either provide a valid VIX observation, set 'vix_regime.manual_stress_weight', or select a non-VIX covariance_estimator ('static_shrinkage' or 'ewma').",
                details={"current_vix": current_vix, "vix_date": None if current_vix_date is None else str(current_vix_date)},
            )
        if current_vix_is_stale:
            raise MissingVixDataError(
                f"Current VIX observation ({current_vix:.2f} on {current_vix_date}) exceeds max_vix_staleness_days={vcfg.max_vix_staleness_days}. "
                "Configure 'manual_stress_weight' or switch to a non-VIX covariance_estimator instead of using stale VIX.",
                details={
                    "current_vix": float(current_vix),
                    "vix_date": None if current_vix_date is None else str(current_vix_date),
                    "max_vix_staleness_days": vcfg.max_vix_staleness_days,
                },
            )
        p_value, sched_note = compute_vix_stress_weight(float(current_vix), vcfg)
        p_source = "vix_schedule"
        discontinuity_note = f"{sched_note} {discontinuity_note}"

    if vcfg.regime_classification_method == "vix_history":
        if vix_history is None or vix_history.empty:
            raise MissingVixDataError(
                "Historical VIX series is required for regime_classification_method='vix_history'. "
                "Provide VIX history or explicitly select regime_classification_method='rolling_volatility_quantile' or covariance_estimator='static_shrinkage'.",
            )
        vix_w = _align_vix_weekly(
            vix_history,
            returns.index,
            cutoff_tz=cfg.weekly_cutoff_tz,
            cutoff_hour=cfg.weekly_cutoff_hour,
        )
        valid_mask = vix_w.notna()
        missing_vix_weeks = int((~valid_mask).sum())
        if missing_vix_weeks > 0:
            warnings_list.append(
                f"Excluded {missing_vix_weeks} weekly observation(s) from regime split due to missing historical VIX alignment."
            )
        classified_returns = returns.loc[valid_mask]
        vix_valid = vix_w.loc[valid_mask]
        crisis_mask = vix_valid > vcfg.crisis_vix_threshold
        calm_returns = classified_returns.loc[~crisis_mask]
        crisis_returns = classified_returns.loc[crisis_mask]
    else:
        rolling_vol = returns.rolling(window=vcfg.rolling_vol_weeks, min_periods=vcfg.rolling_vol_weeks).std().mean(axis=1).dropna()
        if rolling_vol.empty:
            calm_returns = returns.iloc[0:0]
            crisis_returns = returns.iloc[0:0]
        else:
            expanding_q = rolling_vol.expanding(min_periods=max(vcfg.min_calm_weeks // 2, 6)).quantile(vcfg.rolling_vol_quantile)
            valid_q = expanding_q.dropna()
            aligned_rv = rolling_vol.loc[valid_q.index]
            crisis_idx = aligned_rv.index[aligned_rv > valid_q]
            calm_idx = aligned_rv.index[aligned_rv <= valid_q]
            calm_returns = returns.loc[calm_idx]
            crisis_returns = returns.loc[crisis_idx]

    n_calm = len(calm_returns)
    n_crisis = len(crisis_returns)

    if n_calm < vcfg.min_calm_weeks or n_crisis < vcfg.min_crisis_weeks:
        msg = (
            f"Insufficient regime observations for dynamic VIX covariance: "
            f"calm={n_calm} (min {vcfg.min_calm_weeks}), crisis={n_crisis} (min {vcfg.min_crisis_weeks}). "
            "Refusing to fabricate Sigma_crisis via arbitrary scaling."
        )
        if vcfg.insufficient_regime_policy == "error":
            raise InsufficientRegimeSampleError(
                msg,
                details={
                    "calm_samples": n_calm,
                    "crisis_samples": n_crisis,
                    "min_calm_weeks": vcfg.min_calm_weeks,
                    "min_crisis_weeks": vcfg.min_crisis_weeks,
                    "crisis_vix_threshold": vcfg.crisis_vix_threshold,
                },
            )
        warnings_list.append(msg + " Falling back to full-sample shrinkage as explicitly configured.")
        cov_raw, shrink_full = shrinkage_covariance_weekly(returns, method=cfg.shrinkage_method)
        cov_psd, psd_adj, min_b, min_a = ensure_psd_covariance(cov_raw)
        corr_df, vols = covariance_to_correlation(cov_psd)
        return RiskModelResult(
            tickers=tickers,
            cov_matrix=cov_psd,
            corr_matrix=corr_df,
            annualized_volatilities=vols,
            weekly_returns=returns,
            estimator_name="vix_regime_fallback_static_shrinkage",
            shrinkage_method=cfg.shrinkage_method,
            shrinkage_calm=shrink_full,
            vix_value=None if current_vix is None else float(current_vix),
            vix_date=None if current_vix_date is None else pd.Timestamp(current_vix_date).date().isoformat(),
            vix_is_stale=current_vix_is_stale,
            stress_weight_p=p_value,
            p_source=p_source,
            calm_samples=n_calm,
            crisis_samples=n_crisis,
            total_weekly_samples=len(returns),
            start_date=weekly_bundle.start_date,
            end_date=weekly_bundle.end_date,
            psd_adjusted=psd_adj,
            min_eigenvalue_before=min_b,
            min_eigenvalue_after=min_a,
            excluded_dates_count=weekly_bundle.excluded_dates_count,
            excluded_incomplete_week_bins=incomplete_bins,
            last_closed_friday=last_closed_str,
            missing_weeks_by_asset=dict(weekly_bundle.missing_weeks_by_asset),
            duplicate_dates_removed=dict(weekly_bundle.duplicate_dates_removed),
            proxies_used=dict(proxies_used or {}),
            discontinuity_note=discontinuity_note,
            warnings=warnings_list,
        )

    cov_calm_raw, shrink_calm = shrinkage_covariance_weekly(calm_returns, method=cfg.shrinkage_method)
    cov_crisis_raw, shrink_crisis = shrinkage_covariance_weekly(crisis_returns, method=cfg.shrinkage_method)

    cov_calm_psd, adj_c, _, _ = ensure_psd_covariance(cov_calm_raw)
    cov_crisis_psd, adj_k, _, _ = ensure_psd_covariance(cov_crisis_raw)

    cov_blended_raw = (1.0 - p_value) * cov_calm_psd + p_value * cov_crisis_psd
    cov_psd, psd_adj, min_b, min_a = ensure_psd_covariance(cov_blended_raw)

    if adj_c or adj_k or psd_adj:
        warnings_list.append(
            f"PSD spectral adjustment applied (blended min eigenvalue {min_b:.3e} -> {min_a:.3e})."
        )

    corr_df, vols = covariance_to_correlation(cov_psd)

    return RiskModelResult(
        tickers=tickers,
        cov_matrix=cov_psd,
        corr_matrix=corr_df,
        annualized_volatilities=vols,
        weekly_returns=returns,
        cov_calm=cov_calm_psd,
        cov_crisis=cov_crisis_psd,
        estimator_name="vix_regime",
        shrinkage_method=cfg.shrinkage_method,
        shrinkage_calm=shrink_calm,
        shrinkage_crisis=shrink_crisis,
        vix_value=None if current_vix is None else float(current_vix),
        vix_date=None if current_vix_date is None else pd.Timestamp(current_vix_date).date().isoformat(),
        vix_is_stale=current_vix_is_stale,
        stress_weight_p=p_value,
        p_source=p_source,
        calm_samples=n_calm,
        crisis_samples=n_crisis,
        total_weekly_samples=len(returns),
        start_date=weekly_bundle.start_date,
        end_date=weekly_bundle.end_date,
        psd_adjusted=bool(adj_c or adj_k or psd_adj),
        min_eigenvalue_before=min_b,
        min_eigenvalue_after=min_a,
        excluded_dates_count=weekly_bundle.excluded_dates_count,
        excluded_incomplete_week_bins=incomplete_bins,
        last_closed_friday=last_closed_str,
        missing_weeks_by_asset=dict(weekly_bundle.missing_weeks_by_asset),
        duplicate_dates_removed=dict(weekly_bundle.duplicate_dates_removed),
        proxies_used=dict(proxies_used or {}),
        discontinuity_note=discontinuity_note,
        warnings=warnings_list,
    )
