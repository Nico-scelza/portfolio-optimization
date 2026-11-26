"""Market data acquisition, pandas 3.x-safe causal FX conversion, and W-FRI weekly returns.

Key invariants:
- All time series are sorted chronologically ascending with explicit duplicate-date policy.
- FX conversion is strictly causal (backward-only with configurable max staleness),
  handles currency subunits (e.g. GBp/GBX -> GBP -> USD), verifies direction,
  and NEVER silently returns unconverted local prices labeled as USD.
- Multi-asset risk series use weekly Friday-to-Friday (W-FRI) returns aligned in USD,
  without artificial zeros on holidays/weekends and without treating multi-week gaps
  as single-week returns.
- All HTTP/network calls enforce strict TLS certificate verification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .config import FRED_BASE_URL, _redact_secret, get_fred_api_key
from .errors import DataQualityError, FXConversionError, InsufficientHistoryError
from .schemas import InstrumentConfig, RiskModelConfig

CURRENCY_FX_SPECS: Dict[str, Tuple[str, str, float]] = {
    "GBp": ("GBPUSD=X", "multiply", 0.01),
    "GBX": ("GBPUSD=X", "multiply", 0.01),
    "ZAc": ("ZAR=X", "divide", 0.01),
    "ZAC": ("ZAR=X", "divide", 0.01),
    "ILA": ("ILS=X", "divide", 0.01),
    "ILX": ("ILS=X", "divide", 0.01),
    "EUR": ("EURUSD=X", "multiply", 1.0),
    "GBP": ("GBPUSD=X", "multiply", 1.0),
    "AUD": ("AUDUSD=X", "multiply", 1.0),
    "NZD": ("NZDUSD=X", "multiply", 1.0),
    "ARS": ("ARS=X", "divide", 1.0),
    "BRL": ("BRL=X", "divide", 1.0),
    "JPY": ("JPY=X", "divide", 1.0),
    "CAD": ("CAD=X", "divide", 1.0),
    "CHF": ("CHF=X", "divide", 1.0),
    "CNY": ("CNY=X", "divide", 1.0),
    "MXN": ("MXN=X", "divide", 1.0),
    "CLP": ("CLP=X", "divide", 1.0),
    "COP": ("COP=X", "divide", 1.0),
    "PEN": ("PEN=X", "divide", 1.0),
    "INR": ("INR=X", "divide", 1.0),
    "KRW": ("KRW=X", "divide", 1.0),
    "TWD": ("TWD=X", "divide", 1.0),
    "HKD": ("HKD=X", "divide", 1.0),
    "SGD": ("SGD=X", "divide", 1.0),
    "THB": ("THB=X", "divide", 1.0),
    "ZAR": ("ZAR=X", "divide", 1.0),
    "TRY": ("TRY=X", "divide", 1.0),
    "SEK": ("SEK=X", "divide", 1.0),
    "NOK": ("NOK=X", "divide", 1.0),
    "DKK": ("DKK=X", "divide", 1.0),
    "PLN": ("PLN=X", "divide", 1.0),
    "ILS": ("ILS=X", "divide", 1.0),
    "UYU": ("UYU=X", "divide", 1.0),
}


_BASE_CURRENCY_OF_SUBUNIT: Dict[str, str] = {
    "GBp": "GBP",
    "GBX": "GBP",
    "ZAc": "ZAR",
    "ZAC": "ZAR",
    "ILA": "ILS",
    "ILX": "ILS",
}


@dataclass
class SanitizedSeriesResult:
    series: pd.Series
    duplicates_removed: int = 0
    invalid_values_removed: int = 0
    was_reordered: bool = False


@dataclass
class FXConversionDiagnostics:
    asset_id: str
    original_currency: str
    fx_symbol: Optional[str]
    direction: str
    unit_scale: float
    total_input_points: int
    valid_converted_points: int
    coverage_ratio: float
    dropped_stale_or_missing_fx: int
    max_staleness_days: int
    economic_currency: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class WeeklyReturnsBundle:
    weekly_returns: pd.DataFrame
    weekly_prices_usd: pd.DataFrame
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    total_grid_weeks: int
    aligned_weeks: int
    excluded_dates_count: int
    last_closed_friday: Optional[pd.Timestamp] = None
    excluded_incomplete_week_bins: List[str] = field(default_factory=list)
    missing_weeks_by_asset: Dict[str, int] = field(default_factory=dict)
    coverage_by_asset: Dict[str, float] = field(default_factory=dict)
    duplicate_dates_removed: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def resolve_provider_currency(
    asset_id: str,
    provider_currency: Optional[str],
    declared_currency: Optional[str] = None,
) -> Tuple[str, str, List[str]]:
    """Resolve effective quote currency and economic currency while validating conflicts.

    Rules (AUD-01):
    - Differentiates economic currency (e.g., ``GBP``) from subunit quote currency
      (e.g., ``GBp`` / ``GBX`` requiring ``0.01`` scaling).
    - Never assumes ``USD`` when both provider metadata and explicit override are absent.
    - When user declares economic currency ``GBP`` and provider reports ``GBp``/``GBX``,
      preserves the provider subunit quote currency so ``0.01`` scaling is applied.
    - Rejects material currency conflicts (e.g., declared ``USD`` vs provider ``GBp``/``ARS``/``EUR``,
      or declared subunit ``GBp`` vs provider major ``GBP``) with ``FXConversionError``.
    """
    prov = str(provider_currency).strip() if provider_currency is not None and str(provider_currency).strip() else None
    decl = str(declared_currency).strip() if declared_currency is not None and str(declared_currency).strip() else None
    warnings: List[str] = []

    def _norm_code(code: str) -> str:
        return code if code in _BASE_CURRENCY_OF_SUBUNIT else code.upper()

    if decl is None:
        if prov is None:
            raise FXConversionError(
                f"Missing currency metadata from market data provider for '{asset_id}'; cannot assume USD.",
                asset_id=asset_id,
                details={"provider_currency": None, "declared_currency": None},
            )
        eff_quote = _norm_code(prov)
        econ = _BASE_CURRENCY_OF_SUBUNIT.get(eff_quote, eff_quote.upper())
        return eff_quote, econ, warnings

    decl_norm = _norm_code(decl)
    decl_econ = _BASE_CURRENCY_OF_SUBUNIT.get(decl_norm, decl_norm.upper())

    if prov is None:
        warnings.append(
            f"Provider omitted currency metadata for '{asset_id}'; using explicit declared currency '{decl_norm}'."
        )
        return decl_norm, decl_econ, warnings

    prov_norm = _norm_code(prov)
    prov_econ = _BASE_CURRENCY_OF_SUBUNIT.get(prov_norm, prov_norm.upper())

    if prov_norm == decl_norm:
        return prov_norm, prov_econ, warnings

    if prov_econ == decl_econ:
        if prov_norm in _BASE_CURRENCY_OF_SUBUNIT and decl_norm not in _BASE_CURRENCY_OF_SUBUNIT:
            # Declared economic currency (e.g. GBP) matches provider subunit quote (e.g. GBp/GBX): apply 0.01!
            warnings.append(
                f"Asset '{asset_id}' declared economic currency '{decl_norm}' while provider quotes in subunit "
                f"'{prov_norm}'; applying provider subunit scale (0.01) before FX conversion to USD."
            )
            return prov_norm, prov_econ, warnings
        # Material 100x scale conflict (e.g. declared GBp vs provider GBP, or conflicting subunits)
        raise FXConversionError(
            f"Currency scale conflict for '{asset_id}': declared '{decl_norm}' conflicts with provider quote unit '{prov_norm}'.",
            asset_id=asset_id,
            details={"declared_currency": decl_norm, "provider_currency": prov_norm},
        )

    raise FXConversionError(
        f"Material currency conflict for '{asset_id}': declared '{decl_norm}' (economic {decl_econ}) "
        f"conflicts with provider currency '{prov_norm}' (economic {prov_econ}). Refusing to label mismatched prices as USD.",
        asset_id=asset_id,
        details={
            "declared_currency": decl_norm,
            "declared_economic_currency": decl_econ,
            "provider_currency": prov_norm,
            "provider_economic_currency": prov_econ,
        },
    )


def get_currency_fx_spec(currency: str) -> Optional[Tuple[str, str, float]]:
    """Resolve ``(fx_symbol, direction, unit_scale)`` preserving subunit case (e.g. ``GBp``)."""
    raw = str(currency or "").strip()
    if not raw or raw.upper() == "USD":
        return None
    if raw in CURRENCY_FX_SPECS:
        return CURRENCY_FX_SPECS[raw]
    upper = raw.upper()
    if upper in CURRENCY_FX_SPECS:
        return CURRENCY_FX_SPECS[upper]
    return None


def _normalize_datetime_index(
    index: pd.Index,
    *,
    target_tz: Optional[str] = None,
    strip_tz: bool = True,
) -> pd.DatetimeIndex:
    """Normalize any datetime-like index to uniform ``datetime64[ns]`` cleanly on pandas 3.x."""
    dt_idx = pd.to_datetime(index)
    if not isinstance(dt_idx, pd.DatetimeIndex):
        dt_idx = pd.DatetimeIndex(dt_idx)

    if dt_idx.tz is not None:
        if target_tz:
            dt_idx = dt_idx.tz_convert(ZoneInfo(target_tz))
        else:
            dt_idx = dt_idx.tz_convert("UTC")
        if strip_tz:
            dt_idx = dt_idx.tz_localize(None)

    return dt_idx.as_unit("ns")


def sanitize_price_series(
    series: pd.Series,
    *,
    asset_id: str = "",
    duplicate_date_policy: str = "last",
    target_tz: Optional[str] = None,
    as_of_cutoff: Optional[pd.Timestamp] = None,
    require_positive: bool = True,
) -> SanitizedSeriesResult:
    """Sort series chronologically ascending, handle duplicate dates, and filter invalid values."""
    if series is None or len(series) == 0:
        return SanitizedSeriesResult(series=pd.Series(dtype=float, name=asset_id or None))

    s = pd.Series(series.values, index=_normalize_datetime_index(series.index, target_tz=target_tz, strip_tz=True), name=series.name or asset_id)
    s = pd.to_numeric(s, errors="coerce")

    was_reordered = not s.index.is_monotonic_increasing
    if was_reordered:
        s = s.sort_index(ascending=True, kind="mergesort")

    if as_of_cutoff is not None:
        cutoff_ts = pd.Timestamp(as_of_cutoff)
        if cutoff_ts.tz is not None:
            cutoff_ts = cutoff_ts.tz_convert(ZoneInfo(target_tz)).tz_localize(None) if target_tz else cutoff_ts.tz_convert("UTC").tz_localize(None)
        cutoff_ts = cutoff_ts.as_unit("ns")
        s = s[s.index <= cutoff_ts]

    if require_positive:
        valid_mask = np.isfinite(s.to_numpy(dtype=float, na_value=np.nan)) & (s.to_numpy(dtype=float, na_value=np.nan) > 0.0)
    else:
        valid_mask = np.isfinite(s.to_numpy(dtype=float, na_value=np.nan))
    invalid_removed = int((~valid_mask).sum())
    s = s[valid_mask]

    dup_mask = s.index.duplicated(keep="last")
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        if duplicate_date_policy == "error":
            dup_dates = [str(d) for d in s.index[dup_mask][:5]]
            raise DataQualityError(
                f"Duplicate timestamps detected for '{asset_id}' under duplicate_date_policy='error': {dup_dates}.",
                asset_id=asset_id or None,
                details={"duplicate_count": dup_count, "sample_dates": dup_dates},
            )
        s = s[~dup_mask]

    return SanitizedSeriesResult(
        series=s.astype(float),
        duplicates_removed=dup_count,
        invalid_values_removed=invalid_removed,
        was_reordered=was_reordered,
    )


def convert_price_series_to_usd(
    asset_id: str,
    prices: pd.Series,
    currency: str,
    fx_rates: Optional[pd.Series] = None,
    *,
    fx_max_staleness_days: int = 5,
    min_fx_coverage: float = 0.95,
    duplicate_date_policy: str = "last",
    as_of_cutoff: Optional[pd.Timestamp] = None,
) -> Tuple[pd.Series, FXConversionDiagnostics]:
    """Convert an asset price series into USD with strict causality and pandas 3.x compatibility."""
    raw_ccy = str(currency or "").strip()
    if not raw_ccy:
        raise FXConversionError(
            f"Missing currency for asset '{asset_id}'; cannot assume USD without currency metadata.",
            asset_id=asset_id,
        )

    sanitized_prices = sanitize_price_series(
        prices,
        asset_id=asset_id,
        duplicate_date_policy=duplicate_date_policy,
        as_of_cutoff=as_of_cutoff,
        require_positive=True,
    )
    clean_prices = sanitized_prices.series
    if clean_prices.empty:
        raise FXConversionError(
            f"No valid positive prices available for '{asset_id}'.",
            asset_id=asset_id,
        )

    if raw_ccy.upper() == "USD":
        diag = FXConversionDiagnostics(
            asset_id=asset_id,
            original_currency="USD",
            fx_symbol=None,
            direction="identity",
            unit_scale=1.0,
            total_input_points=len(clean_prices),
            valid_converted_points=len(clean_prices),
            coverage_ratio=1.0,
            dropped_stale_or_missing_fx=0,
            max_staleness_days=fx_max_staleness_days,
        )
        return clean_prices.rename(asset_id), diag

    spec = get_currency_fx_spec(raw_ccy)
    if spec is None:
        raise FXConversionError(
            f"Unsupported currency '{raw_ccy}' for asset '{asset_id}'. Cannot convert safely to USD.",
            asset_id=asset_id,
            details={"currency": raw_ccy},
        )
    fx_symbol, direction, unit_scale = spec

    if fx_rates is None or len(fx_rates) == 0:
        raise FXConversionError(
            f"Missing historical FX series '{fx_symbol}' required to convert '{asset_id}' from {raw_ccy} to USD.",
            asset_id=asset_id,
            details={"currency": raw_ccy, "fx_symbol": fx_symbol},
        )

    sanitized_fx = sanitize_price_series(
        fx_rates,
        asset_id=fx_symbol,
        duplicate_date_policy=duplicate_date_policy,
        as_of_cutoff=as_of_cutoff,
        require_positive=True,
    )
    clean_fx = sanitized_fx.series
    if clean_fx.empty:
        raise FXConversionError(
            f"FX series '{fx_symbol}' for '{asset_id}' ({raw_ccy}) contains no finite positive observations.",
            asset_id=asset_id,
            details={"currency": raw_ccy, "fx_symbol": fx_symbol},
        )

    price_df = pd.DataFrame({"ts": clean_prices.index, "price": clean_prices.to_numpy(dtype=float)})
    fx_df = pd.DataFrame({"ts": clean_fx.index, "fx_rate": clean_fx.to_numpy(dtype=float)})

    price_df["ts"] = price_df["ts"].astype("datetime64[ns]")
    fx_df["ts"] = fx_df["ts"].astype("datetime64[ns]")

    tolerance = pd.Timedelta(days=int(fx_max_staleness_days))
    merged = pd.merge_asof(
        price_df,
        fx_df,
        on="ts",
        direction="backward",
        tolerance=tolerance,
    )

    valid_fx_mask = merged["fx_rate"].notna() & np.isfinite(merged["fx_rate"]) & (merged["fx_rate"] > 0.0)
    total_pts = len(merged)
    valid_pts = int(valid_fx_mask.sum())
    dropped_pts = total_pts - valid_pts
    coverage = valid_pts / total_pts if total_pts > 0 else 0.0

    if valid_pts == 0 or coverage < min_fx_coverage:
        raise FXConversionError(
            f"Insufficient causal FX coverage for '{asset_id}' ({raw_ccy} via {fx_symbol}): "
            f"{valid_pts}/{total_pts} ({coverage:.1%}) < required {min_fx_coverage:.1%} "
            f"(max staleness={fx_max_staleness_days}d, no future bfill permitted).",
            asset_id=asset_id,
            details={
                "currency": raw_ccy,
                "fx_symbol": fx_symbol,
                "valid_points": valid_pts,
                "total_points": total_pts,
                "coverage_ratio": coverage,
                "min_fx_coverage": min_fx_coverage,
                "fx_max_staleness_days": fx_max_staleness_days,
            },
        )

    valid_rows = merged.loc[valid_fx_mask].copy()
    scaled_prices = valid_rows["price"] * unit_scale
    if direction == "multiply":
        converted = scaled_prices * valid_rows["fx_rate"]
    else:
        converted = scaled_prices / valid_rows["fx_rate"]

    result_series = pd.Series(
        converted.to_numpy(dtype=float),
        index=pd.DatetimeIndex(valid_rows["ts"]),
        name=asset_id,
    )
    diag = FXConversionDiagnostics(
        asset_id=asset_id,
        original_currency=raw_ccy,
        fx_symbol=fx_symbol,
        direction=direction,
        unit_scale=unit_scale,
        total_input_points=total_pts,
        valid_converted_points=valid_pts,
        coverage_ratio=coverage,
        dropped_stale_or_missing_fx=dropped_pts,
        max_staleness_days=fx_max_staleness_days,
    )
    return result_series, diag


def _extract_close_series(df: pd.DataFrame) -> pd.Series:
    """Extract adjusted/close price Series safely from yfinance DataFrame (single or MultiIndex)."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        col = "Adj Close" if "Adj Close" in lvl0 else ("Close" if "Close" in lvl0 else None)
        if col is None:
            return pd.Series(dtype=float)
        sub = df[col]
        return sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub
    col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if col is None:
        return pd.Series(dtype=float)
    return df[col]


def _download_fx_series(fx_symbol: str, period: str = "10y") -> pd.Series:
    fx_data = yf.download(fx_symbol, period=period, auto_adjust=True, progress=False)
    return _extract_close_series(fx_data)


def _download_prices(
    tickers: List[str],
    period: str = "10y",
    *,
    fx_max_staleness_days: int = 5,
    min_fx_coverage: float = 0.95,
    duplicate_date_policy: str = "last",
    currency_overrides: Optional[Dict[str, Optional[str]]] = None,
    as_of_cutoff: Optional[pd.Timestamp] = None,
    fx_fetcher: Optional[Callable[[str, str], pd.Series]] = None,
    allow_partial: bool = False,
    errors_out: Optional[Dict[str, Dict[str, Any]]] = None,
    fx_diagnostics_out: Optional[Dict[str, FXConversionDiagnostics]] = None,
    unadjusted_close_out: Optional[Dict[str, pd.Series]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Download historical adjusted USD-normalized prices via yfinance.

    By default (``allow_partial=False``), FX/currency failures propagate immediately as
    ``FXConversionError`` and history/provider failures propagate as ``InsufficientHistoryError``
    (never silently dropping failed assets or masking FX root causes).
    """
    fx_cache: Dict[str, pd.Series] = {}
    fetcher = fx_fetcher or _download_fx_series
    overrides = {
        k.upper(): v
        for k, v in (currency_overrides or {}).items()
        if v is not None and str(v).strip()
    }

    dropped: List[str] = []
    series_map: Dict[str, pd.Series] = {}

    def _handle_error(tkr_raw: str, tkr_up: str, exc: Exception) -> None:
        if not allow_partial:
            raise exc
        dropped.append(tkr_raw)
        if errors_out is not None:
            if hasattr(exc, "to_dict"):
                errors_out[tkr_up] = exc.to_dict()  # type: ignore[union-attr]
            else:
                errors_out[tkr_up] = {
                    "code": type(exc).__name__,
                    "asset_id": tkr_up,
                    "message": str(exc),
                }

    for ticker in tickers:
        ticker_upper = ticker.strip().upper()
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info or {}
            provider_currency = info.get("currency")
            declared_currency = overrides.get(ticker_upper)
            eff_currency, econ_currency, ccy_warnings = resolve_provider_currency(
                asset_id=ticker_upper,
                provider_currency=provider_currency,
                declared_currency=declared_currency,
            )

            hist = ticker_obj.history(period=period, auto_adjust=True)
            if hist is None or hist.empty:
                raise InsufficientHistoryError(
                    f"No historical price data returned by provider for '{ticker_upper}'.",
                    asset_id=ticker_upper,
                    details={"ticker": ticker, "period": period},
                )
            raw_series = _extract_close_series(hist)
            if raw_series.empty:
                raise InsufficientHistoryError(
                    f"Provider returned historical table without usable Close/Adj Close column for '{ticker_upper}'.",
                    asset_id=ticker_upper,
                    details={"ticker": ticker},
                )

            fx_series: Optional[pd.Series] = None
            spec = get_currency_fx_spec(eff_currency)
            if spec is not None:
                fx_sym = spec[0]
                if fx_sym not in fx_cache:
                    fx_cache[fx_sym] = fetcher(fx_sym, period)
                fx_series = fx_cache[fx_sym]

            usd_series, diag = convert_price_series_to_usd(
                asset_id=ticker_upper,
                prices=raw_series,
                currency=eff_currency,
                fx_rates=fx_series,
                fx_max_staleness_days=fx_max_staleness_days,
                min_fx_coverage=min_fx_coverage,
                duplicate_date_policy=duplicate_date_policy,
                as_of_cutoff=as_of_cutoff,
            )
            diag.economic_currency = econ_currency
            diag.warnings.extend(ccy_warnings)
            series_map[ticker_upper] = usd_series
            if fx_diagnostics_out is not None:
                fx_diagnostics_out[ticker_upper] = diag

            if unadjusted_close_out is not None:
                try:
                    hist_unadj = ticker_obj.history(period=period, auto_adjust=False)
                    if hist_unadj is not None and not hist_unadj.empty and "Close" in hist_unadj.columns:
                        raw_unadj = hist_unadj["Close"]
                        if isinstance(raw_unadj, pd.DataFrame):
                            raw_unadj = raw_unadj.iloc[:, 0]
                        unadj_usd, _ = convert_price_series_to_usd(
                            asset_id=ticker_upper,
                            prices=raw_unadj,
                            currency=eff_currency,
                            fx_rates=fx_series,
                            fx_max_staleness_days=fx_max_staleness_days,
                            min_fx_coverage=min_fx_coverage,
                            duplicate_date_policy=duplicate_date_policy,
                            as_of_cutoff=as_of_cutoff,
                        )
                        unadjusted_close_out[ticker_upper] = unadj_usd
                except Exception:
                    pass
        except (FXConversionError, InsufficientHistoryError, DataQualityError) as exc:
            _handle_error(ticker, ticker_upper, exc)
        except Exception as exc:
            wrapped = DataQualityError(
                f"Provider error while downloading '{ticker_upper}': {type(exc).__name__}: {exc}",
                asset_id=ticker_upper,
                details={"ticker": ticker, "error_type": type(exc).__name__},
            )
            _handle_error(ticker, ticker_upper, wrapped)

    if not series_map:
        return pd.DataFrame(), dropped

    prices_usd = pd.DataFrame(series_map).sort_index(ascending=True)
    prices_usd = prices_usd.dropna(how="all")
    return prices_usd, dropped


def _assign_weekly_friday_bin(
    index: pd.DatetimeIndex,
    *,
    cutoff_hour: int = 17,
) -> pd.DatetimeIndex:
    """Assign each timestamp to its Friday-ending (W-FRI) bin without post-cutoff leakage."""
    has_intraday_time = bool(((index.hour != 0) | (index.minute != 0) | (index.second != 0)).any())
    adjusted = index
    if has_intraday_time:
        friday_after_cutoff = (adjusted.day_of_week == 4) & (
            (adjusted.hour > cutoff_hour)
            | ((adjusted.hour == cutoff_hour) & ((adjusted.minute > 0) | (adjusted.second > 0)))
        )
        if friday_after_cutoff.any():
            adjusted = pd.DatetimeIndex(
                np.where(friday_after_cutoff, adjusted + pd.Timedelta(days=1), adjusted)
            )
    days_to_fri = (4 - adjusted.day_of_week) % 7
    bin_dates = adjusted.normalize() + pd.to_timedelta(days_to_fri, unit="D")
    return pd.DatetimeIndex(bin_dates).as_unit("ns")


def _normalize_cutoff_to_local_naive(
    cutoff: pd.Timestamp,
    target_tz: str,
) -> pd.Timestamp:
    """Convert a cutoff timestamp into naive wall-clock time in ``target_tz`` (respecting DST)."""
    ts = pd.Timestamp(cutoff)
    if ts.tz is not None:
        ts = ts.tz_convert(ZoneInfo(target_tz)).tz_localize(None)
    return ts.as_unit("ns")


def _compute_last_closed_friday(
    ref_ts_local: pd.Timestamp,
    *,
    cutoff_hour: int = 17,
    is_date_only_ref: bool = False,
) -> pd.Timestamp:
    """Compute the latest W-FRI bin whose Friday ``cutoff_hour:00`` close has occurred by ``ref_ts_local``.

    Semantics (AUD-04):
    - A W-FRI week ending on Friday ``F`` closes at ``F 17:00:00`` (or configured ``cutoff_hour``)
      in ``weekly_cutoff_tz`` (default ``America/New_York``, DST-aware when input timestamps are tz-aware).
    - If ``ref_ts_local`` is date-only (midnight ``00:00:00`` with no intraday timestamps),
      it represents end-of-day on that calendar date (``23:59:59``):
        * Wednesday ``2025-07-02`` -> Friday ``2025-07-04 17:00`` has NOT occurred -> ``2025-06-27``.
        * Friday ``2025-07-04`` or Saturday ``2025-07-05`` -> Friday ``2025-07-04 17:00`` HAS occurred -> ``2025-07-04``.
    - If ``ref_ts_local`` has an explicit intraday time (e.g., Friday ``16:00`` vs ``17:00`` vs ``18:30``),
      it is compared directly against ``cand_fri + Timedelta(hours=cutoff_hour)``:
        * Friday ``2025-07-04 16:00`` -> ``2025-06-27`` (week still open).
        * Friday ``2025-07-04 17:00`` or ``18:30`` -> ``2025-07-04`` (week closed; post-17:00 crypto prints
          belong to ``2025-07-11`` and are excluded as incomplete).
    - A Friday market holiday (e.g. Good Friday ``2025-04-18`` where last equity trade is Thursday ``2025-04-17``)
      is validly INCLUDED once ``ref_ts_local >= 2025-04-18 17:00`` (e.g., Saturday ``2025-04-19`` or any later date).
    """
    eff_ref = ref_ts_local
    if is_date_only_ref and eff_ref.hour == 0 and eff_ref.minute == 0 and eff_ref.second == 0 and eff_ref.nanosecond == 0:
        eff_ref = eff_ref.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)

    days_since_fri = (eff_ref.day_of_week - 4) % 7
    cand_fri = eff_ref.normalize() - pd.Timedelta(days=days_since_fri)
    cand_close = cand_fri + pd.Timedelta(hours=int(cutoff_hour))
    if eff_ref < cand_close:
        return (cand_fri - pd.Timedelta(days=7)).as_unit("ns")
    return cand_fri.as_unit("ns")


def compute_weekly_returns_w_fri(
    prices_usd: pd.DataFrame,
    *,
    risk_config: Optional[RiskModelConfig] = None,
    as_of_cutoff: Optional[pd.Timestamp] = None,
) -> WeeklyReturnsBundle:
    """Build Friday-to-Friday (W-FRI) USD weekly returns excluding incomplete current weeks."""
    cfg = risk_config or RiskModelConfig()
    if prices_usd is None or prices_usd.empty:
        raise InsufficientHistoryError("Cannot compute weekly returns from empty price DataFrame.")

    if prices_usd.columns.duplicated().any():
        raise DataQualityError(
            f"Duplicate asset columns found in price DataFrame: {list(prices_usd.columns[prices_usd.columns.duplicated()])}."
        )

    sanitized_series_map: Dict[str, pd.Series] = {}
    dup_removed: Dict[str, int] = {}
    warnings_list: List[str] = []
    any_intraday_obs = False

    for col in prices_usd.columns:
        san = sanitize_price_series(
            prices_usd[col],
            asset_id=str(col),
            duplicate_date_policy=cfg.duplicate_date_policy,
            target_tz=cfg.weekly_cutoff_tz,
            as_of_cutoff=as_of_cutoff,
            require_positive=True,
        )
        dup_removed[str(col)] = san.duplicates_removed
        if san.duplicates_removed > 0:
            warnings_list.append(
                f"Asset '{col}': removed {san.duplicates_removed} duplicate timestamp(s) using policy='{cfg.duplicate_date_policy}'."
            )
        if san.series.empty:
            raise InsufficientHistoryError(
                f"Asset '{col}' has zero valid positive USD prices after sanitization.",
                asset_id=str(col),
            )
        idx = san.series.index
        if bool(((idx.hour != 0) | (idx.minute != 0) | (idx.second != 0)).any()):
            any_intraday_obs = True
        sanitized_series_map[str(col)] = san.series

    if as_of_cutoff is not None:
        ref_local = _normalize_cutoff_to_local_naive(as_of_cutoff, cfg.weekly_cutoff_tz)
        cutoff_is_date_only = (
            ref_local.hour == 0 and ref_local.minute == 0 and ref_local.second == 0 and ref_local.nanosecond == 0
        )
        last_closed_friday = _compute_last_closed_friday(
            ref_local,
            cutoff_hour=cfg.weekly_cutoff_hour,
            is_date_only_ref=cutoff_is_date_only,
        )
    else:
        max_obs_ts = max(s.index.max() for s in sanitized_series_map.values())
        last_closed_friday = _compute_last_closed_friday(
            max_obs_ts,
            cutoff_hour=cfg.weekly_cutoff_hour,
            is_date_only_ref=not any_intraday_obs,
        )

    weekly_price_cols: Dict[str, pd.Series] = {}
    incomplete_bins_found: set[pd.Timestamp] = set()

    for col, san_s in sanitized_series_map.items():
        bins = _assign_weekly_friday_bin(san_s.index, cutoff_hour=cfg.weekly_cutoff_hour)
        grouped = san_s.groupby(bins, sort=True).last()
        open_bins = grouped.index[grouped.index > last_closed_friday]
        if len(open_bins) > 0:
            for b in open_bins:
                incomplete_bins_found.add(pd.Timestamp(b))
            grouped = grouped.loc[grouped.index <= last_closed_friday]
        if grouped.empty:
            raise InsufficientHistoryError(
                f"Asset '{col}' has zero observations in closed W-FRI weeks (last_closed_friday={last_closed_friday.date().isoformat()}).",
                asset_id=col,
            )
        weekly_price_cols[col] = grouped

    excluded_incomplete_bins = sorted(b.date().isoformat() for b in incomplete_bins_found)
    if excluded_incomplete_bins:
        warnings_list.append(
            f"Excluded incomplete/unclosed W-FRI week bin(s) {excluded_incomplete_bins} after last closed Friday "
            f"{last_closed_friday.date().isoformat()} ({cfg.weekly_cutoff_hour:02d}:00 {cfg.weekly_cutoff_tz})."
        )

    first_common_week = max(s.index.min() for s in weekly_price_cols.values())
    last_common_week = min(min(s.index.max() for s in weekly_price_cols.values()), last_closed_friday)

    if first_common_week >= last_common_week:
        raise InsufficientHistoryError(
            "Assets have no overlapping historical date range for weekly return alignment.",
            details={
                "last_closed_friday": last_closed_friday.date().isoformat(),
                "excluded_incomplete_week_bins": excluded_incomplete_bins,
                "asset_ranges": {
                    k: (str(v.index.min().date()), str(v.index.max().date()))
                    for k, v in weekly_price_cols.items()
                },
            },
        )

    full_friday_grid = pd.date_range(start=first_common_week, end=last_common_week, freq="W-FRI").as_unit("ns")
    weekly_prices_df = pd.DataFrame(index=full_friday_grid)
    missing_weeks_by_asset: Dict[str, int] = {}
    coverage_by_asset: Dict[str, float] = {}

    for col, s in weekly_price_cols.items():
        reindexed = s.reindex(full_friday_grid)
        weekly_prices_df[col] = reindexed
        n_missing = int(reindexed.isna().sum())
        missing_weeks_by_asset[col] = n_missing
        coverage_by_asset[col] = float(len(full_friday_grid) - n_missing) / float(len(full_friday_grid))

    raw_weekly_returns = (weekly_prices_df / weekly_prices_df.shift(1)) - 1.0
    raw_weekly_returns = raw_weekly_returns.iloc[1:]

    low_coverage_assets = {
        col: cov for col, cov in coverage_by_asset.items() if cov < cfg.min_asset_coverage
    }
    if low_coverage_assets:
        raise InsufficientHistoryError(
            f"Asset(s) below minimum weekly coverage ({cfg.min_asset_coverage:.1%}): "
            + ", ".join(f"{k}={v:.1%}" for k, v in low_coverage_assets.items()),
            details={
                "low_coverage_assets": low_coverage_assets,
                "min_asset_coverage": cfg.min_asset_coverage,
                "missing_weeks_by_asset": missing_weeks_by_asset,
            },
        )

    aligned_returns = raw_weekly_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    excluded_dates = int(len(raw_weekly_returns) - len(aligned_returns))

    if excluded_dates > 0:
        warnings_list.append(
            f"Excluded {excluded_dates} week(s) from aligned weekly return matrix due to missing asset observations or post-gap transitions."
        )

    min_required = cfg.min_weekly_observations
    if len(aligned_returns) < min_required:
        raise InsufficientHistoryError(
            f"Insufficient aligned weekly returns ({len(aligned_returns)} weeks < required {min_required}).",
            details={
                "aligned_weeks": len(aligned_returns),
                "min_weekly_observations": min_required,
                "missing_weeks_by_asset": missing_weeks_by_asset,
                "excluded_dates_count": excluded_dates,
                "excluded_incomplete_week_bins": excluded_incomplete_bins,
                "last_closed_friday": last_closed_friday.date().isoformat(),
            },
        )

    n_assets = aligned_returns.shape[1]
    if len(aligned_returns) < max(2 * n_assets, 52):
        warnings_list.append(
            f"Aligned sample ({len(aligned_returns)} weeks for {n_assets} assets) is relatively short; shrinkage regularization is advisable."
        )

    return WeeklyReturnsBundle(
        weekly_returns=aligned_returns,
        weekly_prices_usd=weekly_prices_df,
        start_date=aligned_returns.index.min(),
        end_date=aligned_returns.index.max(),
        total_grid_weeks=len(full_friday_grid),
        aligned_weeks=len(aligned_returns),
        excluded_dates_count=excluded_dates,
        last_closed_friday=last_closed_friday,
        excluded_incomplete_week_bins=excluded_incomplete_bins,
        missing_weeks_by_asset=missing_weeks_by_asset,
        coverage_by_asset=coverage_by_asset,
        duplicate_dates_removed=dup_removed,
        warnings=warnings_list,
    )


def fetch_vix_history(
    period: str = "10y",
    *,
    as_of_cutoff: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """Download historical daily ^VIX close observations sorted chronologically ascending."""
    vix = yf.Ticker("^VIX")
    hist = vix.history(period=period, auto_adjust=False)
    if hist is None or hist.empty:
        return pd.Series(dtype=float, name="^VIX")
    raw = _extract_close_series(hist)
    san = sanitize_price_series(
        raw,
        asset_id="^VIX",
        duplicate_date_policy="last",
        as_of_cutoff=as_of_cutoff,
        require_positive=True,
    )
    return san.series.rename("^VIX")


def get_current_vix_observation(
    *,
    vix_history: Optional[pd.Series] = None,
    as_of_date: Optional[date] = None,
    max_staleness_days: int = 7,
) -> Tuple[Optional[float], Optional[pd.Timestamp], bool]:
    """Return ``(vix_value, observation_timestamp, is_stale)`` using strictly causal data."""
    cutoff = pd.Timestamp(as_of_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) if as_of_date else None
    series = vix_history
    if series is None:
        try:
            series = fetch_vix_history(period="1mo", as_of_cutoff=cutoff)
        except Exception:
            return None, None, True
    else:
        san = sanitize_price_series(series, asset_id="^VIX", as_of_cutoff=cutoff, require_positive=True)
        series = san.series

    if series is None or series.empty:
        return None, None, True

    obs_ts = pd.Timestamp(series.index[-1])
    obs_val = float(series.iloc[-1])
    if not math.isfinite(obs_val) or obs_val <= 0.0:
        return None, obs_ts, True

    ref_date = pd.Timestamp(as_of_date) if as_of_date is not None else pd.Timestamp.now("UTC").tz_localize(None).normalize()
    age_days = (ref_date.normalize() - obs_ts.normalize()).days
    is_stale = age_days > int(max_staleness_days)
    return obs_val, obs_ts, is_stale


def get_current_vix() -> Optional[float]:
    """Convenience helper returning the latest valid non-stale ^VIX level or ``None``."""
    val, _, is_stale = get_current_vix_observation()
    return None if is_stale else val


def get_risk_free_rate(default: float = 0.04) -> float:
    """Fetch annual USD risk-free rate from ^TNX, falling back to FRED DGS10 or default."""
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        if hist is not None and not hist.empty:
            close = _extract_close_series(hist).dropna()
            if not close.empty:
                rate = float(close.iloc[-1]) / 100.0
                if 0.0 <= rate < 0.25:
                    return rate
    except Exception:
        pass

    fred_api_key = get_fred_api_key()
    if fred_api_key:
        try:
            resp = requests.get(
                f"{FRED_BASE_URL}/series/observations",
                params={
                    "series_id": "DGS10",
                    "api_key": fred_api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 5,
                },
                timeout=10,
                verify=True,
            )
            if resp.status_code == 200:
                data = resp.json()
                for obs in data.get("observations", []):
                    val = obs.get("value", ".")
                    if val != ".":
                        rate = float(val) / 100.0
                        if 0.0 <= rate < 0.25:
                            return rate
        except Exception as exc:
            _ = _redact_secret(str(exc), fred_api_key)

    return float(default)


def obtener_tasa_libre_riesgo(default: float = 0.04) -> float:
    """Spanish-named wrapper returning the annual USD risk-free rate."""
    return get_risk_free_rate(default=default)


def descargar_precios_historicos(
    tickers: List[str],
    periodo: str = "10y",
    *,
    fx_max_staleness_days: int = 5,
    min_fx_coverage: float = 0.95,
    currency_overrides: Optional[Dict[str, Optional[str]]] = None,
    as_of_cutoff: Optional[pd.Timestamp] = None,
    fx_fetcher: Optional[Callable[[str, str], pd.Series]] = None,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """Download historical USD-normalized adjusted prices (blocks on FX/data errors by default)."""
    prices, _ = _download_prices(
        tickers,
        period=periodo,
        fx_max_staleness_days=fx_max_staleness_days,
        min_fx_coverage=min_fx_coverage,
        currency_overrides=currency_overrides,
        as_of_cutoff=as_of_cutoff,
        fx_fetcher=fx_fetcher,
        allow_partial=allow_partial,
    )
    return prices


def metricas_portfolio(
    weights: Optional[np.ndarray],
    mu: Optional[pd.Series],
    cov: Optional[pd.DataFrame],
    rf: float,
) -> Tuple[float, float, float]:
    """Compute annual expected return, annual volatility, and Sharpe ratio."""
    if mu is None or cov is None or weights is None:
        return float("nan"), float("nan"), float("nan")
    w = np.asarray(weights, dtype=float)
    ret = float(w @ mu.to_numpy(dtype=float))
    var = float(w @ cov.to_numpy(dtype=float) @ w)
    vol = float(np.sqrt(max(var, 0.0)))
    sharpe = (ret - float(rf)) / vol if vol > 1e-12 else float("nan")
    return ret, vol, sharpe


def calcular_cvar_historico(
    weights: np.ndarray,
    period_returns: pd.DataFrame,
    alpha: float = 0.95,
) -> float:
    """Compute historical Conditional Value-at-Risk (tail mean return) at confidence ``alpha``."""
    arr = period_returns.to_numpy(dtype=float) @ np.asarray(weights, dtype=float)
    if arr.size == 0:
        return float("nan")
    threshold = float(np.percentile(arr, (1.0 - alpha) * 100.0))
    tail = arr[arr <= threshold]
    return float(tail.mean()) if tail.size > 0 else threshold
