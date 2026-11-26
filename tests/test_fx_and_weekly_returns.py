"""Pandas 3.x causal FX conversion and W-FRI weekly return alignment tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization import (
    DataQualityError,
    FXConversionError,
    InsufficientHistoryError,
    RiskModelConfig,
    compute_weekly_returns_w_fri,
    convert_price_series_to_usd,
    sanitize_price_series,
)


def test_running_on_pandas_3x():
    major = int(pd.__version__.split(".")[0])
    assert major >= 3, f"Expected pandas 3.x test runtime, got {pd.__version__}"


def test_fx_conversion_direct_inverse_and_subunit_gbp_with_unsorted_inputs():
    """Verify exact FX direction (ARS divide, EUR multiply, GBp /100 * GBPUSD) on shuffled dates."""
    dates = pd.date_range("2025-01-06", periods=6, freq="B")

    perm = [4, 0, 5, 2, 1, 3]
    ars_prices = pd.Series([1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1250.0], index=dates).iloc[perm]
    ars_fx = pd.Series([1000.0, 1000.0, 1100.0, 1150.0, 1200.0, 1250.0], index=dates).iloc[::-1]

    usd_ars, diag_ars = convert_price_series_to_usd("GGAL.BA", ars_prices, "ARS", ars_fx)
    assert diag_ars.direction == "divide"
    assert diag_ars.unit_scale == 1.0
    assert usd_ars.index.is_monotonic_increasing
    np.testing.assert_allclose(usd_ars.values, [1.0, 1.05, 1.0, 1.0, 1.0, 1.0], rtol=1e-12)

    eur_prices = pd.Series([50.0, 55.0, 60.0, 65.0, 70.0, 75.0], index=dates)
    eur_fx = pd.Series([1.10, 1.10, 1.20, 1.20, 1.10, 1.10], index=dates)
    usd_eur, diag_eur = convert_price_series_to_usd("SAP.DE", eur_prices, "EUR", eur_fx)
    assert diag_eur.direction == "multiply"
    np.testing.assert_allclose(usd_eur.values, [55.0, 60.5, 72.0, 78.0, 77.0, 82.5], rtol=1e-12)

    gbp_pence = pd.Series([2500.0, 2600.0, 2700.0, 2800.0, 2900.0, 3000.0], index=dates)
    gbp_fx = pd.Series([1.25] * 6, index=dates)
    usd_gbp, diag_gbp = convert_price_series_to_usd("AZN.L", gbp_pence, "GBp", gbp_fx)
    assert diag_gbp.unit_scale == 0.01
    assert diag_gbp.direction == "multiply"
    np.testing.assert_allclose(usd_gbp.values, np.array([25.0, 26.0, 27.0, 28.0, 29.0, 30.0]) * 1.25, rtol=1e-12)


def test_fx_conversion_blocks_future_bfill_stale_fx_and_missing_fx():
    """No future FX leakage (no bfill), no stale carry past tolerance, no raw LCY fallback."""
    dates = pd.date_range("2025-01-06", periods=10, freq="B")
    prices = pd.Series(np.full(10, 1000.0), index=dates)

    late_fx = pd.Series(np.full(5, 1000.0), index=dates[5:])
    with pytest.raises(FXConversionError, match="Insufficient causal FX coverage"):
        convert_price_series_to_usd("LOCAL_ARS", prices, "ARS", late_fx, min_fx_coverage=0.90)

    usd_partial, diag = convert_price_series_to_usd("LOCAL_ARS", prices, "ARS", late_fx, min_fx_coverage=0.50)
    assert len(usd_partial) == 5
    assert usd_partial.index.min() == dates[5]
    assert diag.dropped_stale_or_missing_fx == 5

    sparse_fx = pd.Series([1000.0, 1050.0], index=[dates[0], dates[-1]])
    with pytest.raises(FXConversionError, match="Insufficient causal FX coverage"):
        convert_price_series_to_usd("STALE_ARS", prices, "ARS", sparse_fx, fx_max_staleness_days=3, min_fx_coverage=0.90)

    with pytest.raises(FXConversionError, match="Missing historical FX"):
        convert_price_series_to_usd("NO_FX", prices, "ARS", None)

    with pytest.raises(FXConversionError, match="Unsupported currency"):
        convert_price_series_to_usd("EXOTIC", prices, "ZZZ", late_fx)


def test_sanitize_price_series_duplicate_date_policies_and_descending_input():
    idx = pd.to_datetime(["2025-01-03", "2025-01-02", "2025-01-02", "2025-01-01"])
    s = pd.Series([103.0, 102.5, 102.0, 101.0], index=idx)

    san = sanitize_price_series(s, asset_id="TEST", duplicate_date_policy="last")
    assert san.was_reordered is True
    assert san.duplicates_removed == 1
    assert san.series.index.is_monotonic_increasing
    assert list(san.series.index) == list(pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]))

    with pytest.raises(DataQualityError, match="Duplicate timestamps"):
        sanitize_price_series(s, asset_id="TEST", duplicate_date_policy="error")


def test_weekly_w_fri_holidays_24x7_crypto_and_no_multiweek_gap_returns():
    start = pd.Timestamp("2025-01-06")
    all_days = pd.date_range(start, periods=140, freq="D")
    bdays = pd.bdate_range(start, periods=100)

    eq_dates = bdays[
        (bdays != pd.Timestamp("2025-02-14"))
        & ~((bdays >= pd.Timestamp("2025-03-03")) & (bdays <= pd.Timestamp("2025-03-07")))
    ]
    eq_series = pd.Series(np.linspace(100.0, 130.0, len(eq_dates)), index=eq_dates, name="EQ")
    btc_series = pd.Series(np.linspace(50000.0, 65000.0, len(all_days)), index=all_days, name="BTC")

    df = pd.DataFrame({"EQ": eq_series, "BTC": btc_series})
    df_desc = df.iloc[::-1].copy()

    cfg = RiskModelConfig(min_weekly_observations=10, min_asset_coverage=0.80)
    bundle_asc = compute_weekly_returns_w_fri(df, risk_config=cfg)
    bundle_desc = compute_weekly_returns_w_fri(df_desc, risk_config=cfg)

    pd.testing.assert_frame_equal(bundle_asc.weekly_returns, bundle_desc.weekly_returns)

    assert pd.Timestamp("2025-02-14") in bundle_asc.weekly_returns.index
    assert bundle_asc.weekly_prices_usd.loc[pd.Timestamp("2025-02-14"), "EQ"] == pytest.approx(
        eq_series.loc[pd.Timestamp("2025-02-13")]
    )
    assert bundle_asc.weekly_returns.loc[pd.Timestamp("2025-02-14"), "EQ"] != 0.0

    assert pd.Timestamp("2025-03-07") not in bundle_asc.weekly_returns.index
    assert pd.Timestamp("2025-03-14") not in bundle_asc.weekly_returns.index
    assert bundle_asc.missing_weeks_by_asset["EQ"] == 1
    assert bundle_asc.excluded_dates_count >= 2

    tz_idx = pd.DatetimeIndex([
        pd.Timestamp("2025-01-10 16:00:00", tz="America/New_York"),
        pd.Timestamp("2025-01-10 19:00:00", tz="America/New_York"),
        pd.Timestamp("2025-01-17 16:00:00", tz="America/New_York"),
        pd.Timestamp("2025-01-24 16:00:00", tz="America/New_York"),
        pd.Timestamp("2025-01-31 16:00:00", tz="America/New_York"),
        pd.Timestamp("2025-02-07 16:00:00", tz="America/New_York"),
    ])
    crypto_intraday = pd.DataFrame({"BTC": [100.0, 999.0, 110.0, 121.0, 133.1, 146.41]}, index=tz_idx)
    b_intra = compute_weekly_returns_w_fri(
        crypto_intraday,
        risk_config=RiskModelConfig(min_weekly_observations=4, weekly_cutoff_hour=17),
        as_of_cutoff=pd.Timestamp("2025-02-07 18:00:00", tz="America/New_York"),
    )
    assert b_intra.weekly_prices_usd.loc[pd.Timestamp("2025-01-10"), "BTC"] == pytest.approx(100.0)
    assert b_intra.weekly_returns.loc[pd.Timestamp("2025-01-17"), "BTC"] == pytest.approx(0.10)


def test_weekly_returns_rejects_insufficient_sample_size():
    dates = pd.date_range("2025-01-03", periods=10, freq="W-FRI")
    df = pd.DataFrame({"A": np.linspace(100, 110, 10), "B": np.linspace(50, 55, 10)}, index=dates)
    with pytest.raises(InsufficientHistoryError, match="Insufficient aligned weekly returns"):
        compute_weekly_returns_w_fri(df, risk_config=RiskModelConfig(min_weekly_observations=52))
