"""Tests for annualized DCF, multi-asset Black-Litterman, and VIX dynamic regime covariance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization import (
    BlackLittermanConfig,
    ConfigurationError,
    DCFViewConfig,
    InstrumentConfig,
    InsufficientRegimeSampleError,
    MissingVixDataError,
    RiskModelConfig,
    VixRegimeConfig,
    annualize_dcf_view,
    compute_black_litterman,
    compute_expected_returns_pipeline,
    compute_vix_stress_weight,
    compute_weekly_returns_w_fri,
    ensure_psd_covariance,
    estimate_risk_model,
    optimize_erc,
    optimize_hrp,
    optimize_min_variance,
    validate_constraints_feasibility,
)


def test_dcf_annualization_oracles_and_valuation_bases():
    v_target = DCFViewConfig(
        valuation_basis="terminal_target",
        target_price=150.0,
        current_price=100.0,
        horizon_years=5.0,
        confidence=0.60,
    )
    res = annualize_dcf_view("AAPL", v_target)
    expected = (150.0 / 100.0) ** (1.0 / 5.0) - 1.0
    assert res.annualized_expected_return == pytest.approx(expected, rel=1e-12)
    assert res.annualized_expected_return == pytest.approx(0.0844717712, abs=1e-6)
    assert res.active_in_bl is True

    v_iv = DCFViewConfig(
        valuation_basis="present_intrinsic_value",
        present_intrinsic_value=130.0,
        current_price=100.0,
        horizon_years=3.0,
        expected_dividend_yield=0.02,
        confidence=0.40,
    )
    res_iv = annualize_dcf_view("MSFT", v_iv)
    assert res_iv.annualized_expected_return == pytest.approx((1.30 ** (1.0 / 3.0) - 1.0) + 0.02, rel=1e-12)
    assert any("convergence hypothesis" in a.lower() for a in res_iv.assumptions)

    v_cf = DCFViewConfig(
        valuation_basis="dated_cashflows",
        current_price=100.0,
        target_price=100.0,
        horizon_years=2.0,
        distributions=((1.0, 10.0), (2.0, 10.0)),
        confidence=0.50,
    )
    res_cf = annualize_dcf_view("DIVCO", v_cf)
    assert res_cf.annualized_expected_return == pytest.approx(0.10, abs=1e-6)

    v_zero = DCFViewConfig(valuation_basis="terminal_target", total_upside=0.50, horizon_years=5.0, confidence=0.0)
    res_zero = annualize_dcf_view("ZERO", v_zero)
    assert res_zero.active_in_bl is False


def test_black_litterman_confidence_boundaries_partial_views_and_rf_consistency():
    tickers = ["EQ1", "EQ2", "GLD", "BTC"]
    cov = pd.DataFrame(
        [
            [0.040, 0.024, 0.002, 0.015],
            [0.024, 0.040, 0.002, 0.012],
            [0.002, 0.002, 0.015, 0.001],
            [0.015, 0.012, 0.001, 0.360],
        ],
        index=tickers,
        columns=tickers,
    )
    w_ref = pd.Series([0.35, 0.35, 0.20, 0.10], index=tickers)
    rf = 0.04

    bl_none = compute_black_litterman(cov, w_ref, rf_rate=rf, absolute_views_total={})
    assert bl_none.diagnostics["status"] == "no_active_views_prior_returned"
    np.testing.assert_allclose(bl_none.posterior_total_returns.values, bl_none.prior_total_returns.values, atol=1e-12)

    bl_zero = compute_black_litterman(cov, w_ref, rf_rate=rf, absolute_views_total={"EQ1": (0.20, 0.0)})
    assert "EQ1" in bl_zero.inactive_view_assets
    assert bl_zero.active_view_assets == []
    np.testing.assert_allclose(bl_zero.posterior_total_returns.values, bl_zero.prior_total_returns.values, atol=1e-12)

    bl_one = compute_black_litterman(cov, w_ref, rf_rate=rf, absolute_views_total={"EQ1": (0.14, 1.0)})
    assert np.all(np.isfinite(bl_one.posterior_total_returns.values))
    assert bl_one.posterior_total_returns["EQ1"] == pytest.approx(0.14, abs=1e-10)
    assert bl_one.posterior_total_returns["EQ2"] > bl_one.prior_total_returns["EQ2"]

    bl_half = compute_black_litterman(cov, w_ref, rf_rate=rf, absolute_views_total={"EQ1": (0.14, 0.5)})
    assert bl_half.prior_total_returns["EQ1"] < bl_half.posterior_total_returns["EQ1"] < 0.14
    assert all("Black-Litterman" in src for src in bl_half.mu_sources.values())

    bl_rf2 = compute_black_litterman(cov, w_ref, rf_rate=rf + 0.02, absolute_views_total={"EQ1": (0.16, 0.5)})
    np.testing.assert_allclose(
        bl_rf2.posterior_total_returns.values - bl_half.posterior_total_returns.values,
        0.02,
        atol=1e-12,
    )

    inst_map = {
        "EQ1": InstrumentConfig(instrument_id="EQ1", ticker="EQ1", asset_subtype="equity"),
        "EQ2": InstrumentConfig(instrument_id="EQ2", ticker="EQ2", asset_subtype="equity"),
        "GLD": InstrumentConfig(instrument_id="GLD", ticker="GLD", asset_subtype="commodity_gold"),
        "BTC": InstrumentConfig(instrument_id="BTC", ticker="BTC", asset_subtype="crypto"),
    }
    with pytest.raises(ConfigurationError, match="Explicit prior reference weights"):
        compute_expected_returns_pipeline(
            tickers=tickers,
            cov_matrix=cov,
            rf_rate=rf,
            bl_config=BlackLittermanConfig(require_explicit_reference_weights=True),
            instruments_by_id=inst_map,
        )
    res_multi = compute_expected_returns_pipeline(
        tickers=tickers,
        cov_matrix=cov,
        rf_rate=rf,
        bl_config=BlackLittermanConfig(prior_weights_policy="explicit_or_warned_equal_weight"),
        instruments_by_id=inst_map,
    )
    assert res_multi.prior_policy_used == "equal_weight_fallback"
    assert any("without fabricating market caps" in w for w in res_multi.warnings)


def test_vix_stress_weight_schedule_boundaries_and_custom_config():
    expected_cases = [
        (10.0, 0.15),
        (14.99, 0.15),
        (15.0, 0.15),
        (17.5, 0.25),
        (20.0, 0.35),
        (25.0, 0.35),
        (30.0, 0.35),
        (30.01, 0.60),
        (45.0, 0.60),
    ]
    for vix, exp_p in expected_cases:
        p, _ = compute_vix_stress_weight(vix)
        assert p == pytest.approx(exp_p, abs=1e-12), f"Failed for VIX={vix}"

    custom = VixRegimeConfig(vix_low=12.0, vix_mid_low=22.0, vix_high=32.0, p_low=0.10, p_mid=0.40, p_high=0.70)
    assert compute_vix_stress_weight(17.0, custom)[0] == pytest.approx(0.25)
    assert compute_vix_stress_weight(32.0, custom)[0] == pytest.approx(0.40)
    assert compute_vix_stress_weight(32.5, custom)[0] == pytest.approx(0.70)


def test_vix_regime_covariance_distinct_calm_crisis_alters_weights_and_handles_edge_cases():
    rng = np.random.default_rng(2026)
    n_calm = 100
    n_crisis = 50
    n_total = n_calm + n_crisis
    weeks = pd.date_range("2021-01-01", periods=n_total + 1, freq="W-FRI")

    cov_calm_true = np.array([
        [0.010**2, 0.00005, 0.00002],
        [0.00005,  0.040**2, 0.00010],
        [0.00002,  0.00010, 0.020**2],
    ])
    cov_crisis_true = np.array([
        [0.060**2, 0.0002,  0.0022],
        [0.0002,   0.015**2, 0.0001],
        [0.0022,   0.0001,  0.050**2],
    ])

    rets_calm = rng.multivariate_normal(np.zeros(3), cov_calm_true, size=n_calm)
    rets_crisis = rng.multivariate_normal(np.zeros(3), cov_crisis_true, size=n_crisis)
    rets_all = np.vstack([rets_calm, rets_crisis])
    prices = pd.DataFrame(
        100.0 * np.vstack([np.ones((1, 3)), np.cumprod(1.0 + rets_all, axis=0)]),
        index=weeks,
        columns=["A", "B", "C"],
    )
    vix_vals = np.concatenate([[14.0], np.full(n_calm, 14.0), np.full(n_crisis, 35.0)])
    vix_hist = pd.Series(vix_vals, index=weeks, name="^VIX")

    bundle = compute_weekly_returns_w_fri(prices, risk_config=RiskModelConfig(min_weekly_observations=52))

    risk_low = estimate_risk_model(
        bundle,
        risk_config=RiskModelConfig(covariance_estimator="vix_regime", shrinkage_method="sample"),
        vix_history=vix_hist,
        current_vix=12.0,
        current_vix_date=weeks[-1],
        current_vix_is_stale=False,
    )
    risk_high = estimate_risk_model(
        bundle,
        risk_config=RiskModelConfig(covariance_estimator="vix_regime", shrinkage_method="sample"),
        vix_history=vix_hist,
        current_vix=35.0,
        current_vix_date=weeks[-1],
        current_vix_is_stale=False,
    )
    assert risk_low.stress_weight_p == pytest.approx(0.15)
    assert risk_high.stress_weight_p == pytest.approx(0.60)
    assert risk_low.calm_samples == n_calm and risk_low.crisis_samples == n_crisis

    np.testing.assert_allclose(
        risk_high.cov_matrix.values,
        0.40 * risk_high.cov_calm.values + 0.60 * risk_high.cov_crisis.values,
        rtol=1e-10,
        atol=1e-10,
    )

    resolved = validate_constraints_feasibility(["A", "B", "C"])
    mv_low = optimize_min_variance(risk_low.cov_matrix, resolved)
    mv_high = optimize_min_variance(risk_high.cov_matrix, resolved)
    erc_low = optimize_erc(risk_low.cov_matrix, resolved)
    erc_high = optimize_erc(risk_high.cov_matrix, resolved)
    hrp_low = optimize_hrp(risk_low.cov_matrix, resolved)
    hrp_high = optimize_hrp(risk_high.cov_matrix, resolved)

    assert mv_high.weights_by_asset["B"] > mv_low.weights_by_asset["B"] + 0.10
    assert erc_high.weights_by_asset["B"] > erc_low.weights_by_asset["B"] + 0.10
    assert hrp_high.weights_by_asset["B"] > hrp_low.weights_by_asset["B"] + 0.10

    with pytest.raises(MissingVixDataError):
        estimate_risk_model(bundle, risk_config=RiskModelConfig(covariance_estimator="vix_regime"), vix_history=vix_hist, current_vix=None)

    risk_manual = estimate_risk_model(
        bundle,
        risk_config=RiskModelConfig(covariance_estimator="vix_regime", vix_regime=VixRegimeConfig(manual_stress_weight=0.45)),
        vix_history=vix_hist,
        current_vix=None,
    )
    assert risk_manual.stress_weight_p == pytest.approx(0.45)
    assert risk_manual.p_source == "manual_override"

    calm_only_vix = pd.Series(np.full(len(weeks), 14.0), index=weeks)
    with pytest.raises(InsufficientRegimeSampleError):
        estimate_risk_model(
            bundle,
            risk_config=RiskModelConfig(covariance_estimator="vix_regime"),
            vix_history=calm_only_vix,
            current_vix=18.0,
            current_vix_date=weeks[-1],
        )

    risk_ewma = estimate_risk_model(
        bundle,
        risk_config=RiskModelConfig(covariance_estimator="ewma", ewma_half_life_weeks=26.0),
    )
    assert risk_ewma.estimator_name.startswith("ewma")
    assert np.all(np.linalg.eigvalsh(risk_ewma.cov_matrix.values) > 0)

    indef = pd.DataFrame([[0.04, 0.05], [0.05, 0.04]], index=["X", "Y"], columns=["X", "Y"])
    repaired, adj, min_b, min_a = ensure_psd_covariance(indef)
    assert adj is True
    assert min_b < 0.0
    assert min_a > 0.0
    assert np.allclose(repaired.values, repaired.values.T)
