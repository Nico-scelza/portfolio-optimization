"""Tests for constraints, LP feasibility, all 5 optimizers (1..22+ assets),
ERC/HRP oracles, multi-broker fractional rebalancing, and pipeline consistency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization import (
    BlackLittermanConfig,
    BrokerConfig,
    BrokerPosition,
    ConstraintsConfig,
    DCFViewConfig,
    InfeasibleConstraintsError,
    InstrumentConfig,
    OptimizationConfig,
    PortfolioAnalysisConfig,
    RebalancingConfig,
    RiskModelConfig,
    analizar_contribucion_marginal_riesgo,
    analizar_rendimiento_historico,
    analyze_portfolio,
    armar_cartera,
    calculate_broker_trade_cost_usd,
    compute_pure_hrp_weights,
    compute_rebalancing_plan,
    ejecutar_analisis_completo,
    generar_frontera_eficiente,
    generar_grafico_riesgo,
    generar_heatmap_correlaciones,
    generar_underwater_plot,
    optimize_erc,
    optimize_hrp,
    optimize_max_sharpe,
    optimize_min_cvar,
    optimize_min_variance,
    run_portfolio_analysis,
    validate_constraints_feasibility,
    verify_solution_feasibility,
)


def _make_weekly_universe(n_assets: int, n_weeks: int = 120, seed: int = 17):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-07", periods=n_weeks + 1, freq="W-FRI")
    tickers = [f"A{i:02d}" for i in range(n_assets)]
    factor = rng.normal(0.001, 0.012, size=(n_weeks, 1))
    idio_vols = np.linspace(0.008, 0.032, n_assets)
    rets = 0.0015 + 0.5 * factor + rng.normal(0.0, 1.0, size=(n_weeks, n_assets)) * idio_vols
    prices = pd.DataFrame(
        100.0 * np.vstack([np.ones((1, n_assets)), np.cumprod(1.0 + rets, axis=0)]),
        index=dates,
        columns=tickers,
    )
    weekly_rets = (prices / prices.shift(1) - 1.0).iloc[1:]
    cov = pd.DataFrame(np.cov(weekly_rets.values, rowvar=False) * 52.0 if n_assets > 1 else [[np.var(weekly_rets.values, ddof=1) * 52.0]], index=tickers, columns=tickers)
    mu = pd.Series(np.linspace(0.05, 0.14, n_assets), index=tickers)
    return tickers, prices, weekly_rets, cov, mu


@pytest.mark.parametrize("n_assets", [1, 4, 5, 20, 22])
def test_all_five_optimizers_succeed_for_1_4_5_20_22_assets_without_fixed_bounds(n_assets: int):
    tickers, _, weekly_rets, cov, mu = _make_weekly_universe(n_assets, n_weeks=120, seed=100 + n_assets)
    resolved = validate_constraints_feasibility(tickers, ConstraintsConfig())

    res_sh = optimize_max_sharpe(mu, cov, 0.04, resolved, weekly_returns=weekly_rets)
    res_mv = optimize_min_variance(cov, resolved, mu=None, weekly_returns=weekly_rets)
    res_cv = optimize_min_cvar(weekly_rets, cov, resolved, mu=None)
    res_erc = optimize_erc(cov, resolved, mu=None, weekly_returns=weekly_rets)
    res_hrp = optimize_hrp(cov, resolved, mu=None, weekly_returns=weekly_rets)

    for res in (res_sh, res_mv, res_cv, res_erc, res_hrp):
        assert res.converged is True
        assert res.feasible is True
        ok, violations = verify_solution_feasibility(res.weights, resolved)
        assert ok, f"{res.strategy_name} (n={n_assets}) violated constraints: {violations}"
        assert res.weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(res.weights >= -1e-6)
        assert res.risk_contributions_rel.sum() == pytest.approx(1.0, abs=1e-5)

    assert res_erc.expected_return is None and res_hrp.expected_return is None


def test_joint_lp_infeasibility_and_class_constraints_across_all_optimizers():
    tickers = ["EQ1", "EQ2", "BND1", "GLD1", "BTC1"]
    rc_map = {"EQ1": "equity", "EQ2": "equity", "BND1": "fixed_income", "GLD1": "gold_commodities", "BTC1": "crypto"}

    infeasible_lp = ConstraintsConfig(
        asset_bounds={t: (0.0, 0.30) for t in tickers},
        class_bounds={
            "equity": (0.0, 0.40),
            "fixed_income": (0.0, 0.25),
            "gold_commodities": (0.0, 0.15),
            "crypto": (0.0, 0.10),
        },
    )
    with pytest.raises(InfeasibleConstraintsError):
        validate_constraints_feasibility(tickers, infeasible_lp, risk_class_by_ticker=rc_map)

    with pytest.raises(InfeasibleConstraintsError, match="zero instruments"):
        validate_constraints_feasibility(
            ["EQ1", "EQ2"],
            ConstraintsConfig(class_bounds={"crypto": (0.05, 0.20)}),
            risk_class_by_ticker={"EQ1": "equity", "EQ2": "equity"},
        )

    with pytest.raises(InfeasibleConstraintsError, match="HHI"):
        validate_constraints_feasibility(["EQ1", "EQ2"], ConstraintsConfig(hhi_limit=0.40))

    _, _, weekly_rets, cov, mu = _make_weekly_universe(5, seed=33)
    cov.index = cov.columns = tickers
    weekly_rets.columns = tickers
    mu.index = tickers
    con = ConstraintsConfig(
        class_bounds={"crypto": (0.0, 0.10), "fixed_income": (0.25, 0.50), "equity": (0.10, 0.50)}
    )
    resolved = validate_constraints_feasibility(tickers, con, risk_class_by_ticker=rc_map)

    for res in (
        optimize_max_sharpe(mu, cov, 0.04, resolved, weekly_returns=weekly_rets),
        optimize_min_variance(cov, resolved, weekly_returns=weekly_rets),
        optimize_min_cvar(weekly_rets, cov, resolved),
        optimize_erc(cov, resolved, weekly_returns=weekly_rets),
        optimize_hrp(cov, resolved, weekly_returns=weekly_rets),
    ):
        assert res.feasible is True
        assert res.class_exposure["crypto"] <= 0.10 + 1e-6
        assert 0.25 - 1e-6 <= res.class_exposure["fixed_income"] <= 0.50 + 1e-6
        assert 0.10 - 1e-6 <= res.class_exposure["equity"] <= 0.50 + 1e-6


def test_erc_exact_diagonal_oracle_and_constrained_approximate_case():
    vols = np.array([0.10, 0.20, 0.40])
    cov = pd.DataFrame(np.diag(vols ** 2), index=["LOW", "MID", "HIGH"], columns=["LOW", "MID", "HIGH"])
    resolved_free = validate_constraints_feasibility(["LOW", "MID", "HIGH"])

    erc_free = optimize_erc(cov, resolved_free)
    expected_w = (1.0 / vols) / np.sum(1.0 / vols)
    assert erc_free.status == "optimal"
    np.testing.assert_allclose(erc_free.weights, expected_w, atol=1e-6)
    np.testing.assert_allclose(erc_free.risk_contributions_rel.values, [1 / 3, 1 / 3, 1 / 3], atol=1e-5)
    assert erc_free.diagnostics["is_constrained_erc"] is False

    resolved_tight = validate_constraints_feasibility(
        ["LOW", "MID", "HIGH"],
        ConstraintsConfig(asset_bounds={"LOW": (0.0, 0.30)}),
    )
    erc_tight = optimize_erc(cov, resolved_tight)
    assert erc_tight.feasible is True
    assert erc_tight.status == "constrained_approximate"
    assert erc_tight.weights_by_asset["LOW"] <= 0.30 + 1e-6
    assert erc_tight.diagnostics["is_constrained_erc"] is True
    assert erc_tight.diagnostics["max_risk_share_deviation"] > 1e-3


def test_hrp_reproducible_no_sigma_inversion_and_pure_vs_adjusted():
    cov_singular = pd.DataFrame(
        [
            [0.04, 0.04, 0.00],
            [0.04, 0.04, 0.00],
            [0.00, 0.00, 0.09],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    w_pure_1, order_1 = compute_pure_hrp_weights(cov_singular, linkage_method="single")
    w_pure_2, order_2 = compute_pure_hrp_weights(cov_singular, linkage_method="single")
    np.testing.assert_allclose(w_pure_1, w_pure_2, atol=1e-15)
    assert order_1 == order_2
    assert w_pure_1.sum() == pytest.approx(1.0)

    resolved_cap = validate_constraints_feasibility(
        ["A", "B", "C"],
        ConstraintsConfig(asset_bounds={"C": (0.50, 0.80)}),
    )
    hrp_adj = optimize_hrp(cov_singular, resolved_cap)
    assert hrp_adj.feasible is True
    assert hrp_adj.status == "adjusted_to_constraints"
    assert hrp_adj.diagnostics["uses_sigma_inversion"] is False
    assert hrp_adj.diagnostics["is_constrained_adjusted"] is True
    assert hrp_adj.diagnostics["l1_adjustment"] > 0.0
    assert 0.50 - 1e-6 <= hrp_adj.weights_by_asset["C"] <= 0.80 + 1e-6
    assert "pure_hrp_weights" in hrp_adj.diagnostics


def test_broker_commission_rules_and_rebalancing_thresholds_and_multibroker():
    b_pct = BrokerConfig(broker_id="B_PCT", fee_structure="percentage", buy_percentage_fee=0.005, sell_percentage_fee=0.006, min_fee=2.0, estimated_spread_bps=10.0)
    comm_small, sp_small = calculate_broker_trade_cost_usd(100.0, "BUY", b_pct)
    assert comm_small == pytest.approx(2.0)
    assert sp_small == pytest.approx(0.10)

    b_incl = BrokerConfig(broker_id="B_INCL", fee_structure="fixed_plus_percentage", fee_currency="EUR", fee_fx_to_usd=1.10, buy_fixed_fee=5.0, buy_percentage_fee=0.002, estimated_spread_bps=20.0, price_includes_spread=True)
    comm_fx, sp_incl = calculate_broker_trade_cost_usd(1000.0, "BUY", b_incl)
    assert comm_fx == pytest.approx(5.0 * 1.10 + 1000.0 * 0.002)
    assert sp_incl == pytest.approx(0.0)

    brokers = {
        "IBKR": BrokerConfig(broker_id="IBKR", fee_structure="fixed", buy_fixed_fee=1.0, sell_fixed_fee=1.0),
        "IOL": BrokerConfig(broker_id="IOL", fee_structure="percentage", buy_percentage_fee=0.005, sell_percentage_fee=0.005),
        "CRYPTO_BRK": BrokerConfig(broker_id="CRYPTO_BRK", fee_structure="percentage", buy_percentage_fee=0.001, sell_percentage_fee=0.001),
    }
    insts = {
        "AAPL": InstrumentConfig(
            instrument_id="AAPL", ticker="AAPL", asset_subtype="equity",
            default_buy_broker_id="IBKR", sell_broker_priority=("IOL", "IBKR"),
        ),
        "GLD": InstrumentConfig(
            instrument_id="GLD", ticker="GLD", asset_subtype="commodity_gold",
            default_buy_broker_id="IBKR",
        ),
        "BTC": InstrumentConfig(
            instrument_id="BTC", ticker="BTC-USD", asset_subtype="crypto",
            default_buy_broker_id="CRYPTO_BRK",
        ),
    }
    prices = {"AAPL": 200.0, "GLD": 100.0, "BTC": 50000.0}
    positions = [
        BrokerPosition(instrument_id="AAPL", broker_id="IOL", quantity=10.0),
        BrokerPosition(instrument_id="AAPL", broker_id="IBKR", quantity=20.0),
        BrokerPosition(instrument_id="GLD", broker_id="IBKR", quantity=35.0),
        BrokerPosition(instrument_id="BTC", broker_id="CRYPTO_BRK", quantity=0.01),
    ]

    # Multi-broker with zero initial cash: IOL sell proceeds ($1,990) stay segregated in IOL;
    # IBKR can ONLY spend its own AAPL sell proceeds ($999) to buy GLD -> partially_funded!
    reb = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["AAPL", "GLD", "BTC"],
        target_weights={"AAPL": 0.30, "GLD": 0.645, "BTC": 0.055},
        prices_usd=prices,
        instruments_by_id=insts,
        brokers_by_id=brokers,
        positions=positions,
        rebalancing_config=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=25.0),
    )
    assert reb.feasible is False
    assert reb.status == "partially_funded"

    omitted_ids = {o.instrument_id for o in reb.omitted_operations}
    assert "BTC" in omitted_ids

    aapl_sells = [o for o in reb.orders if o.instrument_id == "AAPL" and o.side == "SELL"]
    assert len(aapl_sells) == 2
    assert aapl_sells[0].broker_id == "IOL" and aapl_sells[0].quantity == pytest.approx(10.0)
    assert aapl_sells[1].broker_id == "IBKR" and aapl_sells[1].quantity == pytest.approx(5.0)

    gld_buys = [o for o in reb.orders if o.instrument_id == "GLD" and o.side == "BUY"]
    assert len(gld_buys) == 1
    assert gld_buys[0].broker_id == "IBKR"
    assert gld_buys[0].quantity % 1 != 0
    assert gld_buys[0].gross_amount_usd == pytest.approx(998.0, abs=1e-4)
    assert reb.ending_cash_by_broker["IOL"] == pytest.approx(1990.0, abs=1e-6)
    assert reb.ending_cash_by_broker["IBKR"] == pytest.approx(0.0, abs=1e-6)
    assert reb.ending_cash_usd == pytest.approx(1990.0, abs=1e-6)

    assert reb.final_portfolio_value_usd == pytest.approx(
        reb.initial_portfolio_value_usd - reb.total_transaction_cost_usd,
        abs=1e-6,
    )

    # With IBKR $1 fee and $0 cash, selling $50 yields $49 -> scaled buy is $48 < min_notional_usd ($50) -> omitted (AUD-11)
    reb_eq_fee_omits = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["AAPL", "GLD"],
        target_weights={"AAPL": 0.51, "GLD": 0.49},
        prices_usd={"AAPL": 100.0, "GLD": 100.0},
        instruments_by_id={"AAPL": insts["AAPL"], "GLD": insts["GLD"]},
        brokers_by_id={"IBKR": brokers["IBKR"]},
        positions=[
            BrokerPosition(instrument_id="AAPL", broker_id="IBKR", quantity=25.0),
            BrokerPosition(instrument_id="GLD", broker_id="IBKR", quantity=25.0),
        ],
        rebalancing_config=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=50.0, threshold_comparison=">="),
    )
    assert len(reb_eq_fee_omits.orders) == 1
    assert any(o.instrument_id == "AAPL" and "scaled_notional" in ";".join(o.reasons) for o in reb_eq_fee_omits.omitted_operations)

    # Exact boundary >= vs > tested with zero-fee broker (exact $50.00 notional & 1.00 pp weight diff)
    zero_broker = BrokerConfig(broker_id="ZERO", fee_structure="percentage", buy_percentage_fee=0.0, sell_percentage_fee=0.0)
    inst_zero = {
        "AAPL": InstrumentConfig(instrument_id="AAPL", ticker="AAPL", asset_subtype="equity", default_buy_broker_id="ZERO"),
        "GLD": InstrumentConfig(instrument_id="GLD", ticker="GLD", asset_subtype="commodity_gold", default_buy_broker_id="ZERO"),
    }
    reb_eq = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["AAPL", "GLD"],
        target_weights={"AAPL": 0.51, "GLD": 0.49},
        prices_usd={"AAPL": 100.0, "GLD": 100.0},
        instruments_by_id=inst_zero,
        brokers_by_id={"ZERO": zero_broker},
        positions=[
            BrokerPosition(instrument_id="AAPL", broker_id="ZERO", quantity=25.0),
            BrokerPosition(instrument_id="GLD", broker_id="ZERO", quantity=25.0),
        ],
        rebalancing_config=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=50.0, threshold_comparison=">="),
    )
    assert len(reb_eq.orders) == 2

    reb_gt = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["AAPL", "GLD"],
        target_weights={"AAPL": 0.51, "GLD": 0.49},
        prices_usd={"AAPL": 100.0, "GLD": 100.0},
        instruments_by_id=inst_zero,
        brokers_by_id={"ZERO": zero_broker},
        positions=[
            BrokerPosition(instrument_id="AAPL", broker_id="ZERO", quantity=25.0),
            BrokerPosition(instrument_id="GLD", broker_id="ZERO", quantity=25.0),
        ],
        rebalancing_config=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=50.0, threshold_comparison=">"),
    )
    assert len(reb_gt.orders) == 0

    inst_ambig = dict(insts)
    inst_ambig["AAPL"] = InstrumentConfig(instrument_id="AAPL", ticker="AAPL", default_buy_broker_id="IBKR")
    reb_ambig = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["AAPL", "GLD", "BTC"],
        target_weights={"AAPL": 0.30, "GLD": 0.65, "BTC": 0.05},
        prices_usd=prices,
        instruments_by_id=inst_ambig,
        brokers_by_id=brokers,
        positions=positions,
    )
    assert reb_ambig.status == "broker_routing_required"
    assert reb_ambig.feasible is False

    reb_conflict = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["GLD", "BTC"],
        target_weights={"GLD": 0.90, "BTC": 0.10},
        prices_usd={"GLD": 100.0, "BTC": 100.0},
        instruments_by_id={"GLD": insts["GLD"], "BTC": insts["BTC"]},
        brokers_by_id=brokers,
        positions=[
            BrokerPosition(instrument_id="GLD", broker_id="IBKR", quantity=89.5),
            BrokerPosition(instrument_id="BTC", broker_id="CRYPTO_BRK", quantity=10.5),
        ],
        rebalancing_config=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=25.0),
        constraints_config=ConstraintsConfig(class_bounds={"crypto": (0.0, 0.10)}),
    )
    assert reb_conflict.feasible is False
    assert reb_conflict.status == "infeasible_hard_limits"
    assert len(reb_conflict.hard_limit_conflicts) >= 1

    reb_empty = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["GLD", "BTC"],
        target_weights={"GLD": 0.90, "BTC": 0.10},
        prices_usd={"GLD": 100.0, "BTC": 100.0},
        instruments_by_id={"GLD": insts["GLD"], "BTC": insts["BTC"]},
        brokers_by_id=brokers,
        positions=[],
    )
    assert reb_empty.status == "skipped_no_positions"

    inst_nonfrac = {
        "BOND1": InstrumentConfig(
            instrument_id="BOND1", ticker="BOND1", section="fixed_income", asset_subtype="sovereign_bond",
            fractional_allowed=False, lot_size=10.0, min_nominal=100.0, default_buy_broker_id="IBKR",
        ),
        "GLD": insts["GLD"],
    }
    reb_lot = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["BOND1", "GLD"],
        target_weights={"BOND1": 0.50, "GLD": 0.50},
        prices_usd={"BOND1": 1.0, "GLD": 100.0},
        instruments_by_id=inst_nonfrac,
        brokers_by_id={"IBKR": brokers["IBKR"]},
        positions=[BrokerPosition(instrument_id="GLD", broker_id="IBKR", quantity=1.5)],
        rebalancing_config=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=10.0),
    )
    assert any(o.instrument_id == "BOND1" for o in reb_lot.omitted_operations)
    assert all(o.instrument_id != "BOND1" for o in reb_lot.orders)

    reb_no_wt = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["AAPL", "GLD"],
        target_weights={"AAPL": 0.505, "GLD": 0.495},
        prices_usd={"AAPL": 100.0, "GLD": 100.0},
        instruments_by_id={"AAPL": insts["AAPL"], "GLD": insts["GLD"]},
        brokers_by_id={"IBKR": brokers["IBKR"]},
        positions=[
            BrokerPosition(instrument_id="AAPL", broker_id="IBKR", quantity=50.0),
            BrokerPosition(instrument_id="GLD", broker_id="IBKR", quantity=50.0),
        ],
        rebalancing_config=RebalancingConfig(use_weight_threshold=False, min_notional_usd=25.0),
    )
    assert len(reb_no_wt.orders) == 2

    reb_withdraw = compute_rebalancing_plan(
        strategy_name="erc",
        tickers=["GLD"],
        target_weights={"GLD": 1.0},
        prices_usd={"GLD": 100.0},
        instruments_by_id={"GLD": insts["GLD"]},
        brokers_by_id={"IBKR": brokers["IBKR"]},
        positions=[BrokerPosition(instrument_id="GLD", broker_id="IBKR", quantity=1.0)],
        rebalancing_config=RebalancingConfig(available_cash_usd=0.0, cash_flow_usd=-200.0),
    )
    assert reb_withdraw.status == "insufficient_cash"
    assert reb_withdraw.feasible is False

    cfg_mb = PortfolioAnalysisConfig(
        variable_income=(insts["AAPL"], insts["GLD"]),
        brokers=(brokers["IBKR"], brokers["IOL"]),
        positions=(
            BrokerPosition(instrument_id="AAPL", broker_id="IBKR", quantity=5.0),
            BrokerPosition(instrument_id="AAPL", broker_id="IOL", quantity=3.0),
            BrokerPosition(instrument_id="GLD", broker_id="IBKR", quantity=10.0),
        ),
        risk=RiskModelConfig(covariance_estimator="static_shrinkage", min_weekly_observations=52),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
    )
    _, mb_prices, _, _, _ = _make_weekly_universe(2, n_weeks=80, seed=9)
    mb_prices.columns = ["AAPL", "GLD"]
    res_mb = analyze_portfolio(cfg_mb, prices_usd_override=mb_prices)
    assert res_mb.tickers == ["AAPL", "GLD"]
    assert res_mb.cov_matrix.shape == (2, 2)


def test_direct_analyze_portfolio_and_master_pipeline_are_identical_and_analytics_stateless():
    tickers, prices, _, _, _ = _make_weekly_universe(5, n_weeks=110, seed=55)
    vix_hist = pd.Series(
        np.where(np.arange(len(prices)) % 4 == 0, 32.0, 14.0),
        index=prices.index,
        name="^VIX",
    )

    cfg = PortfolioAnalysisConfig(
        variable_income=(
            InstrumentConfig(
                instrument_id=tickers[0], ticker=tickers[0], asset_subtype="equity",
                default_buy_broker_id="MAIN",
                dcf_view=DCFViewConfig(valuation_basis="terminal_target", total_upside=0.50, horizon_years=5.0, confidence=0.5),
            ),
            InstrumentConfig(instrument_id=tickers[1], ticker=tickers[1], asset_subtype="equity", default_buy_broker_id="MAIN"),
            InstrumentConfig(instrument_id=tickers[2], ticker=tickers[2], asset_subtype="commodity_gold", default_buy_broker_id="MAIN"),
            InstrumentConfig(instrument_id=tickers[3], ticker=tickers[3], asset_subtype="crypto", default_buy_broker_id="MAIN"),
        ),
        fixed_income=(
            InstrumentConfig(instrument_id=tickers[4], ticker=tickers[4], section="fixed_income", asset_subtype="fixed_income_etf", default_buy_broker_id="MAIN"),
        ),
        brokers=(BrokerConfig(broker_id="MAIN", fee_structure="percentage", buy_percentage_fee=0.002),),
        positions=tuple(BrokerPosition(instrument_id=t, broker_id="MAIN", quantity=10.0) for t in tickers),
        black_litterman=BlackLittermanConfig(rf_rate=0.04),
        risk=RiskModelConfig(covariance_estimator="vix_regime", min_weekly_observations=52),
        constraints=ConstraintsConfig(class_bounds={"crypto": (0.0, 0.10)}),
        optimization=OptimizationConfig(selected_strategy="erc"),
        rebalancing=RebalancingConfig(min_weight_diff=0.01, min_notional_usd=25.0),
    )

    curr_vix = (22.0, prices.index[-1], False)
    r_direct = analyze_portfolio(cfg, prices_usd_override=prices, vix_history_override=vix_hist, current_vix_override=curr_vix)
    r_master = ejecutar_analisis_completo(cfg, render_ui=False, prices_usd_override=prices, vix_history_override=vix_hist, current_vix_override=curr_vix)
    r_backend = run_portfolio_analysis(cfg, render_ui=False, prices_usd_override=prices, vix_history_override=vix_hist, current_vix_override=curr_vix)

    assert r_direct.tickers == r_master.tickers == r_backend.tickers
    np.testing.assert_allclose(r_direct.expected_returns.values, r_master.expected_returns.values, atol=1e-12)
    np.testing.assert_allclose(r_direct.cov_matrix.values, r_backend.cov_matrix.values, atol=1e-12)
    for sname in ("max_sharpe", "min_variance", "min_cvar", "erc", "hrp"):
        np.testing.assert_allclose(r_direct.strategies[sname].weights, r_master.strategies[sname].weights, atol=1e-12)
        np.testing.assert_allclose(r_direct.strategies[sname].weights, r_backend.strategies[sname].weights, atol=1e-12)

    front = generar_frontera_eficiente(r_direct, n_samples=200, render=False)
    assert front["weights"].shape == (200, 5)
    perf = analizar_rendimiento_historico(r_direct, render=False)
    assert list(perf.index) == tickers
    corr = generar_heatmap_correlaciones(r_direct, render=False)
    assert corr.shape == (5, 5)
    dd = generar_underwater_plot(r_direct, render=False)
    assert "PORTFOLIO_ERC" in dd.columns
    mcr_df = analizar_contribucion_marginal_riesgo(r_direct, render=False)
    assert "rc_rel_erc" in mcr_df.columns
    risk_df = generar_grafico_riesgo(r_direct, render=False)
    assert set(risk_df.index) == {"max_sharpe", "min_variance", "min_cvar", "erc", "hrp"}
    alloc = armar_cartera(r_direct, capital_disponible=10000.0)
    assert set(alloc.keys()) == {"max_sharpe", "min_variance", "min_cvar", "erc", "hrp"}
    for df_a in alloc.values():
        assert df_a["Amount (USD)"].sum() == pytest.approx(10000.0, abs=1e-4)
