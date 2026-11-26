"""Multi-broker, commission-aware fractional portfolio rebalancing engine.

Key invariants:
- Differentiates economic instrument identity from broker-level holdings.
- Uses the shared canonical dirty-USD-per-quantity-unit normalization
  (``normalize_execution_price_to_usd_per_unit``) so individual bonds/ONs quoted
  per 100 nominal (original or residual) or per bond unit reconcile identically
  across valuation, YTM, quantities, ``min_nominal``, ``lot_size``, and orders.
- Enforces strict per-broker cash segregation: each broker maintains an
  independent ledger; sells only occur where positions are held; net sell
  proceeds remain in that broker's ledger (never implicitly transferred across
  brokers); buys never cause negative broker cash.
- Supports fixed, percentage, combined (explicit), minimum fee, buy/sell
  asymmetry, non-USD fee FX conversion, and optional estimated spread without
  double-counting.
- Revalidates operational limits (``min_notional_usd``, ``min_nominal``,
  ``lot_size``) after any cash-scaling or rounding, and verifies ALL hard
  constraints (asset bounds, class bounds, ``max_sector_weight``, ``hhi_limit``)
  post-rebalance.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import ConfigurationError, RebalancingError
from .fixed_income import normalize_execution_price_to_usd_per_unit
from .schemas import (
    BrokerConfig,
    BrokerPosition,
    ConstraintsConfig,
    InstrumentConfig,
    OmittedTrade,
    ProposedOrder,
    RebalancingConfig,
    RebalancingResult,
)

_ZERO_FEE_FALLBACK_BROKER = BrokerConfig(
    broker_id="DEFAULT_UNSPECIFIED",
    name="Unspecified Broker (0% fee placeholder)",
    fee_currency="USD",
    fee_structure="percentage",
    buy_percentage_fee=0.0,
    sell_percentage_fee=0.0,
    buy_fixed_fee=0.0,
    sell_fixed_fee=0.0,
    min_fee=0.0,
    estimated_spread_bps=0.0,
)


def calculate_broker_trade_cost_usd(
    gross_amount_usd: float,
    side: str,
    broker: BrokerConfig,
) -> Tuple[float, float]:
    """Compute ``(commission_usd, spread_cost_usd)`` for a single trade."""
    gross = float(gross_amount_usd)
    if gross <= 0.0:
        return 0.0, 0.0

    side_u = side.upper()
    if side_u not in ("BUY", "SELL"):
        raise RebalancingError(f"Invalid order side '{side}'. Expected 'BUY' or 'SELL'.")

    pct = broker.buy_percentage_fee if side_u == "BUY" else float(broker.sell_percentage_fee)
    fixed_local = broker.buy_fixed_fee if side_u == "BUY" else float(broker.sell_fixed_fee)
    fx = float(broker.fee_fx_to_usd)
    fixed_usd = fixed_local * fx
    min_usd = broker.min_fee * fx

    if broker.fee_structure == "percentage":
        raw_comm = gross * pct
    elif broker.fee_structure == "fixed":
        raw_comm = fixed_usd
    elif broker.fee_structure == "fixed_plus_percentage":
        raw_comm = fixed_usd + gross * pct
    else:
        raise RebalancingError(f"Unsupported fee_structure '{broker.fee_structure}'.")

    commission_usd = max(min_usd, raw_comm) if min_usd > 0.0 else raw_comm
    spread_usd = 0.0 if broker.price_includes_spread else gross * (broker.estimated_spread_bps / 10000.0)
    return float(commission_usd), float(spread_usd)


def _passes_threshold(value: float, threshold: float, comparison: str) -> bool:
    if comparison == ">=":
        return value >= threshold - 1e-12
    return value > threshold + 1e-12


def _apply_operational_quantity_rules(
    desired_qty: float,
    instrument: InstrumentConfig,
    *,
    max_qty: Optional[float] = None,
) -> Tuple[float, List[str]]:
    """Enforce fractional vs lot_size / min_nominal rules without premature rounding."""
    notes: List[str] = []
    qty = float(desired_qty)
    if max_qty is not None:
        qty = min(qty, float(max_qty))
    if qty <= 0.0:
        return 0.0, notes

    if not instrument.fractional_allowed:
        step = float(instrument.lot_size) if instrument.lot_size is not None else 1.0
        exec_qty = math.floor((qty + 1e-12) / step) * step
        if not math.isclose(exec_qty, qty, abs_tol=1e-9):
            notes.append(
                f"Instrument '{instrument.instrument_id}' does not allow fractional execution (step={step:g}); "
                f"floored quantity from {qty:.6f} to {exec_qty:.6f}."
            )
        qty = exec_qty
    elif instrument.lot_size is not None:
        step = float(instrument.lot_size)
        exec_qty = math.floor((qty + 1e-12) / step) * step
        if not math.isclose(exec_qty, qty, abs_tol=1e-9):
            notes.append(
                f"Instrument '{instrument.instrument_id}' requires lot_size={step:g}; floored quantity from {qty:.6f} to {exec_qty:.6f}."
            )
        qty = exec_qty

    if instrument.min_nominal is not None and qty > 0.0 and qty < instrument.min_nominal - 1e-12:
        notes.append(
            f"Proposed quantity ({qty:.6f}) is below min_nominal ({instrument.min_nominal:g}) for '{instrument.instrument_id}'; order omitted."
        )
        return 0.0, notes

    return float(round(qty, 10)), notes


def _check_post_rebalance_hard_limits(
    achievable_weights: Dict[str, float],
    instruments_by_id: Dict[str, InstrumentConfig],
    constraints: Optional[ConstraintsConfig],
    *,
    sector_by_ticker: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Verify ALL hard allocation limits on achievable post-rebalance weights (AUD-11).

    Checks:
    - Individual ``asset_bounds``
    - Risk-class ``class_bounds``
    - ``max_sector_weight`` (across known non-``"Unknown"`` sectors)
    - ``hhi_limit`` (Herfindahl-Hirschman Index ``sum(w_i^2)``)
    """
    if constraints is None:
        return []
    tol = float(constraints.feasibility_tolerance)
    conflicts: List[str] = []

    for iid, (lo, hi) in constraints.asset_bounds.items():
        if iid not in achievable_weights:
            continue
        w = achievable_weights[iid]
        if w < lo - tol:
            conflicts.append(
                f"Hard asset minimum violated for '{iid}': achievable weight {w:.4%} < min {lo:.4%}."
            )
        if w > hi + tol:
            conflicts.append(
                f"Hard asset maximum violated for '{iid}': achievable weight {w:.4%} > max {hi:.4%}."
            )

    class_totals: Dict[str, float] = {}
    for iid, w in achievable_weights.items():
        inst = instruments_by_id.get(iid)
        rc = inst.risk_class if inst is not None else "other"
        class_totals[rc] = class_totals.get(rc, 0.0) + w

    for rc, (lo, hi) in constraints.class_bounds.items():
        cw = class_totals.get(rc, 0.0)
        if cw < lo - tol:
            conflicts.append(
                f"Hard class minimum violated for '{rc}': achievable exposure {cw:.4%} < min {lo:.4%}."
            )
        if cw > hi + tol:
            conflicts.append(
                f"Hard class maximum violated for '{rc}': achievable exposure {cw:.4%} > max {hi:.4%}."
            )

    if constraints.max_sector_weight is not None and sector_by_ticker:
        sec_map = {str(k).strip().upper(): str(v) for k, v in sector_by_ticker.items()}
        sector_totals: Dict[str, float] = {}
        for iid, w in achievable_weights.items():
            sec = sec_map.get(iid, "Unknown")
            if sec and sec != "Unknown":
                sector_totals[sec] = sector_totals.get(sec, 0.0) + w
        for sec, sw in sector_totals.items():
            if sw > constraints.max_sector_weight + tol:
                conflicts.append(
                    f"Hard sector maximum violated for '{sec}': achievable exposure {sw:.4%} > max {constraints.max_sector_weight:.4%}."
                )

    if constraints.hhi_limit is not None:
        hhi_val = float(sum(w * w for w in achievable_weights.values()))
        if hhi_val > constraints.hhi_limit + tol:
            conflicts.append(
                f"Hard HHI limit violated: achievable HHI {hhi_val:.6f} > limit {constraints.hhi_limit:.6f}."
            )

    return conflicts


def _resolve_segregated_broker_cash(
    cfg: RebalancingConfig,
    effective_brokers: Dict[str, BrokerConfig],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Resolve and validate per-broker initial cash and cash flows without ambiguity (AUD-03)."""
    broker_ids = list(effective_brokers.keys())
    broker_set = set(broker_ids)

    for bid in cfg.available_cash_by_broker:
        if bid not in broker_set:
            raise ConfigurationError(
                f"available_cash_by_broker references unknown broker_id '{bid}' (known: {broker_ids})."
            )
    for bid in cfg.cash_flow_by_broker:
        if bid not in broker_set:
            raise ConfigurationError(
                f"cash_flow_by_broker references unknown broker_id '{bid}' (known: {broker_ids})."
            )

    if len(broker_ids) > 1:
        if abs(cfg.available_cash_usd) > 0.0:
            raise ConfigurationError(
                "Scalar 'available_cash_usd' is ambiguous with multiple brokers; "
                "assign non-zero cash balances explicitly via 'available_cash_by_broker'."
            )
        if abs(cfg.cash_flow_usd) > 0.0:
            raise ConfigurationError(
                "Scalar 'cash_flow_usd' is ambiguous with multiple brokers; "
                "assign non-zero cash flows explicitly via 'cash_flow_by_broker'."
            )
        init_cash = {bid: float(cfg.available_cash_by_broker.get(bid, 0.0)) for bid in broker_ids}
        flows = {bid: float(cfg.cash_flow_by_broker.get(bid, 0.0)) for bid in broker_ids}
        return init_cash, flows

    # Single-broker compatibility (reject double-counting if both scalar and map are set)
    only_bid = broker_ids[0]
    if abs(cfg.available_cash_usd) > 0.0 and cfg.available_cash_by_broker:
        raise ConfigurationError(
            "Provide either scalar 'available_cash_usd' or 'available_cash_by_broker', not both."
        )
    if abs(cfg.cash_flow_usd) > 0.0 and cfg.cash_flow_by_broker:
        raise ConfigurationError(
            "Provide either scalar 'cash_flow_usd' or 'cash_flow_by_broker', not both."
        )

    init_cash = {
        only_bid: float(cfg.available_cash_by_broker[only_bid])
        if only_bid in cfg.available_cash_by_broker
        else float(cfg.available_cash_usd)
    }
    flows = {
        only_bid: float(cfg.cash_flow_by_broker[only_bid])
        if only_bid in cfg.cash_flow_by_broker
        else float(cfg.cash_flow_usd)
    }
    return init_cash, flows


def _build_broker_ledgers(
    broker_ids: Sequence[str],
    initial_holdings_by_broker: Dict[str, float],
    final_holdings_by_broker: Dict[str, float],
    initial_cash_by_broker: Dict[str, float],
    cash_flow_by_broker: Dict[str, float],
    sell_gross_by_broker: Dict[str, float],
    sell_comm_by_broker: Dict[str, float],
    sell_spread_by_broker: Dict[str, float],
    buy_gross_by_broker: Dict[str, float],
    buy_comm_by_broker: Dict[str, float],
    buy_spread_by_broker: Dict[str, float],
    ending_cash_by_broker: Dict[str, float],
    shortfall_by_broker: Dict[str, float],
    *,
    desired_buy_gross_by_broker: Optional[Dict[str, float]] = None,
    cost_adjusted_buy_reduction_by_broker: Optional[Dict[str, float]] = None,
    threshold_omitted_buy_by_broker: Optional[Dict[str, float]] = None,
    lot_rounding_residual_by_broker: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    ledgers: Dict[str, Dict[str, Any]] = {}
    desired_map = desired_buy_gross_by_broker or {}
    cost_adj_map = cost_adjusted_buy_reduction_by_broker or {}
    thresh_map = threshold_omitted_buy_by_broker or {}
    lot_map = lot_rounding_residual_by_broker or {}
    for bid in broker_ids:
        s_gross = sell_gross_by_broker.get(bid, 0.0)
        s_comm = sell_comm_by_broker.get(bid, 0.0)
        s_spread = sell_spread_by_broker.get(bid, 0.0)
        s_net = s_gross - s_comm - s_spread

        b_gross = buy_gross_by_broker.get(bid, 0.0)
        b_comm = buy_comm_by_broker.get(bid, 0.0)
        b_spread = buy_spread_by_broker.get(bid, 0.0)
        b_out = b_gross + b_comm + b_spread

        ledgers[bid] = {
            "broker_id": bid,
            "initial_holdings_value_usd": float(initial_holdings_by_broker.get(bid, 0.0)),
            "final_holdings_value_usd": float(final_holdings_by_broker.get(bid, 0.0)),
            "initial_cash_usd": float(initial_cash_by_broker.get(bid, 0.0)),
            "cash_flow_usd": float(cash_flow_by_broker.get(bid, 0.0)),
            "sell_gross_usd": float(s_gross),
            "sell_commission_usd": float(s_comm),
            "sell_spread_usd": float(s_spread),
            "sell_net_proceeds_usd": float(s_net),
            "desired_buy_gross_usd": float(desired_map.get(bid, 0.0)),
            "buy_gross_usd": float(b_gross),
            "buy_commission_usd": float(b_comm),
            "buy_spread_usd": float(b_spread),
            "buy_total_outflow_usd": float(b_out),
            "total_transaction_cost_usd": float(s_comm + s_spread + b_comm + b_spread),
            "ending_cash_usd": float(ending_cash_by_broker.get(bid, 0.0)),
            "cost_adjusted_buy_reduction_usd": float(cost_adj_map.get(bid, 0.0)),
            "threshold_omitted_buy_usd": float(thresh_map.get(bid, 0.0)),
            "lot_rounding_residual_usd": float(lot_map.get(bid, 0.0)),
            "unfunded_buy_shortfall_usd": float(shortfall_by_broker.get(bid, 0.0)),
        }
    return ledgers


def compute_rebalancing_plan(
    *,
    strategy_name: str,
    tickers: Sequence[str],
    target_weights: Dict[str, float],
    prices_usd: Dict[str, float],
    instruments_by_id: Dict[str, InstrumentConfig],
    brokers_by_id: Dict[str, BrokerConfig],
    positions: Sequence[BrokerPosition],
    rebalancing_config: Optional[RebalancingConfig] = None,
    constraints_config: Optional[ConstraintsConfig] = None,
    sector_by_ticker: Optional[Dict[str, str]] = None,
    quote_units_by_id: Optional[Dict[str, str]] = None,
    prices_are_dirty_per_unit: bool = False,
) -> RebalancingResult:
    """Compute an indicative multi-broker rebalancing proposal with strict broker cash segregation.

    Capital definitions (consistent across ALL normal and error branches):
    - ``initial_holdings_value_usd``: sum of current holdings valued at dirty USD per unit.
    - ``initial_cash_usd``: sum of pre-flow starting cash balances across all brokers.
    - ``initial_portfolio_value_usd``: ``initial_holdings_value_usd + initial_cash_usd`` (pre-flow).
    - ``cash_flow_usd``: net external deposit (+) or withdrawal (-) across all brokers.
    - ``deployable_capital_usd``: ``initial_portfolio_value_usd + cash_flow_usd`` (target weight base).
    """
    cfg = rebalancing_config or RebalancingConfig()
    ordered_tickers = [t.strip().upper() for t in tickers]
    theoretical = {t: float(target_weights.get(t, 0.0)) for t in ordered_tickers}
    warnings: List[str] = []

    effective_brokers: Dict[str, BrokerConfig] = dict(brokers_by_id)
    if not effective_brokers:
        effective_brokers[_ZERO_FEE_FALLBACK_BROKER.broker_id] = _ZERO_FEE_FALLBACK_BROKER
        warnings.append(
            "No brokers configured; using 0% commission placeholder broker. Configure brokers to include realistic transaction costs."
        )
    broker_ids = list(effective_brokers.keys())

    # Resolve & validate segregated per-broker cash balances and flows
    initial_cash_by_broker, cash_flow_by_broker = _resolve_segregated_broker_cash(cfg, effective_brokers)
    total_initial_cash = sum(initial_cash_by_broker.values())
    total_cash_flow = sum(cash_flow_by_broker.values())
    total_cash_start = total_initial_cash + total_cash_flow

    has_positions = any(p.quantity > 0.0 for p in positions)
    if not has_positions and total_cash_start <= 0.0:
        ending_map = {bid: initial_cash_by_broker[bid] + cash_flow_by_broker[bid] for bid in broker_ids}
        ledgers = _build_broker_ledgers(
            broker_ids, {}, {}, initial_cash_by_broker, cash_flow_by_broker,
            {}, {}, {}, {}, {}, {}, ending_map, {},
        )
        return RebalancingResult(
            strategy_name=strategy_name,
            status="skipped_no_positions",
            feasible=True,
            theoretical_target_weights=theoretical,
            initial_holdings_value_usd=0.0,
            initial_portfolio_value_usd=total_initial_cash,
            deployable_capital_usd=total_cash_start,
            final_portfolio_value_usd=total_cash_start,
            initial_cash_usd=total_initial_cash,
            cash_flow_usd=total_cash_flow,
            ending_cash_usd=total_cash_start,
            initial_cash_by_broker=dict(initial_cash_by_broker),
            cash_flow_by_broker=dict(cash_flow_by_broker),
            ending_cash_by_broker=ending_map,
            broker_ledgers=ledgers,
            warnings=warnings + [
                "Current positions and investable cash were not provided; optimized target weights are available, but rebalancing orders were not simulated."
            ],
        )

    # Normalize every instrument's price to dirty USD per 1 quantity unit using shared helper (AUD-02)
    exec_prices_usd: Dict[str, float] = {}
    quoted_prices_usd: Dict[str, float] = {}
    quantity_units: Dict[str, str] = {}
    q_unit_overrides = quote_units_by_id or {}

    for t in ordered_tickers:
        raw_px = prices_usd.get(t)
        if raw_px is None or not math.isfinite(float(raw_px)) or float(raw_px) <= 0.0:
            raise RebalancingError(
                f"Missing or invalid USD price for '{t}'; cannot compute rebalancing orders.",
                asset_id=t,
            )
        inst = instruments_by_id[t]
        quoted_prices_usd[t] = float(raw_px)
        if prices_are_dirty_per_unit:
            exec_prices_usd[t] = float(raw_px)
            quantity_units[t] = (
                "original_nominal_units" if inst.asset_subtype in ("sovereign_bond", "corporate_bond_on", "tbill") else "shares_or_units"
            )
        else:
            unit_px, meta = normalize_execution_price_to_usd_per_unit(
                inst,
                float(raw_px),
                quote_unit_override=q_unit_overrides.get(t),
            )
            exec_prices_usd[t] = float(unit_px)
            quantity_units[t] = str(meta["quantity_unit"])
            warnings.extend(meta.get("warnings", []))

    pos_by_inst_broker: Dict[Tuple[str, str], float] = {}
    holdings_by_inst: Dict[str, float] = {t: 0.0 for t in ordered_tickers}
    brokers_holding_inst: Dict[str, List[str]] = {t: [] for t in ordered_tickers}
    initial_holdings_by_broker: Dict[str, float] = {bid: 0.0 for bid in broker_ids}

    for pos in positions:
        if pos.instrument_id not in holdings_by_inst:
            raise RebalancingError(
                f"Position instrument '{pos.instrument_id}' is not in the optimization universe.",
                asset_id=pos.instrument_id,
            )
        if pos.quantity > 0.0:
            bid = pos.broker_id
            if bid not in effective_brokers:
                if len(brokers_by_id) == 0:
                    bid = _ZERO_FEE_FALLBACK_BROKER.broker_id
                else:
                    raise RebalancingError(
                        f"Position for '{pos.instrument_id}' references unknown broker_id '{bid}'.",
                        asset_id=pos.instrument_id,
                    )
            pos_by_inst_broker[(pos.instrument_id, bid)] = pos_by_inst_broker.get((pos.instrument_id, bid), 0.0) + pos.quantity
            holdings_by_inst[pos.instrument_id] += pos.quantity
            if bid not in brokers_holding_inst[pos.instrument_id]:
                brokers_holding_inst[pos.instrument_id].append(bid)
            initial_holdings_by_broker[bid] = initial_holdings_by_broker.get(bid, 0.0) + pos.quantity * exec_prices_usd[pos.instrument_id]

    holdings_value_by_inst = {t: holdings_by_inst[t] * exec_prices_usd[t] for t in ordered_tickers}
    holdings_total_usd = sum(holdings_value_by_inst.values())
    initial_portfolio_value_usd = holdings_total_usd + total_initial_cash
    deployable_capital_usd = initial_portfolio_value_usd + total_cash_flow

    if deployable_capital_usd <= 0.0:
        ending_map = {bid: initial_cash_by_broker[bid] + cash_flow_by_broker[bid] for bid in broker_ids}
        ledgers = _build_broker_ledgers(
            broker_ids, initial_holdings_by_broker, initial_holdings_by_broker,
            initial_cash_by_broker, cash_flow_by_broker,
            {}, {}, {}, {}, {}, {}, ending_map, {},
        )
        return RebalancingResult(
            strategy_name=strategy_name,
            status="insufficient_cash",
            feasible=False,
            theoretical_target_weights=theoretical,
            initial_holdings_value_usd=holdings_total_usd,
            initial_portfolio_value_usd=initial_portfolio_value_usd,
            deployable_capital_usd=deployable_capital_usd,
            final_portfolio_value_usd=deployable_capital_usd,
            initial_cash_usd=total_initial_cash,
            cash_flow_usd=total_cash_flow,
            ending_cash_usd=total_cash_start,
            initial_cash_by_broker=dict(initial_cash_by_broker),
            cash_flow_by_broker=dict(cash_flow_by_broker),
            ending_cash_by_broker=ending_map,
            broker_ledgers=ledgers,
            warnings=warnings + [
                f"Deployable portfolio capital after cash_flow ({deployable_capital_usd:.2f} USD) is non-positive; withdrawal exceeds portfolio value."
            ],
        )

    current_weights = {t: holdings_value_by_inst[t] / deployable_capital_usd for t in ordered_tickers}

    def _resolve_buy_broker(inst: InstrumentConfig) -> Optional[BrokerConfig]:
        if inst.default_buy_broker_id:
            if inst.default_buy_broker_id not in effective_brokers:
                raise RebalancingError(
                    f"Instrument '{inst.instrument_id}' references unknown default_buy_broker_id '{inst.default_buy_broker_id}'.",
                    asset_id=inst.instrument_id,
                )
            return effective_brokers[inst.default_buy_broker_id]
        if len(effective_brokers) == 1:
            return next(iter(effective_brokers.values()))
        return None

    def _omitted_sell_broker_label(inst: InstrumentConfig, holding_brokers: List[str]) -> Optional[str]:
        if len(holding_brokers) == 1:
            return holding_brokers[0]
        if len(holding_brokers) > 1:
            ordered = [b for b in inst.sell_broker_priority if b in holding_brokers] if inst.sell_broker_priority else holding_brokers
            return ",".join(ordered)
        return None

    omitted: List[OmittedTrade] = []
    sell_candidates: List[Dict[str, Any]] = []
    buy_candidates: List[Dict[str, Any]] = []
    routing_errors: List[str] = []

    for t in ordered_tickers:
        inst = instruments_by_id[t]
        px = exec_prices_usd[t]
        w_curr = current_weights[t]
        w_tgt = theoretical[t]
        abs_w_diff = abs(w_tgt - w_curr)
        desired_value = w_tgt * deployable_capital_usd
        delta_usd = desired_value - holdings_value_by_inst[t]
        desired_notional = abs(delta_usd)
        holding_brokers = brokers_holding_inst[t]

        if desired_notional <= 1e-8:
            omitted.append(
                OmittedTrade(
                    instrument_id=t,
                    broker_id=_omitted_sell_broker_label(inst, holding_brokers) or inst.default_buy_broker_id,
                    side="HOLD",
                    current_weight=w_curr,
                    target_weight=w_tgt,
                    abs_weight_diff=abs_w_diff,
                    desired_notional_usd=0.0,
                    reasons=["already_at_target"],
                )
            )
            continue

        side = "BUY" if delta_usd > 0.0 else "SELL"

        fail_reasons: List[str] = []
        if cfg.use_weight_threshold and cfg.min_weight_diff is not None:
            if not _passes_threshold(abs_w_diff, cfg.min_weight_diff, cfg.threshold_comparison):
                fail_reasons.append(
                    f"abs_weight_diff ({abs_w_diff:.4%}) fails {cfg.threshold_comparison} min_weight_diff ({cfg.min_weight_diff:.4%})"
                )
        if cfg.use_notional_threshold and cfg.min_notional_usd is not None:
            if not _passes_threshold(desired_notional, cfg.min_notional_usd, cfg.threshold_comparison):
                fail_reasons.append(
                    f"notional ({desired_notional:.2f} USD) fails {cfg.threshold_comparison} min_notional_usd ({cfg.min_notional_usd:.2f} USD)"
                )

        if fail_reasons:
            if side == "SELL":
                omitted_bid = _omitted_sell_broker_label(inst, holding_brokers)
            else:
                bb = _resolve_buy_broker(inst)
                omitted_bid = bb.broker_id if bb is not None else inst.default_buy_broker_id
            omitted.append(
                OmittedTrade(
                    instrument_id=t,
                    broker_id=omitted_bid,
                    side=side,
                    current_weight=w_curr,
                    target_weight=w_tgt,
                    abs_weight_diff=abs_w_diff,
                    desired_notional_usd=desired_notional,
                    reasons=fail_reasons,
                )
            )
            continue

        if side == "SELL":
            if not holding_brokers:
                omitted.append(
                    OmittedTrade(
                        instrument_id=t,
                        broker_id=None,
                        side="SELL",
                        current_weight=w_curr,
                        target_weight=w_tgt,
                        abs_weight_diff=abs_w_diff,
                        desired_notional_usd=desired_notional,
                        reasons=["no_position_available_to_sell"],
                    )
                )
                continue

            if len(holding_brokers) == 1:
                ordered_sell_brokers = [holding_brokers[0]]
            else:
                if not inst.sell_broker_priority or not set(holding_brokers).issubset(set(inst.sell_broker_priority)):
                    routing_errors.append(
                        f"Instrument '{t}' is held across multiple brokers ({holding_brokers}) without complete explicit 'sell_broker_priority'; refusing to guess sell route or assume free transfers."
                    )
                    continue
                ordered_sell_brokers = [b for b in inst.sell_broker_priority if b in holding_brokers]

            remaining_sell_qty = min(desired_notional / px, holdings_by_inst[t])
            for bid in ordered_sell_brokers:
                if remaining_sell_qty <= 1e-10:
                    break
                held_here = pos_by_inst_broker[(t, bid)]
                qty_here = min(remaining_sell_qty, held_here)
                exec_qty, op_notes = _apply_operational_quantity_rules(qty_here, inst, max_qty=held_here)
                gross_here = exec_qty * px
                if exec_qty <= 0.0:
                    omitted.append(
                        OmittedTrade(
                            instrument_id=t,
                            broker_id=bid,
                            side="SELL",
                            current_weight=w_curr,
                            target_weight=w_tgt,
                            abs_weight_diff=abs_w_diff,
                            desired_notional_usd=qty_here * px,
                            reasons=op_notes or ["below_operational_lot_or_min_nominal"],
                        )
                    )
                    continue
                if cfg.use_notional_threshold and cfg.min_notional_usd is not None and not _passes_threshold(
                    gross_here, cfg.min_notional_usd, cfg.threshold_comparison
                ):
                    omitted.append(
                        OmittedTrade(
                            instrument_id=t,
                            broker_id=bid,
                            side="SELL",
                            current_weight=w_curr,
                            target_weight=w_tgt,
                            abs_weight_diff=abs_w_diff,
                            desired_notional_usd=gross_here,
                            reasons=[
                                f"post_rounding_notional ({gross_here:.2f} USD) fails {cfg.threshold_comparison} min_notional_usd ({cfg.min_notional_usd:.2f} USD)"
                            ],
                        )
                    )
                    continue
                sell_candidates.append({
                    "instrument": inst,
                    "broker": effective_brokers[bid],
                    "qty": exec_qty,
                    "price_usd": px,
                    "notes": op_notes,
                })
                remaining_sell_qty -= exec_qty
        else:
            buy_broker = _resolve_buy_broker(inst)
            if buy_broker is None:
                routing_errors.append(
                    f"Instrument '{t}' requires a BUY order across multiple configured brokers without an explicit 'default_buy_broker_id'."
                )
                continue
            desired_qty = desired_notional / px
            exec_qty, op_notes = _apply_operational_quantity_rules(desired_qty, inst)
            gross_after_round = exec_qty * px
            if exec_qty <= 0.0:
                omitted.append(
                    OmittedTrade(
                        instrument_id=t,
                        broker_id=buy_broker.broker_id,
                        side="BUY",
                        current_weight=w_curr,
                        target_weight=w_tgt,
                        abs_weight_diff=abs_w_diff,
                        desired_notional_usd=desired_notional,
                        reasons=op_notes or ["below_operational_lot_or_min_nominal"],
                    )
                )
                continue
            if cfg.use_notional_threshold and cfg.min_notional_usd is not None and not _passes_threshold(
                gross_after_round, cfg.min_notional_usd, cfg.threshold_comparison
            ):
                omitted.append(
                    OmittedTrade(
                        instrument_id=t,
                        broker_id=buy_broker.broker_id,
                        side="BUY",
                        current_weight=w_curr,
                        target_weight=w_tgt,
                        abs_weight_diff=abs_w_diff,
                        desired_notional_usd=gross_after_round,
                        reasons=[
                            f"post_rounding_notional ({gross_after_round:.2f} USD) fails {cfg.threshold_comparison} min_notional_usd ({cfg.min_notional_usd:.2f} USD)"
                        ],
                    )
                )
                continue
            buy_candidates.append({
                "instrument": inst,
                "broker": buy_broker,
                "qty": exec_qty,
                "price_usd": px,
                "notes": op_notes,
            })

    if routing_errors:
        ending_map = {bid: initial_cash_by_broker[bid] + cash_flow_by_broker[bid] for bid in broker_ids}
        ledgers = _build_broker_ledgers(
            broker_ids, initial_holdings_by_broker, initial_holdings_by_broker,
            initial_cash_by_broker, cash_flow_by_broker,
            {}, {}, {}, {}, {}, {}, ending_map, {},
        )
        return RebalancingResult(
            strategy_name=strategy_name,
            status="broker_routing_required",
            feasible=False,
            omitted_operations=omitted,
            theoretical_target_weights=theoretical,
            current_weights=current_weights,
            achievable_weights=current_weights,
            weight_deviations={t: current_weights[t] - theoretical[t] for t in ordered_tickers},
            initial_holdings_value_usd=holdings_total_usd,
            initial_portfolio_value_usd=initial_portfolio_value_usd,
            deployable_capital_usd=deployable_capital_usd,
            final_portfolio_value_usd=deployable_capital_usd,
            initial_cash_usd=total_initial_cash,
            cash_flow_usd=total_cash_flow,
            ending_cash_usd=total_cash_start,
            initial_cash_by_broker=dict(initial_cash_by_broker),
            cash_flow_by_broker=dict(cash_flow_by_broker),
            ending_cash_by_broker=ending_map,
            broker_ledgers=ledgers,
            warnings=warnings + routing_errors,
        )

    # Execute sells strictly per broker ledger
    executed_orders: List[ProposedOrder] = []
    post_qty_by_inst = dict(holdings_by_inst)
    post_pos_by_inst_broker = dict(pos_by_inst_broker)

    sell_gross_by_broker: Dict[str, float] = {bid: 0.0 for bid in broker_ids}
    sell_comm_by_broker: Dict[str, float] = {bid: 0.0 for bid in broker_ids}
    sell_spread_by_broker: Dict[str, float] = {bid: 0.0 for bid in broker_ids}
    sell_net_by_broker: Dict[str, float] = {bid: 0.0 for bid in broker_ids}

    for cand in sell_candidates:
        inst: InstrumentConfig = cand["instrument"]
        broker: BrokerConfig = cand["broker"]
        bid = broker.broker_id
        qty = float(cand["qty"])
        px = float(cand["price_usd"])
        notes: List[str] = list(cand["notes"])
        gross = qty * px
        comm, spread = calculate_broker_trade_cost_usd(gross, "SELL", broker)
        net_inflow = gross - comm - spread
        if net_inflow < 0.0:
            warnings.append(
                f"SELL order for '{inst.instrument_id}' at '{bid}' omitted because costs ({comm + spread:.2f} USD) exceed gross proceeds ({gross:.2f} USD)."
            )
            omitted.append(
                OmittedTrade(
                    instrument_id=inst.instrument_id,
                    broker_id=bid,
                    side="SELL",
                    current_weight=current_weights[inst.instrument_id],
                    target_weight=theoretical[inst.instrument_id],
                    abs_weight_diff=abs(theoretical[inst.instrument_id] - current_weights[inst.instrument_id]),
                    desired_notional_usd=gross,
                    reasons=["transaction_costs_exceed_sell_proceeds"],
                )
            )
            continue

        post_qty_by_inst[inst.instrument_id] -= qty
        post_pos_by_inst_broker[(inst.instrument_id, bid)] = max(
            0.0, post_pos_by_inst_broker.get((inst.instrument_id, bid), 0.0) - qty
        )
        sell_gross_by_broker[bid] += gross
        sell_comm_by_broker[bid] += comm
        sell_spread_by_broker[bid] += spread
        sell_net_by_broker[bid] += net_inflow
        executed_orders.append(
            ProposedOrder(
                broker_id=bid,
                instrument_id=inst.instrument_id,
                ticker=inst.ticker,
                side="SELL",
                quantity=qty,
                price_usd=px,
                gross_amount_usd=gross,
                commission_usd=comm,
                spread_cost_usd=spread,
                net_amount_usd=net_inflow,
                pre_weight=current_weights[inst.instrument_id],
                target_weight=theoretical[inst.instrument_id],
                achievable_weight=0.0,
                fractional=inst.fractional_allowed,
                quantity_unit=quantity_units[inst.instrument_id],
                quoted_price_usd=quoted_prices_usd[inst.instrument_id],
                notes=notes,
            )
        )

    # Available buy cash STRICTLY segregated per broker (ZERO implicit cross-broker transfers!)
    avail_cash_by_broker: Dict[str, float] = {
        bid: initial_cash_by_broker[bid] + cash_flow_by_broker[bid] + sell_net_by_broker[bid]
        for bid in broker_ids
    }

    buy_gross_by_broker: Dict[str, float] = {bid: 0.0 for bid in broker_ids}
    buy_comm_by_broker: Dict[str, float] = {bid: 0.0 for bid in broker_ids}
    buy_spread_by_broker: Dict[str, float] = {bid: 0.0 for bid in broker_ids}
    buy_outflow_by_broker: Dict[str, float] = {bid: 0.0 for bid in broker_ids}
    desired_buy_gross_by_broker: Dict[str, float] = {bid: 0.0 for bid in broker_ids}
    shortfall_by_broker: Dict[str, float] = {bid: 0.0 for bid in broker_ids}
    cost_adjusted_buy_reduction_by_broker: Dict[str, float] = {bid: 0.0 for bid in broker_ids}
    threshold_omitted_buy_by_broker: Dict[str, float] = {bid: 0.0 for bid in broker_ids}
    lot_rounding_residual_by_broker: Dict[str, float] = {bid: 0.0 for bid in broker_ids}

    buys_by_broker: Dict[str, List[Dict[str, Any]]] = {bid: [] for bid in broker_ids}
    for cand in buy_candidates:
        buys_by_broker[cand["broker"].broker_id].append(cand)

    def _eval_scaled_buys(
        cands: List[Dict[str, Any]],
        scale: float,
    ) -> Tuple[float, List[Tuple[float, float, float, float, List[str]]]]:
        total_out = 0.0
        evals: List[Tuple[float, float, float, float, List[str]]] = []
        for c in cands:
            inst: InstrumentConfig = c["instrument"]
            broker: BrokerConfig = c["broker"]
            px = float(c["price_usd"])
            raw_q = float(c["qty"]) * scale
            q_exec, op_notes = _apply_operational_quantity_rules(raw_q, inst)
            gross = q_exec * px
            if q_exec <= 0.0 or gross <= 1e-10:
                evals.append((0.0, 0.0, 0.0, 0.0, op_notes))
                continue
            comm, spread = calculate_broker_trade_cost_usd(gross, "BUY", broker)
            total_out += gross + comm + spread
            evals.append((q_exec, gross, comm, spread, op_notes))
        return total_out, evals

    used_same_broker_sell_proceeds = False

    for bid in broker_ids:
        cands_here = buys_by_broker[bid]
        if not cands_here:
            continue
        broker_avail = avail_cash_by_broker[bid]
        desired_buy_gross_b = sum(float(c["qty"]) * float(c["price_usd"]) for c in cands_here)
        desired_buy_gross_by_broker[bid] = desired_buy_gross_b
        gross_pre_cost_cash_b = initial_cash_by_broker[bid] + cash_flow_by_broker[bid] + sell_gross_by_broker[bid]
        pre_cost_deficit_b = max(0.0, desired_buy_gross_b - max(gross_pre_cost_cash_b, 0.0))
        if pre_cost_deficit_b <= 1e-8:
            pre_cost_deficit_b = 0.0

        desired_full_outflow, _ = _eval_scaled_buys(cands_here, 1.0)

        if broker_avail <= 1e-8:
            other_brokers_with_cash = [b for b in broker_ids if b != bid and avail_cash_by_broker[b] > 1e-6]
            if other_brokers_with_cash:
                warnings.append(
                    f"Broker '{bid}' has {broker_avail:.2f} USD available to fund BUY orders ({desired_full_outflow:.2f} USD required); "
                    f"cash held in other broker(s) {other_brokers_with_cash} is strictly segregated (no implicit inter-broker transfer)."
                )
            else:
                warnings.append(
                    f"Broker '{bid}' has {broker_avail:.2f} USD available cash; BUY order(s) omitted to avoid unfunded purchases."
                )
            for c in cands_here:
                inst = c["instrument"]
                desired_g = float(c["qty"]) * float(c["price_usd"])
                shortfall_by_broker[bid] += desired_g
                omitted.append(
                    OmittedTrade(
                        instrument_id=inst.instrument_id,
                        broker_id=bid,
                        side="BUY",
                        current_weight=current_weights[inst.instrument_id],
                        target_weight=theoretical[inst.instrument_id],
                        abs_weight_diff=abs(theoretical[inst.instrument_id] - current_weights[inst.instrument_id]),
                        desired_notional_usd=desired_g,
                        reasons=["insufficient_cash_in_broker"],
                    )
                )
            continue

        # Iterative per-broker cash scaling + post-scaling min_notional_usd revalidation (AUD-03, AUD-11, NEW-02)
        active_cands = list(cands_here)
        final_evals: List[Tuple[Dict[str, Any], float, float, float, float, List[str]]] = []
        cost_infeasible_omitted_b = 0.0
        threshold_omitted_b = 0.0

        while active_cands:
            full_out, evals = _eval_scaled_buys(active_cands, 1.0)
            chosen_scale = 1.0
            if full_out > broker_avail + 1e-9:
                # 60 bisection steps achieve scale precision 2^-60 (~8.7e-19), guaranteed monotone convergence.
                lo, hi = 0.0, 1.0
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    mid_out, _ = _eval_scaled_buys(active_cands, mid)
                    if mid_out <= broker_avail + 1e-12:
                        lo = mid
                    else:
                        hi = mid
                chosen_scale = lo
                _, evals = _eval_scaled_buys(active_cands, chosen_scale)

            # Revalidate operational rules and min_notional_usd AFTER scaling
            dropped_this_round = False
            next_active: List[Dict[str, Any]] = []
            for c, (q_exec, gross, comm, spread, op_notes) in zip(active_cands, evals):
                inst = c["instrument"]
                px = float(c["price_usd"])
                desired_g = float(c["qty"]) * px
                if q_exec <= 0.0 or gross <= 1e-10:
                    cost_infeasible_omitted_b += desired_g
                    omitted.append(
                        OmittedTrade(
                            instrument_id=inst.instrument_id,
                            broker_id=bid,
                            side="BUY",
                            current_weight=current_weights[inst.instrument_id],
                            target_weight=theoretical[inst.instrument_id],
                            abs_weight_diff=abs(theoretical[inst.instrument_id] - current_weights[inst.instrument_id]),
                            desired_notional_usd=desired_g,
                            reasons=op_notes or ["insufficient_cash_in_broker_after_costs"],
                        )
                    )
                    dropped_this_round = True
                    continue
                if cfg.use_notional_threshold and cfg.min_notional_usd is not None and not _passes_threshold(
                    gross, cfg.min_notional_usd, cfg.threshold_comparison
                ):
                    threshold_omitted_b += desired_g
                    omitted.append(
                        OmittedTrade(
                            instrument_id=inst.instrument_id,
                            broker_id=bid,
                            side="BUY",
                            current_weight=current_weights[inst.instrument_id],
                            target_weight=theoretical[inst.instrument_id],
                            abs_weight_diff=abs(theoretical[inst.instrument_id] - current_weights[inst.instrument_id]),
                            desired_notional_usd=desired_g,
                            reasons=[
                                f"scaled_notional ({gross:.2f} USD) fails {cfg.threshold_comparison} min_notional_usd ({cfg.min_notional_usd:.2f} USD)"
                            ],
                        )
                    )
                    dropped_this_round = True
                    continue
                next_active.append(c)

            if dropped_this_round:
                active_cands = next_active
                continue

            if chosen_scale < 1.0 - 1e-9:
                if pre_cost_deficit_b > 1e-8:
                    warnings.append(
                        f"Broker '{bid}': BUY order(s) scaled by {chosen_scale:.4f} due to segregated broker funding deficit "
                        f"({pre_cost_deficit_b:.2f} USD pre-cost shortfall) and transaction costs (available net cash: {broker_avail:.2f} USD)."
                    )
                else:
                    warnings.append(
                        f"Broker '{bid}': BUY order(s) scaled by {chosen_scale:.4f} to absorb broker transaction costs "
                        f"within available segregated cash ({broker_avail:.2f} USD; pre-cost funding shortfall: 0.00 USD)."
                    )
            final_evals = [
                (c, q_exec, gross, comm, spread, op_notes)
                for c, (q_exec, gross, comm, spread, op_notes) in zip(active_cands, evals)
            ]
            break

        exec_reduction_b = 0.0
        for c, q_exec, gross, comm, spread, op_notes in final_evals:
            inst = c["instrument"]
            px = float(c["price_usd"])
            desired_g = float(c["qty"]) * px
            cash_out = gross + comm + spread
            if buy_outflow_by_broker[bid] + cash_out > broker_avail + 1e-9:
                cost_infeasible_omitted_b += desired_g
                omitted.append(
                    OmittedTrade(
                        instrument_id=inst.instrument_id,
                        broker_id=bid,
                        side="BUY",
                        current_weight=current_weights[inst.instrument_id],
                        target_weight=theoretical[inst.instrument_id],
                        abs_weight_diff=abs(theoretical[inst.instrument_id] - current_weights[inst.instrument_id]),
                        desired_notional_usd=desired_g,
                        reasons=["insufficient_cash_in_broker_for_buy_and_fee"],
                    )
                )
                continue

            exec_reduction_b += max(0.0, desired_g - gross)

            post_qty_by_inst[inst.instrument_id] += q_exec
            post_pos_by_inst_broker[(inst.instrument_id, bid)] = (
                post_pos_by_inst_broker.get((inst.instrument_id, bid), 0.0) + q_exec
            )
            buy_gross_by_broker[bid] += gross
            buy_comm_by_broker[bid] += comm
            buy_spread_by_broker[bid] += spread
            buy_outflow_by_broker[bid] += cash_out

            if sell_net_by_broker[bid] > 1e-8 and buy_outflow_by_broker[bid] > (
                initial_cash_by_broker[bid] + cash_flow_by_broker[bid] + 1e-8
            ):
                used_same_broker_sell_proceeds = True

            base_notes = list(dict.fromkeys(list(c["notes"]) + op_notes))
            executed_orders.append(
                ProposedOrder(
                    broker_id=bid,
                    instrument_id=inst.instrument_id,
                    ticker=inst.ticker,
                    side="BUY",
                    quantity=q_exec,
                    price_usd=px,
                    gross_amount_usd=gross,
                    commission_usd=comm,
                    spread_cost_usd=spread,
                    net_amount_usd=-cash_out,
                    pre_weight=current_weights[inst.instrument_id],
                    target_weight=theoretical[inst.instrument_id],
                    achievable_weight=0.0,
                    fractional=inst.fractional_allowed,
                    quantity_unit=quantity_units[inst.instrument_id],
                    quoted_price_usd=quoted_prices_usd[inst.instrument_id],
                    notes=base_notes,
                )
            )

        # Causal per-broker decomposition of unexecuted buy gross (NEW-02):
        # Exact identity: desired_buy_gross_b - buy_gross_b
        #   == (desired_buy_gross_b - gross_pre_cost_cash_b) + total_tx_cost_b + ending_cash_b
        # Separates real funding shortfall from broker transaction-cost friction, operational
        # min_notional threshold omissions, and discrete lot rounding residuals.
        total_unexec_b = max(0.0, desired_buy_gross_b - buy_gross_by_broker[bid])
        if total_unexec_b <= 1e-10:
            total_unexec_b = 0.0

        shortfall_b = max(pre_cost_deficit_b, cost_infeasible_omitted_b)
        if shortfall_b <= 1e-8:
            shortfall_b = 0.0
        shortfall_b = min(shortfall_b, total_unexec_b)

        rem_after_shortfall = max(0.0, total_unexec_b - shortfall_b)
        broker_tx_cost_b = (
            sell_comm_by_broker[bid]
            + sell_spread_by_broker[bid]
            + buy_comm_by_broker[bid]
            + buy_spread_by_broker[bid]
        )
        cost_adj_b = min(exec_reduction_b, rem_after_shortfall, broker_tx_cost_b)
        if cost_adj_b <= 1e-10:
            cost_adj_b = 0.0

        rem_after_cost = max(0.0, rem_after_shortfall - cost_adj_b)
        threshold_net_b = min(rem_after_cost, threshold_omitted_b)
        if threshold_net_b <= 1e-10:
            threshold_net_b = 0.0

        lot_resid_b = max(0.0, rem_after_cost - threshold_net_b)
        if lot_resid_b <= 1e-10:
            lot_resid_b = 0.0

        shortfall_by_broker[bid] = shortfall_b
        cost_adjusted_buy_reduction_by_broker[bid] = cost_adj_b
        threshold_omitted_buy_by_broker[bid] = threshold_net_b
        lot_rounding_residual_by_broker[bid] = lot_resid_b

    if used_same_broker_sell_proceeds:
        warnings.append(
            "Same-broker sell proceeds are assumed available to fund buys within the same broker ledger; "
            "real-world settlement timing (e.g. T+1/T+2) is not guaranteed and no inter-broker transfers are assumed."
        )

    ending_cash_by_broker: Dict[str, float] = {}
    for bid in broker_ids:
        end_b = avail_cash_by_broker[bid] - buy_outflow_by_broker[bid]
        if abs(end_b) < 1e-10:
            end_b = 0.0
        ending_cash_by_broker[bid] = end_b

    ending_cash_usd = sum(ending_cash_by_broker.values())
    if abs(ending_cash_usd) < 1e-10:
        ending_cash_usd = 0.0

    final_holdings_by_broker: Dict[str, float] = {bid: 0.0 for bid in broker_ids}
    for (iid, bid), q_val in post_pos_by_inst_broker.items():
        if q_val > 0.0:
            final_holdings_by_broker[bid] = final_holdings_by_broker.get(bid, 0.0) + q_val * exec_prices_usd[iid]

    final_val_by_inst = {t: post_qty_by_inst[t] * exec_prices_usd[t] for t in ordered_tickers}
    final_portfolio_value_usd = sum(final_val_by_inst.values()) + ending_cash_usd
    achievable_weights = {
        t: (final_val_by_inst[t] / final_portfolio_value_usd if final_portfolio_value_usd > 0 else 0.0)
        for t in ordered_tickers
    }
    weight_deviations = {t: achievable_weights[t] - theoretical[t] for t in ordered_tickers}

    for order in executed_orders:
        order.achievable_weight = achievable_weights[order.instrument_id]

    max_dev = max((abs(v) for v in weight_deviations.values()), default=0.0)
    if max_dev > 0.005:
        warnings.append(
            f"Achievable post-rebalance weights deviate from theoretical targets by up to {max_dev:.2%} due to thresholds, transaction costs, or segregated broker cash limits."
        )

    hard_conflicts: List[str] = []
    if cfg.enforce_hard_bounds_after_rebalance:
        hard_conflicts = _check_post_rebalance_hard_limits(
            achievable_weights,
            instruments_by_id,
            constraints_config,
            sector_by_ticker=sector_by_ticker,
        )

    negative_cash_brokers = [bid for bid, cval in ending_cash_by_broker.items() if cval < -1e-6]
    total_shortfall = sum(shortfall_by_broker.values())
    if abs(total_shortfall) < 1e-10:
        total_shortfall = 0.0

    if negative_cash_brokers:
        status = "insufficient_cash"
        feasible = False
        warnings.append(
            f"Broker(s) {negative_cash_brokers} ended with negative cash balance after requested withdrawal/costs."
        )
    elif hard_conflicts:
        status = "infeasible_hard_limits"
        feasible = False
        warnings.extend(hard_conflicts)
    elif total_shortfall > 1e-6:
        status = "partially_funded"
        feasible = False
    else:
        status = "ok"
        feasible = True

    total_buy_gross = sum(buy_gross_by_broker.values())
    total_sell_gross = sum(sell_gross_by_broker.values())
    total_commission = sum(sell_comm_by_broker.values()) + sum(buy_comm_by_broker.values())
    total_spread = sum(sell_spread_by_broker.values()) + sum(buy_spread_by_broker.values())
    total_tx_cost = total_commission + total_spread

    two_way_turnover = (total_buy_gross + total_sell_gross) / deployable_capital_usd
    one_way_turnover = 0.5 * two_way_turnover

    broker_ledgers = _build_broker_ledgers(
        broker_ids,
        initial_holdings_by_broker,
        final_holdings_by_broker,
        initial_cash_by_broker,
        cash_flow_by_broker,
        sell_gross_by_broker,
        sell_comm_by_broker,
        sell_spread_by_broker,
        buy_gross_by_broker,
        buy_comm_by_broker,
        buy_spread_by_broker,
        ending_cash_by_broker,
        shortfall_by_broker,
        desired_buy_gross_by_broker=desired_buy_gross_by_broker,
        cost_adjusted_buy_reduction_by_broker=cost_adjusted_buy_reduction_by_broker,
        threshold_omitted_buy_by_broker=threshold_omitted_buy_by_broker,
        lot_rounding_residual_by_broker=lot_rounding_residual_by_broker,
    )

    return RebalancingResult(
        strategy_name=strategy_name,
        status=status,
        feasible=feasible,
        orders=executed_orders,
        omitted_operations=omitted,
        theoretical_target_weights=theoretical,
        current_weights=current_weights,
        achievable_weights=achievable_weights,
        weight_deviations=weight_deviations,
        initial_holdings_value_usd=holdings_total_usd,
        initial_portfolio_value_usd=initial_portfolio_value_usd,
        deployable_capital_usd=deployable_capital_usd,
        final_portfolio_value_usd=final_portfolio_value_usd,
        initial_cash_usd=total_initial_cash,
        cash_flow_usd=total_cash_flow,
        ending_cash_usd=ending_cash_usd,
        initial_cash_by_broker=dict(initial_cash_by_broker),
        cash_flow_by_broker=dict(cash_flow_by_broker),
        ending_cash_by_broker=ending_cash_by_broker,
        broker_ledgers=broker_ledgers,
        unfunded_buy_shortfall_usd=total_shortfall,
        total_buy_gross_usd=total_buy_gross,
        total_sell_gross_usd=total_sell_gross,
        total_commission_usd=total_commission,
        total_spread_cost_usd=total_spread,
        total_transaction_cost_usd=total_tx_cost,
        one_way_turnover=one_way_turnover,
        two_way_turnover=two_way_turnover,
        hard_limit_conflicts=hard_conflicts,
        warnings=warnings,
    )
