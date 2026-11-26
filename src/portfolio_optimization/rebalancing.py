"""Multi-broker, commission-aware fractional portfolio rebalancing engine.

Key invariants:
- Differentiates economic instrument identity from broker-level holdings.
- Supports multiple brokers with explicit buy/sell routing (never assumes free
  inter-broker transfers and never picks a broker silently when ambiguous).
- Supports fixed, percentage, combined (only when explicitly configured),
  minimum fee, buy/sell asymmetry, non-USD fee FX conversion, and optional
  estimated spread without double-counting.
- Fractional execution by default (no integer rounding or MILP); validates
  non-fractional / lot_size / min_nominal operational restrictions explicitly.
- Editable dual thresholds (absolute percentage-point weight difference AND
  gross USD notional), omitting trades failing any enabled threshold.
- Full cash and cost accounting reconciliation: never creates unfunded buys
  or oversells broker positions; flags hard-limit conflicts explicitly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .errors import RebalancingError
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
    """Enforce fractional vs lot_size / min_nominal rules without simulating impossible fills."""
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

    return float(round(qty, 8)), notes


def _check_post_rebalance_hard_limits(
    achievable_weights: Dict[str, float],
    instruments_by_id: Dict[str, InstrumentConfig],
    constraints: Optional[ConstraintsConfig],
) -> List[str]:
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

    return conflicts


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
) -> RebalancingResult:
    """Compute an indicative multi-broker rebalancing proposal with full reconciliation."""
    cfg = rebalancing_config or RebalancingConfig()
    ordered_tickers = [t.strip().upper() for t in tickers]
    theoretical = {t: float(target_weights.get(t, 0.0)) for t in ordered_tickers}
    warnings: List[str] = []

    has_positions = any(p.quantity > 0.0 for p in positions)
    total_cash_start = cfg.available_cash_usd + cfg.cash_flow_usd
    if not has_positions and total_cash_start <= 0.0:
        return RebalancingResult(
            strategy_name=strategy_name,
            status="skipped_no_positions",
            feasible=True,
            theoretical_target_weights=theoretical,
            initial_cash_usd=cfg.available_cash_usd,
            cash_flow_usd=cfg.cash_flow_usd,
            ending_cash_usd=total_cash_start,
            warnings=[
                "Current positions and investable cash were not provided; optimized target weights are available, but rebalancing orders were not simulated."
            ],
        )

    for t in ordered_tickers:
        px = prices_usd.get(t)
        if px is None or not math.isfinite(float(px)) or float(px) <= 0.0:
            raise RebalancingError(
                f"Missing or invalid USD price for '{t}'; cannot compute rebalancing orders.",
                asset_id=t,
            )

    pos_by_inst_broker: Dict[Tuple[str, str], float] = {}
    holdings_by_inst: Dict[str, float] = {t: 0.0 for t in ordered_tickers}
    brokers_holding_inst: Dict[str, List[str]] = {t: [] for t in ordered_tickers}

    for pos in positions:
        if pos.instrument_id not in holdings_by_inst:
            raise RebalancingError(
                f"Position instrument '{pos.instrument_id}' is not in the optimization universe.",
                asset_id=pos.instrument_id,
            )
        if pos.quantity > 0.0:
            pos_by_inst_broker[(pos.instrument_id, pos.broker_id)] = pos.quantity
            holdings_by_inst[pos.instrument_id] += pos.quantity
            brokers_holding_inst[pos.instrument_id].append(pos.broker_id)

    holdings_value_by_inst = {t: holdings_by_inst[t] * float(prices_usd[t]) for t in ordered_tickers}
    holdings_total_usd = sum(holdings_value_by_inst.values())
    total_portfolio_value_usd = holdings_total_usd + cfg.available_cash_usd + cfg.cash_flow_usd

    if total_portfolio_value_usd <= 0.0:
        return RebalancingResult(
            strategy_name=strategy_name,
            status="insufficient_cash",
            feasible=False,
            theoretical_target_weights=theoretical,
            initial_portfolio_value_usd=holdings_total_usd + cfg.available_cash_usd,
            initial_cash_usd=cfg.available_cash_usd,
            cash_flow_usd=cfg.cash_flow_usd,
            ending_cash_usd=total_cash_start,
            warnings=[
                f"Total portfolio value after cash_flow ({total_portfolio_value_usd:.2f} USD) is non-positive; withdrawal exceeds portfolio value."
            ],
        )

    current_weights = {t: holdings_value_by_inst[t] / total_portfolio_value_usd for t in ordered_tickers}

    effective_brokers: Dict[str, BrokerConfig] = dict(brokers_by_id)
    if not effective_brokers:
        effective_brokers[_ZERO_FEE_FALLBACK_BROKER.broker_id] = _ZERO_FEE_FALLBACK_BROKER
        warnings.append(
            "No brokers configured; using 0% commission placeholder broker. Configure brokers to include realistic transaction costs."
        )

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

    omitted: List[OmittedTrade] = []
    sell_candidates: List[Dict[str, object]] = []
    buy_candidates: List[Dict[str, object]] = []
    routing_errors: List[str] = []

    for t in ordered_tickers:
        inst = instruments_by_id[t]
        px = float(prices_usd[t])
        w_curr = current_weights[t]
        w_tgt = theoretical[t]
        abs_w_diff = abs(w_tgt - w_curr)
        desired_value = w_tgt * total_portfolio_value_usd
        delta_usd = desired_value - holdings_value_by_inst[t]
        desired_notional = abs(delta_usd)

        if desired_notional <= 1e-8:
            omitted.append(
                OmittedTrade(
                    instrument_id=t,
                    broker_id=inst.default_buy_broker_id,
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
            omitted.append(
                OmittedTrade(
                    instrument_id=t,
                    broker_id=inst.default_buy_broker_id,
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
            holding_brokers = brokers_holding_inst[t]
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
                if bid not in effective_brokers:
                    routing_errors.append(f"Position for '{t}' references unknown broker '{bid}'.")
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
            buy_candidates.append({
                "instrument": inst,
                "broker": buy_broker,
                "qty": exec_qty,
                "price_usd": px,
                "notes": op_notes,
            })

    if routing_errors:
        return RebalancingResult(
            strategy_name=strategy_name,
            status="broker_routing_required",
            feasible=False,
            omitted_operations=omitted,
            theoretical_target_weights=theoretical,
            current_weights=current_weights,
            achievable_weights=current_weights,
            weight_deviations={t: current_weights[t] - theoretical[t] for t in ordered_tickers},
            initial_portfolio_value_usd=total_portfolio_value_usd,
            final_portfolio_value_usd=total_portfolio_value_usd,
            initial_cash_usd=cfg.available_cash_usd,
            cash_flow_usd=cfg.cash_flow_usd,
            ending_cash_usd=total_cash_start,
            warnings=warnings + routing_errors,
        )

    executed_orders: List[ProposedOrder] = []
    post_qty_by_inst = dict(holdings_by_inst)
    sell_net_proceeds = 0.0
    total_sell_gross = 0.0
    total_commission = 0.0
    total_spread = 0.0

    for cand in sell_candidates:
        inst: InstrumentConfig = cand["instrument"]  # type: ignore[assignment]
        broker: BrokerConfig = cand["broker"]  # type: ignore[assignment]
        qty = float(cand["qty"])  # type: ignore[arg-type]
        px = float(cand["price_usd"])  # type: ignore[arg-type]
        notes: List[str] = list(cand["notes"])  # type: ignore[arg-type]
        gross = qty * px
        comm, spread = calculate_broker_trade_cost_usd(gross, "SELL", broker)
        net_inflow = gross - comm - spread
        if net_inflow < 0.0:
            warnings.append(
                f"SELL order for '{inst.instrument_id}' at '{broker.broker_id}' omitted because costs ({comm + spread:.2f} USD) exceed gross proceeds ({gross:.2f} USD)."
            )
            omitted.append(
                OmittedTrade(
                    instrument_id=inst.instrument_id,
                    broker_id=broker.broker_id,
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
        total_sell_gross += gross
        total_commission += comm
        total_spread += spread
        sell_net_proceeds += net_inflow
        executed_orders.append(
            ProposedOrder(
                broker_id=broker.broker_id,
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
                notes=notes,
            )
        )

    available_cash_for_buys = cfg.available_cash_usd + cfg.cash_flow_usd + sell_net_proceeds

    if available_cash_for_buys < -1e-6:
        final_val_by_inst = {t: post_qty_by_inst[t] * float(prices_usd[t]) for t in ordered_tickers}
        final_total = sum(final_val_by_inst.values()) + available_cash_for_buys
        ach_w = {t: final_val_by_inst[t] / final_total if final_total > 0 else 0.0 for t in ordered_tickers}
        return RebalancingResult(
            strategy_name=strategy_name,
            status="insufficient_cash",
            feasible=False,
            orders=executed_orders,
            omitted_operations=omitted,
            theoretical_target_weights=theoretical,
            current_weights=current_weights,
            achievable_weights=ach_w,
            weight_deviations={t: ach_w[t] - theoretical[t] for t in ordered_tickers},
            initial_portfolio_value_usd=total_portfolio_value_usd,
            final_portfolio_value_usd=final_total,
            initial_cash_usd=cfg.available_cash_usd,
            cash_flow_usd=cfg.cash_flow_usd,
            ending_cash_usd=available_cash_for_buys,
            total_sell_gross_usd=total_sell_gross,
            total_commission_usd=total_commission,
            total_spread_cost_usd=total_spread,
            total_transaction_cost_usd=total_commission + total_spread,
            warnings=warnings + [
                f"Insufficient cash after sells ({available_cash_for_buys:.2f} USD) to cover requested withdrawal/costs."
            ],
        )

    def _buy_total_cost(cands: List[Dict[str, object]], scale: float) -> Tuple[float, List[Tuple[float, float, float, float, List[str]]]]:
        total_out = 0.0
        evals: List[Tuple[float, float, float, float, List[str]]] = []
        for c in cands:
            inst: InstrumentConfig = c["instrument"]  # type: ignore[assignment]
            broker: BrokerConfig = c["broker"]  # type: ignore[assignment]
            px = float(c["price_usd"])  # type: ignore[arg-type]
            raw_q = float(c["qty"]) * scale  # type: ignore[arg-type]
            q_exec, op_notes = _apply_operational_quantity_rules(raw_q, inst)
            gross = q_exec * px
            if gross <= 1e-10:
                evals.append((0.0, 0.0, 0.0, 0.0, op_notes))
                continue
            comm, spread = calculate_broker_trade_cost_usd(gross, "BUY", broker)
            total_out += gross + comm + spread
            evals.append((q_exec, gross, comm, spread, op_notes))
        return total_out, evals

    total_buy_gross = 0.0
    total_buy_cash_outflow = 0.0

    if buy_candidates:
        if available_cash_for_buys <= 1e-8:
            warnings.append("Available cash after sells is 0.00 USD; all BUY orders omitted to avoid unfunded purchases.")
            for c in buy_candidates:
                inst: InstrumentConfig = c["instrument"]  # type: ignore[assignment]
                broker: BrokerConfig = c["broker"]  # type: ignore[assignment]
                desired_g = float(c["qty"]) * float(c["price_usd"])  # type: ignore[arg-type]
                omitted.append(
                    OmittedTrade(
                        instrument_id=inst.instrument_id,
                        broker_id=broker.broker_id,
                        side="BUY",
                        current_weight=current_weights[inst.instrument_id],
                        target_weight=theoretical[inst.instrument_id],
                        abs_weight_diff=abs(theoretical[inst.instrument_id] - current_weights[inst.instrument_id]),
                        desired_notional_usd=desired_g,
                        reasons=["insufficient_cash_for_buy"],
                    )
                )
        else:
            full_outflow, evals = _buy_total_cost(buy_candidates, 1.0)
            chosen_scale = 1.0
            if full_outflow > available_cash_for_buys + 1e-8:
                lo, hi = 0.0, 1.0
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    mid_out, _ = _buy_total_cost(buy_candidates, mid)
                    if mid_out <= available_cash_for_buys:
                        lo = mid
                    else:
                        hi = mid
                chosen_scale = lo
                _, evals = _buy_total_cost(buy_candidates, chosen_scale)
                warnings.append(
                    f"Buy orders scaled by {chosen_scale:.4f} to respect available cash ({available_cash_for_buys:.2f} USD) after transaction costs."
                )

            for c, (q_exec, gross, comm, spread, op_notes) in zip(buy_candidates, evals):
                inst = c["instrument"]  # type: ignore[assignment]
                broker = c["broker"]  # type: ignore[assignment]
                px = float(c["price_usd"])  # type: ignore[arg-type]
                base_notes: List[str] = list(c["notes"]) + op_notes  # type: ignore[arg-type]
                if q_exec <= 0.0 or gross <= 1e-10:
                    omitted.append(
                        OmittedTrade(
                            instrument_id=inst.instrument_id,
                            broker_id=broker.broker_id,
                            side="BUY",
                            current_weight=current_weights[inst.instrument_id],
                            target_weight=theoretical[inst.instrument_id],
                            abs_weight_diff=abs(theoretical[inst.instrument_id] - current_weights[inst.instrument_id]),
                            desired_notional_usd=float(c["qty"]) * px,  # type: ignore[arg-type]
                            reasons=op_notes or ["insufficient_cash_after_costs"],
                        )
                    )
                    continue
                cash_out = gross + comm + spread
                if total_buy_cash_outflow + cash_out > available_cash_for_buys + 1e-6:
                    omitted.append(
                        OmittedTrade(
                            instrument_id=inst.instrument_id,
                            broker_id=broker.broker_id,
                            side="BUY",
                            current_weight=current_weights[inst.instrument_id],
                            target_weight=theoretical[inst.instrument_id],
                            abs_weight_diff=abs(theoretical[inst.instrument_id] - current_weights[inst.instrument_id]),
                            desired_notional_usd=gross,
                            reasons=["insufficient_cash_for_buy_and_fixed_min_fee"],
                        )
                    )
                    continue
                post_qty_by_inst[inst.instrument_id] += q_exec
                total_buy_gross += gross
                total_commission += comm
                total_spread += spread
                total_buy_cash_outflow += cash_out
                executed_orders.append(
                    ProposedOrder(
                        broker_id=broker.broker_id,
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
                        notes=base_notes,
                    )
                )

    ending_cash_usd = available_cash_for_buys - total_buy_cash_outflow
    if abs(ending_cash_usd) < 1e-9:
        ending_cash_usd = 0.0

    final_val_by_inst = {t: post_qty_by_inst[t] * float(prices_usd[t]) for t in ordered_tickers}
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
            f"Achievable post-rebalance weights deviate from theoretical targets by up to {max_dev:.2%} due to thresholds, transaction costs, or cash limits."
        )

    hard_conflicts: List[str] = []
    if cfg.enforce_hard_bounds_after_rebalance:
        hard_conflicts = _check_post_rebalance_hard_limits(
            achievable_weights,
            instruments_by_id,
            constraints_config,
        )

    feasible = len(hard_conflicts) == 0 and ending_cash_usd >= -1e-6
    status = "ok"
    if ending_cash_usd < -1e-6:
        status = "insufficient_cash"
    elif hard_conflicts:
        status = "infeasible_hard_limits"
        warnings.extend(hard_conflicts)

    two_way_turnover = (total_buy_gross + total_sell_gross) / total_portfolio_value_usd
    one_way_turnover = 0.5 * two_way_turnover
    total_tx_cost = total_commission + total_spread

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
        initial_portfolio_value_usd=total_portfolio_value_usd,
        final_portfolio_value_usd=final_portfolio_value_usd,
        initial_cash_usd=cfg.available_cash_usd,
        cash_flow_usd=cfg.cash_flow_usd,
        ending_cash_usd=ending_cash_usd,
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
