"""Manual contractual fixed-income valuation (YTM/IRR, dirty price, duration).

Manual contractual schedules are the sole source of truth for individual bonds,
corporate obligations (ONs), and T-bills. Published yields never replace manual
cashflows, and schedules or floating/indexed projections are never fabricated.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .errors import FixedIncomeValidationError
from .schemas import (
    INDIVIDUAL_DEBT_SUBTYPES,
    CashFlowEvent,
    FixedIncomeMetricsResult,
    FixedIncomeTerms,
    InstrumentConfig,
)


def day_count_fraction(start: date, end: date, convention: str = "ACT/365") -> float:
    """Compute exact year fraction between two dates under the chosen convention."""
    if end < start:
        raise FixedIncomeValidationError(
            f"End date ({end}) cannot precede start date ({start}) in day_count_fraction."
        )
    if start == end:
        return 0.0
    if convention == "ACT/365":
        return (end - start).days / 365.0
    if convention == "ACT/360":
        return (end - start).days / 360.0
    if convention == "30/360":
        d1 = min(start.day, 30)
        d2 = 30 if (end.day == 31 and d1 == 30) else end.day
        return (360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)) / 360.0
    if convention == "ACT/ACT":
        if start.year == end.year:
            denom = 366.0 if _is_leap(start.year) else 365.0
            return (end - start).days / denom
        y1_end = date(start.year + 1, 1, 1)
        y2_start = date(end.year, 1, 1)
        frac1 = (y1_end - start).days / (366.0 if _is_leap(start.year) else 365.0)
        frac2 = (end - y2_start).days / (366.0 if _is_leap(end.year) else 365.0)
        middle_years = end.year - start.year - 1
        return frac1 + float(middle_years) + frac2
    raise FixedIncomeValidationError(f"Unsupported day_count_convention '{convention}'.")


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def compute_dirty_price_per_reference(
    quoted_price: float,
    terms: FixedIncomeTerms,
    *,
    asset_id: Optional[str] = None,
) -> Tuple[float, float]:
    """Convert a clean or dirty quoted price into dirty price per ``reference_nominal``.

    Price convention definitions (all in ``terms.cashflow_currency``):
    - ``"per_100_original_nominal"``: quote per 100 units of original nominal
      (e.g., BYMA/IOL ``AL30D = 49.16`` -> ``49.16 * (reference_nominal / 100)`` per
      reference block, or ``0.4916`` per 1 original VN).
    - ``"per_100_residual_nominal"``: quote per 100 units of current residual nominal
      (e.g., ``amortization_factor = 0.6``, quote ``80.0`` -> ``80.0 * (residual_nominal / 100)``
      per reference block, or ``(80.0 / 100) * 0.6 = 0.48`` per 1 original VN).
    - ``"per_unit"``: quote for **1 reference bond unit of face value ``reference_nominal``**
      (example: if ``reference_nominal = 1000.0`` and ``quoted_price = 950.0``, 1 bond unit of
      1000 original VN costs ``950.0`` -> ``dirty_per_ref = 950.0`` and ``0.95`` per 1 original VN;
      if ``reference_nominal = 1.0``, ``quoted_price`` is directly per 1 original VN).

    Accrued interest conversion:
    - When ``price_type == "clean"``, ``accrued_interest`` is converted from
      ``terms.accrued_interest_basis`` (``"quote_convention"``, ``"per_reference_nominal"``,
      or ``"per_1_original_nominal"``) into the ``reference_nominal`` basis before summing.
    - When ``price_type == "dirty"``, accrued interest is NOT added again (no double counting).

    Returns:
        ``(dirty_price_per_reference, accrued_per_reference_added)``
    """
    if not isinstance(quoted_price, (int, float)) or not math.isfinite(float(quoted_price)) or float(quoted_price) <= 0.0:
        raise FixedIncomeValidationError(
            f"Quoted price must be finite and strictly positive (got {quoted_price}).",
            asset_id=asset_id,
        )
    qp = float(quoted_price)

    def _scale_quote_to_reference(val: float) -> float:
        if terms.price_convention == "per_100_original_nominal":
            return val * (terms.reference_nominal / 100.0)
        if terms.price_convention == "per_100_residual_nominal":
            return val * (terms.residual_nominal / 100.0)
        if terms.price_convention == "per_unit":
            return val
        raise FixedIncomeValidationError(
            f"Unsupported price_convention '{terms.price_convention}'.",
            asset_id=asset_id,
        )

    def _accrued_to_reference(acc_val: float) -> float:
        basis = terms.accrued_interest_basis
        if basis == "quote_convention":
            return _scale_quote_to_reference(acc_val)
        if basis == "per_reference_nominal":
            return acc_val
        if basis == "per_1_original_nominal":
            return acc_val * terms.reference_nominal
        raise FixedIncomeValidationError(
            f"Unsupported accrued_interest_basis '{basis}'.",
            asset_id=asset_id,
        )

    base_per_ref = _scale_quote_to_reference(qp)

    if terms.price_type == "clean":
        if terms.accrued_interest is None:
            if terms.coupon_structure == "zero_coupon":
                accrued_per_ref = 0.0
            else:
                raise FixedIncomeValidationError(
                    "Clean price requires explicit 'accrued_interest' (cannot be omitted for coupon-bearing debt).",
                    asset_id=asset_id,
                )
        else:
            accrued_per_ref = _accrued_to_reference(float(terms.accrued_interest))
        dirty_per_ref = base_per_ref + accrued_per_ref
    else:
        accrued_per_ref = 0.0 if terms.accrued_interest is None else _accrued_to_reference(float(terms.accrued_interest))
        dirty_per_ref = base_per_ref

    if not math.isfinite(dirty_per_ref) or dirty_per_ref <= 0.0:
        raise FixedIncomeValidationError(
            f"Computed dirty price per reference nominal is invalid ({dirty_per_ref}).",
            asset_id=asset_id,
        )
    return dirty_per_ref, accrued_per_ref


def normalize_execution_price_to_usd_per_unit(
    instrument: InstrumentConfig,
    quoted_price_usd: float,
    *,
    quote_unit_override: Optional[str] = None,
    quoted_price_currency: str = "USD",
    fx_to_usd: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Shared canonical normalization from a quoted market price to **dirty USD per 1 quantity unit**.

    Quantity unit contract (shared across valuation, YTM consistency, and rebalancing):
    - Individual debt (``sovereign_bond``, ``corporate_bond_on``, ``tbill``):
      1 quantity unit = **1 unit of original nominal (1 VN original)**.
      * ``per_100_original_nominal``: ``dirty_usd / 100.0``
      * ``per_100_residual_nominal``: ``(dirty_usd / 100.0) * amortization_factor``
      * ``per_unit`` (1 bond of ``reference_nominal``): ``dirty_usd / reference_nominal``
      All three equal ``dirty_price_usd_per_reference / reference_nominal``.
    - All other instruments (equities, ETFs including ``fixed_income_etf``, gold, crypto):
      1 quantity unit = **1 share / unit** (price already USD per unit).
    """
    iid = instrument.instrument_id
    if not isinstance(quoted_price_usd, (int, float)) or not math.isfinite(float(quoted_price_usd)) or float(quoted_price_usd) <= 0.0:
        raise FixedIncomeValidationError(
            f"Quoted price for '{iid}' must be finite and strictly positive (got {quoted_price_usd}).",
            asset_id=iid,
        )
    qp = float(quoted_price_usd)
    q_ccy = str(quoted_price_currency or "USD").strip().upper()

    if instrument.asset_subtype not in INDIVIDUAL_DEBT_SUBTYPES:
        if q_ccy != "USD":
            raise FixedIncomeValidationError(
                f"Execution price for '{iid}' must be in USD (got '{q_ccy}').",
                asset_id=iid,
            )
        return qp, {
            "instrument_id": iid,
            "quantity_unit": "shares_or_units",
            "quote_unit": "per_unit",
            "price_type": "market",
            "quoted_price_usd": qp,
            "dirty_price_usd_per_unit": qp,
            "reference_nominal": 1.0,
            "amortization_factor": 1.0,
            "warnings": [],
        }

    terms = instrument.fixed_income_terms
    warnings_list: List[str] = []

    if terms is not None:
        if quote_unit_override is not None and quote_unit_override != terms.price_convention:
            raise FixedIncomeValidationError(
                f"Quote unit conflict for '{iid}': provider/override reports '{quote_unit_override}' "
                f"but FixedIncomeTerms declares price_convention='{terms.price_convention}'.",
                asset_id=iid,
            )
        if instrument.quote_unit is not None and instrument.quote_unit != terms.price_convention:
            raise FixedIncomeValidationError(
                f"Instrument '{iid}' has quote_unit='{instrument.quote_unit}' conflicting with "
                f"FixedIncomeTerms.price_convention='{terms.price_convention}'.",
                asset_id=iid,
            )

        if q_ccy == "USD" and terms.cashflow_currency == "USD":
            dirty_per_ref, accrued_ref = compute_dirty_price_per_reference(qp, terms, asset_id=iid)
            dirty_usd_per_ref = dirty_per_ref
        elif q_ccy == terms.cashflow_currency:
            eff_fx = fx_to_usd if fx_to_usd is not None else terms.cashflow_fx_to_usd
            if eff_fx is None or not math.isfinite(float(eff_fx)) or float(eff_fx) <= 0.0:
                raise FixedIncomeValidationError(
                    f"Instrument '{iid}' is denominated in '{terms.cashflow_currency}' without a valid positive fx_to_usd.",
                    asset_id=iid,
                )
            dirty_per_ref, accrued_ref = compute_dirty_price_per_reference(qp, terms, asset_id=iid)
            dirty_usd_per_ref = dirty_per_ref * float(eff_fx)
        elif q_ccy == "USD" and terms.cashflow_currency != "USD":
            if terms.price_type == "clean" and terms.accrued_interest not in (None, 0.0):
                eff_fx = fx_to_usd if fx_to_usd is not None else terms.cashflow_fx_to_usd
                if eff_fx is None or not math.isfinite(float(eff_fx)) or float(eff_fx) <= 0.0:
                    raise FixedIncomeValidationError(
                        f"Clean USD quote for non-USD debt '{iid}' ({terms.cashflow_currency}) requires fx_to_usd to convert accrued interest.",
                        asset_id=iid,
                    )
                qp_in_cf_ccy = qp / float(eff_fx)
                dirty_per_ref, accrued_ref = compute_dirty_price_per_reference(qp_in_cf_ccy, terms, asset_id=iid)
                dirty_usd_per_ref = dirty_per_ref * float(eff_fx)
            else:
                dirty_per_ref, accrued_ref = compute_dirty_price_per_reference(qp, terms, asset_id=iid)
                dirty_usd_per_ref = dirty_per_ref
        else:
            raise FixedIncomeValidationError(
                f"Quoted price currency '{q_ccy}' does not match USD or cashflow_currency '{terms.cashflow_currency}' for '{iid}'.",
                asset_id=iid,
            )

        dirty_usd_per_vn = dirty_usd_per_ref / float(terms.reference_nominal)
        return float(dirty_usd_per_vn), {
            "instrument_id": iid,
            "quantity_unit": "original_nominal_units",
            "quote_unit": terms.price_convention,
            "price_type": terms.price_type,
            "quoted_price_usd": qp if q_ccy == "USD" else None,
            "dirty_price_usd_per_reference": float(dirty_usd_per_ref),
            "dirty_price_usd_per_unit": float(dirty_usd_per_vn),
            "accrued_per_reference": float(accrued_ref),
            "reference_nominal": float(terms.reference_nominal),
            "residual_nominal": float(terms.residual_nominal),
            "amortization_factor": float(terms.amortization_factor),
            "warnings": warnings_list,
        }

    # Individual debt WITHOUT FixedIncomeTerms
    if q_ccy != "USD":
        raise FixedIncomeValidationError(
            f"Execution price for '{iid}' must be in USD (got '{q_ccy}').",
            asset_id=iid,
        )
    eff_unit = quote_unit_override or instrument.quote_unit
    if eff_unit is None and instrument.quote_source in ("iol_bonos", "iol_ons"):
        eff_unit = "per_100_original_nominal"

    if eff_unit == "per_100_original_nominal":
        dirty_usd_per_vn = qp / 100.0
    elif eff_unit == "per_100_residual_nominal":
        raise FixedIncomeValidationError(
            f"Individual bond '{iid}' uses quote_unit='per_100_residual_nominal' without FixedIncomeTerms; "
            "cannot determine amortization_factor / residual_nominal safely.",
            asset_id=iid,
        )
    elif eff_unit == "per_unit":
        dirty_usd_per_vn = qp
    else:
        dirty_usd_per_vn = qp
        eff_unit = "per_unit"
        warnings_list.append(
            f"Individual debt '{iid}' lacks FixedIncomeTerms and explicit quote_unit; treating price ({qp}) as dirty USD per 1 original nominal unit."
        )

    return float(dirty_usd_per_vn), {
        "instrument_id": iid,
        "quantity_unit": "original_nominal_units",
        "quote_unit": eff_unit,
        "price_type": "dirty_assumed",
        "quoted_price_usd": qp,
        "dirty_price_usd_per_unit": float(dirty_usd_per_vn),
        "reference_nominal": 1.0,
        "amortization_factor": 1.0,
        "warnings": warnings_list,
    }


def expand_and_validate_cashflows(
    terms: FixedIncomeTerms,
    *,
    asset_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate manual contractual cashflows and expand rate-based coupons explicitly."""
    warnings: List[str] = []

    if not terms.cashflows:
        raise FixedIncomeValidationError(
            "FixedIncomeTerms requires an explicit manual cashflow schedule; schedules are never fabricated.",
            asset_id=asset_id,
        )

    if terms.coupon_structure in ("floating", "indexed"):
        missing_monetary = [cf for cf in terms.cashflows if cf.payment_date > terms.settlement_date and cf.coupon_amount is None and cf.coupon_rate is None]
        if missing_monetary or not terms.index_assumptions:
            raise FixedIncomeValidationError(
                f"Instrument with coupon_structure='{terms.coupon_structure}' lacks explicit index/rate assumptions or monetary cashflows; cannot value without fabricating projections.",
                asset_id=asset_id,
                code="not_evaluable_under_available_method",
            )
        warnings.append(
            f"Floating/indexed instrument evaluated strictly under user-supplied assumptions: {terms.index_assumptions}."
        )

    sorted_events = sorted(terms.cashflows, key=lambda e: e.payment_date)
    future_events = [e for e in sorted_events if e.payment_date > terms.settlement_date]
    if not future_events:
        raise FixedIncomeValidationError(
            f"No future contractual cashflows remain after settlement_date ({terms.settlement_date}).",
            asset_id=asset_id,
        )

    last_cf_date = future_events[-1].payment_date
    if last_cf_date > terms.maturity_date:
        raise FixedIncomeValidationError(
            f"Cashflow date ({last_cf_date}) exceeds maturity_date ({terms.maturity_date}).",
            asset_id=asset_id,
        )

    grouped: Dict[date, List[CashFlowEvent]] = {}
    for ev in future_events:
        grouped.setdefault(ev.payment_date, []).append(ev)

    outstanding_principal = float(terms.residual_nominal)
    prev_date: Optional[date] = None
    past_dates = [e.payment_date for e in sorted_events if e.payment_date <= terms.settlement_date]
    if past_dates:
        prev_date = past_dates[-1]

    resolved_schedule: List[Dict[str, Any]] = []
    total_amort = 0.0
    total_coupon = 0.0

    for pay_date in sorted(grouped.keys()):
        day_events = grouped[pay_date]
        day_coupon = 0.0
        day_amort = 0.0

        for ev in day_events:
            if ev.coupon_amount is not None:
                if terms.coupon_structure == "zero_coupon" and ev.coupon_amount > 0.0:
                    raise FixedIncomeValidationError(
                        f"Zero-coupon instrument cannot have positive coupon_amount ({ev.coupon_amount}) on {pay_date}.",
                        asset_id=asset_id,
                    )
                day_coupon += float(ev.coupon_amount)
            elif ev.coupon_rate is not None:
                if terms.coupon_structure == "zero_coupon" and ev.coupon_rate > 0.0:
                    raise FixedIncomeValidationError(
                        f"Zero-coupon instrument cannot have positive coupon_rate ({ev.coupon_rate}) on {pay_date}.",
                        asset_id=asset_id,
                    )
                period_start = ev.period_start_date or prev_date
                if period_start is not None:
                    if terms.day_count_convention is None:
                        raise FixedIncomeValidationError(
                            "Rate-based coupon with accrual dates requires explicit 'day_count_convention'.",
                            asset_id=asset_id,
                        )
                    yf = day_count_fraction(period_start, pay_date, terms.day_count_convention)
                    if terms.coupon_frequency is not None:
                        expected_period = 1.0 / float(terms.coupon_frequency)
                        # Validate schedule spacing matches declared coupon_frequency (allowing reasonable stub/calendar tolerance)
                        if abs(yf - expected_period) > 0.35 * expected_period:
                            raise FixedIncomeValidationError(
                                f"Coupon accrual fraction ({yf:.4f}y from {period_start} to {pay_date}) is inconsistent with "
                                f"declared coupon_frequency={terms.coupon_frequency} (expected ~{expected_period:.4f}y). "
                                "Correct coupon_frequency or supply explicit monetary coupon_amount for irregular stubs.",
                                asset_id=asset_id,
                            )
                    day_coupon += outstanding_principal * float(ev.coupon_rate) * yf
                elif terms.coupon_frequency is not None:
                    # Periodic rate = annual coupon_rate / coupon_frequency when period_start_date is omitted
                    yf = 1.0 / float(terms.coupon_frequency)
                    day_coupon += outstanding_principal * float(ev.coupon_rate) * yf
                else:
                    raise FixedIncomeValidationError(
                        f"Rate-based coupon on {pay_date} requires either 'period_start_date' (with 'day_count_convention') "
                        "or explicit 'coupon_frequency' so monetary coupon can be computed deterministically.",
                        asset_id=asset_id,
                    )
            else:
                if terms.coupon_structure != "zero_coupon" and ev.amortization_amount == 0.0:
                    raise FixedIncomeValidationError(
                        f"CashFlowEvent on {pay_date} specifies neither coupon nor amortization.",
                        asset_id=asset_id,
                    )

            day_amort += float(ev.amortization_amount)

        if day_amort > outstanding_principal + 1e-6 * terms.reference_nominal:
            raise FixedIncomeValidationError(
                f"Amortization ({day_amort}) on {pay_date} exceeds remaining principal balance ({outstanding_principal}).",
                asset_id=asset_id,
            )

        outstanding_principal = max(0.0, outstanding_principal - day_amort)
        total_amort += day_amort
        total_coupon += day_coupon
        t_years = day_count_fraction(terms.settlement_date, pay_date, terms.day_count_convention or "ACT/365")

        resolved_schedule.append({
            "payment_date": pay_date.isoformat(),
            "time_years": t_years,
            "coupon_amount": day_coupon,
            "amortization_amount": day_amort,
            "total_cashflow": day_coupon + day_amort,
            "remaining_principal_after": outstanding_principal,
        })
        prev_date = pay_date

    tol = max(1e-4 * terms.reference_nominal, 1e-6)
    if not math.isclose(total_amort, terms.residual_nominal, abs_tol=tol):
        raise FixedIncomeValidationError(
            f"Sum of future amortizations ({total_amort:.6f}) does not reconcile with residual_nominal ({terms.residual_nominal:.6f}).",
            asset_id=asset_id,
            details={"total_future_amortizations": total_amort, "residual_nominal": terms.residual_nominal},
        )

    amort_dates = [r["payment_date"] for r in resolved_schedule if r["amortization_amount"] > 0.0]
    if not amort_dates or amort_dates[-1] != terms.maturity_date.isoformat():
        raise FixedIncomeValidationError(
            f"Final principal amortization date ({amort_dates[-1] if amort_dates else None}) must match maturity_date ({terms.maturity_date.isoformat()}).",
            asset_id=asset_id,
        )

    return resolved_schedule, warnings


def solve_xirr_with_root_check(
    dirty_price: float,
    dated_cashflows: Sequence[Tuple[float, float]],
    *,
    y_min: float = -0.95,
    y_max: float = 10.0,
    grid_points: int = 400,
) -> Tuple[Optional[float], int, str]:
    """Solve ``-dirty_price + sum(cf_k / (1 + y)^t_k) = 0`` and detect root multiplicity."""
    if not math.isfinite(dirty_price) or dirty_price <= 0.0 or not dated_cashflows:
        return None, 0, "insufficient_data"

    times = np.array([float(t) for t, _ in dated_cashflows], dtype=float)
    cfs = np.array([float(c) for _, c in dated_cashflows], dtype=float)
    if np.any(times <= 0.0) or not np.all(np.isfinite(cfs)):
        return None, 0, "insufficient_data"

    def npv(y: float) -> float:
        return float(-dirty_price + np.sum(cfs / np.power(1.0 + y, times)))

    grid = np.unique(np.concatenate([
        np.linspace(y_min, -0.20, 60),
        np.linspace(-0.20, 0.50, 180),
        np.linspace(0.50, 2.00, 80),
        np.linspace(2.00, y_max, max(20, grid_points - 320)),
    ]))

    vals = np.array([npv(float(y)) for y in grid], dtype=float)
    brackets: List[Tuple[float, float]] = []
    exact_roots: List[float] = []

    for i in range(len(grid)):
        if abs(vals[i]) < 1e-12:
            exact_roots.append(float(grid[i]))
        elif i > 0 and math.isfinite(vals[i - 1]) and math.isfinite(vals[i]):
            if vals[i - 1] * vals[i] < 0.0:
                brackets.append((float(grid[i - 1]), float(grid[i])))

    roots: List[float] = list(exact_roots)
    for lo, hi in brackets:
        try:
            r = float(brentq(npv, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200))
            if not any(abs(r - existing) < 1e-6 for existing in roots):
                roots.append(r)
        except ValueError:
            continue

    if len(roots) == 0:
        return None, 0, "no_root"
    if len(roots) > 1:
        return None, len(roots), "multiple_roots"
    return roots[0], 1, "ok"


def evaluate_fixed_income_instrument(
    instrument_id: str,
    terms: FixedIncomeTerms,
    quoted_price: Optional[float],
    *,
    quoted_price_currency: Optional[str] = None,
    fx_to_usd: Optional[float] = None,
    quote_source_used: Optional[str] = None,
    valuation_price_date: Optional[str] = None,
    extra_warnings: Optional[List[str]] = None,
    not_evaluable_reason: Optional[str] = None,
) -> FixedIncomeMetricsResult:
    """Compute contractual dirty price, YTM/IRR, duration, and WAL from manual terms."""
    iid = instrument_id.strip().upper()
    warnings: List[str] = list(extra_warnings or [])

    if terms.coupon_structure in ("floating", "indexed"):
        missing_monetary = [
            cf for cf in terms.cashflows
            if cf.payment_date > terms.settlement_date and cf.coupon_amount is None and cf.coupon_rate is None
        ]
        if missing_monetary or not terms.index_assumptions:
            return FixedIncomeMetricsResult(
                instrument_id=iid,
                status="not_evaluable_under_available_method",
                cashflow_currency=terms.cashflow_currency,
                settlement_date=terms.settlement_date,
                maturity_date=terms.maturity_date,
                reference_nominal=terms.reference_nominal,
                residual_nominal=terms.residual_nominal,
                amortization_factor=terms.amortization_factor,
                price_convention=terms.price_convention,
                price_type=terms.price_type,
                quoted_price=quoted_price,
                quote_source_used=quote_source_used,
                valuation_price_date=valuation_price_date,
                warnings=warnings + [
                    f"Instrument '{iid}' has coupon_structure='{terms.coupon_structure}' without explicit index/rate projections; marked not evaluable under fixed-schedule YTM."
                ],
                reason="Missing explicit index/rate assumptions for floating or indexed instrument.",
            )

    schedule, sched_warnings = expand_and_validate_cashflows(terms, asset_id=iid)
    warnings.extend(sched_warnings)

    total_coupons = sum(r["coupon_amount"] for r in schedule)
    total_amorts = sum(r["amortization_amount"] for r in schedule)
    wal = (
        sum(r["time_years"] * r["amortization_amount"] for r in schedule) / total_amorts
        if total_amorts > 0
        else None
    )

    if not_evaluable_reason is not None:
        return FixedIncomeMetricsResult(
            instrument_id=iid,
            status="not_evaluable_under_available_method",
            cashflow_currency=terms.cashflow_currency,
            settlement_date=terms.settlement_date,
            maturity_date=terms.maturity_date,
            reference_nominal=terms.reference_nominal,
            residual_nominal=terms.residual_nominal,
            amortization_factor=terms.amortization_factor,
            price_convention=terms.price_convention,
            price_type=terms.price_type,
            quoted_price=None,
            quote_source_used=quote_source_used,
            valuation_price_date=valuation_price_date,
            weighted_average_life_years=wal,
            future_cashflows=schedule,
            total_future_coupons=total_coupons,
            total_future_amortizations=total_amorts,
            warnings=warnings + [not_evaluable_reason],
            reason=not_evaluable_reason,
        )

    if quoted_price is None:
        return FixedIncomeMetricsResult(
            instrument_id=iid,
            status="insufficient_data",
            cashflow_currency=terms.cashflow_currency,
            settlement_date=terms.settlement_date,
            maturity_date=terms.maturity_date,
            reference_nominal=terms.reference_nominal,
            residual_nominal=terms.residual_nominal,
            amortization_factor=terms.amortization_factor,
            price_convention=terms.price_convention,
            price_type=terms.price_type,
            quoted_price=None,
            quote_source_used=quote_source_used,
            valuation_price_date=valuation_price_date,
            weighted_average_life_years=wal,
            future_cashflows=schedule,
            total_future_coupons=total_coupons,
            total_future_amortizations=total_amorts,
            warnings=warnings + ["No valid quoted price available; cannot compute YTM."],
            reason="Missing quoted price.",
        )

    if quoted_price_currency and quoted_price_currency.upper() != terms.cashflow_currency:
        raise FixedIncomeValidationError(
            f"Quoted price currency ({quoted_price_currency}) does not match cashflow_currency ({terms.cashflow_currency}) for '{iid}'. Convert to a common currency first.",
            asset_id=iid,
        )

    dirty_per_ref, accrued_used = compute_dirty_price_per_reference(quoted_price, terms, asset_id=iid)

    effective_fx = fx_to_usd if fx_to_usd is not None else terms.cashflow_fx_to_usd
    if terms.cashflow_currency == "USD":
        dirty_usd = dirty_per_ref
    elif effective_fx is not None and effective_fx > 0:
        dirty_usd = dirty_per_ref * effective_fx
    else:
        dirty_usd = None
        warnings.append(
            f"Cashflows are in {terms.cashflow_currency} and no FX rate to USD was supplied; dirty_price_usd unavailable."
        )
    dirty_usd_per_vn = None if dirty_usd is None else dirty_usd / float(terms.reference_nominal)

    dated_cfs = [(r["time_years"], r["total_cashflow"]) for r in schedule]
    ytm, root_count, status = solve_xirr_with_root_check(dirty_per_ref, dated_cfs)

    mac_dur: Optional[float] = None
    mod_dur: Optional[float] = None
    reason: Optional[str] = None

    if status == "ok" and ytm is not None:
        pv_weights = [
            (r["time_years"], r["total_cashflow"] / ((1.0 + ytm) ** r["time_years"]))
            for r in schedule
        ]
        pv_total = sum(pv for _, pv in pv_weights)
        if pv_total > 0:
            mac_dur = sum(t * pv for t, pv in pv_weights) / pv_total
            mod_dur = mac_dur / (1.0 + ytm) if (1.0 + ytm) > 0 else None
        warnings.append(
            "Contractual YTM assumes timely contractual payments, holding to maturity, and reinvestment at YTM; it is not a guaranteed 1-year expected return."
        )
    elif status == "multiple_roots":
        reason = f"Cashflow profile produced {root_count} IRR roots; YTM is ambiguous."
        warnings.append(reason)
    elif status == "no_root":
        reason = "No real YTM root found within [-95%, +1000%] for the supplied price and cashflows."
        warnings.append(reason)
    else:
        reason = "Insufficient data to compute YTM."
        warnings.append(reason)

    return FixedIncomeMetricsResult(
        instrument_id=iid,
        status=status,
        cashflow_currency=terms.cashflow_currency,
        settlement_date=terms.settlement_date,
        maturity_date=terms.maturity_date,
        reference_nominal=terms.reference_nominal,
        residual_nominal=terms.residual_nominal,
        amortization_factor=terms.amortization_factor,
        price_convention=terms.price_convention,
        price_type=terms.price_type,
        quoted_price=float(quoted_price),
        accrued_interest=accrued_used,
        dirty_price_per_reference=dirty_per_ref,
        dirty_price_usd=dirty_usd,
        dirty_price_usd_per_original_nominal=dirty_usd_per_vn,
        quote_source_used=quote_source_used,
        valuation_price_date=valuation_price_date,
        ytm_annual=ytm,
        macaulay_duration_years=mac_dur,
        modified_duration_years=mod_dur,
        weighted_average_life_years=wal,
        future_cashflows=schedule,
        total_future_coupons=total_coupons,
        total_future_amortizations=total_amorts,
        root_count_detected=root_count,
        warnings=warnings,
        reason=reason,
    )


def build_fixed_income_total_return_index(
    historical_dirty_prices_usd: pd.Series,
    paid_cashflows_usd: Optional[pd.Series] = None,
) -> pd.Series:
    """Build a total-return wealth index for a bond from real historical dirty prices
    and actual paid cashflows (coupons + amortizations) per reference nominal."""
    if historical_dirty_prices_usd is None or historical_dirty_prices_usd.empty:
        raise FixedIncomeValidationError("Cannot build bond total return index from an empty price series.")

    s = historical_dirty_prices_usd.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index(ascending=True)
    s = s[~s.index.duplicated(keep="last")]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0.0]

    if len(s) < 2:
        raise FixedIncomeValidationError("At least 2 valid historical dirty prices are required for bond total return.")

    cf_aligned = pd.Series(0.0, index=s.index, dtype=float)
    if paid_cashflows_usd is not None and not paid_cashflows_usd.empty:
        cf = paid_cashflows_usd.copy()
        cf.index = pd.to_datetime(cf.index)
        cf = cf.sort_index(ascending=True)
        idx_arr = s.index
        for cf_dt, cf_amt in cf.items():
            if not math.isfinite(float(cf_amt)) or float(cf_amt) == 0.0:
                continue
            pos = idx_arr.searchsorted(cf_dt, side="left")
            if 1 <= pos < len(idx_arr):
                cf_aligned.iloc[pos] += float(cf_amt)

    prev_p = s.shift(1)
    period_ret = (s + cf_aligned - prev_p) / prev_p
    period_ret.iloc[0] = 0.0
    wealth_index = (1.0 + period_ret).cumprod() * 100.0
    wealth_index.name = s.name
    return wealth_index
