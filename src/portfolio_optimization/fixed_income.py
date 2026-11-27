"""Manual contractual fixed-income valuation (YTM/IRR, dirty price, duration).

Manual contractual schedules are the sole source of truth for individual bonds,
corporate obligations (ONs), and T-bills. Published yields never replace manual
cashflows, and schedules or floating/indexed projections are never fabricated.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field, replace
import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .errors import DataQualityError, FixedIncomeValidationError
from .schemas import (
    INDIVIDUAL_DEBT_SUBTYPES,
    CashFlowEvent,
    FixedIncomeMetricsResult,
    FixedIncomeTerms,
    InstrumentConfig,
)

WEEKS_PER_YEAR_EFFECTIVE = 52.0


def effective_detachment_date(ev: CashFlowEvent) -> date:
    """Return the economic detachment date (``ex_date`` if set, else ``payment_date``)."""
    return ev.ex_date if ev.ex_date is not None else ev.payment_date


def annual_rate_to_weekly_effective(annual_rate: float, weeks_per_year: float = WEEKS_PER_YEAR_EFFECTIVE) -> float:
    """Convert an annual effective rate to its equivalent weekly effective compounded rate."""
    r = float(annual_rate)
    if not math.isfinite(r) or r <= -1.0:
        raise FixedIncomeValidationError(
            f"Annual rate ({annual_rate}) must be finite and > -1.0 for effective weekly conversion."
        )
    return float((1.0 + r) ** (1.0 / float(weeks_per_year)) - 1.0)


def resolve_fixed_income_terms_for_valuation(
    terms: FixedIncomeTerms,
    *,
    valuation_date: Optional[date] = None,
    asset_id: Optional[str] = None,
) -> FixedIncomeTerms:
    """Resolve ``settlement_date`` and ``residual_nominal`` consistently for a valuation date.

    When ``terms.settlement_date`` is already supplied, it is preserved. When omitted
    (``None``), ``valuation_date`` (typically ``as_of_date`` or the market quote date)
    is used and ``residual_nominal`` is inferred from ``reference_nominal - past_amortizations``
    if the supplied schedule reconciles total amortizations with ``reference_nominal``.
    """
    settle = terms.settlement_date if terms.settlement_date is not None else valuation_date
    if settle is None:
        raise FixedIncomeValidationError(
            "FixedIncomeTerms requires either explicit 'settlement_date' or a valuation/as_of date.",
            asset_id=asset_id,
        )
    if settle > terms.maturity_date:
        raise FixedIncomeValidationError(
            f"Settlement/valuation date ({settle}) cannot be after maturity_date ({terms.maturity_date}).",
            asset_id=asset_id,
        )
    if terms.issue_date is not None and settle < terms.issue_date:
        raise FixedIncomeValidationError(
            f"Settlement/valuation date ({settle}) cannot precede issue_date ({terms.issue_date}).",
            asset_id=asset_id,
        )

    res_nom = terms.residual_nominal
    amort_fac = terms.amortization_factor
    if res_nom is None or amort_fac is None:
        ref_nom = float(terms.reference_nominal)
        tol = max(1e-4 * ref_nom, 1e-6)
        total_all_amorts = sum(ev.amortization_amount for ev in terms.cashflows)
        past_amorts = sum(
            ev.amortization_amount for ev in terms.cashflows if effective_detachment_date(ev) <= settle
        )
        if past_amorts > 0.0 and math.isclose(total_all_amorts, ref_nom, abs_tol=tol):
            res_nom = ref_nom - past_amorts
        else:
            res_nom = ref_nom
        if res_nom <= 0.0:
            raise FixedIncomeValidationError(
                f"No remaining residual_nominal after past amortizations on settlement_date ({settle}).",
                asset_id=asset_id,
            )
        amort_fac = res_nom / ref_nom

    if (
        terms.settlement_date == settle
        and terms.residual_nominal == res_nom
        and terms.amortization_factor == amort_fac
    ):
        return terms
    return replace(
        terms,
        settlement_date=settle,
        residual_nominal=float(res_nom),
        amortization_factor=float(amort_fac),
    )


def day_count_fraction(start: date, end: date, convention: str = "ACT/365") -> float:
    """Compute exact year fraction between two dates under the chosen convention."""
    if end < start:
        raise FixedIncomeValidationError(
            f"End date ({end}) cannot precede start date ({start}) in day_count_fraction."
        )
    if start == end:
        return 0.0
    if convention in ("ACT/365", "ACT/365F"):
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
    valuation_date: Optional[date] = None,
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
    if terms.price_convention == "per_100_residual_nominal" and terms.residual_nominal is None:
        terms = resolve_fixed_income_terms_for_valuation(
            terms,
            valuation_date=valuation_date,
            asset_id=asset_id,
        )
    qp = float(quoted_price)

    def _scale_quote_to_reference(val: float) -> float:
        if terms.price_convention == "per_100_original_nominal":
            return val * (terms.reference_nominal / 100.0)
        if terms.price_convention == "per_100_residual_nominal":
            assert terms.residual_nominal is not None
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
    valuation_date: Optional[date] = None,
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
        if (terms.settlement_date is None or terms.residual_nominal is None) and (
            valuation_date is not None or terms.price_convention == "per_100_residual_nominal"
        ):
            terms = resolve_fixed_income_terms_for_valuation(
                terms,
                valuation_date=valuation_date,
                asset_id=iid,
            )
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
            dirty_per_ref, accrued_ref = compute_dirty_price_per_reference(
                qp, terms, asset_id=iid, valuation_date=valuation_date
            )
            dirty_usd_per_ref = dirty_per_ref
        elif q_ccy == terms.cashflow_currency:
            eff_fx = fx_to_usd if fx_to_usd is not None else terms.cashflow_fx_to_usd
            if eff_fx is None or not math.isfinite(float(eff_fx)) or float(eff_fx) <= 0.0:
                raise FixedIncomeValidationError(
                    f"Instrument '{iid}' is denominated in '{terms.cashflow_currency}' without a valid positive fx_to_usd.",
                    asset_id=iid,
                )
            dirty_per_ref, accrued_ref = compute_dirty_price_per_reference(
                qp, terms, asset_id=iid, valuation_date=valuation_date
            )
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
                dirty_per_ref, accrued_ref = compute_dirty_price_per_reference(
                    qp_in_cf_ccy, terms, asset_id=iid, valuation_date=valuation_date
                )
                dirty_usd_per_ref = dirty_per_ref * float(eff_fx)
            else:
                dirty_per_ref, accrued_ref = compute_dirty_price_per_reference(
                    qp, terms, asset_id=iid, valuation_date=valuation_date
                )
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
            "residual_nominal": None if terms.residual_nominal is None else float(terms.residual_nominal),
            "amortization_factor": None if terms.amortization_factor is None else float(terms.amortization_factor),
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
    valuation_date: Optional[date] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate manual contractual cashflows and expand rate-based coupons explicitly."""
    warnings: List[str] = []
    terms = resolve_fixed_income_terms_for_valuation(
        terms,
        valuation_date=valuation_date,
        asset_id=asset_id,
    )
    assert terms.settlement_date is not None
    assert terms.residual_nominal is not None

    if not terms.cashflows:
        raise FixedIncomeValidationError(
            "FixedIncomeTerms requires an explicit manual cashflow schedule; schedules are never fabricated.",
            asset_id=asset_id,
        )

    sorted_events = sorted(terms.cashflows, key=lambda e: (effective_detachment_date(e), e.payment_date))
    future_events = [e for e in sorted_events if effective_detachment_date(e) > terms.settlement_date]
    detached_pending = [
        e for e in sorted_events
        if e.ex_date is not None and e.ex_date <= terms.settlement_date < e.payment_date
    ]
    if detached_pending:
        warnings.append(
            f"Excluded {len(detached_pending)} cashflow event(s) with ex_date <= settlement_date "
            f"({terms.settlement_date.isoformat()}) < payment_date from future YTM inflows "
            "(already detached from market quote; prevents double counting)."
        )

    if terms.coupon_structure in ("floating", "indexed"):
        missing_monetary = [cf for cf in future_events if cf.coupon_amount is None and cf.coupon_rate is None]
        if missing_monetary or not terms.index_assumptions:
            raise FixedIncomeValidationError(
                f"Instrument with coupon_structure='{terms.coupon_structure}' lacks explicit index/rate assumptions or monetary cashflows; cannot value without fabricating projections.",
                asset_id=asset_id,
                code="not_evaluable_under_available_method",
            )
        warnings.append(
            f"Floating/indexed instrument evaluated strictly under user-supplied assumptions: {terms.index_assumptions}."
        )

    if not future_events:
        raise FixedIncomeValidationError(
            f"No future contractual cashflows remain after settlement_date ({terms.settlement_date}).",
            asset_id=asset_id,
        )

    last_cf_date = max(e.payment_date for e in future_events)
    if last_cf_date > terms.maturity_date:
        raise FixedIncomeValidationError(
            f"Cashflow date ({last_cf_date}) exceeds maturity_date ({terms.maturity_date}).",
            asset_id=asset_id,
        )

    grouped: Dict[date, List[CashFlowEvent]] = {}
    for ev in future_events:
        grouped.setdefault(ev.payment_date, []).append(ev)

    outstanding_principal = float(terms.residual_nominal)
    past_dates = [e.payment_date for e in sorted_events if effective_detachment_date(e) <= terms.settlement_date]
    prev_date: Optional[date] = max(past_dates) if past_dates else None

    resolved_schedule: List[Dict[str, Any]] = []
    total_amort = 0.0
    total_coupon = 0.0

    for pay_date in sorted(grouped.keys()):
        day_events = grouped[pay_date]
        day_coupon = 0.0
        day_amort = 0.0
        day_ex_dates = sorted({e.ex_date.isoformat() for e in day_events if e.ex_date is not None})

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
            "ex_date": day_ex_dates[0] if len(day_ex_dates) == 1 else (day_ex_dates or None),
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


def _parse_optional_valuation_date(
    valuation_date: Optional[date],
    valuation_price_date: Optional[str],
) -> Optional[date]:
    if valuation_date is not None:
        return valuation_date
    if valuation_price_date:
        text = str(valuation_price_date).strip()
        if len(text) >= 10:
            try:
                return date.fromisoformat(text[:10])
            except ValueError:
                pass
    return None


def evaluate_fixed_income_instrument(
    instrument_id: str,
    terms: FixedIncomeTerms,
    quoted_price: Optional[float],
    *,
    quoted_price_currency: Optional[str] = None,
    fx_to_usd: Optional[float] = None,
    quote_source_used: Optional[str] = None,
    valuation_price_date: Optional[str] = None,
    valuation_date: Optional[date] = None,
    extra_warnings: Optional[List[str]] = None,
    not_evaluable_reason: Optional[str] = None,
) -> FixedIncomeMetricsResult:
    """Compute contractual dirty price, YTM/IRR, duration, and WAL from manual terms."""
    iid = instrument_id.strip().upper()
    warnings: List[str] = list(extra_warnings or [])

    eff_val_date = _parse_optional_valuation_date(valuation_date, valuation_price_date)
    terms = resolve_fixed_income_terms_for_valuation(
        terms,
        valuation_date=eff_val_date,
        asset_id=iid,
    )
    assert terms.settlement_date is not None
    assert terms.residual_nominal is not None
    assert terms.amortization_factor is not None

    if terms.coupon_structure in ("floating", "indexed"):
        missing_monetary = [
            cf for cf in terms.cashflows
            if effective_detachment_date(cf) > terms.settlement_date
            and cf.coupon_amount is None
            and cf.coupon_rate is None
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
    ytm_weekly: Optional[float] = None
    reason: Optional[str] = None

    if status == "ok" and ytm is not None:
        ytm_weekly = annual_rate_to_weekly_effective(ytm) if ytm > -1.0 else None
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
        ytm_weekly_effective=ytm_weekly,
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


@dataclass(frozen=True)
class FixedIncomeTotalReturnResult:
    """Result and provenance metadata for fixed-income historical return series construction."""

    instrument_id: str
    series: pd.Series
    series_type: str  # "adjusted_total_return" | "partial_total_return" | "market_price"
    is_total_return: bool
    coverage_status: str  # "complete" | "partial" | "unconfirmed" | "convention_blocked"
    price_convention_used: Optional[str] = None
    price_type_used: Optional[str] = None
    detachment_date_convention: Optional[str] = None  # "ex_date" | "payment_date" | "mixed" | "none_in_window"
    confirmed_start_date: Optional[date] = None
    confirmed_end_date: Optional[date] = None
    price_start_date: Optional[date] = None
    price_end_date: Optional[date] = None
    cashflow_events_applied: int = 0
    cashflow_events_with_ex_date: int = 0
    cashflow_events_payment_date_only: int = 0
    total_coupons_applied: float = 0.0
    total_amortizations_applied: float = 0.0
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _sanitize_historical_series(
    historical_prices: pd.Series,
    *,
    asset_id: str,
    as_of_date: Optional[date] = None,
    duplicate_date_policy: str = "last",
    target_tz: Optional[str] = None,
) -> pd.Series:
    """Sort chronologically ascending, filter valid positive closes, and clip to ``as_of_date``."""
    if historical_prices is None or len(historical_prices) == 0:
        return pd.Series(dtype=float, name=asset_id)

    dt_idx = pd.to_datetime(historical_prices.index)
    if not isinstance(dt_idx, pd.DatetimeIndex):
        dt_idx = pd.DatetimeIndex(dt_idx)
    if dt_idx.tz is not None:
        dt_idx = dt_idx.tz_convert(ZoneInfo(target_tz) if target_tz else "UTC").tz_localize(None)
    dt_idx = dt_idx.as_unit("ns")

    s = pd.Series(historical_prices.to_numpy(), index=dt_idx, name=historical_prices.name or asset_id)
    s = pd.to_numeric(s, errors="coerce")
    if not s.index.is_monotonic_increasing:
        s = s.sort_index(ascending=True, kind="mergesort")

    if as_of_date is not None:
        cutoff_ts = (pd.Timestamp(as_of_date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)).as_unit("ns")
        s = s[s.index <= cutoff_ts]

    vals = s.to_numpy(dtype=float, na_value=np.nan)
    s = s[np.isfinite(vals) & (vals > 0.0)]

    dup_mask = s.index.duplicated(keep="last")
    if dup_mask.any():
        if duplicate_date_policy == "error":
            dup_dates = [str(d) for d in s.index[dup_mask][:5]]
            raise DataQualityError(
                f"Duplicate timestamps detected for '{asset_id}' under duplicate_date_policy='error': {dup_dates}.",
                asset_id=asset_id,
            )
        s = s[~dup_mask]
    return s.astype(float)


def _resolve_event_coupon_for_history(
    ev: CashFlowEvent,
    *,
    terms: FixedIncomeTerms,
    rn_at_fn: Any,
    all_events: Sequence[CashFlowEvent],
    prev_payment_date: Optional[date],
    asset_id: str,
) -> Tuple[Optional[float], Optional[str]]:
    """Resolve monetary coupon per ``reference_nominal`` for a historical event without fabrication."""
    if ev.coupon_amount is not None:
        if terms.coupon_structure == "zero_coupon" and ev.coupon_amount > 0.0:
            raise FixedIncomeValidationError(
                f"Zero-coupon instrument cannot have positive coupon_amount ({ev.coupon_amount}) on {ev.payment_date}.",
                asset_id=asset_id,
            )
        return float(ev.coupon_amount), None

    if ev.coupon_rate is not None:
        if terms.coupon_structure == "zero_coupon" and ev.coupon_rate > 0.0:
            raise FixedIncomeValidationError(
                f"Zero-coupon instrument cannot have positive coupon_rate ({ev.coupon_rate}) on {ev.payment_date}.",
                asset_id=asset_id,
            )
        detach_dt = effective_detachment_date(ev)
        same_day_amort = sum(
            e2.amortization_amount for e2 in all_events if effective_detachment_date(e2) == detach_dt
        )
        rn_before = rn_at_fn(detach_dt) + same_day_amort
        p_start = ev.period_start_date or prev_payment_date
        if p_start is not None:
            if terms.day_count_convention is None:
                return None, "Rate-based historical coupon with accrual dates requires explicit 'day_count_convention'."
            yf = day_count_fraction(p_start, ev.payment_date, terms.day_count_convention)
        elif terms.coupon_frequency is not None:
            yf = 1.0 / float(terms.coupon_frequency)
        else:
            return None, (
                f"Rate-based historical coupon on {ev.payment_date.isoformat()} lacks 'period_start_date' "
                "or 'coupon_frequency' to determine accrual fraction."
            )
        return float(rn_before * float(ev.coupon_rate) * yf), None

    if terms.coupon_structure in ("floating", "indexed"):
        return None, (
            f"Floating/indexed historical cashflow on {ev.payment_date.isoformat()} lacks realized "
            "'coupon_amount' or 'coupon_rate'."
        )
    if terms.coupon_structure != "zero_coupon" and ev.amortization_amount == 0.0:
        return None, f"Historical cashflow event on {ev.payment_date.isoformat()} specifies neither coupon nor amortization."
    return 0.0, None


def _resolve_clean_accrued_interest_values(
    obs_dates: Sequence[date],
    cov_indices: Sequence[int],
    *,
    terms: FixedIncomeTerms,
    all_events: Sequence[CashFlowEvent],
    rn_at_fn: Any,
    asset_id: str,
) -> Tuple[Optional[Dict[int, float]], Optional[str]]:
    """Compute exact historical accrued interest per ``reference_nominal`` for clean-quoted series."""
    if terms.coupon_structure == "zero_coupon":
        return {idx: 0.0 for idx in cov_indices}, None
    if terms.day_count_convention is None:
        return None, "Clean historical price series requires explicit 'day_count_convention' to compute daily accrued interest."

    grouped: Dict[date, List[CashFlowEvent]] = {}
    for ev in all_events:
        grouped.setdefault(ev.payment_date, []).append(ev)

    sorted_pay_dates = sorted(grouped.keys())
    if not sorted_pay_dates:
        return None, "Clean historical price series requires coupon schedule periods to reconstruct daily accrued interest."

    periods: List[Tuple[date, date, Optional[date], float]] = []
    prev_pay: Optional[date] = None
    for pay_dt in sorted_pay_dates:
        evs = grouped[pay_dt]
        explicit_starts = {e.period_start_date for e in evs if e.period_start_date is not None}
        if len(explicit_starts) > 1:
            return None, f"Conflicting 'period_start_date' values on {pay_dt.isoformat()} prevent unambiguous accrued interest reconstruction."
        p_start = next(iter(explicit_starts)) if explicit_starts else (prev_pay or terms.issue_date)
        period_coupon = 0.0
        for ev in evs:
            c_amt, err = _resolve_event_coupon_for_history(
                ev,
                terms=terms,
                rn_at_fn=rn_at_fn,
                all_events=all_events,
                prev_payment_date=prev_pay,
                asset_id=asset_id,
            )
            if err is not None or c_amt is None:
                return None, err
            period_coupon += c_amt
        ex_dates = [e.ex_date for e in evs if e.ex_date is not None]
        min_ex = min(ex_dates) if ex_dates else None
        if p_start is not None and p_start < pay_dt:
            periods.append((p_start, pay_dt, min_ex, period_coupon))
        elif period_coupon > 0.0:
            return None, (
                f"Coupon on {pay_dt.isoformat()} lacks 'period_start_date' (or prior coupon / issue_date) "
                "required to reconstruct clean-price accrued interest."
            )
        prev_pay = pay_dt

    dcc = terms.day_count_convention
    ai_by_idx: Dict[int, float] = {}
    for idx in cov_indices:
        d = obs_dates[idx]
        matched: Optional[Tuple[date, date, Optional[date], float]] = None
        for p_start, pay_dt, min_ex, p_coup in periods:
            if p_start <= d <= pay_dt:
                matched = (p_start, pay_dt, min_ex, p_coup)
                break
        if matched is None:
            return None, (
                f"Observation date {d.isoformat()} falls outside known coupon accrual periods; "
                "cannot reconstruct clean-price accrued interest without gaps."
            )
        p_start, pay_dt, min_ex, p_coup = matched
        if d == pay_dt or (min_ex is not None and d >= min_ex) or p_coup == 0.0:
            ai_by_idx[idx] = 0.0
            continue
        denom = day_count_fraction(p_start, pay_dt, dcc)
        if denom <= 0.0:
            return None, f"Non-positive accrual period duration for {p_start.isoformat()}..{pay_dt.isoformat()}."
        ai_by_idx[idx] = p_coup * (day_count_fraction(p_start, d, dcc) / denom)

    return ai_by_idx, None


def build_fixed_income_historical_return_series(
    instrument_id: str,
    historical_prices_usd: pd.Series,
    terms: Optional[FixedIncomeTerms],
    *,
    price_source: str = "iol_historical",
    quote_unit: Optional[str] = None,
    as_of_date: Optional[date] = None,
    duplicate_date_policy: str = "last",
    target_tz: Optional[str] = None,
    adjusted_close_differs_count: int = 0,
    price_field: str = "close",
) -> FixedIncomeTotalReturnResult:
    """Build a Total Return wealth series from historical market prices and manual cashflows.

    Preserves unadjusted Price Return (``is_total_return=False``) with explicit
    warnings whenever historical cashflow coverage is unconfirmed, incomplete, or
    incompatible with pricing/currency/clean-dirty conventions.
    """
    iid = instrument_id.strip().upper()
    s = _sanitize_historical_series(
        historical_prices_usd,
        asset_id=iid,
        as_of_date=as_of_date,
        duplicate_date_policy=duplicate_date_policy,
        target_tz=target_tz,
    )
    if s.empty:
        return FixedIncomeTotalReturnResult(
            instrument_id=iid,
            series=s,
            series_type="market_price",
            is_total_return=False,
            coverage_status="unconfirmed",
            warnings=(
                f"Historical series for '{iid}' is empty after sanitization; keeping unadjusted Price Return.",
            ),
        )

    obs_dates = [pd.Timestamp(idx).date() for idx in s.index]
    price_start = obs_dates[0]
    price_end = obs_dates[-1]
    hist_cutoff = min(price_end, as_of_date) if as_of_date is not None else price_end

    def _price_return_fallback(
        status: str,
        warning_msg: str,
        *,
        price_conv: Optional[str] = None,
        price_tp: Optional[str] = None,
        cov_start_dt: Optional[date] = None,
        cov_end_dt: Optional[date] = None,
    ) -> FixedIncomeTotalReturnResult:
        return FixedIncomeTotalReturnResult(
            instrument_id=iid,
            series=s.rename(iid),
            series_type="market_price",
            is_total_return=False,
            coverage_status=status,
            price_convention_used=price_conv,
            price_type_used=price_tp,
            confirmed_start_date=cov_start_dt,
            confirmed_end_date=cov_end_dt,
            price_start_date=price_start,
            price_end_date=price_end,
            warnings=(warning_msg,),
        )

    if len(s) < 2:
        return _price_return_fallback(
            "unconfirmed",
            f"Fewer than 2 valid historical price observations for '{iid}'; keeping unadjusted Price Return.",
        )

    if terms is None:
        return _price_return_fallback(
            "unconfirmed",
            f"No manual FixedIncomeTerms cashflow schedule configured for '{iid}'; historical returns reflect "
            "unadjusted Price Return only (results may be distorted by omitted coupons and amortizations).",
        )

    if quote_unit is not None and quote_unit != terms.price_convention:
        raise FixedIncomeValidationError(
            f"Historical quote_unit '{quote_unit}' for '{iid}' conflicts with FixedIncomeTerms.price_convention='{terms.price_convention}'.",
            asset_id=iid,
        )

    if price_source == "iol_historical" and price_field == "adjusted_close" and adjusted_close_differs_count > 0:
        return _price_return_fallback(
            "convention_blocked",
            f"IOL historical series for '{iid}' uses 'Cierre ajustado' differing from 'Cierre' on "
            f"{adjusted_close_differs_count} row(s); refusing manual cashflow adjustment to avoid double counting. "
            "Set IOLConfig(historical_price_field='close') for exact Total Return.",
            price_conv=terms.price_convention,
            price_tp=terms.price_type,
        )

    if not terms.historical_cashflows_confirmed:
        has_past_events = any(effective_detachment_date(ev) <= hist_cutoff for ev in terms.cashflows)
        if has_past_events:
            msg = (
                f"Historical cashflow events exist for '{iid}' but 'historical_cashflows_confirmed=False'; "
                "cannot assume unrecorded periods had zero payments. Keeping Price Return only "
                "(results may be distorted)."
            )
        else:
            msg = (
                f"Historical cashflow coverage is not confirmed for '{iid}' ('historical_cashflows_confirmed=False'); "
                "absence of records does not imply zero payments. Keeping Price Return only "
                "(results may be distorted)."
            )
        return _price_return_fallback(
            "unconfirmed",
            msg,
            price_conv=terms.price_convention,
            price_tp=terms.price_type,
        )

    anchor_date = as_of_date if as_of_date is not None else price_end
    eff_terms = resolve_fixed_income_terms_for_valuation(
        terms,
        valuation_date=anchor_date,
        asset_id=iid,
    )
    assert eff_terms.settlement_date is not None
    assert eff_terms.residual_nominal is not None

    cov_start = eff_terms.historical_cashflow_start_date or eff_terms.issue_date or price_start
    cov_end = (
        min(eff_terms.historical_cashflow_end_date, hist_cutoff)
        if eff_terms.historical_cashflow_end_date is not None
        else hist_cutoff
    )

    if cov_start > cov_end or cov_end < price_start or cov_start > price_end:
        return _price_return_fallback(
            "unconfirmed",
            f"Confirmed historical cashflow window ({cov_start.isoformat()}..{cov_end.isoformat()}) for '{iid}' "
            f"does not overlap observed market prices ({price_start.isoformat()}..{price_end.isoformat()}); "
            "keeping Price Return only (results may be distorted).",
            price_conv=eff_terms.price_convention,
            price_tp=eff_terms.price_type,
            cov_start_dt=cov_start,
            cov_end_dt=cov_end,
        )

    cov_indices = [i for i, d in enumerate(obs_dates) if cov_start <= d <= cov_end]
    if len(cov_indices) < 2:
        return _price_return_fallback(
            "unconfirmed",
            f"Confirmed historical cashflow window ({cov_start.isoformat()}..{cov_end.isoformat()}) for '{iid}' "
            "covers fewer than 2 price observations; keeping Price Return only (results may be distorted).",
            price_conv=eff_terms.price_convention,
            price_tp=eff_terms.price_type,
            cov_start_dt=cov_start,
            cov_end_dt=cov_end,
        )

    first_cov_idx = cov_indices[0]
    last_cov_idx = cov_indices[-1]
    cov_obs_start = obs_dates[first_cov_idx]
    cov_obs_end = obs_dates[last_cov_idx]
    is_full_coverage = (cov_start <= price_start) and (cov_end >= price_end)

    all_events = sorted(eff_terms.cashflows, key=lambda e: (effective_detachment_date(e), e.payment_date))
    covered_events = [
        ev for ev in all_events if cov_obs_start < effective_detachment_date(ev) <= cov_obs_end
    ]

    if eff_terms.cashflow_currency != "USD":
        has_nonzero_or_rate = any(
            ev.amortization_amount > 0.0
            or (ev.coupon_amount is not None and ev.coupon_amount > 0.0)
            or (ev.coupon_rate is not None and ev.coupon_rate > 0.0)
            for ev in covered_events
        )
        if has_nonzero_or_rate or (eff_terms.price_convention == "per_100_residual_nominal" and covered_events):
            return _price_return_fallback(
                "convention_blocked",
                f"Historical cashflows for '{iid}' are denominated in '{eff_terms.cashflow_currency}' while market prices "
                "are in USD, and no historical FX series is available for payment dates (spot cashflow_fx_to_usd is never "
                "applied historically); preserving Price Return only.",
                price_conv=eff_terms.price_convention,
                price_tp=eff_terms.price_type,
                cov_start_dt=cov_start,
                cov_end_dt=cov_end,
            )

    if eff_terms.historical_price_type is not None:
        eff_hist_price_type = eff_terms.historical_price_type
    elif price_source == "iol_historical" and eff_terms.price_type == "clean":
        return _price_return_fallback(
            "convention_blocked",
            f"FixedIncomeTerms for '{iid}' sets price_type='clean' with historical_price_type=None while reading IOL "
            "DatosHistoricos (which normally reports dirty BCBA market prices). Specify historical_price_type='dirty' or "
            "'clean' explicitly to avoid double counting or missing accrued interest; preserving Price Return only.",
            price_conv=eff_terms.price_convention,
            price_tp=eff_terms.price_type,
            cov_start_dt=cov_start,
            cov_end_dt=cov_end,
        )
    else:
        eff_hist_price_type = eff_terms.price_type

    anchor_settle = eff_terms.settlement_date
    anchor_rn = float(eff_terms.residual_nominal)

    def _rn_at(d: date) -> float:
        if d < anchor_settle:
            return anchor_rn + sum(
                ev.amortization_amount for ev in all_events if d < effective_detachment_date(ev) <= anchor_settle
            )
        if d == anchor_settle:
            return anchor_rn
        return anchor_rn - sum(
            ev.amortization_amount for ev in all_events if anchor_settle < effective_detachment_date(ev) <= d
        )

    rn_by_idx: Dict[int, float] = {}
    for idx in cov_indices:
        rn_val = _rn_at(obs_dates[idx])
        if rn_val <= 0.0 or rn_val > eff_terms.reference_nominal * (1.0 + 1e-6):
            return _price_return_fallback(
                "convention_blocked",
                f"Derived historical residual nominal ({rn_val:.6f}) on {obs_dates[idx].isoformat()} for '{iid}' "
                f"is inconsistent with reference_nominal ({eff_terms.reference_nominal:.6f}); preserving Price Return only.",
                price_conv=eff_terms.price_convention,
                price_tp=eff_hist_price_type,
                cov_start_dt=cov_start,
                cov_end_dt=cov_end,
            )
        rn_by_idx[idx] = rn_val

    ai_by_idx: Dict[int, float] = {idx: 0.0 for idx in cov_indices}
    if eff_hist_price_type == "clean":
        resolved_ai, ai_err = _resolve_clean_accrued_interest_values(
            obs_dates,
            cov_indices,
            terms=eff_terms,
            all_events=all_events,
            rn_at_fn=_rn_at,
            asset_id=iid,
        )
        if resolved_ai is None:
            return _price_return_fallback(
                "convention_blocked",
                f"Cannot construct exact Total Return for clean-quoted '{iid}': {ai_err} Preserving Price Return only.",
                price_conv=eff_terms.price_convention,
                price_tp=eff_hist_price_type,
                cov_start_dt=cov_start,
                cov_end_dt=cov_end,
            )
        ai_by_idx = resolved_ai

    n_obs = len(obs_dates)
    interval_coupon = np.zeros(n_obs, dtype=float)
    interval_amort = np.zeros(n_obs, dtype=float)
    events_applied = 0
    with_ex_count = 0
    pay_only_count = 0

    for ev in covered_events:
        detach_dt = effective_detachment_date(ev)
        prior_pays = [e2.payment_date for e2 in all_events if e2.payment_date < ev.payment_date]
        prev_pay = max(prior_pays) if prior_pays else eff_terms.issue_date
        c_amt, c_err = _resolve_event_coupon_for_history(
            ev,
            terms=eff_terms,
            rn_at_fn=_rn_at,
            all_events=all_events,
            prev_payment_date=prev_pay,
            asset_id=iid,
        )
        if c_err is not None or c_amt is None:
            return _price_return_fallback(
                "convention_blocked",
                f"Cannot resolve historical cashflow for '{iid}': {c_err} Preserving Price Return only.",
                price_conv=eff_terms.price_convention,
                price_tp=eff_hist_price_type,
                cov_start_dt=cov_start,
                cov_end_dt=cov_end,
            )

        pos = bisect_left(obs_dates, detach_dt)
        if not (first_cov_idx + 1 <= pos <= last_cov_idx):
            continue
        interval_coupon[pos] += float(c_amt)
        interval_amort[pos] += float(ev.amortization_amount)
        events_applied += 1
        if ev.ex_date is not None:
            with_ex_count += 1
        else:
            pay_only_count += 1

    mv = np.zeros(n_obs, dtype=float)
    for idx in cov_indices:
        q_i = float(s.iloc[idx])
        if eff_terms.price_convention == "per_100_original_nominal":
            mv_base = q_i * (eff_terms.reference_nominal / 100.0)
        elif eff_terms.price_convention == "per_100_residual_nominal":
            mv_base = q_i * (rn_by_idx[idx] / 100.0)
        else:
            mv_base = q_i
        mv[idx] = mv_base + ai_by_idx[idx]

    rets = np.zeros(n_obs, dtype=float)
    raw_vals = s.to_numpy(dtype=float)
    for i in range(1, n_obs):
        if first_cov_idx + 1 <= i <= last_cov_idx:
            if mv[i - 1] <= 0.0:
                return _price_return_fallback(
                    "convention_blocked",
                    f"Non-positive market value encountered for '{iid}' on {obs_dates[i - 1].isoformat()}; preserving Price Return only.",
                    price_conv=eff_terms.price_convention,
                    price_tp=eff_hist_price_type,
                    cov_start_dt=cov_start,
                    cov_end_dt=cov_end,
                )
            r_i = (mv[i] + interval_coupon[i] + interval_amort[i]) / mv[i - 1] - 1.0
        else:
            r_i = (raw_vals[i] / raw_vals[i - 1]) - 1.0
        if not math.isfinite(r_i) or (1.0 + r_i) <= 0.0:
            return _price_return_fallback(
                "convention_blocked",
                f"Invalid total return step ({r_i}) encountered for '{iid}' on {obs_dates[i].isoformat()}; preserving Price Return only.",
                price_conv=eff_terms.price_convention,
                price_tp=eff_hist_price_type,
                cov_start_dt=cov_start,
                cov_end_dt=cov_end,
            )
        rets[i] = r_i

    wealth_index = pd.Series(100.0 * np.cumprod(1.0 + rets), index=s.index, name=iid, dtype=float)
    total_c = float(np.sum(interval_coupon))
    total_a = float(np.sum(interval_amort))

    if events_applied == 0:
        detach_conv = "none_in_window"
    elif pay_only_count == 0:
        detach_conv = "ex_date"
    elif with_ex_count == 0:
        detach_conv = "payment_date"
    else:
        detach_conv = "mixed"

    warnings_out: List[str] = []
    if pay_only_count > 0:
        warnings_out.append(
            f"Used payment_date as detachment proxy for {pay_only_count} historical cashflow(s) of '{iid}' "
            "lacking explicit ex_date; if ex-date and payment-date span different W-FRI weeks, weekly returns "
            "and volatility may reflect timing mismatch."
        )

    post_cutoff_detached = [
        ev for ev in all_events if cov_obs_end < effective_detachment_date(ev) <= hist_cutoff
    ]
    if post_cutoff_detached:
        warnings_out.append(
            f"Excluded {len(post_cutoff_detached)} cashflow event(s) for '{iid}' detaching after last valid "
            f"historical trade date ({cov_obs_end.isoformat()}) up to cutoff ({hist_cutoff.isoformat()}); "
            "last observed price precedes detachment."
        )

    src_label = "IOL historical series" if price_source == "iol_historical" else f"Historical series ({price_source})"
    if is_full_coverage:
        series_type = "adjusted_total_return"
        is_tr = True
        cov_status = "complete"
        if events_applied == 0:
            warnings_out.append(
                f"{src_label} for '{iid}' verified as Total Return (confirmed payment-free historical period "
                f"{cov_start.isoformat()}..{cov_end.isoformat()}; price_convention={eff_terms.price_convention}, "
                f"price_type={eff_hist_price_type})."
            )
        else:
            warnings_out.append(
                f"{src_label} for '{iid}' adjusted to Total Return using {events_applied} confirmed manual cashflow(s) "
                f"over {cov_start.isoformat()}..{cov_end.isoformat()} (coupons={total_c:.4f}, amortizations={total_a:.4f} "
                f"{eff_terms.cashflow_currency}, price_convention={eff_terms.price_convention}, "
                f"price_type={eff_hist_price_type}, detachment={detach_conv})."
            )
    else:
        series_type = "partial_total_return"
        is_tr = False
        cov_status = "partial"
        warnings_out.append(
            f"Partial historical cashflow coverage for '{iid}': confirmed cashflows applied only over "
            f"{cov_obs_start.isoformat()}..{cov_obs_end.isoformat()} ({events_applied} event(s), coupons={total_c:.4f}, "
            f"amortizations={total_a:.4f} {eff_terms.cashflow_currency}, detachment={detach_conv}); unconfirmed sub-periods "
            f"within {price_start.isoformat()}..{price_end.isoformat()} remain unadjusted Price Return. Series is NOT "
            "labeled full Total Return and results may still be distorted."
        )

    return FixedIncomeTotalReturnResult(
        instrument_id=iid,
        series=wealth_index,
        series_type=series_type,
        is_total_return=is_tr,
        coverage_status=cov_status,
        price_convention_used=eff_terms.price_convention,
        price_type_used=eff_hist_price_type,
        detachment_date_convention=detach_conv,
        confirmed_start_date=cov_start,
        confirmed_end_date=cov_end,
        price_start_date=price_start,
        price_end_date=price_end,
        cashflow_events_applied=events_applied,
        cashflow_events_with_ex_date=with_ex_count,
        cashflow_events_payment_date_only=pay_only_count,
        total_coupons_applied=total_c,
        total_amortizations_applied=total_a,
        warnings=tuple(warnings_out),
        diagnostics={
            "price_source": price_source,
            "series_type": series_type,
            "is_total_return": is_tr,
            "coverage_status": cov_status,
            "price_convention": eff_terms.price_convention,
            "price_type": eff_hist_price_type,
            "detachment_date_convention": detach_conv,
            "confirmed_window": [cov_start.isoformat(), cov_end.isoformat()],
            "price_window": [price_start.isoformat(), price_end.isoformat()],
            "cashflow_events_applied": events_applied,
            "total_coupons_applied": total_c,
            "total_amortizations_applied": total_a,
        },
    )
