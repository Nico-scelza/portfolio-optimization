"""Validated configuration and result contracts for stateless portfolio analysis.

This module is the single source of truth for all user-editable settings,
financial defaults, units, validation rules, and structured output payloads.
All configurations and results live strictly in memory per request with zero
persistence, user accounts, or process-wide mutable state.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from .errors import ConfigurationError, FixedIncomeValidationError

# =============================================================================
# Domain vocabularies (UI sections separated from optimizer risk classes)
# =============================================================================

VALID_SECTIONS = ("variable_income", "fixed_income")

VARIABLE_INCOME_SUBTYPES = (
    "equity",
    "equity_etf",
    "commodity_gold",
    "crypto",
    "other_variable",
)
FIXED_INCOME_SUBTYPES = (
    "sovereign_bond",
    "corporate_bond_on",
    "tbill",
    "fixed_income_etf",
    "other_fixed_income",
)
VALID_ASSET_SUBTYPES = VARIABLE_INCOME_SUBTYPES + FIXED_INCOME_SUBTYPES
INDIVIDUAL_DEBT_SUBTYPES = ("sovereign_bond", "corporate_bond_on", "tbill")

VALID_RISK_CLASSES = (
    "equity",
    "fixed_income",
    "gold_commodities",
    "crypto",
    "cash",
    "other",
)

DEFAULT_RISK_CLASS_BY_SUBTYPE: Dict[str, str] = {
    "equity": "equity",
    "equity_etf": "equity",
    "commodity_gold": "gold_commodities",
    "crypto": "crypto",
    "other_variable": "other",
    "sovereign_bond": "fixed_income",
    "corporate_bond_on": "fixed_income",
    "tbill": "fixed_income",
    "fixed_income_etf": "fixed_income",
    "other_fixed_income": "fixed_income",
}

VALID_QUOTE_SOURCES = ("yfinance", "iol_bonos", "iol_ons", "manual")
VALID_STRATEGIES = ("max_sharpe", "min_variance", "min_cvar", "erc", "hrp")
VALID_PRICE_CONVENTIONS = (
    "per_100_original_nominal",
    "per_100_residual_nominal",
    "per_unit",
)
VALID_PRICE_TYPES = ("dirty", "clean")
VALID_COUPON_STRUCTURES = ("fixed", "zero_coupon", "step_up", "floating", "indexed")
VALID_DAY_COUNTS = ("30/360", "ACT/365", "ACT/360", "ACT/ACT")
VALID_FEE_STRUCTURES = ("percentage", "fixed", "fixed_plus_percentage")
VALID_COV_ESTIMATORS = ("vix_regime", "static_shrinkage", "ewma")
VALID_SHRINKAGE_METHODS = ("ledoit_wolf", "oas", "sample")
VALID_REGIME_CLASSIFIERS = ("vix_history", "rolling_volatility_quantile")
VALID_INSUFFICIENT_REGIME_POLICIES = ("error", "fallback_full_sample_shrinkage")
VALID_DUPLICATE_DATE_POLICIES = ("last", "error")
VALID_PRIOR_POLICIES = (
    "explicit_or_warned_equal_weight",
    "explicit",
    "market_cap_equity_only",
    "equal_weight",
    "inverse_volatility",
)
VALID_FI_BL_MODES = ("metric_only", "explicit_ytm_view")
VALID_DCF_BASES = ("terminal_target", "present_intrinsic_value", "dated_cashflows")
VALID_HRP_LINKAGES = ("single", "complete", "average", "ward")
VALID_NON_POSITIVE_EXCESS_POLICIES = ("block", "allow_with_warning")
VALID_ACCRUED_BASES = ("quote_convention", "per_reference_nominal", "per_1_original_nominal")
VALID_HISTORICAL_PRICE_TYPES = ("market_price", "adjusted_total_return")
VALID_IOL_HISTORICAL_PRICE_FIELDS = ("close", "adjusted_close")
VALID_IOL_HISTORICAL_FREQUENCIES = ("daily", "weekly", "monthly")
VALID_EWMA_MEAN_MODES = ("zero", "causal_recursive")
VALID_EWMA_GAP_POLICIES = ("elapsed_weeks_decay", "error_on_gap")
SUBUNIT_CURRENCY_CODES = frozenset({"GBp", "GBX", "ZAc", "ZAC", "ILA", "ILX"})

# Removed legacy concepts that must never be silently accepted
REMOVED_LEGACY_KEYS = frozenset({
    "pais_riesgo",
    "country_risk_argentina",
    "country_risk_map",
    "country_risk_value",
    "prima_riesgo_mas_reciente",
    "country_risk_premia",
    "ejecutar_ml",
    "activar_ml",
    "ml_predictions_5y",
    "predicciones_ml",
    "predicciones_5y",
})


def _parse_date(value: Union[date, datetime, str, pd.Timestamp], field_name: str, asset_id: Optional[str] = None) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise ConfigurationError(f"Invalid NaT date for '{field_name}'.", asset_id=asset_id)
        return value.date()
    if isinstance(value, str):
        text = value.strip()
        try:
            return date.fromisoformat(text[:10])
        except ValueError as exc:
            raise ConfigurationError(
                f"Invalid ISO date '{value}' for '{field_name}'. Expected YYYY-MM-DD.",
                asset_id=asset_id,
            ) from exc
    raise ConfigurationError(f"Unsupported date type for '{field_name}': {type(value).__name__}", asset_id=asset_id)


def _check_finite(value: float, field_name: str, *, asset_id: Optional[str] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"'{field_name}' must be numeric.", asset_id=asset_id)
    val = float(value)
    if not math.isfinite(val):
        raise ConfigurationError(f"'{field_name}' must be finite.", asset_id=asset_id)
    return val


def _serialize_value(obj: Any) -> Any:
    """Convert dataclasses, numpy/pandas objects, and dates to plain JSON-safe types."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    if isinstance(obj, np.generic):
        py_val = obj.item()
        if isinstance(py_val, float) and not math.isfinite(py_val):
            return None
        return py_val
    if isinstance(obj, np.ndarray):
        return [_serialize_value(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return {str(k): _serialize_value(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return {
            "index": [str(idx) for idx in obj.index],
            "columns": [str(col) for col in obj.columns],
            "data": [[_serialize_value(v) for v in row] for row in obj.values.tolist()],
        }
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): _serialize_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_value(v) for v in obj]
    return str(obj)


# =============================================================================
# Broker configuration
# =============================================================================

@dataclass(frozen=True)
class BrokerConfig:
    """Editable broker commission and estimated execution spread configuration.

    Units:
        - Percentage fees: decimal fraction of gross trade notional (e.g., 0.005 = 0.50%).
        - Fixed fees / min_fee: monetary amount in ``fee_currency``.
        - ``fee_fx_to_usd``: USD per 1 unit of ``fee_currency`` (required if ``fee_currency != 'USD'``).
        - ``estimated_spread_bps``: one-way half-spread in basis points (10 bps = 0.0010).
    """

    broker_id: str
    name: str = ""
    fee_currency: str = "USD"
    fee_fx_to_usd: Optional[float] = None
    fee_structure: str = "percentage"
    buy_percentage_fee: float = 0.0
    sell_percentage_fee: Optional[float] = None
    buy_fixed_fee: float = 0.0
    sell_fixed_fee: Optional[float] = None
    min_fee: float = 0.0
    estimated_spread_bps: float = 0.0
    price_includes_spread: bool = False

    def __post_init__(self) -> None:
        if not self.broker_id or not str(self.broker_id).strip():
            raise ConfigurationError("Broker 'broker_id' must be a non-empty string.")
        object.__setattr__(self, "broker_id", str(self.broker_id).strip())
        if not self.name:
            object.__setattr__(self, "name", self.broker_id)
        fee_ccy = str(self.fee_currency or "USD").strip().upper()
        object.__setattr__(self, "fee_currency", fee_ccy)

        if self.fee_structure not in VALID_FEE_STRUCTURES:
            raise ConfigurationError(
                f"Invalid broker fee_structure '{self.fee_structure}'. Expected one of {VALID_FEE_STRUCTURES}."
            )

        buy_pct = _check_finite(self.buy_percentage_fee, "buy_percentage_fee")
        sell_pct = buy_pct if self.sell_percentage_fee is None else _check_finite(self.sell_percentage_fee, "sell_percentage_fee")
        buy_fix = _check_finite(self.buy_fixed_fee, "buy_fixed_fee")
        sell_fix = buy_fix if self.sell_fixed_fee is None else _check_finite(self.sell_fixed_fee, "sell_fixed_fee")
        min_fee = _check_finite(self.min_fee, "min_fee")
        spread_bps = _check_finite(self.estimated_spread_bps, "estimated_spread_bps")

        for fname, val in (
            ("buy_percentage_fee", buy_pct),
            ("sell_percentage_fee", sell_pct),
            ("buy_fixed_fee", buy_fix),
            ("sell_fixed_fee", sell_fix),
            ("min_fee", min_fee),
            ("estimated_spread_bps", spread_bps),
        ):
            if val < 0.0:
                raise ConfigurationError(f"Broker '{self.broker_id}': '{fname}' must be >= 0.")
        if buy_pct >= 1.0 or sell_pct >= 1.0:
            raise ConfigurationError(f"Broker '{self.broker_id}': percentage fee must be < 1.0 (decimal).")

        if fee_ccy == "USD":
            fx_to_usd = 1.0
        else:
            if self.fee_fx_to_usd is None:
                raise ConfigurationError(
                    f"Broker '{self.broker_id}': fee_currency='{fee_ccy}' requires explicit positive 'fee_fx_to_usd'."
                )
            fx_to_usd = _check_finite(self.fee_fx_to_usd, "fee_fx_to_usd")
            if fx_to_usd <= 0.0:
                raise ConfigurationError(
                    f"Broker '{self.broker_id}': 'fee_fx_to_usd' must be strictly positive."
                )

        object.__setattr__(self, "buy_percentage_fee", buy_pct)
        object.__setattr__(self, "sell_percentage_fee", sell_pct)
        object.__setattr__(self, "buy_fixed_fee", buy_fix)
        object.__setattr__(self, "sell_fixed_fee", sell_fix)
        object.__setattr__(self, "min_fee", min_fee)
        object.__setattr__(self, "estimated_spread_bps", spread_bps)
        object.__setattr__(self, "fee_fx_to_usd", fx_to_usd)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrokerConfig":
        return cls(**data)


# =============================================================================
# Manual contractual fixed-income terms
# =============================================================================

@dataclass(frozen=True)
class CashFlowEvent:
    """Explicit contractual payment event for an individual bond / ON / T-bill.

    Amounts are expressed per ``reference_nominal`` in ``cashflow_currency``.
    If ``coupon_rate`` is supplied instead of ``coupon_amount``, explicit
    ``day_count_convention``, ``coupon_frequency``, and ``period_start_date``
    are required on ``FixedIncomeTerms`` / ``CashFlowEvent``.
    """

    payment_date: date
    coupon_amount: Optional[float] = None
    amortization_amount: float = 0.0
    coupon_rate: Optional[float] = None
    period_start_date: Optional[date] = None

    def __post_init__(self) -> None:
        pdate = _parse_date(self.payment_date, "payment_date")
        object.__setattr__(self, "payment_date", pdate)

        if self.period_start_date is not None:
            sdate = _parse_date(self.period_start_date, "period_start_date")
            if sdate >= pdate:
                raise FixedIncomeValidationError(
                    f"CashFlowEvent period_start_date ({sdate}) must be strictly before payment_date ({pdate})."
                )
            object.__setattr__(self, "period_start_date", sdate)

        amort = _check_finite(self.amortization_amount, "amortization_amount")
        if amort < 0.0:
            raise FixedIncomeValidationError("amortization_amount must be >= 0.")
        object.__setattr__(self, "amortization_amount", amort)

        if self.coupon_amount is not None:
            camt = _check_finite(self.coupon_amount, "coupon_amount")
            if camt < 0.0:
                raise FixedIncomeValidationError("coupon_amount must be >= 0.")
            object.__setattr__(self, "coupon_amount", camt)

        if self.coupon_rate is not None:
            crate = _check_finite(self.coupon_rate, "coupon_rate")
            if crate < 0.0:
                raise FixedIncomeValidationError("coupon_rate must be >= 0.")
            object.__setattr__(self, "coupon_rate", crate)

        if self.coupon_amount is not None and self.coupon_rate is not None:
            raise FixedIncomeValidationError(
                "Provide either monetary 'coupon_amount' or 'coupon_rate', not both, to avoid ambiguity."
            )

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CashFlowEvent":
        return cls(**data)


@dataclass(frozen=True)
class FixedIncomeTerms:
    """Manual contractual specification for an individual bond, ON, or T-bill.

    Manual cashflows are the source of truth; published yields never override them.
    """

    cashflow_currency: str
    reference_nominal: float
    maturity_date: date
    settlement_date: date
    cashflows: Tuple[CashFlowEvent, ...] = field(default_factory=tuple)
    residual_nominal: Optional[float] = None
    amortization_factor: Optional[float] = None
    price_convention: str = "per_100_original_nominal"
    price_type: str = "dirty"
    accrued_interest: Optional[float] = None
    accrued_interest_basis: str = "quote_convention"
    coupon_structure: str = "fixed"
    day_count_convention: Optional[str] = "ACT/365"
    coupon_frequency: Optional[int] = None
    index_assumptions: Optional[Dict[str, Any]] = None
    cashflow_fx_to_usd: Optional[float] = None

    def __post_init__(self) -> None:
        ccy = str(self.cashflow_currency or "").strip().upper()
        if not ccy:
            raise FixedIncomeValidationError("FixedIncomeTerms requires non-empty 'cashflow_currency'.")
        object.__setattr__(self, "cashflow_currency", ccy)

        ref_nom = _check_finite(self.reference_nominal, "reference_nominal")
        if ref_nom <= 0.0:
            raise FixedIncomeValidationError("'reference_nominal' must be strictly positive.")
        object.__setattr__(self, "reference_nominal", ref_nom)

        mat_date = _parse_date(self.maturity_date, "maturity_date")
        settle_date = _parse_date(self.settlement_date, "settlement_date")
        if settle_date > mat_date:
            raise FixedIncomeValidationError(
                f"settlement_date ({settle_date}) cannot be after maturity_date ({mat_date})."
            )
        object.__setattr__(self, "maturity_date", mat_date)
        object.__setattr__(self, "settlement_date", settle_date)

        # Resolve residual_nominal and amortization_factor consistently
        res_nom = self.residual_nominal
        amort_fac = self.amortization_factor
        if res_nom is None and amort_fac is None:
            res_nom = ref_nom
            amort_fac = 1.0
        elif res_nom is not None and amort_fac is None:
            res_nom = _check_finite(res_nom, "residual_nominal")
            amort_fac = res_nom / ref_nom
        elif res_nom is None and amort_fac is not None:
            amort_fac = _check_finite(amort_fac, "amortization_factor")
            res_nom = ref_nom * amort_fac
        else:
            res_nom = _check_finite(res_nom, "residual_nominal")
            amort_fac = _check_finite(amort_fac, "amortization_factor")
            if not math.isclose(res_nom, ref_nom * amort_fac, rel_tol=1e-4, abs_tol=1e-6):
                raise FixedIncomeValidationError(
                    f"Inconsistent residual_nominal ({res_nom}) vs reference_nominal * amortization_factor ({ref_nom * amort_fac})."
                )
        if res_nom <= 0.0 or res_nom > ref_nom * (1.0 + 1e-9):
            raise FixedIncomeValidationError(
                f"residual_nominal ({res_nom}) must satisfy 0 < residual_nominal <= reference_nominal ({ref_nom})."
            )
        if amort_fac <= 0.0 or amort_fac > 1.0 + 1e-9:
            raise FixedIncomeValidationError(
                f"amortization_factor ({amort_fac}) must satisfy 0 < amortization_factor <= 1."
            )
        object.__setattr__(self, "residual_nominal", float(res_nom))
        object.__setattr__(self, "amortization_factor", float(amort_fac))

        if self.price_convention not in VALID_PRICE_CONVENTIONS:
            raise FixedIncomeValidationError(
                f"Invalid price_convention '{self.price_convention}'. Expected one of {VALID_PRICE_CONVENTIONS}."
            )
        if self.price_type not in VALID_PRICE_TYPES:
            raise FixedIncomeValidationError(
                f"Invalid price_type '{self.price_type}'. Expected one of {VALID_PRICE_TYPES}."
            )
        if self.coupon_structure not in VALID_COUPON_STRUCTURES:
            raise FixedIncomeValidationError(
                f"Invalid coupon_structure '{self.coupon_structure}'. Expected one of {VALID_COUPON_STRUCTURES}."
            )
        if self.day_count_convention is not None and self.day_count_convention not in VALID_DAY_COUNTS:
            raise FixedIncomeValidationError(
                f"Invalid day_count_convention '{self.day_count_convention}'. Expected one of {VALID_DAY_COUNTS}."
            )
        if self.coupon_frequency is not None and self.coupon_frequency not in (1, 2, 4, 12):
            raise FixedIncomeValidationError(
                f"Invalid coupon_frequency '{self.coupon_frequency}'. Expected one of (1, 2, 4, 12)."
            )

        if self.accrued_interest is not None:
            acc = _check_finite(self.accrued_interest, "accrued_interest")
            if acc < 0.0:
                raise FixedIncomeValidationError("'accrued_interest' must be >= 0.")
            object.__setattr__(self, "accrued_interest", acc)
        if self.accrued_interest_basis not in VALID_ACCRUED_BASES:
            raise FixedIncomeValidationError(
                f"Invalid accrued_interest_basis '{self.accrued_interest_basis}'. Expected one of {VALID_ACCRUED_BASES}."
            )

        if self.cashflow_fx_to_usd is not None:
            fx = _check_finite(self.cashflow_fx_to_usd, "cashflow_fx_to_usd")
            if fx <= 0.0:
                raise FixedIncomeValidationError("'cashflow_fx_to_usd' must be strictly positive.")
            object.__setattr__(self, "cashflow_fx_to_usd", fx)

        normalized_cfs = tuple(
            cf if isinstance(cf, CashFlowEvent) else CashFlowEvent.from_dict(cf)
            for cf in self.cashflows
        )
        object.__setattr__(self, "cashflows", normalized_cfs)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FixedIncomeTerms":
        payload = dict(data)
        if "cashflows" in payload and payload["cashflows"] is not None:
            payload["cashflows"] = tuple(
                cf if isinstance(cf, CashFlowEvent) else CashFlowEvent.from_dict(cf)
                for cf in payload["cashflows"]
            )
        return cls(**payload)


# =============================================================================
# DCF valuation view input
# =============================================================================

@dataclass(frozen=True)
class DCFViewConfig:
    """Validated DCF view specification for Black-Litterman integration.

    Supports:
        - ``valuation_basis="terminal_target"``: target price V_T at horizon T
          (or ``total_upside`` over horizon T), annualized via ``(V_T / P_0)^(1/T) - 1``.
        - ``valuation_basis="present_intrinsic_value"``: discounted intrinsic value V_0
          with explicit convergence horizon T (documented as convergence hypothesis,
          not guaranteed CAGR). Optional ``expected_dividend_yield`` only when not
          already embedded in V_0.
        - ``valuation_basis="dated_cashflows"``: explicit dated distributions
          ``(time_years, amount)`` plus terminal value at horizon T, solved via IRR.
    """

    valuation_basis: str = "terminal_target"
    horizon_years: Optional[float] = None
    confidence: Optional[float] = None
    target_price: Optional[float] = None
    present_intrinsic_value: Optional[float] = None
    total_upside: Optional[float] = None
    current_price: Optional[float] = None
    currency: str = "USD"
    expected_dividend_yield: float = 0.0
    distributions: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.valuation_basis not in VALID_DCF_BASES:
            raise ConfigurationError(
                f"Invalid DCF valuation_basis '{self.valuation_basis}'. Expected one of {VALID_DCF_BASES}."
            )
        if self.horizon_years is not None:
            horizon = _check_finite(self.horizon_years, "horizon_years")
            if horizon <= 0.0:
                raise ConfigurationError("'horizon_years' must be strictly positive.")
            object.__setattr__(self, "horizon_years", horizon)

        if self.confidence is not None:
            conf = _check_finite(self.confidence, "confidence")
            if conf < 0.0 or conf > 1.0:
                raise ConfigurationError("'confidence' must be in [0.0, 1.0].")
            object.__setattr__(self, "confidence", conf)

        if self.target_price is not None:
            tp = _check_finite(self.target_price, "target_price")
            if tp <= 0.0:
                raise ConfigurationError("'target_price' must be strictly positive.")
            object.__setattr__(self, "target_price", tp)

        if self.present_intrinsic_value is not None:
            piv = _check_finite(self.present_intrinsic_value, "present_intrinsic_value")
            if piv <= 0.0:
                raise ConfigurationError("'present_intrinsic_value' must be strictly positive.")
            object.__setattr__(self, "present_intrinsic_value", piv)

        if self.total_upside is not None:
            up = _check_finite(self.total_upside, "total_upside")
            if up <= -1.0:
                raise ConfigurationError("'total_upside' must be > -1.0 (-100%).")
            object.__setattr__(self, "total_upside", up)

        if self.current_price is not None:
            cp = _check_finite(self.current_price, "current_price")
            if cp <= 0.0:
                raise ConfigurationError("'current_price' must be strictly positive.")
            object.__setattr__(self, "current_price", cp)

        div_y = _check_finite(self.expected_dividend_yield, "expected_dividend_yield")
        if div_y < 0.0:
            raise ConfigurationError("'expected_dividend_yield' must be >= 0.")
        object.__setattr__(self, "expected_dividend_yield", div_y)

        object.__setattr__(self, "currency", str(self.currency or "USD").strip().upper())

        clean_dists = []
        for item in self.distributions:
            if len(item) != 2:
                raise ConfigurationError("Each DCF distribution must be a (time_years, cashflow) pair.")
            t_yr = _check_finite(item[0], "distribution.time_years")
            amt = _check_finite(item[1], "distribution.cashflow")
            if t_yr <= 0.0:
                raise ConfigurationError("Distribution time_years must be > 0.")
            if amt < 0.0:
                raise ConfigurationError("Distribution cashflow must be >= 0.")
            clean_dists.append((t_yr, amt))
        object.__setattr__(self, "distributions", tuple(clean_dists))

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DCFViewConfig":
        payload = dict(data)
        if "distributions" in payload and payload["distributions"] is not None:
            payload["distributions"] = tuple((float(t), float(a)) for t, a in payload["distributions"])
        return cls(**payload)


# =============================================================================
# Instrument & Position configuration (instrument identity != broker position)
# =============================================================================

@dataclass(frozen=True)
class InstrumentConfig:
    """Unique economic instrument definition independent of broker holdings.

    Currency & units contract:
    - ``currency`` defaults to ``None`` (auto-detect quote currency and subunit scale
      from the market data provider without overriding provider metadata to ``"USD"``).
      Provide an explicit value ONLY as a verified economic/quote currency override.
    - For individual debt (``sovereign_bond``, ``corporate_bond_on``, ``tbill``),
      ``BrokerPosition.quantity``, ``ProposedOrder.quantity``, ``min_nominal``, and
      ``lot_size`` are expressed in **original nominal units (1 VN original)**.
    - For equities, ETFs (including ``fixed_income_etf``), gold/commodities, and crypto,
      quantities are expressed in **shares / units**.
    """

    instrument_id: str
    ticker: str
    section: str = "variable_income"
    asset_subtype: str = "equity"
    risk_class: Optional[str] = None
    quote_source: str = "yfinance"
    currency: Optional[str] = None
    quote_unit: Optional[str] = None
    historical_price_series_type: str = "market_price"
    market: str = "US"
    country: Optional[str] = None  # Descriptive/calendar metadata only, never country-risk spread
    exchange: Optional[str] = None
    name: Optional[str] = None
    fractional_allowed: bool = True
    lot_size: Optional[float] = None
    min_nominal: Optional[float] = None
    default_buy_broker_id: Optional[str] = None
    sell_broker_priority: Tuple[str, ...] = field(default_factory=tuple)
    iol_species_map: Optional[Dict[str, str]] = None
    iol_symbol: Optional[str] = None
    iol_market: Optional[str] = None
    fixed_income_terms: Optional[FixedIncomeTerms] = None
    dcf_view: Optional[DCFViewConfig] = None
    historical_proxy_ticker: Optional[str] = None
    manual_price_usd: Optional[float] = None

    def __post_init__(self) -> None:
        iid = str(self.instrument_id or "").strip().upper()
        tkr = str(self.ticker or "").strip()
        if not iid:
            raise ConfigurationError("Instrument 'instrument_id' must be non-empty.")
        if not tkr:
            raise ConfigurationError(f"Instrument '{iid}': 'ticker' must be non-empty.", asset_id=iid)
        object.__setattr__(self, "instrument_id", iid)
        object.__setattr__(self, "ticker", tkr)

        if self.section not in VALID_SECTIONS:
            raise ConfigurationError(
                f"Instrument '{iid}': invalid section '{self.section}'. Expected one of {VALID_SECTIONS}.",
                asset_id=iid,
            )
        if self.asset_subtype not in VALID_ASSET_SUBTYPES:
            raise ConfigurationError(
                f"Instrument '{iid}': invalid asset_subtype '{self.asset_subtype}'. Expected one of {VALID_ASSET_SUBTYPES}.",
                asset_id=iid,
            )

        if self.section == "variable_income" and self.asset_subtype not in VARIABLE_INCOME_SUBTYPES:
            raise ConfigurationError(
                f"Instrument '{iid}': subtype '{self.asset_subtype}' does not belong to section 'variable_income'.",
                asset_id=iid,
            )
        if self.section == "fixed_income" and self.asset_subtype not in FIXED_INCOME_SUBTYPES:
            raise ConfigurationError(
                f"Instrument '{iid}': subtype '{self.asset_subtype}' does not belong to section 'fixed_income'.",
                asset_id=iid,
            )

        resolved_risk_class = self.risk_class or DEFAULT_RISK_CLASS_BY_SUBTYPE[self.asset_subtype]
        if resolved_risk_class not in VALID_RISK_CLASSES:
            raise ConfigurationError(
                f"Instrument '{iid}': invalid risk_class '{resolved_risk_class}'. Expected one of {VALID_RISK_CLASSES}.",
                asset_id=iid,
            )
        object.__setattr__(self, "risk_class", resolved_risk_class)

        if self.quote_source not in VALID_QUOTE_SOURCES:
            raise ConfigurationError(
                f"Instrument '{iid}': invalid quote_source '{self.quote_source}'. Expected one of {VALID_QUOTE_SOURCES}.",
                asset_id=iid,
            )

        # Preserve case for subunit currencies like GBp/GBX/ZAc/ZAC/ILA/ILX; keep None as auto-detect
        if self.currency is None or not str(self.currency).strip():
            norm_ccy: Optional[str] = None
        else:
            raw_ccy = str(self.currency).strip()
            if raw_ccy in SUBUNIT_CURRENCY_CODES:
                norm_ccy = raw_ccy
            else:
                norm_ccy = raw_ccy.upper()
        object.__setattr__(self, "currency", norm_ccy)

        if self.quote_unit is not None:
            qu = str(self.quote_unit).strip()
            if qu not in VALID_PRICE_CONVENTIONS:
                raise ConfigurationError(
                    f"Instrument '{iid}': invalid quote_unit '{qu}'. Expected one of {VALID_PRICE_CONVENTIONS}.",
                    asset_id=iid,
                )
            object.__setattr__(self, "quote_unit", qu)

        if self.historical_price_series_type not in VALID_HISTORICAL_PRICE_TYPES:
            raise ConfigurationError(
                f"Instrument '{iid}': invalid historical_price_series_type '{self.historical_price_series_type}'. "
                f"Expected one of {VALID_HISTORICAL_PRICE_TYPES}.",
                asset_id=iid,
            )

        object.__setattr__(self, "market", str(self.market or "US").strip().upper())

        if self.lot_size is not None:
            ls = _check_finite(self.lot_size, "lot_size", asset_id=iid)
            if ls <= 0.0:
                raise ConfigurationError(f"Instrument '{iid}': 'lot_size' must be > 0.", asset_id=iid)
            object.__setattr__(self, "lot_size", ls)

        if self.min_nominal is not None:
            mn = _check_finite(self.min_nominal, "min_nominal", asset_id=iid)
            if mn <= 0.0:
                raise ConfigurationError(f"Instrument '{iid}': 'min_nominal' must be > 0.", asset_id=iid)
            object.__setattr__(self, "min_nominal", mn)

        if self.manual_price_usd is not None:
            mp = _check_finite(self.manual_price_usd, "manual_price_usd", asset_id=iid)
            if mp <= 0.0:
                raise ConfigurationError(f"Instrument '{iid}': 'manual_price_usd' must be > 0.", asset_id=iid)
            object.__setattr__(self, "manual_price_usd", mp)

        if self.default_buy_broker_id is not None:
            object.__setattr__(self, "default_buy_broker_id", str(self.default_buy_broker_id).strip() or None)

        object.__setattr__(
            self,
            "sell_broker_priority",
            tuple(str(b).strip() for b in self.sell_broker_priority if str(b).strip()),
        )

        if self.iol_species_map is not None:
            clean_map = {
                str(k).strip().upper(): str(v).strip().upper()
                for k, v in self.iol_species_map.items()
                if str(k).strip() and str(v).strip()
            }
            object.__setattr__(self, "iol_species_map", clean_map)

        if self.iol_symbol is not None:
            sym_clean = str(self.iol_symbol).strip().upper()
            object.__setattr__(self, "iol_symbol", sym_clean or None)

        if self.iol_market is not None:
            mkt_clean = str(self.iol_market).strip().upper()
            object.__setattr__(self, "iol_market", mkt_clean or None)

        if self.fixed_income_terms is not None and not isinstance(self.fixed_income_terms, FixedIncomeTerms):
            object.__setattr__(self, "fixed_income_terms", FixedIncomeTerms.from_dict(self.fixed_income_terms))
        if self.dcf_view is not None and not isinstance(self.dcf_view, DCFViewConfig):
            object.__setattr__(self, "dcf_view", DCFViewConfig.from_dict(self.dcf_view))

        # Fixed-income ETFs must not be forced to provide contractual schedules;
        # individual debt subtypes should carry FixedIncomeTerms when valued contractually.
        if self.asset_subtype == "fixed_income_etf" and self.fixed_income_terms is not None:
            raise ConfigurationError(
                f"Instrument '{iid}': fixed_income_etf must not carry an individual contractual bond schedule.",
                asset_id=iid,
            )

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstrumentConfig":
        payload = dict(data)
        forbidden = REMOVED_LEGACY_KEYS.intersection(payload.keys())
        if forbidden:
            raise ConfigurationError(
                f"Removed legacy parameter(s) are not supported: {sorted(forbidden)}.",
                asset_id=payload.get("instrument_id") or payload.get("ticker"),
            )
        if "fixed_income_terms" in payload and isinstance(payload["fixed_income_terms"], dict):
            payload["fixed_income_terms"] = FixedIncomeTerms.from_dict(payload["fixed_income_terms"])
        if "dcf_view" in payload and isinstance(payload["dcf_view"], dict):
            payload["dcf_view"] = DCFViewConfig.from_dict(payload["dcf_view"])
        if "sell_broker_priority" in payload and payload["sell_broker_priority"] is not None:
            payload["sell_broker_priority"] = tuple(payload["sell_broker_priority"])
        return cls(**payload)


@dataclass(frozen=True)
class BrokerPosition:
    """Holding of an instrument inside a specific broker (distinct from instrument risk identity)."""

    instrument_id: str
    broker_id: str
    quantity: float
    average_cost: Optional[float] = None
    cost_currency: str = "USD"

    def __post_init__(self) -> None:
        iid = str(self.instrument_id or "").strip().upper()
        bid = str(self.broker_id or "").strip()
        if not iid:
            raise ConfigurationError("BrokerPosition requires non-empty 'instrument_id'.")
        if not bid:
            raise ConfigurationError(f"BrokerPosition for '{iid}' requires non-empty 'broker_id'.", asset_id=iid)
        qty = _check_finite(self.quantity, "quantity", asset_id=iid)
        if qty < 0.0:
            raise ConfigurationError(f"BrokerPosition for '{iid}' in '{bid}': quantity must be >= 0.", asset_id=iid)
        if self.average_cost is not None:
            ac = _check_finite(self.average_cost, "average_cost", asset_id=iid)
            if ac < 0.0:
                raise ConfigurationError(f"BrokerPosition for '{iid}': average_cost must be >= 0.", asset_id=iid)
            object.__setattr__(self, "average_cost", ac)
        object.__setattr__(self, "instrument_id", iid)
        object.__setattr__(self, "broker_id", bid)
        object.__setattr__(self, "quantity", qty)
        object.__setattr__(self, "cost_currency", str(self.cost_currency or "USD").strip().upper())

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrokerPosition":
        return cls(**data)


# =============================================================================
# Sub-configurations: Black-Litterman, VIX/Risk, Constraints, Optimization, Rebalancing, IOL
# =============================================================================

@dataclass(frozen=True)
class BlackLittermanConfig:
    """Editable expected-returns and Black-Litterman configuration (annual USD basis)."""

    risk_aversion: float = 2.5
    tau: float = 0.05
    rf_rate: Optional[float] = None
    default_rf_rate: float = 0.04
    default_dcf_horizon_years: float = 5.0
    default_dcf_confidence: float = 0.50
    prior_weights_policy: str = "explicit_or_warned_equal_weight"
    reference_weights: Dict[str, float] = field(default_factory=dict)
    require_explicit_reference_weights: bool = False
    fixed_income_bl_view_mode: str = "metric_only"
    fixed_income_view_confidence: float = 0.35

    def __post_init__(self) -> None:
        ra = _check_finite(self.risk_aversion, "risk_aversion")
        if ra <= 0.0:
            raise ConfigurationError("'risk_aversion' must be > 0.")
        tau = _check_finite(self.tau, "tau")
        if tau <= 0.0 or tau > 1.0:
            raise ConfigurationError("'tau' must be in (0, 1].")
        if self.rf_rate is not None:
            rf = _check_finite(self.rf_rate, "rf_rate")
            if rf < -0.10 or rf > 0.50:
                raise ConfigurationError("'rf_rate' must be in [-0.10, 0.50].")
            object.__setattr__(self, "rf_rate", rf)
        def_rf = _check_finite(self.default_rf_rate, "default_rf_rate")
        if def_rf < -0.10 or def_rf > 0.50:
            raise ConfigurationError("'default_rf_rate' must be in [-0.10, 0.50].")
        dh = _check_finite(self.default_dcf_horizon_years, "default_dcf_horizon_years")
        if dh <= 0.0:
            raise ConfigurationError("'default_dcf_horizon_years' must be > 0.")
        dc = _check_finite(self.default_dcf_confidence, "default_dcf_confidence")
        if dc < 0.0 or dc > 1.0:
            raise ConfigurationError("'default_dcf_confidence' must be in [0.0, 1.0].")
        fic = _check_finite(self.fixed_income_view_confidence, "fixed_income_view_confidence")
        if fic < 0.0 or fic > 1.0:
            raise ConfigurationError("'fixed_income_view_confidence' must be in [0.0, 1.0].")

        if self.prior_weights_policy not in VALID_PRIOR_POLICIES:
            raise ConfigurationError(
                f"Invalid prior_weights_policy '{self.prior_weights_policy}'. Expected one of {VALID_PRIOR_POLICIES}."
            )
        if self.fixed_income_bl_view_mode not in VALID_FI_BL_MODES:
            raise ConfigurationError(
                f"Invalid fixed_income_bl_view_mode '{self.fixed_income_bl_view_mode}'. Expected one of {VALID_FI_BL_MODES}."
            )

        clean_ref: Dict[str, float] = {}
        for k, v in (self.reference_weights or {}).items():
            fv = _check_finite(v, f"reference_weights[{k}]")
            if fv < 0.0:
                raise ConfigurationError(f"reference_weights['{k}'] must be >= 0.")
            clean_ref[str(k).strip().upper()] = fv

        object.__setattr__(self, "risk_aversion", ra)
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "default_rf_rate", def_rf)
        object.__setattr__(self, "default_dcf_horizon_years", dh)
        object.__setattr__(self, "default_dcf_confidence", dc)
        object.__setattr__(self, "fixed_income_view_confidence", fic)
        object.__setattr__(self, "reference_weights", clean_ref)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlackLittermanConfig":
        return cls(**data)


@dataclass(frozen=True)
class VixRegimeConfig:
    """Editable VIX dynamic stress-weight schedule and regime classification settings.

    Default schedule for current stress weight p in [0, 1]:
        - VIX < 15.0: p = 0.15
        - 15.0 <= VIX <= 20.0: linear interpolation from 0.15 to 0.35
        - 20.0 < VIX <= 30.0: p = 0.35
        - VIX > 30.0: p = 0.60 (deliberate regime-shift discontinuity above 30)
    """

    vix_low: float = 15.0
    vix_mid_low: float = 20.0
    vix_high: float = 30.0
    p_low: float = 0.15
    p_mid: float = 0.35
    p_high: float = 0.60
    crisis_vix_threshold: float = 25.0
    max_vix_staleness_days: int = 7
    manual_stress_weight: Optional[float] = None
    regime_classification_method: str = "vix_history"
    rolling_vol_weeks: int = 12
    rolling_vol_quantile: float = 0.75
    min_calm_weeks: int = 26
    min_crisis_weeks: int = 12
    insufficient_regime_policy: str = "error"

    def __post_init__(self) -> None:
        vl = _check_finite(self.vix_low, "vix_low")
        vml = _check_finite(self.vix_mid_low, "vix_mid_low")
        vh = _check_finite(self.vix_high, "vix_high")
        if not (0.0 < vl < vml < vh):
            raise ConfigurationError(
                f"VIX thresholds must satisfy 0 < vix_low ({vl}) < vix_mid_low ({vml}) < vix_high ({vh})."
            )

        pl = _check_finite(self.p_low, "p_low")
        pm = _check_finite(self.p_mid, "p_mid")
        ph = _check_finite(self.p_high, "p_high")
        for name, val in (("p_low", pl), ("p_mid", pm), ("p_high", ph)):
            if val < 0.0 or val > 1.0:
                raise ConfigurationError(f"VIX stress weight '{name}' ({val}) must be in [0.0, 1.0].")
        if not (pl <= pm <= ph):
            raise ConfigurationError(
                f"VIX stress weights must be non-decreasing: p_low ({pl}) <= p_mid ({pm}) <= p_high ({ph})."
            )

        cth = _check_finite(self.crisis_vix_threshold, "crisis_vix_threshold")
        if cth <= 0.0:
            raise ConfigurationError("'crisis_vix_threshold' must be > 0.")

        if self.manual_stress_weight is not None:
            mp = _check_finite(self.manual_stress_weight, "manual_stress_weight")
            if mp < 0.0 or mp > 1.0:
                raise ConfigurationError("'manual_stress_weight' must be in [0.0, 1.0].")
            object.__setattr__(self, "manual_stress_weight", mp)

        if self.max_vix_staleness_days < 0:
            raise ConfigurationError("'max_vix_staleness_days' must be >= 0.")
        if self.regime_classification_method not in VALID_REGIME_CLASSIFIERS:
            raise ConfigurationError(
                f"Invalid regime_classification_method '{self.regime_classification_method}'. Expected one of {VALID_REGIME_CLASSIFIERS}."
            )
        if self.rolling_vol_weeks < 2:
            raise ConfigurationError("'rolling_vol_weeks' must be >= 2.")
        rvq = _check_finite(self.rolling_vol_quantile, "rolling_vol_quantile")
        if rvq <= 0.0 or rvq >= 1.0:
            raise ConfigurationError("'rolling_vol_quantile' must be in (0.0, 1.0).")
        if self.min_calm_weeks < 2 or self.min_crisis_weeks < 2:
            raise ConfigurationError("'min_calm_weeks' and 'min_crisis_weeks' must be >= 2.")
        if self.insufficient_regime_policy not in VALID_INSUFFICIENT_REGIME_POLICIES:
            raise ConfigurationError(
                f"Invalid insufficient_regime_policy '{self.insufficient_regime_policy}'."
            )

        object.__setattr__(self, "vix_low", vl)
        object.__setattr__(self, "vix_mid_low", vml)
        object.__setattr__(self, "vix_high", vh)
        object.__setattr__(self, "p_low", pl)
        object.__setattr__(self, "p_mid", pm)
        object.__setattr__(self, "p_high", ph)
        object.__setattr__(self, "crisis_vix_threshold", cth)
        object.__setattr__(self, "rolling_vol_quantile", rvq)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VixRegimeConfig":
        return cls(**data)


@dataclass(frozen=True)
class RiskModelConfig:
    """Editable weekly multi-asset risk and covariance estimation configuration."""

    return_frequency: str = "W-FRI"
    weekly_cutoff_tz: str = "America/New_York"
    weekly_cutoff_hour: int = 17
    period: str = "10y"
    duplicate_date_policy: str = "last"
    fx_max_staleness_days: int = 5
    min_fx_coverage: float = 0.95
    min_weekly_observations: int = 52
    min_asset_coverage: float = 0.85
    covariance_estimator: str = "vix_regime"
    shrinkage_method: str = "ledoit_wolf"
    ewma_half_life_weeks: float = 26.0
    ewma_mean_mode: str = "zero"
    ewma_gap_policy: str = "elapsed_weeks_decay"
    allow_explicit_proxies: bool = False
    vix_regime: VixRegimeConfig = field(default_factory=VixRegimeConfig)

    def __post_init__(self) -> None:
        if self.return_frequency != "W-FRI":
            raise ConfigurationError("Only 'W-FRI' weekly Friday-to-Friday return frequency is supported.")
        if not (0 <= self.weekly_cutoff_hour <= 23):
            raise ConfigurationError("'weekly_cutoff_hour' must be between 0 and 23.")
        if self.duplicate_date_policy not in VALID_DUPLICATE_DATE_POLICIES:
            raise ConfigurationError(f"Invalid duplicate_date_policy '{self.duplicate_date_policy}'.")
        if self.fx_max_staleness_days < 0:
            raise ConfigurationError("'fx_max_staleness_days' must be >= 0.")
        mfc = _check_finite(self.min_fx_coverage, "min_fx_coverage")
        if mfc <= 0.0 or mfc > 1.0:
            raise ConfigurationError("'min_fx_coverage' must be in (0.0, 1.0].")
        if self.min_weekly_observations < 4:
            raise ConfigurationError("'min_weekly_observations' must be >= 4.")
        mac = _check_finite(self.min_asset_coverage, "min_asset_coverage")
        if mac <= 0.0 or mac > 1.0:
            raise ConfigurationError("'min_asset_coverage' must be in (0.0, 1.0].")
        if self.covariance_estimator not in VALID_COV_ESTIMATORS:
            raise ConfigurationError(
                f"Invalid covariance_estimator '{self.covariance_estimator}'. Expected one of {VALID_COV_ESTIMATORS}."
            )
        if self.shrinkage_method not in VALID_SHRINKAGE_METHODS:
            raise ConfigurationError(
                f"Invalid shrinkage_method '{self.shrinkage_method}'. Expected one of {VALID_SHRINKAGE_METHODS}."
            )
        hl = _check_finite(self.ewma_half_life_weeks, "ewma_half_life_weeks")
        if hl <= 0.0:
            raise ConfigurationError("'ewma_half_life_weeks' must be > 0.")
        if self.ewma_mean_mode not in VALID_EWMA_MEAN_MODES:
            raise ConfigurationError(
                f"Invalid ewma_mean_mode '{self.ewma_mean_mode}'. Expected one of {VALID_EWMA_MEAN_MODES}."
            )
        if self.ewma_gap_policy not in VALID_EWMA_GAP_POLICIES:
            raise ConfigurationError(
                f"Invalid ewma_gap_policy '{self.ewma_gap_policy}'. Expected one of {VALID_EWMA_GAP_POLICIES}."
            )

        vr = self.vix_regime if isinstance(self.vix_regime, VixRegimeConfig) else VixRegimeConfig.from_dict(self.vix_regime)
        object.__setattr__(self, "min_fx_coverage", mfc)
        object.__setattr__(self, "min_asset_coverage", mac)
        object.__setattr__(self, "ewma_half_life_weeks", hl)
        object.__setattr__(self, "vix_regime", vr)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskModelConfig":
        payload = dict(data)
        if "vix_regime" in payload and isinstance(payload["vix_regime"], dict):
            payload["vix_regime"] = VixRegimeConfig.from_dict(payload["vix_regime"])
        return cls(**payload)


@dataclass(frozen=True)
class ConstraintsConfig:
    """Editable portfolio allocation constraints (neutral defaults, no fixed (0.05, 0.20)).

    Defaults:
        - Long-only (``w_i >= 0``), full investment (``sum(w_i) == 1.0``).
        - No mandatory individual minimums or arbitrary individual caps.
        - Optional per-asset bounds ``{instrument_id: (min_w, max_w)}``.
        - Optional per-risk-class bounds ``{risk_class: (min_w, max_w)}`` (e.g. ``{"crypto": (0.0, 0.10)}``).
        - Optional ``hhi_limit`` and ``max_sector_weight`` (disabled by default).
    """

    long_only: bool = True
    budget_sum: float = 1.0
    asset_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    class_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    hhi_limit: Optional[float] = None
    max_sector_weight: Optional[float] = None
    solver_tolerance: float = 1e-8
    feasibility_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if not self.long_only:
            raise ConfigurationError("Short-selling / leverage is not supported; 'long_only' must be True.")
        bsum = _check_finite(self.budget_sum, "budget_sum")
        if not math.isclose(bsum, 1.0, abs_tol=1e-9):
            raise ConfigurationError("'budget_sum' must be 1.0 (100% unlevered budget).")

        clean_ab: Dict[str, Tuple[float, float]] = {}
        for k, pair in (self.asset_bounds or {}).items():
            if len(pair) != 2:
                raise ConfigurationError(f"asset_bounds['{k}'] must be a (min_weight, max_weight) tuple.")
            lo = _check_finite(pair[0], f"asset_bounds[{k}].min")
            hi = _check_finite(pair[1], f"asset_bounds[{k}].max")
            if lo < 0.0 or hi > 1.0 or lo > hi:
                raise ConfigurationError(
                    f"asset_bounds['{k}'] = ({lo}, {hi}) must satisfy 0.0 <= min <= max <= 1.0.",
                    asset_id=str(k).upper(),
                )
            clean_ab[str(k).strip().upper()] = (lo, hi)

        clean_cb: Dict[str, Tuple[float, float]] = {}
        for k, pair in (self.class_bounds or {}).items():
            rk = str(k).strip().lower()
            if rk not in VALID_RISK_CLASSES:
                raise ConfigurationError(
                    f"class_bounds key '{k}' is not a valid risk_class ({VALID_RISK_CLASSES})."
                )
            if len(pair) != 2:
                raise ConfigurationError(f"class_bounds['{k}'] must be a (min_weight, max_weight) tuple.")
            lo = _check_finite(pair[0], f"class_bounds[{k}].min")
            hi = _check_finite(pair[1], f"class_bounds[{k}].max")
            if lo < 0.0 or hi > 1.0 or lo > hi:
                raise ConfigurationError(
                    f"class_bounds['{k}'] = ({lo}, {hi}) must satisfy 0.0 <= min <= max <= 1.0."
                )
            clean_cb[rk] = (lo, hi)

        if self.hhi_limit is not None:
            hl = _check_finite(self.hhi_limit, "hhi_limit")
            if hl <= 0.0 or hl > 1.0:
                raise ConfigurationError("'hhi_limit' must be in (0.0, 1.0].")
            object.__setattr__(self, "hhi_limit", hl)

        if self.max_sector_weight is not None:
            msw = _check_finite(self.max_sector_weight, "max_sector_weight")
            if msw <= 0.0 or msw > 1.0:
                raise ConfigurationError("'max_sector_weight' must be in (0.0, 1.0].")
            object.__setattr__(self, "max_sector_weight", msw)

        stol = _check_finite(self.solver_tolerance, "solver_tolerance")
        ftol = _check_finite(self.feasibility_tolerance, "feasibility_tolerance")
        if stol <= 0.0 or ftol <= 0.0:
            raise ConfigurationError("Tolerances must be strictly positive.")

        object.__setattr__(self, "budget_sum", bsum)
        object.__setattr__(self, "asset_bounds", clean_ab)
        object.__setattr__(self, "class_bounds", clean_cb)
        object.__setattr__(self, "solver_tolerance", stol)
        object.__setattr__(self, "feasibility_tolerance", ftol)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstraintsConfig":
        payload = dict(data)
        if "asset_bounds" in payload and payload["asset_bounds"] is not None:
            payload["asset_bounds"] = {k: tuple(v) for k, v in payload["asset_bounds"].items()}
        if "class_bounds" in payload and payload["class_bounds"] is not None:
            payload["class_bounds"] = {k: tuple(v) for k, v in payload["class_bounds"].items()}
        return cls(**payload)


@dataclass(frozen=True)
class OptimizationConfig:
    """Editable optimizer selection and solver settings."""

    selected_strategy: str = "max_sharpe"
    strategies_to_run: Tuple[str, ...] = VALID_STRATEGIES
    hrp_linkage_method: str = "single"
    cvar_confidence: float = 0.95
    non_positive_excess_return_policy: str = "block"
    include_stress_tests: bool = True

    def __post_init__(self) -> None:
        if self.selected_strategy not in VALID_STRATEGIES:
            raise ConfigurationError(
                f"Invalid selected_strategy '{self.selected_strategy}'. Expected one of {VALID_STRATEGIES}."
            )
        strats = tuple(dict.fromkeys(str(s).strip() for s in self.strategies_to_run))
        if not strats:
            raise ConfigurationError("'strategies_to_run' must contain at least one strategy.")
        for s in strats:
            if s not in VALID_STRATEGIES:
                raise ConfigurationError(f"Invalid strategy '{s}' in strategies_to_run.")
        if self.selected_strategy not in strats:
            strats = (self.selected_strategy,) + strats
        if self.hrp_linkage_method not in VALID_HRP_LINKAGES:
            raise ConfigurationError(
                f"Invalid hrp_linkage_method '{self.hrp_linkage_method}'. Expected one of {VALID_HRP_LINKAGES}."
            )
        conf = _check_finite(self.cvar_confidence, "cvar_confidence")
        if conf <= 0.5 or conf >= 1.0:
            raise ConfigurationError("'cvar_confidence' must be in (0.5, 1.0).")
        if self.non_positive_excess_return_policy not in VALID_NON_POSITIVE_EXCESS_POLICIES:
            raise ConfigurationError(
                f"Invalid non_positive_excess_return_policy '{self.non_positive_excess_return_policy}'. "
                f"Expected one of {VALID_NON_POSITIVE_EXCESS_POLICIES}."
            )

        object.__setattr__(self, "strategies_to_run", strats)
        object.__setattr__(self, "cvar_confidence", conf)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationConfig":
        payload = dict(data)
        removed = {"simulation_samples", "random_seed"}.intersection(payload.keys())
        if removed:
            raise ConfigurationError(
                f"Removed decorative OptimizationConfig field(s) are no longer supported: {sorted(removed)} "
                "(this engine uses deterministic optimization with no Monte Carlo simulation)."
            )
        if "strategies_to_run" in payload and payload["strategies_to_run"] is not None:
            payload["strategies_to_run"] = tuple(payload["strategies_to_run"])
        return cls(**payload)


@dataclass(frozen=True)
class RebalancingConfig:
    """Editable fractional rebalancing thresholds and segregated broker cash settings.

    Rule:
        When both thresholds are enabled, a trade is eligible ONLY if
        ``abs(target_weight - current_weight) >= min_weight_diff`` AND
        ``gross_notional_usd >= min_notional_usd`` (using ``threshold_comparison``).
        Cash is strictly segregated per broker: scalar ``available_cash_usd`` and
        ``cash_flow_usd`` are valid only for single-broker portfolios (or when 0.0).
        Multi-broker setups MUST assign non-zero cash/flows via
        ``available_cash_by_broker`` and ``cash_flow_by_broker``.
    """

    min_weight_diff: Optional[float] = 0.01  # 1 percentage point (absolute weight difference)
    use_weight_threshold: bool = True
    min_notional_usd: Optional[float] = 25.0  # Editable (e.g., 50.0 USD)
    use_notional_threshold: bool = True
    threshold_comparison: str = ">="
    available_cash_usd: float = 0.0
    cash_flow_usd: float = 0.0
    available_cash_by_broker: Dict[str, float] = field(default_factory=dict)
    cash_flow_by_broker: Dict[str, float] = field(default_factory=dict)
    enforce_hard_bounds_after_rebalance: bool = True

    def __post_init__(self) -> None:
        if self.min_weight_diff is not None:
            mwd = _check_finite(self.min_weight_diff, "min_weight_diff")
            if mwd < 0.0 or mwd > 1.0:
                raise ConfigurationError("'min_weight_diff' must be in [0.0, 1.0] (absolute weight points).")
            object.__setattr__(self, "min_weight_diff", mwd)
        if self.min_notional_usd is not None:
            mn = _check_finite(self.min_notional_usd, "min_notional_usd")
            if mn < 0.0:
                raise ConfigurationError("'min_notional_usd' must be >= 0.")
            object.__setattr__(self, "min_notional_usd", mn)
        if self.threshold_comparison not in (">=", ">"):
            raise ConfigurationError("'threshold_comparison' must be '>=' or '>'.")
        cash = _check_finite(self.available_cash_usd, "available_cash_usd")
        if cash < 0.0:
            raise ConfigurationError("'available_cash_usd' must be >= 0.")
        cf = _check_finite(self.cash_flow_usd, "cash_flow_usd")

        clean_cash_by_broker: Dict[str, float] = {}
        for bid, val in (self.available_cash_by_broker or {}).items():
            key = str(bid).strip()
            if not key:
                raise ConfigurationError("available_cash_by_broker keys must be non-empty broker IDs.")
            fval = _check_finite(val, f"available_cash_by_broker[{key}]")
            if fval < 0.0:
                raise ConfigurationError(f"available_cash_by_broker['{key}'] must be >= 0.")
            clean_cash_by_broker[key] = fval

        clean_flow_by_broker: Dict[str, float] = {}
        for bid, val in (self.cash_flow_by_broker or {}).items():
            key = str(bid).strip()
            if not key:
                raise ConfigurationError("cash_flow_by_broker keys must be non-empty broker IDs.")
            fval = _check_finite(val, f"cash_flow_by_broker[{key}]")
            clean_flow_by_broker[key] = fval

        object.__setattr__(self, "available_cash_usd", cash)
        object.__setattr__(self, "cash_flow_usd", cf)
        object.__setattr__(self, "available_cash_by_broker", clean_cash_by_broker)
        object.__setattr__(self, "cash_flow_by_broker", clean_flow_by_broker)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RebalancingConfig":
        return cls(**data)


def _validate_https_url(raw_url: Any, field_name: str) -> str:
    """Validate that a configurable endpoint URL is a well-formed HTTPS URL without credentials."""
    url = str(raw_url or "").strip()
    if not url:
        raise ConfigurationError(f"'{field_name}' must be a non-empty HTTPS URL.")
    if any(ch.isspace() or ord(ch) < 32 for ch in url):
        raise ConfigurationError(f"'{field_name}' contains whitespace or control characters.")
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ConfigurationError(
            f"'{field_name}' must use the 'https://' scheme with a valid hostname (got '{url}')."
        )
    if parsed.username is not None or parsed.password is not None:
        raise ConfigurationError(
            f"'{field_name}' must not contain embedded URL credentials."
        )
    return url


@dataclass(frozen=True)
class IOLConfig:
    """Editable public IOL fixed-income quote and historical price acquisition settings."""

    bonos_url: str = "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"
    ons_url: str = "https://iol.invertironline.com/mercado/cotizaciones/argentina/obligaciones-negociables/todos"
    historical_url: str = "https://iol.invertironline.com/Titulo/DatosHistoricos"
    default_market: str = "BCBA"
    timeout_seconds: float = 15.0
    max_mep_time_skew_seconds: float = 300.0
    max_mep_market_time_diff_seconds: float = 3600.0
    max_mep_quote_age_seconds: Optional[float] = None
    allow_mep_without_market_timestamp: bool = False
    mep_plausible_min: float = 50.0
    mep_plausible_max: float = 50000.0
    low_liquidity_amount_threshold: float = 1000.0
    wide_spread_bps_threshold: float = 300.0
    min_valid_rows: int = 1
    fetch_historical_prices: bool = True
    historical_price_field: str = "close"
    historical_chunk_days: int = 1825
    historical_max_retries: int = 2
    historical_retry_backoff_seconds: float = 0.5
    historical_request_delay_seconds: float = 0.15
    allow_usd_cable_history: bool = False
    drop_zero_volume_rows: bool = True

    def __post_init__(self) -> None:
        t = _check_finite(self.timeout_seconds, "timeout_seconds")
        if t <= 0.0:
            raise ConfigurationError("'timeout_seconds' must be > 0.")
        skew = _check_finite(self.max_mep_time_skew_seconds, "max_mep_time_skew_seconds")
        if skew < 0.0:
            raise ConfigurationError("'max_mep_time_skew_seconds' must be >= 0.")
        mkt_diff = _check_finite(self.max_mep_market_time_diff_seconds, "max_mep_market_time_diff_seconds")
        if mkt_diff < 0.0:
            raise ConfigurationError("'max_mep_market_time_diff_seconds' must be >= 0.")
        if self.max_mep_quote_age_seconds is not None:
            age = _check_finite(self.max_mep_quote_age_seconds, "max_mep_quote_age_seconds")
            if age < 0.0:
                raise ConfigurationError("'max_mep_quote_age_seconds' must be >= 0.")
            object.__setattr__(self, "max_mep_quote_age_seconds", age)
        m_min = _check_finite(self.mep_plausible_min, "mep_plausible_min")
        m_max = _check_finite(self.mep_plausible_max, "mep_plausible_max")
        if m_min <= 0.0 or m_max <= m_min:
            raise ConfigurationError(
                f"MEP plausible bounds must satisfy 0 < mep_plausible_min ({m_min}) < mep_plausible_max ({m_max})."
            )
        liq = _check_finite(self.low_liquidity_amount_threshold, "low_liquidity_amount_threshold")
        if liq < 0.0:
            raise ConfigurationError("'low_liquidity_amount_threshold' must be >= 0.")
        sp = _check_finite(self.wide_spread_bps_threshold, "wide_spread_bps_threshold")
        if sp < 0.0:
            raise ConfigurationError("'wide_spread_bps_threshold' must be >= 0.")
        if self.min_valid_rows < 1:
            raise ConfigurationError("'min_valid_rows' must be >= 1.")

        bonos_url = _validate_https_url(self.bonos_url, "bonos_url")
        ons_url = _validate_https_url(self.ons_url, "ons_url")
        hist_url = _validate_https_url(self.historical_url, "historical_url")
        def_mkt = str(self.default_market or "").strip().upper()
        if not def_mkt:
            raise ConfigurationError("'default_market' must be a non-empty market code.")
        if self.historical_price_field not in VALID_IOL_HISTORICAL_PRICE_FIELDS:
            raise ConfigurationError(
                f"Invalid historical_price_field '{self.historical_price_field}'. "
                f"Expected one of {VALID_IOL_HISTORICAL_PRICE_FIELDS}."
            )
        if not isinstance(self.historical_chunk_days, int) or isinstance(self.historical_chunk_days, bool) or self.historical_chunk_days < 1:
            raise ConfigurationError("'historical_chunk_days' must be a positive integer.")
        if not isinstance(self.historical_max_retries, int) or isinstance(self.historical_max_retries, bool) or self.historical_max_retries < 0:
            raise ConfigurationError("'historical_max_retries' must be an integer >= 0.")
        backoff = _check_finite(self.historical_retry_backoff_seconds, "historical_retry_backoff_seconds")
        if backoff < 0.0:
            raise ConfigurationError("'historical_retry_backoff_seconds' must be >= 0.")
        delay = _check_finite(self.historical_request_delay_seconds, "historical_request_delay_seconds")
        if delay < 0.0:
            raise ConfigurationError("'historical_request_delay_seconds' must be >= 0.")

        object.__setattr__(self, "bonos_url", bonos_url)
        object.__setattr__(self, "ons_url", ons_url)
        object.__setattr__(self, "historical_url", hist_url)
        object.__setattr__(self, "default_market", def_mkt)
        object.__setattr__(self, "timeout_seconds", t)
        object.__setattr__(self, "max_mep_time_skew_seconds", skew)
        object.__setattr__(self, "max_mep_market_time_diff_seconds", mkt_diff)
        object.__setattr__(self, "mep_plausible_min", m_min)
        object.__setattr__(self, "mep_plausible_max", m_max)
        object.__setattr__(self, "low_liquidity_amount_threshold", liq)
        object.__setattr__(self, "wide_spread_bps_threshold", sp)
        object.__setattr__(self, "historical_retry_backoff_seconds", backoff)
        object.__setattr__(self, "historical_request_delay_seconds", delay)
        object.__setattr__(self, "drop_zero_volume_rows", bool(self.drop_zero_volume_rows))

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IOLConfig":
        return cls(**data)


# =============================================================================
# Master stateless request configuration
# =============================================================================

@dataclass(frozen=True)
class PortfolioAnalysisConfig:
    """Single validated top-level configuration contract for portfolio analysis.

    Exposes two primary input sections (``variable_income`` and ``fixed_income``)
    alongside brokers, positions, Black-Litterman, weekly risk/VIX regime,
    constraints, optimization, rebalancing, and IOL quote settings.
    """

    variable_income: Tuple[InstrumentConfig, ...] = field(default_factory=tuple)
    fixed_income: Tuple[InstrumentConfig, ...] = field(default_factory=tuple)
    brokers: Tuple[BrokerConfig, ...] = field(default_factory=tuple)
    positions: Tuple[BrokerPosition, ...] = field(default_factory=tuple)
    black_litterman: BlackLittermanConfig = field(default_factory=BlackLittermanConfig)
    risk: RiskModelConfig = field(default_factory=RiskModelConfig)
    constraints: ConstraintsConfig = field(default_factory=ConstraintsConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    rebalancing: RebalancingConfig = field(default_factory=RebalancingConfig)
    iol: IOLConfig = field(default_factory=IOLConfig)
    as_of_date: Optional[date] = None

    def __post_init__(self) -> None:
        vi = tuple(
            inst if isinstance(inst, InstrumentConfig) else InstrumentConfig.from_dict({**inst, "section": "variable_income"})
            for inst in self.variable_income
        )
        fi = tuple(
            inst if isinstance(inst, InstrumentConfig) else InstrumentConfig.from_dict({**inst, "section": "fixed_income"})
            for inst in self.fixed_income
        )
        for inst in vi:
            if inst.section != "variable_income":
                raise ConfigurationError(
                    f"Instrument '{inst.instrument_id}' in 'variable_income' section has section='{inst.section}'.",
                    asset_id=inst.instrument_id,
                )
        for inst in fi:
            if inst.section != "fixed_income":
                raise ConfigurationError(
                    f"Instrument '{inst.instrument_id}' in 'fixed_income' section has section='{inst.section}'.",
                    asset_id=inst.instrument_id,
                )

        all_instruments = vi + fi
        if not all_instruments:
            raise ConfigurationError("At least one instrument must be configured across variable_income or fixed_income.")

        seen_ids: set[str] = set()
        for inst in all_instruments:
            if inst.instrument_id in seen_ids:
                raise ConfigurationError(
                    f"Duplicate instrument_id '{inst.instrument_id}'. Represent multi-broker holdings in 'positions', not duplicate instruments.",
                    asset_id=inst.instrument_id,
                )
            seen_ids.add(inst.instrument_id)

        brokers_tuple = tuple(
            b if isinstance(b, BrokerConfig) else BrokerConfig.from_dict(b)
            for b in self.brokers
        )
        broker_ids: set[str] = set()
        for b in brokers_tuple:
            if b.broker_id in broker_ids:
                raise ConfigurationError(f"Duplicate broker_id '{b.broker_id}'.")
            broker_ids.add(b.broker_id)

        positions_tuple = tuple(
            p if isinstance(p, BrokerPosition) else BrokerPosition.from_dict(p)
            for p in self.positions
        )

        # Validate broker references
        for inst in all_instruments:
            if inst.default_buy_broker_id is not None and broker_ids and inst.default_buy_broker_id not in broker_ids:
                raise ConfigurationError(
                    f"Instrument '{inst.instrument_id}' references unknown default_buy_broker_id '{inst.default_buy_broker_id}'.",
                    asset_id=inst.instrument_id,
                )
            for sbid in inst.sell_broker_priority:
                if broker_ids and sbid not in broker_ids:
                    raise ConfigurationError(
                        f"Instrument '{inst.instrument_id}' references unknown broker '{sbid}' in sell_broker_priority.",
                        asset_id=inst.instrument_id,
                    )

        seen_pos_keys: set[Tuple[str, str]] = set()
        for pos in positions_tuple:
            if pos.instrument_id not in seen_ids:
                raise ConfigurationError(
                    f"Position references unknown instrument_id '{pos.instrument_id}'.",
                    asset_id=pos.instrument_id,
                )
            if broker_ids and pos.broker_id not in broker_ids:
                raise ConfigurationError(
                    f"Position for '{pos.instrument_id}' references unknown broker_id '{pos.broker_id}'.",
                    asset_id=pos.instrument_id,
                )
            key = (pos.instrument_id, pos.broker_id)
            if key in seen_pos_keys:
                raise ConfigurationError(
                    f"Duplicate position for instrument '{pos.instrument_id}' at broker '{pos.broker_id}'. Aggregate within the same broker.",
                    asset_id=pos.instrument_id,
                )
            seen_pos_keys.add(key)

        bl_cfg = self.black_litterman if isinstance(self.black_litterman, BlackLittermanConfig) else BlackLittermanConfig.from_dict(self.black_litterman)
        risk_cfg = self.risk if isinstance(self.risk, RiskModelConfig) else RiskModelConfig.from_dict(self.risk)
        con_cfg = self.constraints if isinstance(self.constraints, ConstraintsConfig) else ConstraintsConfig.from_dict(self.constraints)
        opt_cfg = self.optimization if isinstance(self.optimization, OptimizationConfig) else OptimizationConfig.from_dict(self.optimization)
        reb_cfg = self.rebalancing if isinstance(self.rebalancing, RebalancingConfig) else RebalancingConfig.from_dict(self.rebalancing)
        iol_cfg = self.iol if isinstance(self.iol, IOLConfig) else IOLConfig.from_dict(self.iol)

        # Validate per-broker cash / flow assignments against configured brokers (AUD-03)
        for bid in reb_cfg.available_cash_by_broker:
            if broker_ids and bid not in broker_ids:
                raise ConfigurationError(
                    f"rebalancing.available_cash_by_broker references unknown broker_id '{bid}'."
                )
        for bid in reb_cfg.cash_flow_by_broker:
            if broker_ids and bid not in broker_ids:
                raise ConfigurationError(
                    f"rebalancing.cash_flow_by_broker references unknown broker_id '{bid}'."
                )
        if len(brokers_tuple) > 1:
            if abs(reb_cfg.available_cash_usd) > 0.0:
                raise ConfigurationError(
                    "Scalar 'rebalancing.available_cash_usd' is ambiguous with multiple brokers; "
                    "assign non-zero balances explicitly via 'rebalancing.available_cash_by_broker'."
                )
            if abs(reb_cfg.cash_flow_usd) > 0.0:
                raise ConfigurationError(
                    "Scalar 'rebalancing.cash_flow_usd' is ambiguous with multiple brokers; "
                    "assign non-zero cash flows explicitly via 'rebalancing.cash_flow_by_broker'."
                )
        elif len(brokers_tuple) == 1:
            if abs(reb_cfg.available_cash_usd) > 0.0 and reb_cfg.available_cash_by_broker:
                raise ConfigurationError(
                    "Specify either 'available_cash_usd' or 'available_cash_by_broker', not both, to prevent double-counting."
                )
            if abs(reb_cfg.cash_flow_usd) > 0.0 and reb_cfg.cash_flow_by_broker:
                raise ConfigurationError(
                    "Specify either 'cash_flow_usd' or 'cash_flow_by_broker', not both, to prevent double-counting."
                )

        as_of = None if self.as_of_date is None else _parse_date(self.as_of_date, "as_of_date")

        object.__setattr__(self, "variable_income", vi)
        object.__setattr__(self, "fixed_income", fi)
        object.__setattr__(self, "brokers", brokers_tuple)
        object.__setattr__(self, "positions", positions_tuple)
        object.__setattr__(self, "black_litterman", bl_cfg)
        object.__setattr__(self, "risk", risk_cfg)
        object.__setattr__(self, "constraints", con_cfg)
        object.__setattr__(self, "optimization", opt_cfg)
        object.__setattr__(self, "rebalancing", reb_cfg)
        object.__setattr__(self, "iol", iol_cfg)
        object.__setattr__(self, "as_of_date", as_of)

    @property
    def instruments(self) -> Tuple[InstrumentConfig, ...]:
        """All configured instruments in deterministic order (variable_income then fixed_income)."""
        return self.variable_income + self.fixed_income

    @property
    def instrument_map(self) -> Dict[str, InstrumentConfig]:
        return {inst.instrument_id: inst for inst in self.instruments}

    @property
    def broker_map(self) -> Dict[str, BrokerConfig]:
        return {b.broker_id: b for b in self.brokers}

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioAnalysisConfig":
        payload = dict(data)
        forbidden = REMOVED_LEGACY_KEYS.intersection(payload.keys())
        if forbidden:
            raise ConfigurationError(
                f"Removed legacy configuration parameter(s) are not supported: {sorted(forbidden)}."
            )
        if "instruments" in payload and ("variable_income" not in payload and "fixed_income" not in payload):
            vi_list, fi_list = [], []
            for raw in payload.pop("instruments"):
                inst = raw if isinstance(raw, InstrumentConfig) else InstrumentConfig.from_dict(raw)
                if inst.section == "fixed_income":
                    fi_list.append(inst)
                else:
                    vi_list.append(inst)
            payload["variable_income"] = tuple(vi_list)
            payload["fixed_income"] = tuple(fi_list)
        return cls(**payload)


# =============================================================================
# Structured result dataclasses (all JSON-serializable, stateless)
# =============================================================================

@dataclass
class IOLQuoteResult:
    """Point-in-time USD quote and metadata acquired from public IOL panels."""

    instrument_id: str
    requested_ticker: str
    species_used: str
    panel_source: str  # "iol_bonos" | "iol_ons"
    original_currency: str
    final_currency: str
    usd_settlement_type: str  # "USD_LOCAL_D" | "USD_CABLE_C" | "USD_MEP_IMPLIED"
    conversion_method: str  # "D" | "C" | "ARS/MEP"
    price_original: float
    price_usd: float
    quote_unit: str
    observed_at: str  # UTC ISO timestamp of fetch
    market_timestamp: Optional[str] = None  # Only when explicitly provided by source
    instrument_name: str = ""
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_close: Optional[float] = None
    daily_change_pct: Optional[float] = None
    volume_amount: Optional[float] = None
    published_tir: Optional[float] = None
    published_md: Optional[float] = None
    mep_reference: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))


@dataclass
class FixedIncomeMetricsResult:
    """Contractual fixed-income valuation metrics derived from manual cashflows."""

    instrument_id: str
    status: str  # "ok" | "not_evaluable_under_available_method" | "multiple_roots" | "no_root" | "insufficient_data"
    cashflow_currency: str
    settlement_date: date
    maturity_date: date
    reference_nominal: float
    residual_nominal: float
    amortization_factor: float
    price_convention: str
    price_type: str
    quoted_price: Optional[float] = None
    accrued_interest: float = 0.0
    dirty_price_per_reference: Optional[float] = None
    dirty_price_usd: Optional[float] = None
    dirty_price_usd_per_original_nominal: Optional[float] = None
    quote_source_used: Optional[str] = None
    valuation_price_date: Optional[str] = None
    ytm_annual: Optional[float] = None
    macaulay_duration_years: Optional[float] = None
    modified_duration_years: Optional[float] = None
    weighted_average_life_years: Optional[float] = None
    future_cashflows: List[Dict[str, Any]] = field(default_factory=list)
    total_future_coupons: float = 0.0
    total_future_amortizations: float = 0.0
    root_count_detected: int = 0
    warnings: List[str] = field(default_factory=list)
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))


@dataclass
class DCFViewResult:
    """Computed annualized DCF view and audit metadata for a single asset."""

    instrument_id: str
    valuation_basis: str
    horizon_years: float
    current_price_usd: Optional[float]
    target_or_intrinsic_usd: Optional[float]
    annualized_expected_return: float
    confidence: float
    active_in_bl: bool
    currency: str = "USD"
    assumptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))


@dataclass
class BlackLittermanResult:
    """Complete Black-Litterman prior, views, posterior, and diagnostic trace."""

    tickers: List[str]
    rf_rate: float
    risk_aversion: float
    tau: float
    prior_policy_used: str
    reference_weights: pd.Series
    prior_excess_returns: pd.Series
    prior_total_returns: pd.Series
    posterior_excess_returns: pd.Series
    posterior_total_returns: pd.Series
    mu_sources: Dict[str, str]
    active_view_assets: List[str] = field(default_factory=list)
    inactive_view_assets: List[str] = field(default_factory=list)
    P: Optional[np.ndarray] = None
    Q_excess: Optional[np.ndarray] = None
    omega: Optional[np.ndarray] = None
    view_confidences: Dict[str, float] = field(default_factory=dict)
    dcf_view_results: Dict[str, DCFViewResult] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value({
            "tickers": self.tickers,
            "rf_rate": self.rf_rate,
            "risk_aversion": self.risk_aversion,
            "tau": self.tau,
            "prior_policy_used": self.prior_policy_used,
            "reference_weights": self.reference_weights,
            "prior_excess_returns": self.prior_excess_returns,
            "prior_total_returns": self.prior_total_returns,
            "posterior_excess_returns": self.posterior_excess_returns,
            "posterior_total_returns": self.posterior_total_returns,
            "mu_sources": self.mu_sources,
            "active_view_assets": self.active_view_assets,
            "inactive_view_assets": self.inactive_view_assets,
            "P": self.P,
            "Q_excess": self.Q_excess,
            "omega": self.omega,
            "view_confidences": self.view_confidences,
            "dcf_view_results": self.dcf_view_results,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
        })


@dataclass
class RiskModelResult:
    """Annualized weekly multi-asset covariance and dynamic VIX regime diagnostics."""

    tickers: List[str]
    cov_matrix: pd.DataFrame
    corr_matrix: pd.DataFrame
    annualized_volatilities: pd.Series
    weekly_returns: pd.DataFrame
    cov_calm: Optional[pd.DataFrame] = None
    cov_crisis: Optional[pd.DataFrame] = None
    estimator_name: str = "vix_regime"
    shrinkage_method: str = "ledoit_wolf"
    shrinkage_calm: Optional[float] = None
    shrinkage_crisis: Optional[float] = None
    vix_value: Optional[float] = None
    vix_date: Optional[str] = None
    vix_is_stale: bool = False
    stress_weight_p: Optional[float] = None
    p_source: str = "vix_schedule"
    calm_samples: int = 0
    crisis_samples: int = 0
    total_weekly_samples: int = 0
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    psd_adjusted: bool = False
    min_eigenvalue_before: Optional[float] = None
    min_eigenvalue_after: Optional[float] = None
    excluded_dates_count: int = 0
    excluded_incomplete_week_bins: List[str] = field(default_factory=list)
    last_closed_friday: Optional[str] = None
    missing_weeks_by_asset: Dict[str, int] = field(default_factory=dict)
    duplicate_dates_removed: Dict[str, int] = field(default_factory=dict)
    proxies_used: Dict[str, str] = field(default_factory=dict)
    discontinuity_note: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value({
            "tickers": self.tickers,
            "cov_matrix": self.cov_matrix,
            "corr_matrix": self.corr_matrix,
            "annualized_volatilities": self.annualized_volatilities,
            "estimator_name": self.estimator_name,
            "shrinkage_method": self.shrinkage_method,
            "shrinkage_calm": self.shrinkage_calm,
            "shrinkage_crisis": self.shrinkage_crisis,
            "vix_value": self.vix_value,
            "vix_date": self.vix_date,
            "vix_is_stale": self.vix_is_stale,
            "stress_weight_p": self.stress_weight_p,
            "p_source": self.p_source,
            "calm_samples": self.calm_samples,
            "crisis_samples": self.crisis_samples,
            "total_weekly_samples": self.total_weekly_samples,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "psd_adjusted": self.psd_adjusted,
            "min_eigenvalue_before": self.min_eigenvalue_before,
            "min_eigenvalue_after": self.min_eigenvalue_after,
            "excluded_dates_count": self.excluded_dates_count,
            "excluded_incomplete_week_bins": self.excluded_incomplete_week_bins,
            "last_closed_friday": self.last_closed_friday,
            "missing_weeks_by_asset": self.missing_weeks_by_asset,
            "duplicate_dates_removed": self.duplicate_dates_removed,
            "proxies_used": self.proxies_used,
            "discontinuity_note": self.discontinuity_note,
            "warnings": self.warnings,
        })


@dataclass
class StrategyAllocationResult:
    """Homogeneous output contract for every portfolio optimization method."""

    strategy_name: str
    tickers: List[str]
    weights: np.ndarray
    weights_by_asset: Dict[str, float]
    converged: bool
    feasible: bool
    status: str  # "optimal" | "constrained_approximate" | "adjusted_to_constraints" | "failed"
    expected_return: Optional[float]
    volatility: float
    sharpe_ratio: Optional[float]
    cvar_95: Optional[float]
    marginal_risk_contributions: pd.Series
    risk_contributions_abs: pd.Series
    risk_contributions_rel: pd.Series
    class_exposure: Dict[str, float]
    class_risk_exposure: Dict[str, float]
    hhi: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value({
            "strategy_name": self.strategy_name,
            "tickers": self.tickers,
            "weights_by_asset": self.weights_by_asset,
            "converged": self.converged,
            "feasible": self.feasible,
            "status": self.status,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "cvar_95": self.cvar_95,
            "marginal_risk_contributions": self.marginal_risk_contributions,
            "risk_contributions_abs": self.risk_contributions_abs,
            "risk_contributions_rel": self.risk_contributions_rel,
            "class_exposure": self.class_exposure,
            "class_risk_exposure": self.class_risk_exposure,
            "hhi": self.hhi,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
        })


@dataclass
class ProposedOrder:
    """Indicative trade order (never auto-executed)."""

    broker_id: str
    instrument_id: str
    ticker: str
    side: str  # "BUY" | "SELL"
    quantity: float
    price_usd: float
    gross_amount_usd: float
    commission_usd: float
    spread_cost_usd: float
    net_amount_usd: float  # Cash impact: negative for BUY outflow, positive for SELL inflow
    pre_weight: float
    target_weight: float
    achievable_weight: float
    fractional: bool = True
    quantity_unit: str = "shares_or_units"
    quoted_price_usd: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))


@dataclass
class OmittedTrade:
    """Diagnostic record for a desired trade omitted due to thresholds or constraints."""

    instrument_id: str
    broker_id: Optional[str]
    side: str  # "BUY" | "SELL" | "HOLD"
    current_weight: float
    target_weight: float
    abs_weight_diff: float
    desired_notional_usd: float
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))


@dataclass
class RebalancingResult:
    """Indicative multi-broker rebalancing plan with full cash & cost reconciliation."""

    strategy_name: str
    status: str  # "ok" | "partially_funded" | "skipped_no_positions" | "skipped_invalid_strategy_target" | "missing_tradable_market_price" | "infeasible_hard_limits" | "insufficient_cash" | "broker_routing_required"
    feasible: bool
    orders: List[ProposedOrder] = field(default_factory=list)
    omitted_operations: List[OmittedTrade] = field(default_factory=list)
    theoretical_target_weights: Dict[str, float] = field(default_factory=dict)
    current_weights: Dict[str, float] = field(default_factory=dict)
    achievable_weights: Dict[str, float] = field(default_factory=dict)
    weight_deviations: Dict[str, float] = field(default_factory=dict)
    initial_holdings_value_usd: float = 0.0
    initial_portfolio_value_usd: float = 0.0
    deployable_capital_usd: float = 0.0
    final_portfolio_value_usd: float = 0.0
    initial_cash_usd: float = 0.0
    cash_flow_usd: float = 0.0
    ending_cash_usd: float = 0.0
    initial_cash_by_broker: Dict[str, float] = field(default_factory=dict)
    cash_flow_by_broker: Dict[str, float] = field(default_factory=dict)
    ending_cash_by_broker: Dict[str, float] = field(default_factory=dict)
    broker_ledgers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    unfunded_buy_shortfall_usd: float = 0.0
    total_buy_gross_usd: float = 0.0
    total_sell_gross_usd: float = 0.0
    total_commission_usd: float = 0.0
    total_spread_cost_usd: float = 0.0
    total_transaction_cost_usd: float = 0.0
    one_way_turnover: float = 0.0
    two_way_turnover: float = 0.0
    hard_limit_conflicts: List[str] = field(default_factory=list)
    unpriced_instruments: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))


@dataclass
class AnalysisResult:
    """Stateless, web-ready master output separating every analytical layer."""

    tickers: List[str]
    selected_strategy: str
    config_snapshot: Dict[str, Any]
    quotes_and_data_quality: Dict[str, Any]
    dcf_views: Dict[str, DCFViewResult]
    fixed_income_metrics: Dict[str, FixedIncomeMetricsResult]
    black_litterman: BlackLittermanResult
    risk_model: RiskModelResult
    constraints_summary: Dict[str, Any]
    strategies: Dict[str, StrategyAllocationResult]
    stress_test_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rebalancing: Dict[str, RebalancingResult] = field(default_factory=dict)
    prices_hist: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # Convenience accessors matching canonical engine consumers
    @property
    def expected_returns(self) -> pd.Series:
        return self.black_litterman.posterior_total_returns

    @property
    def cov_matrix(self) -> pd.DataFrame:
        return self.risk_model.cov_matrix

    @property
    def mu_sources(self) -> Dict[str, str]:
        return self.black_litterman.mu_sources

    @property
    def vix_level(self) -> Optional[float]:
        return self.risk_model.vix_value

    @property
    def stress_weight_p(self) -> Optional[float]:
        return self.risk_model.stress_weight_p

    @property
    def start_date(self) -> Optional[pd.Timestamp]:
        return self.risk_model.start_date

    @property
    def end_date(self) -> Optional[pd.Timestamp]:
        return self.risk_model.end_date

    @property
    def sample_weeks(self) -> int:
        return self.risk_model.total_weekly_samples

    def _strategy_weights(self, name: str) -> Optional[np.ndarray]:
        strat = self.strategies.get(name)
        return strat.weights if strat is not None else None

    def _strategy_stats(self, name: str) -> Dict[str, Any]:
        strat = self.strategies.get(name)
        if strat is None:
            return {}
        return {
            "ret": strat.expected_return,
            "vol": strat.volatility,
            "sharpe": strat.sharpe_ratio,
            "cvar": strat.cvar_95,
        }

    @property
    def weights_sharpe(self) -> Optional[np.ndarray]:
        return self._strategy_weights("max_sharpe")

    @property
    def weights_minvar(self) -> Optional[np.ndarray]:
        return self._strategy_weights("min_variance")

    @property
    def weights_cvar(self) -> Optional[np.ndarray]:
        return self._strategy_weights("min_cvar")

    @property
    def weights_erc(self) -> Optional[np.ndarray]:
        return self._strategy_weights("erc")

    @property
    def weights_hrp(self) -> Optional[np.ndarray]:
        return self._strategy_weights("hrp")

    @property
    def stats_sharpe(self) -> Dict[str, Any]:
        return self._strategy_stats("max_sharpe")

    @property
    def stats_minvar(self) -> Dict[str, Any]:
        return self._strategy_stats("min_variance")

    @property
    def stats_cvar(self) -> Dict[str, Any]:
        return self._strategy_stats("min_cvar")

    @property
    def stats_erc(self) -> Dict[str, Any]:
        return self._strategy_stats("erc")

    @property
    def stats_hrp(self) -> Dict[str, Any]:
        return self._strategy_stats("hrp")

    @property
    def mcr_sharpe(self) -> Optional[pd.Series]:
        s = self.strategies.get("max_sharpe")
        return s.risk_contributions_abs if s is not None else None

    @property
    def mcr_minvar(self) -> Optional[pd.Series]:
        s = self.strategies.get("min_variance")
        return s.risk_contributions_abs if s is not None else None

    @property
    def mcr_cvar(self) -> Optional[pd.Series]:
        s = self.strategies.get("min_cvar")
        return s.risk_contributions_abs if s is not None else None

    @property
    def mcr_erc(self) -> Optional[pd.Series]:
        s = self.strategies.get("erc")
        return s.risk_contributions_abs if s is not None else None

    @property
    def mcr_hrp(self) -> Optional[pd.Series]:
        s = self.strategies.get("hrp")
        return s.risk_contributions_abs if s is not None else None

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value({
            "tickers": self.tickers,
            "selected_strategy": self.selected_strategy,
            "config_snapshot": self.config_snapshot,
            "quotes_and_data_quality": self.quotes_and_data_quality,
            "dcf_views": self.dcf_views,
            "fixed_income_metrics": self.fixed_income_metrics,
            "black_litterman": self.black_litterman,
            "risk_model": self.risk_model,
            "constraints_summary": self.constraints_summary,
            "strategies": self.strategies,
            "stress_test_results": self.stress_test_results,
            "rebalancing": self.rebalancing,
            "warnings": self.warnings,
            "errors": self.errors,
        })
