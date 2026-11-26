"""Structured error hierarchy for portfolio_optimization.

All domain errors expose a sanitized, JSON-serializable payload suitable for
stateless web API responses without leaking secrets or internal stack traces.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class PortfolioAnalysisError(Exception):
    """Base structured exception for the portfolio optimization engine."""

    default_code: str = "portfolio_analysis_error"
    default_phase: str = "general"

    def __init__(
        self,
        message: str,
        *,
        phase: Optional[str] = None,
        asset_id: Optional[str] = None,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        self.phase = phase or self.default_phase
        self.asset_id = asset_id
        self.code = code or self.default_code
        self.details: Dict[str, Any] = dict(details) if details else {}

    def to_dict(self) -> Dict[str, Any]:
        """Return a client-safe structured representation of the error."""
        return {
            "code": self.code,
            "phase": self.phase,
            "asset_id": self.asset_id,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(PortfolioAnalysisError):
    default_code = "configuration_error"
    default_phase = "configuration"


class DataQualityError(PortfolioAnalysisError):
    default_code = "data_quality_error"
    default_phase = "data_quality"


class FXConversionError(DataQualityError):
    default_code = "fx_conversion_error"
    default_phase = "fx_conversion"


class IOLQuoteError(DataQualityError):
    default_code = "iol_quote_error"
    default_phase = "iol_quotes"


class FixedIncomeValidationError(PortfolioAnalysisError):
    default_code = "fixed_income_validation_error"
    default_phase = "fixed_income"


class InsufficientHistoryError(DataQualityError):
    default_code = "insufficient_history_error"
    default_phase = "weekly_returns"


class MissingVixDataError(DataQualityError):
    default_code = "missing_vix_data_error"
    default_phase = "risk_model"


class InsufficientRegimeSampleError(DataQualityError):
    default_code = "insufficient_regime_sample_error"
    default_phase = "risk_model"


class InfeasibleConstraintsError(PortfolioAnalysisError):
    default_code = "infeasible_constraints_error"
    default_phase = "constraints_validation"


class OptimizationConvergenceError(PortfolioAnalysisError):
    default_code = "optimization_convergence_error"
    default_phase = "optimization"


class RebalancingError(PortfolioAnalysisError):
    default_code = "rebalancing_error"
    default_phase = "rebalancing"
