"""Shared global state migrated from notebook cells (03, 04, 06, 15, 19).

Centralizing these module-level variables preserves the exact cross-cell state
semantics of the original notebook while avoiding import-time execution.
Note for future web backends: this module holds process-wide mutable state and
is not isolated per user/request session in this initial migration phase.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Cell 03: Argentina country risk
prima_riesgo_mas_reciente: Optional[float] = None
fecha_actualizacion: Optional[str] = None
fuente_datos: Optional[str] = None

# Cell 04: Relative country risk premia via ETFs
country_risk_premia: Dict[str, float] = {}

# Cell 06: Interactive portfolio state
activos_portfolio: List[Dict[str, Any]] = []
indice_editando: Optional[int] = None

# Cell 15: ML ingestion global state
SECTOR_MAPPING: Dict[str, Any] = {}
_ultima_llamada_av: Dict[str, float] = {"tiempo": 0}
_indice_api_key: int = 0

# Cell 19: Main execution global results
resultado_analisis: Any = None
predicciones_ml: Any = None
predicciones_5y: Any = None
datos_procesados_ml: Any = None

tickers: Any = None
cov: Any = None
ret_esp: Any = None
pesos_sharpe: Any = None
pesos_minvar: Any = None
pesos_cvar: Any = None
mcr_sharpe: Any = None
mcr_minvar: Any = None
mcr_cvar: Any = None
rebalanceo_sharpe: List[Dict[str, Any]] = []
rebalanceo_minvar: List[Dict[str, Any]] = []
rebalanceo_cvar: List[Dict[str, Any]] = []
