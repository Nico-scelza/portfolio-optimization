# portfolio-optimization

Faithful initial Python migration of the **Portfolio Optimization & ML Projection** notebook (`Optimización de Portfolio 2.ipynb`) into a single reusable, importable Python package (`portfolio_optimization`) plus a sanitized interactive notebook adapter (`notebooks/portfolio_optimization.ipynb`).

> **Scope note:** This migration strictly preserves the original financial, mathematical, statistical, machine learning, and presentation logic without refactoring algorithms, changing parameters/defaults, or adding web/API infrastructure.

## Repository Structure

```text
portfolio-optimization/
├── .env.example                           # Environment variable names with fictitious placeholders
├── .gitignore                             # Excludes .env, caches, generated charts, and local data
├── pyproject.toml                         # Package metadata and dependencies
├── requirements.txt                       # Core + optional notebook/legacy dependencies
├── docs/
│   └── MIGRATION_TRACEABILITY.md          # Complete 37-cell traceability matrix & known limitations
├── notebooks/
│   └── portfolio_optimization.ipynb       # Sanitized interactive adapter using the canonical package
├── src/
│   └── portfolio_optimization/            # Single canonical active implementation
│       ├── __init__.py                    # Side-effect-free public exports
│       ├── _display.py                    # Optional IPython.display wrapper with headless fallback
│       ├── analytics.py                   # Frontier, 10y performance, correlations, drawdown, MCR, risk, allocation
│       ├── config.py                      # Environment-based secret loaders & explicit runtime setup
│       ├── country_risk.py                # Argentina country risk & ETF relative volatility risk premia
│       ├── market_data.py                 # USD-normalized price downloader, risk-free rate, metric wrappers
│       ├── ml.py                          # DataIngestor, PortfolioProcessor, PortfolioPredictor + Cell 16/17 patches
│       ├── notebook_ui.py                 # Lazy ipywidgets notebook UI & callbacks
│       ├── optimization.py                # Sharpe/HHI, MinVar, CVaR, Black-Litterman, shrinkage, stress, analyze_portfolio
│       ├── pipeline.py                    # _calcular_rebalanceo, ejecutar_analisis_completo, run_portfolio_analysis
│       └── state.py                       # Shared global state migrated from notebook cells
└── tests/
    ├── test_smoke_and_security.py         # Import smoke test, monkey-patch order, secret & notebook sanitization checks
    ├── test_parity_with_notebook.py       # Side-by-side parity & AST verification against source notebook blocks
    └── test_engine_behavior.py            # Characterization tests (country risk, FX, BL, covariance, stress, ML, rebalance)
```

## Configuration & Secrets

All API credentials are externalized to server-side environment variables and **are never required at import time**:

| Environment Variable | Purpose | Format |
| :--- | :--- | :--- |
| `ALPHA_VANTAGE_API_KEYS` | Alpha Vantage API key rotation list (order preserved verbatim) | Comma-separated keys |
| `FRED_API_KEY` | Federal Reserve Economic Data (FRED) API key (`DGS10`, macro series) | Single key |
| `GEMINI_API_KEY` | Reserved Gemini configuration slot from source notebook | Single key |

Copy `.env.example` to `.env` for local development (never commit `.env`), or export the variables in your server environment.

## Installation

```bash
# Core headless engine (suitable for future backend consumption)
pip install -e .

# With interactive notebook / legacy UI dependencies
pip install -e ".[notebook]"

# With test dependencies
pip install -e ".[dev]"
```

### Dependency Version Notes

- Source notebook runtime metadata reported **Python `3.12.12`**.
- Cell 00 execution logs explicitly observed `feedparser==6.0.12` (`sgmllib3k==1.0.0`), `tkhtmlview==0.3.2`, `requests==2.32.4`, `urllib3==2.5.0`, `pillow==11.3.0`, `charset_normalizer==3.4.6`, `idna==3.11`, and `certifi==2026.2.25`.
- Versions of pre-installed Colab scientific libraries (`numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `matplotlib`, `plotly`, `beautifulsoup4`, `yfinance`, `tqdm`, `jinja2`, `ipython`, `ipywidgets`) were **not observable** in the source notebook; therefore no lockfile is fabricated (`jinja2` is required at runtime by `pandas.DataFrame.style` in `ejecutar_analisis_completo`).

## Programmatic Usage (Headless Backend Entrypoint)

Importing `portfolio_optimization` performs **no** network requests, ML training, installations, GUI initialization, or API key validation:

```python
import portfolio_optimization as po

assets = [
    {"ticker": "AAPL", "cantidad": 10, "precio_compra": 180.0, "upside_esperado": 12.0},
    {"ticker": "MSFT", "cantidad": 5, "precio_compra": 390.0, "upside_esperado": 0.0},
    {"ticker": "GOOGL", "cantidad": 8, "precio_compra": 140.0, "upside_esperado": 0.0},
    {"ticker": "AMZN", "cantidad": 6, "precio_compra": 170.0, "upside_esperado": 0.0},
    {"ticker": "NVDA", "cantidad": 4, "precio_compra": 110.0, "upside_esperado": 0.0},
]

# Execute the exact same canonical engine used by the notebook, suppressing IPython HTML output:
result = po.run_portfolio_analysis(
    tickers_list=[a["ticker"] for a in assets],
    activos_dict=assets,
    ejecutar_ml=False,   # Set True to also run DataIngestor -> PortfolioProcessor -> PortfolioPredictor
    render_ui=False,
)

# Access outputs (exact AnalysisResult dataclass and state attributes):
# result.weights_sharpe, result.weights_minvar, result.weights_cvar
# result.stats_sharpe, result.stress_test_results, result.mu_sources
# po.state.rebalanceo_sharpe, po.state.predicciones_5y
```

You can also call `po.analyze_portfolio(...)` directly for lower-level optimization control.

## Running Tests

```bash
pytest -q
```

All tests run completely offline using synthetic fixtures and mocks (zero real API calls, zero API keys required).

## Known Pre-Existing Limitations

See [`docs/MIGRATION_TRACEABILITY.md`](docs/MIGRATION_TRACEABILITY.md) for the complete cell-by-cell audit and pre-existing notebook caveats (process-wide mutable state in `state.py`, `verify=False` in Alpha Vantage HTTP fetch, pandas 3.x deprecation in `_convert_to_usd`, unseeded Monte Carlo simulations, and unverified live `ipywidgets` UI rendering).
