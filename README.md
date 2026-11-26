# portfolio-optimization

Stateless, web-ready multi-asset portfolio optimization engine in Python (`portfolio_optimization`), verified on **pandas 3.x**.

## Key Architectural Guarantees

- **Stateless & Account-Free**: Zero user accounts, logins, cookies, sessions, `localStorage`/`sessionStorage`/`IndexedDB`, databases, or persistent user files. All configuration, quotes, optimization outputs, and rebalancing proposals live exclusively in memory per request and are JSON-serializable via `.to_dict()`.
- **No Country-Risk Spreads & No ML**: Explicit country/sovereign risk spreads (`pais_riesgo`, `country_risk.py`) and all XGBoost/ML pipelines (`ml.py`, `TimeSeriesSplit`, `predicciones_ml`) have been completely removed. Geographic fields (`country`, `market`, `currency`) remain purely descriptive/calendar/FX metadata, and sovereign bonds remain eligible fixed-income instruments.
- **Unified Validated Configuration**: Every editable parameter lives in [`PortfolioAnalysisConfig`](src/portfolio_optimization/schemas.py) and is consumed identically by `analyze_portfolio`, `ejecutar_analisis_completo`, and `run_portfolio_analysis` (no divergent code paths and no notebook-only logic).
- **Pandas 3.x Native**: Causal backward-only FX alignment (no `fillna(method=...)`, no future `bfill`, subunit-aware `GBp`/`GBX` handling, zero silent fallback to raw local prices).

## Repository Structure

```text
portfolio-optimization/
├── .env.example                            # Optional server-side FRED_API_KEY only
├── .gitignore
├── pyproject.toml                          # Clean core dependencies (pandas>=2.1, no xgboost)
├── requirements.txt
├── docs/
│   └── MIGRATION_TRACEABILITY.md           # Superseding revision audit & historical notebook trace
├── notebooks/
│   └── portfolio_optimization.ipynb        # Sanitized stateless notebook adapter (no outputs/secrets)
├── src/portfolio_optimization/
│   ├── __init__.py                         # Side-effect-free public API exports
│   ├── _display.py                         # Optional IPython display wrapper with headless fallback
│   ├── analytics.py                        # Stateless analytics/plotting helpers accepting AnalysisResult
│   ├── config.py                           # Server-side FRED fallback config + re-exported schemas
│   ├── errors.py                           # Client-safe structured domain error hierarchy
│   ├── expected_returns.py                 # Annualized DCF views + multi-asset Black-Litterman
│   ├── fixed_income.py                     # Manual contractual cashflows, dirty price, YTM/IRR, duration
│   ├── iol.py                              # Public IOL Argentine Bonds/ONs parser & AL30/AL30D MEP
│   ├── market_data.py                      # Causal FX conversion, W-FRI weekly returns, VIX, RF rate
│   ├── notebook_ui.py                      # Thin stateless optional notebook display helpers
│   ├── optimization.py                     # Joint LP feasibility, Sharpe/MinVar/CVaR/ERC/HRP, analyze_portfolio
│   ├── pipeline.py                         # ejecutar_analisis_completo & run_portfolio_analysis
│   ├── rebalancing.py                      # Multi-broker fractional rebalancing & cost reconciliation
│   ├── risk_model.py                       # Dynamic VIX-regime covariance Sigma(p), shrinkage, PSD, EWMA
│   └── schemas.py                          # Single validated config & result contract (from_dict/to_dict)
└── tests/
    ├── test_smoke_and_security.py          # Statelessness, isolation, dead-code/ML/country-risk removal, secrets
    ├── test_engine_behavior.py             # Risk-free paths, _download_prices FX, stress tests, sovereign bonds
    ├── test_fx_and_weekly_returns.py       # Pandas 3.x FX causality, subunits, W-FRI holidays/crypto/gaps
    ├── test_iol_and_fixed_income.py        # IOL parser, D->C->ARS/MEP, manual bond schedules, YTM roots
    ├── test_dcf_bl_and_vix_regime.py       # DCF annualization, Idzorek/Woodbury BL, VIX schedule & regime blend
    ├── test_optimizers_and_rebalancing.py  # 1/4/5/20/22 assets, LP feasibility, ERC/HRP, multi-broker rebalancing
    └── test_audit_remediation.py           # Oracle regression suite for AUD-01..AUD-15 & multi-asset E2E
```

## Configuration Parameters, Defaults & Financial Assumptions

| Section | Parameter | Default | Unit / Range | Financial Note |
| :--- | :--- | :--- | :--- | :--- |
| **Instruments** | `variable_income` / `fixed_income` | Required ($\ge 1$ total) | Distinct UI section & `risk_class` | Gold/commodities and crypto reside in `variable_income` without being classified as equities; fixed-income ETFs require no contractual schedule. |
| **Instruments** | `currency` / `quote_unit` | `None` / `None` | Auto-detect from provider | `currency=None` avoids overriding provider metadata (e.g. Yahoo `GBp`/`ARS`/`EUR`) with `USD`; material provider conflicts raise `FXConversionError`; `GBP` declared with `GBp` feed still applies `0.01`. |
| **Fixed Income** | `quantity`, `min_nominal`, `lot_size`, `price_convention` | Original VN (debt) / shares (other) | `per_100_original_nominal` \| `per_100_residual_nominal` \| `per_unit` | Individual debt quantities/minimums/lots are in **original nominal units (1 VN original)**; `per_unit` denotes price per 1 bond unit of `reference_nominal`. Valuation and rebalancing share `normalize_execution_price_to_usd_per_unit`. |
| **Black-Litterman** | `risk_aversion` ($\delta$) / `tau` ($\tau$) | `2.5` / `0.05` | $> 0$ / $(0, 1]$ | Equilibrium excess prior $\Pi = \delta \Sigma w_{\text{ref}}$; diagonal equilibration $D^{-1} S D^{-1}$ preserves accuracy under extreme mixed confidences without scalar jitter corruption. |
| **Black-Litterman** | `default_dcf_horizon_years` / `DCFViewConfig.horizon_years` | `5.0` / `None` (inherits global) | years $> 0$ | Individual `horizon_years=None` inherits `default_dcf_horizon_years`; explicit individual value overrides global. |
| **Black-Litterman** | `default_dcf_confidence` | `0.50` | $[0, 1]$ | Idzorek $\omega_k = \frac{1-c_k}{c_k} P_k (\tau\Sigma) P_k^\top$; Woodbury posterior is non-singular at $c=1.0$; $c=0.0$ is inactive. |
| **Black-Litterman** | `fixed_income_bl_view_mode` | `"metric_only"` | `"metric_only"` \| `"explicit_ytm_view"` | Contractual YTM is reported separately by default and never treated as a guaranteed 1-year return unless explicitly opted in. |
| **Weekly Risk** | `return_frequency` / cutoff | `"W-FRI"` / `17:00 America/New_York` | Weekly USD | Last valid observation per **closed** Friday-ending week (excludes incomplete current weeks after last closed Friday); no `ffill` across empty weeks; annualization $\times 52$ ($\Sigma$) and $\sqrt{52}$ ($\sigma$). Optional EWMA is strictly causal and gap-aware ($\lambda^{\Delta t}$). |
| **Weekly Risk** | `fx_max_staleness_days` / `min_fx_coverage` | `5` days / `0.95` | calendar days / ratio | Strictly causal backward FX alignment; missing/stale FX or unknown currency raises typed `FXConversionError` directly (no silent drops). |
| **Weekly Risk** | `min_weekly_observations` / `min_asset_coverage` | `52` weeks / `0.85` | weeks / ratio | Explicit alignment & incomplete-week exclusion diagnostics. |
| **VIX Regime** | `vix_low`, `vix_mid_low`, `vix_high` | `15.0`, `20.0`, `30.0` | VIX points | `<15: p=0.15`; `15..20: linear 0.15->0.35`; `(20,30]: p=0.35`; `>30: p=0.60` (deliberate upward step discontinuity above 30). |
| **VIX Regime** | `crisis_vix_threshold`, `min_calm_weeks`, `min_crisis_weeks` | `25.0`, `26`, `12` | VIX / weeks | Separate historical regime classifier from current $p(\text{VIX})$; never fabricates $\Sigma_{\text{crisis}} = 1.5 \Sigma_{\text{calm}}$. |
| **Constraints** | `long_only`, `budget_sum`, `asset_bounds`, `class_bounds`, `solver_tolerance` | `True`, `1.0`, `{}`, `{}`, `1e-8` | weights in $[0, 1]$ | Neutral long-only defaults; no fixed `(0.05, 0.20)` bounds; exact joint LP feasibility check; `solver_tolerance` wired directly to solvers; zero silent `1/n` fallback. |
| **Optimization** | `selected_strategy` / `non_positive_excess_return_policy` | `"max_sharpe"` / `"block"` | `max_sharpe`, `min_variance`, `min_cvar`, `erc`, `hrp` | Same annual weekly $\Sigma$ across all strategies; strict `res.success` convergence gate; singular zero-variance & non-positive excess return guards in Max Sharpe; canonical alphabetical HRP tie-breaking; ERC & HRP do not require $\mu$; HRP never inverts $\Sigma$. |
| **Rebalancing** | `available_cash_by_broker`, `cash_flow_by_broker`, `min_weight_diff`, `min_notional_usd` | `{}`, `{}`, `0.01`, `25.0` USD | per-broker USD / weight | Strict per-broker cash segregation (zero implicit inter-broker transfers; scalar cash only valid for single-broker); post-scaling `min_notional_usd` revalidation; post-rebalance verification of asset, class, sector, and `hhi_limit` bounds. |
| **IOL Quotes** | `allow_mep_without_market_timestamp`, `max_mep_market_time_diff_seconds`, `mep_plausible_min/max` | `False`, `3600s`, `50..50000` | ARS/USD | Validates `AL30`/`AL30D` `market_timestamp` contemporaneity, verified nominal units, issuer identity on all branches, and exact header matching first. Missing timestamps block implicit MEP unless explicitly opted in (`allow_mep_without_market_timestamp=True`); valid species `D`/`C` remain unaffected. |

## Known Data Limitations

1. **IOL Argentine Bonds & ONs are point-in-time snapshots only**: Public IOL tables (`bonos/todos`, `obligaciones-negociables/todos`) do not supply historical price series. The engine NEVER fabricates historical returns from coupon schedules or snapshot quotes and NEVER applies today's implicit `AL30/AL30D` MEP rate to historical dates. To include an Argentine bond/ON in portfolio covariance optimization, either supply a real historical USD total-return series or explicitly approve a proxy (`historical_proxy_ticker` + `risk.allow_explicit_proxies=True`) with visible limitations. Standalone quote acquisition (`fetch_iol_quotes_for_instruments`) and contractual YTM valuation (`evaluate_fixed_income_instrument`) remain available regardless.
2. **USD Cable (`C`) vs Local USD (`D`)**: When species `D` is unavailable, species `C` is reported with `usd_settlement_type="USD_CABLE_C"` and an explicit warning; no 1:1 economic parity with local USD is assumed.
3. **Server-side Credentials**: Only `FRED_API_KEY` remains as an optional server-side fallback for the USD risk-free rate (`DGS10`) if `^TNX` fails. `GEMINI_API_KEY` and `ALPHA_VANTAGE_API_KEYS` were removed.

## Installation & Testing

```bash
pip install -e ".[dev]"
pytest -q
```

All tests in `tests/` run deterministically offline with synthetic fixtures and mocked providers (zero API keys and zero live financial calls required).
