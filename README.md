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
│   ├── iol.py                              # Public IOL Argentine Bonds/ONs panel quotes, DatosHistoricos history & AL30/AL30D MEP
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
    ├── test_iol_historical_prices.py       # IOL DatosHistoricos query, parsing, segmentation, currency guards & mixed E2E
    ├── test_dcf_bl_and_vix_regime.py       # DCF annualization, Idzorek/Woodbury BL, VIX schedule & regime blend
    ├── test_optimizers_and_rebalancing.py  # 1/4/5/20/22 assets, LP feasibility, ERC/HRP, multi-broker rebalancing
    ├── test_audit_remediation.py           # Oracle regression suite for AUD-01..AUD-15 & multi-asset E2E
    ├── test_reaudit_remediation.py         # Oracle regression suite for AUD-04 residual, NEW-01..NEW-03 & E2E variants
    └── test_new04_timezone_alignment.py    # Regression suite for NEW-04 tz-aware FX/yfinance daily & intraday alignment
```

## Configuration Parameters, Defaults & Financial Assumptions

| Section | Parameter | Default | Unit / Range | Financial Note |
| :--- | :--- | :--- | :--- | :--- |
| **Instruments** | `variable_income` / `fixed_income` | Required ($\ge 1$ total) | Distinct UI section & `risk_class` | Gold/commodities and crypto reside in `variable_income` without being classified as equities; fixed-income ETFs require no contractual schedule. |
| **Instruments** | `currency` / `quote_unit` | `None` / `None` | Auto-detect from provider | `currency=None` avoids overriding provider metadata (e.g. Yahoo `GBp`/`ARS`/`EUR`) with `USD`; material provider conflicts raise `FXConversionError`; `GBP` declared with `GBp` feed still applies `0.01`. |
| **Fixed Income** | `quantity`, `min_nominal`, `lot_size`, `price_convention` | Original VN (debt) / shares (other) | `per_100_original_nominal` \| `per_100_residual_nominal` \| `per_unit` | Individual debt quantities/minimums/lots are in **original nominal units (1 VN original)**; `per_unit` denotes price per 1 bond unit of `reference_nominal`. Contractual YTM, `latest_prices_usd`, `execution_prices_usd_per_unit`, and rebalancing share one centralized tradable market-price resolver (`IOL -> manual_price_usd -> unadjusted yfinance Close / market_price override`); `adjusted_total_return` series and risk proxies remain usable for covariance risk but never fabricate tradable prices (blocking rebalancing with `status="missing_tradable_market_price"` and `unpriced_instruments`). |
| **Black-Litterman** | `risk_aversion` ($\delta$) / `tau` ($\tau$) | `2.5` / `0.05` | $> 0$ / $(0, 1]$ | Equilibrium excess prior $\Pi = \delta \Sigma w_{\text{ref}}$; diagonal equilibration $D^{-1} S D^{-1}$ preserves accuracy under extreme mixed confidences without scalar jitter corruption. |
| **Black-Litterman** | `default_dcf_horizon_years` / `DCFViewConfig.horizon_years` | `5.0` / `None` (inherits global) | years $> 0$ | Individual `horizon_years=None` inherits `default_dcf_horizon_years`; explicit individual value overrides global. |
| **Black-Litterman** | `default_dcf_confidence` | `0.50` | $[0, 1]$ | Idzorek $\omega_k = \frac{1-c_k}{c_k} P_k (\tau\Sigma) P_k^\top$; Woodbury posterior is non-singular at $c=1.0$; $c=0.0$ is inactive. |
| **Black-Litterman** | `fixed_income_bl_view_mode` | `"metric_only"` | `"metric_only"` \| `"explicit_ytm_view"` | Contractual YTM is reported separately by default and never treated as a guaranteed 1-year return unless explicitly opted in. |
| **Weekly Risk** | `return_frequency` / cutoff | `"W-FRI"` / `17:00 America/New_York` | Weekly USD | Last valid observation per **closed** Friday-ending week (excludes incomplete current weeks after last closed Friday); no `ffill` across empty weeks; annualization $\times 52$ ($\Sigma$) and $\sqrt{52}$ ($\sigma$). Any tz-aware `as_of_cutoff` (including local midnight such as `04:00Z` EDT / `05:00Z` EST) or explicit time string is evaluated as an exact DST-aware instant (never promoted to end-of-day); pure `datetime.date`, `YYYY-MM-DD`, and legacy naive midnight `pd.Timestamp` retain end-of-day semantics (`23:59:59.999999999`; pass tz-aware timestamps for unambiguous midnight instants). `convert_price_series_to_usd` preserves tz-awareness (`UTC` when `target_tz=None`, or `ZoneInfo(target_tz)` when specified), while `yfinance` daily exchange-midnight bars retain exact exchange trading dates (preventing cross-timezone `as_of_date` leakage and +1 week intraday shifts). Optional EWMA is strictly causal and gap-aware ($\lambda^{\Delta t}$). |
| **Weekly Risk** | `fx_max_staleness_days` / `min_fx_coverage` | `5` days / `0.95` | calendar days / ratio | Strictly causal backward FX alignment; missing/stale FX or unknown currency raises typed `FXConversionError` directly (no silent drops). |
| **Weekly Risk** | `min_weekly_observations` / `min_asset_coverage` | `52` weeks / `0.85` | weeks / ratio | Explicit alignment & incomplete-week exclusion diagnostics. |
| **VIX Regime** | `vix_low`, `vix_mid_low`, `vix_high` | `15.0`, `20.0`, `30.0` | VIX points | `<15: p=0.15`; `15..20: linear 0.15->0.35`; `(20,30]: p=0.35`; `>30: p=0.60` (deliberate upward step discontinuity above 30). |
| **VIX Regime** | `crisis_vix_threshold`, `min_calm_weeks`, `min_crisis_weeks` | `25.0`, `26`, `12` | VIX / weeks | Separate historical regime classifier from current $p(\text{VIX})$; never fabricates $\Sigma_{\text{crisis}} = 1.5 \Sigma_{\text{calm}}$. |
| **Constraints** | `long_only`, `budget_sum`, `asset_bounds`, `class_bounds`, `solver_tolerance` | `True`, `1.0`, `{}`, `{}`, `1e-8` | weights in $[0, 1]$ | Neutral long-only defaults; no fixed `(0.05, 0.20)` bounds; exact joint LP feasibility check; `solver_tolerance` wired directly to solvers; zero silent `1/n` fallback. |
| **Optimization** | `selected_strategy` / `non_positive_excess_return_policy` | `"max_sharpe"` / `"block"` | `max_sharpe`, `min_variance`, `min_cvar`, `erc`, `hrp` | Same annual weekly $\Sigma$ across all strategies; strict `res.success` convergence gate; singular zero-variance & non-positive excess return guards in Max Sharpe unified across $n=1$ and $n\ge 2$ (`suboptimal_non_positive_excess` reports `is_valid_target=False` and is skipped as a rebalancing target); canonical alphabetical HRP tie-breaking; ERC & HRP do not require $\mu$; HRP never inverts $\Sigma$. |
| **Rebalancing** | `available_cash_by_broker`, `cash_flow_by_broker`, `min_weight_diff`, `min_notional_usd` | `{}`, `{}`, `0.01`, `25.0` USD | per-broker USD / weight | Strict per-broker cash segregation (zero implicit inter-broker transfers; scalar cash only valid for single-broker); causal per-broker separation of real funding shortfalls (`unfunded_buy_shortfall_usd`) vs broker transaction-cost friction (`cost_adjusted_buy_reduction_usd`), operational threshold omissions (`threshold_omitted_buy_usd`), and lot rounding residuals (`lot_rounding_residual_usd`); post-scaling `min_notional_usd` revalidation; post-rebalance verification of asset, class, sector, and `hhi_limit` bounds. |
| **IOL Quotes** | `allow_mep_without_market_timestamp`, `max_mep_market_time_diff_seconds`, `mep_plausible_min/max` | `False`, `3600s`, `50..50000` | ARS/USD | Validates `AL30`/`AL30D` `market_timestamp` contemporaneity, verified nominal units, issuer identity on all branches, and exact header matching first. Missing timestamps block implicit MEP unless explicitly opted in (`allow_mep_without_market_timestamp=True`); valid species `D`/`C` remain unaffected. |
| **IOL History** | `fetch_historical_prices`, `default_market`, `historical_price_field`, `historical_chunk_days`, `allow_usd_cable_history`, `drop_zero_volume_rows` | `True`, `"BCBA"`, `"close"`, `1825` days, `False`, `True` | Daily market price (`Cierre`) | Automatically fetches historical daily prices for explicit `quote_source in ("iol_bonos", "iol_ons")` instruments via public `GET` + segmented `POST /Titulo/DatosHistoricos` (bypassing the 500-row default GET cap with adaptive truncation detection). Preserves exact symbols (`iol_symbol`/`iol_market` overrides supported), drops `0.00`-price and zero-volume server-forward-filled no-trade rows without forward-filling, rejects negative/corrupted price cells, blocks ARS-only and default `USD_CABLE_C` mixing, and tags series strictly as unadjusted market prices (`is_total_return=False`). |

## IOL Historical Fixed-Income Data (`/Titulo/DatosHistoricos`)

### Acquisition Mechanism & Usage

Public Argentine sovereign bonds and corporate bonds (ONs) on InvertirOnline expose historical OHLCV tables without login or API keys:

1. **Metadata discovery**: `GET https://iol.invertironline.com/Titulo/DatosHistoricos?simbolo={SIMBOLO}&mercado={MERCADO}` (default `mercado=BCBA`) initializes session cookies and resolves the instrument's internal `IdTitulo` and `Simbolo` from the page form (missing `IdTitulo` unambiguously identifies a non-existent symbol). Note that the default GET table is capped at 500 trading rows.
2. **Segmented range retrieval**: Contiguous date windows of up to `IOLConfig.historical_chunk_days` (default `1825` days / ~5 years) are queried via `POST https://iol.invertironline.com/Titulo/DatosHistoricos` with form fields `desdehasta="DD/MM/YYYY - DD/MM/YYYY"`, `idtitulo`, and `idfrecuencias` (`"1"` daily, `"2"` weekly, `"3"` monthly). Because IOL evaluates the right boundary of `desdehasta` exclusively, each segment requests `seg_end + 1 day` and filters parsed dates strictly into `[requested_start_date, effective_end_date]`. If a POST segment hits the 500-row cap while leaving an uncovered leading prefix, bounded adaptive subdivision probes and recovers earlier observations (or raises `IOLHistoryError(code="iol_truncated_response")` if truncation persists).
3. **Automatic pipeline routing & standalone usage**:
   - In `analyze_portfolio`, `ejecutar_analisis_completo`, and `run_portfolio_analysis`, instruments explicitly configured with `quote_source in ("iol_bonos", "iol_ons")` automatically fetch daily IOL history when `IOLConfig.fetch_historical_prices=True` (unless `prices_usd_override` or an explicit user-approved proxy is supplied). Equities, ETFs, CEDEARs, gold, and crypto (`quote_source="yfinance"`) remain on their existing routes—ticker suffixes such as trailing `"D"` are never used to guess bond routing.
   - Optional explicit mapping fields `InstrumentConfig.iol_symbol` and `InstrumentConfig.iol_market` allow mapping custom internal IDs to exact IOL symbols without mutating suffixes.
   - Standalone consumers can invoke `fetch_iol_historical_prices("AEC3D", start_date=..., end_date=...)` or `fetch_iol_historical_prices_for_instruments(...)` directly with optional request-scoped in-memory `cache` dicts (zero disk or global state persistence).

### Price Field Selection, Price Return vs. Total Return & Currency Policy

- **Selected field (`"close"` / `Cierre`)**: Empirical inspection on live IOL fixed-income histories (`AEC3D`, `BPA8D`, `AL30D`, `AL30`, `AL30C`, `YMCHD`) confirmed that `Cierre ajustado` is identical to unadjusted market `Cierre` across coupon payments and principal amortizations (for example, `AL30D`'s January 2025 amortization/coupon ex-date drop from `76.95` to `68.60` appears identically in both columns). The provider therefore selects unadjusted `Cierre` by default, audits `adjusted_close` differences in `quotes_and_data_quality["iol_historical_metadata"]`, always uses unadjusted `Cierre` (never `adjusted_close`) as the tradable market price for contractual YTM/valuation/rebalancing, and tags every IOL fixed-income series as `series_type="market_price"` and `is_total_return=False`.
- **Total-return limitation & manual cashflow integration**: IOL fixed-income prices alone are **price returns only** (coupons and principal amortizations are not adjusted by IOL). When the user supplies manual cashflows on `InstrumentConfig.fixed_income_terms` (`FixedIncomeTerms` with `CashFlowEvent`s) and explicitly sets `historical_cashflows_confirmed=True` (optionally bounded by `historical_cashflow_start_date` / `historical_cashflow_end_date`), the engine integrates IOL unadjusted `Cierre` with those confirmed historical cashflows—preserving nominal/residual and clean/dirty conventions—to produce Total Return before `W-FRI` aggregation and replaces the Price Return warning with Total Return diagnostics. Unconfirmed or partial coverage preserves Price Return on unconfirmed intervals with explicit warnings, and requesting `historical_price_series_type="adjusted_total_return"` without complete confirmed Total Return (or external `prices_usd_override`) raises `IOLHistoryError(code="iol_total_return_unavailable")`.
- **Currency & species comparability**: Only verified local USD (`USD_LOCAL_D`) series enter the USD portfolio by default. USD Cable (`USD_CABLE_C`) is blocked unless explicitly opted in via `IOLConfig(allow_usd_cable_history=True)` (with warning and explicit settlement metadata), and ARS-only bonds/ONs (`USD_MEP_IMPLIED`) are always blocked rather than converted with spot `AL30/AL30D` MEP or official `ARS=X` FX.
- **Operational controls**: HTTPS-only configurable endpoint URL validation (no embedded credentials, no insecure HTTP redirect downgrades), strict TLS (`verify=True`), explicit request timeouts, bounded retries with capped exponential backoff on transient network/HTTP errors (`429`/`5xx`, honoring numeric and HTTP-date `Retry-After`), immediate fail-closed errors on `401`/`403`, non-existent symbols, schema changes, negative/corrupted price cells, and conflicting duplicate closes, polite inter-request pacing, zero-price and zero-volume server-forward-filled no-trade row filtering (`drop_zero_volume_rows=True`) without forward-filling, leading and trailing partial-coverage detection, fail-closed `as_of_cutoff` snapshot filtering, and intraday/future bar exclusion prior to BCBA close (`17:00 ART`).

## Manual Fixed-Income Cashflows, Total Return & Automatic YTM (TIR)

Individual bonds and corporate obligations (`sovereign_bond`, `corporate_bond_on`, `tbill`) accept a single manual contractual schedule in `InstrumentConfig.fixed_income_terms` (`FixedIncomeTerms` + `CashFlowEvent`), used for both **historical Total Return** and **automatic YTM (TIR)**—never fetched from external cashflow APIs and never requiring manual TIR input:

- **Cashflow schedule (`CashFlowEvent`)**: Supply `payment_date`, `coupon_amount` (or `coupon_rate` with `period_start_date` / `coupon_frequency` and `day_count_convention`), `amortization_amount`, and optional `ex_date` (economic detachment date). Historical and future cashflows may share the same `cashflows` tuple; the engine splits them automatically at `settlement_date` (or `as_of_date` / quote date when `settlement_date=None`) using `ex_date` when available (else `payment_date`), preventing double counting on boundary dates.
- **Historical Total Return & confirmed zero-payment periods**: Absence of historical cashflow records does **not** imply zero payments. Set `historical_cashflows_confirmed=True` (and optionally `issue_date`, `historical_cashflow_start_date`, `historical_cashflow_end_date`) to confirm full or partial coverage—including payment-free intervals. Returns `R_t = (P_t + C_t + A_t) / P_{t-1} - 1` normalize quotes to consistent monetary, original/residual nominal (`per_100_original_nominal`, `per_100_residual_nominal`, `per_unit`), and dirty/clean bases (`price_type`, `historical_price_type`), assign each payment once via `ex_date` (or `payment_date` with an explicit timing-limitation note), preserve holiday/no-trade gaps without forward-fill, and enter `W-FRI` weekly aggregation via a base-100 wealth index.
- **Automatic YTM (TIR)**: Solved automatically from the tradable market price `P_0` (dirty, adding `accrued_interest` only when `price_type="clean"`) and future contractual cashflows via XIRR (`0 = -P_0 + sum(CF_i / (1 + y)^t_i)`, effective annual rate using `day_count_convention`, default `ACT/365` / `ACT/365F`), exposed in `AnalysisResult.fixed_income_metrics[id].ytm_annual` (plus `ytm_weekly_effective = (1 + ytm_annual)^(1/52) - 1`). Invalid/incomplete schedules or missing/no-root prices never produce synthetic rates or enter Black-Litterman (`fixed_income_bl_view_mode="explicit_ytm_view"` opts USD contractual YTM into BL with explicit hold-to-maturity caveats).

```python
from datetime import date
import portfolio_optimization as po

al30d = po.InstrumentConfig(
    instrument_id="AL30D",
    ticker="AL30D",
    section="fixed_income",
    asset_subtype="sovereign_bond",
    quote_source="iol_bonos",
    currency="USD",
    fixed_income_terms=po.FixedIncomeTerms(
        cashflow_currency="USD",
        reference_nominal=100.0,
        maturity_date=date(2030, 7, 9),
        # Optional when settlement_date=None: inferred from as_of_date / quote date;
        # residual_nominal is inferred if all amortizations from issuance reconcile with reference_nominal,
        # otherwise provide residual_nominal=... or amortization_factor=... explicitly.
        price_convention="per_100_original_nominal",
        price_type="dirty",
        day_count_convention="ACT/365",
        historical_cashflows_confirmed=True,
        historical_cashflow_start_date=date(2024, 1, 1),
        cashflows=(
            # Historical and future cashflows in one schedule (per 100 original VN):
            po.CashFlowEvent(payment_date=date(2024, 7, 9), ex_date=date(2024, 7, 5), coupon_amount=0.375, amortization_amount=4.0),
            po.CashFlowEvent(payment_date=date(2025, 1, 9), ex_date=date(2025, 1, 7), coupon_amount=0.36, amortization_amount=8.0),
            # ... remaining semi-annual coupon + amortization events through 2030-07-09
        ),
    ),
)
```

## Known Data Limitations

1. **IOL Argentine Bonds & ONs Historical Price Return & Coverage**: Public IOL `/Titulo/DatosHistoricos` supplies daily unadjusted market prices (`Cierre`) in the instrument's native currency/species (quoted per 100 original nominal on BCBA) and does **not** adjust for coupon payments or principal amortizations. Newly issued instruments (for example `AEC3D` listed in August 2026) have legitimate partial history (`partial_coverage=True`) and may be blocked by `RiskModelConfig.min_weekly_observations` (default `52`) unless a shorter window, external total-return series (`prices_usd_override`), or explicit proxy (`historical_proxy_ticker` + `risk.allow_explicit_proxies=True`) is configured. The engine NEVER fabricates historical returns from coupon schedules or snapshot quotes and NEVER applies today's implicit `AL30/AL30D` MEP rate or official `ARS=X` to historical ARS bond prices. Standalone quote acquisition (`fetch_iol_quotes_for_instruments`) and contractual YTM valuation (`evaluate_fixed_income_instrument`) remain available regardless.
2. **USD Cable (`C`) vs Local USD (`D`)**: When species `D` is unavailable, species `C` is reported with `usd_settlement_type="USD_CABLE_C"` and an explicit warning; no 1:1 economic parity with local USD is assumed.
3. **Server-side Credentials**: Only `FRED_API_KEY` remains as an optional server-side fallback for the USD risk-free rate (`DGS10`) if `^TNX` fails. `GEMINI_API_KEY` and `ALPHA_VANTAGE_API_KEYS` were removed.

## Installation & Testing

```bash
pip install -e ".[dev]"
pytest -q
```

All tests in `tests/` run deterministically offline with synthetic fixtures and mocked providers (zero API keys and zero live financial calls required).
