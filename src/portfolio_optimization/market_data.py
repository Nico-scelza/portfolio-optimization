"""Market data downloading, USD normalization, and compatibility wrappers.

Migrated faithfully from Cells 08 and 11 of the source notebook.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import urllib3
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import FRED_BASE_URL, get_fred_api_key


def _download_prices(tickers: List[str], period: str = "10y") -> tuple[pd.DataFrame, List[str]]:
    """
    Descarga precios históricos y los normaliza FORZOSAMENTE a USD.

    Args:
        tickers: Lista de símbolos a descargar
        period: Período de descarga ('10y', 'max', etc.)

    Returns:
        tuple: (DataFrame con precios en USD, Lista de tickers fallidos)
    """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    dropped: List[str] = []
    prices_usd = pd.DataFrame()

    CURRENCY_PAIRS = {
        'ARS': 'ARS=X',
        'BRL': 'BRL=X',
        'EUR': 'EURUSD=X',
        'GBP': 'GBPUSD=X',
        'JPY': 'JPY=X',
        'CAD': 'CAD=X',
        'CHF': 'CHF=X',
        'CNY': 'CNY=X',
        'MXN': 'MXN=X',
        'CLP': 'CLP=X',
        'COP': 'COP=X',
        'PEN': 'PEN=X',
        'AUD': 'AUDUSD=X',
        'NZD': 'NZDUSD=X',
        'INR': 'INR=X',
        'KRW': 'KRW=X',
        'TWD': 'TWD=X',
        'HKD': 'HKD=X',
        'SGD': 'SGD=X',
        'THB': 'THB=X',
        'ZAR': 'ZAR=X',
        'RUB': 'RUB=X',
        'TRY': 'TRY=X',
        'SEK': 'SEK=X',
        'NOK': 'NOK=X',
        'DKK': 'DKK=X',
        'PLN': 'PLN=X',
        'ILS': 'ILS=X'
    }

    INVERSE_PAIRS = {'EUR', 'GBP', 'AUD', 'NZD'}

    print(f"📥 Descargando precios para {len(tickers)} activos con normalización a USD...")

    for ticker in tickers:
        ticker_upper = ticker.upper()

        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            currency = info.get('currency', 'USD')

            hist = ticker_obj.history(period=period)

            if hist.empty or 'Close' not in hist.columns:
                print(f"⚠️ {ticker}: Sin datos históricos")
                dropped.append(ticker)
                continue

            prices_series = hist['Close']

            if prices_series.index.tz is not None:
                prices_series.index = prices_series.index.tz_localize(None)

            if currency == 'USD':
                print(f"✅ {ticker}: En USD (sin conversión necesaria)")
                prices_converted = prices_series

            else:
                print(f"🔄 {ticker}: Moneda {currency} → Convirtiendo a USD...")

                if currency not in CURRENCY_PAIRS:
                    print(f"⚠️ {ticker}: Moneda {currency} no soportada para conversión")
                    dropped.append(ticker)
                    continue

                fx_symbol = CURRENCY_PAIRS[currency]

                try:
                    fx_data = yf.download(fx_symbol, period=period, progress=False)

                    if fx_data.empty:
                        print(f"⚠️ {ticker}: No se pudo descargar tipo de cambio {fx_symbol}")
                        dropped.append(ticker)
                        continue

                    if fx_data.index.tz is not None:
                        fx_data.index = fx_data.index.tz_localize(None)

                    if isinstance(fx_data.columns, pd.MultiIndex):
                        if 'Close' in fx_data.columns.get_level_values(0):
                            fx_rates = fx_data['Close'].iloc[:, 0] if fx_data['Close'].ndim > 1 else fx_data['Close']
                        else:
                            print(f"⚠️ {ticker}: No se encontró columna 'Close' en tipo de cambio {fx_symbol}")
                            dropped.append(ticker)
                            continue
                    else:
                        if 'Close' not in fx_data.columns:
                            print(f"⚠️ {ticker}: No se encontró columna 'Close' en tipo de cambio {fx_symbol}")
                            dropped.append(ticker)
                            continue
                        fx_rates = fx_data['Close']

                    fx_rates_aligned = fx_rates.reindex(prices_series.index)
                    fx_rates_aligned = fx_rates_aligned.ffill()
                    fx_rates_aligned = fx_rates_aligned.bfill()
                    fx_rates_aligned = fx_rates_aligned.interpolate(method='linear', limit_direction='both')

                    nan_pct = fx_rates_aligned.isna().sum() / len(fx_rates_aligned)
                    if nan_pct > 0.10:
                        print(f"   ⚠️ {ticker}: {nan_pct*100:.1f}% de tipo de cambio faltante. Usando promedio disponible.")
                        mean_fx = fx_rates.mean()
                        fx_rates_aligned = fx_rates_aligned.fillna(mean_fx)

                    if fx_rates_aligned.isna().any():
                        print(f"⚠️ {ticker}: Tipo de cambio incompleto después de todos los intentos de alineación")
                        dropped.append(ticker)
                        continue

                    if currency in INVERSE_PAIRS:
                        prices_converted = prices_series * fx_rates_aligned
                    else:
                        prices_converted = prices_series / fx_rates_aligned

                    if prices_converted.isna().all():
                        print(f"⚠️ {ticker}: Conversión resultó en valores nulos")
                        dropped.append(ticker)
                        continue

                    print(f"✅ {ticker}: Convertido exitosamente de {currency} a USD")

                except Exception as e:
                    print(f"❌ {ticker}: Error en conversión de moneda: {str(e)}")
                    dropped.append(ticker)
                    continue

            if prices_usd.empty:
                prices_usd = prices_converted.to_frame(name=ticker_upper)
            else:
                prices_usd = prices_usd.join(prices_converted.to_frame(name=ticker_upper), how='outer')

        except Exception as e:
            print(f"❌ {ticker}: Error general: {str(e)}")
            dropped.append(ticker)
            continue

    if not prices_usd.empty:
        prices_usd = prices_usd.dropna(how='all')
        prices_usd = prices_usd.loc[:, ~prices_usd.columns.duplicated()]
        print(f"\n🎯 Resultado: {len(prices_usd.columns)} activos descargados, {len(dropped)} fallidos")
        print(f"   Todas las columnas están expresadas en USD 💵")
    else:
        print(f"\n❌ No se pudieron descargar precios para ningún activo")

    return prices_usd, dropped


def get_risk_free_rate(default: float = 0.03) -> float:
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        if hist is not None and not hist.empty:
            rate = float(hist["Close"].dropna().iloc[-1]) / 100.0
            if 0 < rate < 0.20:
                print(f"📈 Tasa libre de riesgo (TNX): {rate:.2%}")
                return rate
    except Exception:
        pass

    fred_api_key = get_fred_api_key()
    if fred_api_key:
        try:
            resp = requests.get(
                f"{FRED_BASE_URL}/series/observations",
                params={
                    "series_id": "DGS10",
                    "api_key": fred_api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 5,
                },
                timeout=10,
            )
            data = resp.json()
            for obs in data.get("observations", []):
                if obs["value"] != ".":
                    rate = float(obs["value"]) / 100.0
                    if 0 < rate < 0.20:
                        print(f"📈 Tasa libre de riesgo (FRED DGS10): {rate:.2%}")
                        return rate
        except Exception:
            pass

    print(f"⚠️ No se pudo obtener la tasa libre de riesgo. Usando default: {default:.2%}")
    return float(default)


def obtener_tasa_libre_riesgo() -> float:
    try:
        return get_risk_free_rate()
    except Exception:
        return 0.03


def descargar_precios_historicos(tickers: List[str], periodo: str = "10y") -> pd.DataFrame:
    prices, dropped = _download_prices(tickers, period=periodo)
    if dropped:
        print(f"[WARNING] Tickers sin datos: {', '.join(dropped)}")
    return prices


def metricas_portfolio(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float) -> Tuple[float, float, float]:
    if mu is None or cov is None or weights is None:
        return float("nan"), float("nan"), float("nan")
    ret = float(weights @ mu.values)
    vol = float(np.sqrt(weights @ cov.values @ weights))
    sharpe = (ret - rf) / vol if vol else float("nan")
    return ret, vol, sharpe


def calcular_cvar_historico(weights: np.ndarray, daily_returns: pd.DataFrame, alpha: float = 0.95) -> float:
    from .optimization import _calc_cvar

    portfolio_returns = daily_returns.values @ weights
    return _calc_cvar(portfolio_returns, alpha)
