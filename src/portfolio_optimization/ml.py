"""ML data ingestion, feature engineering, and XGBoost projection engine.

Migrated faithfully from Cells 15, 16, and 17 of the source notebook.
Preserves the exact method monkey-patches applied in Cells 16 and 17 in order:
- PortfolioPredictor._train_xgb = _train_xgb_robust
- PortfolioPredictor.train_short_term_model = _train_short_term_model_robust
- PortfolioPredictor.train_long_term_model = _train_long_term_model_robust
- PortfolioProcessor._process_macro = _process_macro_with_lag
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import yfinance as yf
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from . import state
from .config import (
    ALPHA_VANTAGE_BASE_URL,
    FRED_BASE_URL,
    GOOGLE_NEWS_RSS_URL,
    _redact_secret,
    get_alpha_vantage_api_keys,
    get_fred_api_key,
)

SECTOR_MAPPING = state.SECTOR_MAPPING
_ultima_llamada_av = state._ultima_llamada_av


@dataclass
class IngestedData:
    financials: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)
    prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    economic: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Dict[str, Dict] = field(default_factory=dict)


def _normalizar_df_yf(df: pd.DataFrame, mapping: dict) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    try:
        df_t = df.copy()
        df_t.columns = pd.to_datetime(df_t.columns)
        df_t = df_t.T
        df_t.index = pd.to_datetime(df_t.index)
        df_t = df_t.sort_index(ascending=False)
        df_t = df_t.rename(columns=mapping)
        df_t = df_t.reset_index().rename(columns={'index': 'fiscalDateEnding'})
        return df_t
    except Exception as e:
        print(f"⚠️ Error normalizando datos de Yahoo Finance: {e}")
        return None


class DataIngestor:

    def __init__(self, tickers: List[str]):
        self.tickers = [t.upper() for t in tickers if t]
        self.session = self._create_session()
        self.data = IngestedData()

    def _create_session(self) -> requests.Session:
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
        return session

    def _get_currency_pair(self, currency: str) -> Optional[str]:
        """
        Obtiene el símbolo del par cambiario para convertir a USD.

        Args:
            currency: Código de moneda de 3 letras (ej: 'ARS', 'EUR', 'BRL')

        Returns:
            Símbolo del par cambiario en yfinance (ej: 'ARS=X', 'EURUSD=X') o None si es USD
        """
        if not currency or currency.upper() == 'USD':
            return None

        currency_upper = currency.upper()

        currency_mapping = {
            'ARS': 'ARS=X',
            'BRL': 'BRL=X',
            'MXN': 'MXN=X',
            'CLP': 'CLP=X',
            'COP': 'COP=X',
            'PEN': 'PEN=X',
            'EUR': 'EURUSD=X',
            'GBP': 'GBPUSD=X',
            'JPY': 'JPY=X',
            'CNY': 'CNY=X',
            'KRW': 'KRW=X',
            'CAD': 'CAD=X',
            'AUD': 'AUD=X',
            'INR': 'INR=X',
            'CHF': 'CHF=X',
            'HKD': 'HKD=X',
            'SGD': 'SGD=X',
            'ZAR': 'ZAR=X',
            'RUB': 'RUB=X',
            'TRY': 'TRY=X'
        }

        return currency_mapping.get(currency_upper, f'{currency_upper}=X')

    def _convert_to_usd(self, ticker: str, prices: pd.Series, currency: str) -> pd.Series:
        """
        Convierte una serie de precios a dólares americanos.

        Args:
            ticker: Símbolo del activo (para logging)
            prices: Serie de precios con índice datetime
            currency: Moneda original del activo

        Returns:
            Serie de precios convertida a USD
        """
        try:
            if not currency or currency.upper() == 'USD':
                return prices

            currency_pair = self._get_currency_pair(currency)
            if not currency_pair:
                return prices

            print(f"  💱 Convirtiendo {ticker} de {currency} a USD usando {currency_pair}")

            fx_ticker = yf.Ticker(currency_pair)
            fx_hist = fx_ticker.history(period='max')

            if fx_hist.empty or 'Close' not in fx_hist.columns:
                print(f"  ⚠️ No se pudo obtener el par cambiario {currency_pair}. Usando precios originales.")
                return prices

            fx_rates = fx_hist['Close']

            prices_df = pd.DataFrame({'price': prices})
            fx_df = pd.DataFrame({'fx_rate': fx_rates})

            merged = pd.merge_asof(
                prices_df.sort_index(),
                fx_df.sort_index(),
                left_index=True,
                right_index=True,
                direction='backward'
            )

            if 'EURUSD' in currency_pair or 'GBPUSD' in currency_pair:
                converted_prices = merged['price'] * merged['fx_rate']
            else:
                converted_prices = merged['price'] / merged['fx_rate']

            converted_prices = converted_prices.fillna(method='ffill').fillna(method='bfill')

            print(f"  ✅ Conversión exitosa para {ticker}")
            return converted_prices

        except Exception as e:
            logging.error(f"Error convirtiendo {ticker} de {currency} a USD: {e}")
            print(f"  ⚠️ Error en conversión para {ticker}. Usando precios originales.")
            return prices

    def _get_next_alpha_vantage_api_key(self) -> str:
        api_keys = get_alpha_vantage_api_keys()
        if not api_keys:
            raise RuntimeError(
                "ALPHA_VANTAGE_API_KEYS is not configured. Set the ALPHA_VANTAGE_API_KEYS environment variable."
            )
        api_key = api_keys[state._indice_api_key % len(api_keys)]
        state._indice_api_key += 1
        return api_key

    def _wait_for_rate_limit(self):
        tiempo_transcurrido = time.time() - state._ultima_llamada_av['tiempo']
        if tiempo_transcurrido < 12:
            time.sleep(12 - tiempo_transcurrido)
        state._ultima_llamada_av['tiempo'] = time.time()

    def _fetch_alpha_vantage_financials(self, ticker: str) -> Dict[str, pd.DataFrame]:
        functions = {
            'income_statement': 'INCOME_STATEMENT',
            'balance_sheet': 'BALANCE_SHEET',
            'cash_flow': 'CASH_FLOW'
        }

        result = {}

        for key, function in functions.items():
            self._wait_for_rate_limit()
            api_key = self._get_next_alpha_vantage_api_key()

            params = {
                'function': function,
                'symbol': ticker,
                'apikey': api_key
            }

            try:
                response = self.session.get(
                    ALPHA_VANTAGE_BASE_URL,
                    params=params,
                    timeout=30,
                    verify=False
                )

                if response.status_code != 200:
                    continue

                data = response.json()

                if 'annualReports' not in data:
                    continue

                df = pd.DataFrame(data['annualReports'])

                if df.empty:
                    continue

                df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
                df = df.sort_values('fiscalDateEnding', ascending=False)

                for col in df.columns:
                    if col not in ['fiscalDateEnding', 'reportedCurrency']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                result[key] = df

            except Exception as e:
                logging.error(f"Error fetching {function} for {ticker}: {_redact_secret(str(e), api_key)}")
                continue

        return result

    def _fetch_prices_batch(self, tickers: List[str]) -> pd.DataFrame:
        """Descarga precios en batch usando yfinance para mayor eficiencia y los normaliza a USD"""
        try:
            print(f"📥 Descargando precios en batch para {len(tickers)} activos...")
            data = yf.download(tickers, period='max', progress=False, group_by='ticker', threads=True)

            if data.empty:
                return pd.DataFrame()

            prices_dict = {}

            if len(tickers) == 1:
                if 'Close' in data.columns:
                    ticker = tickers[0]
                    ticker_upper = ticker.upper()

                    try:
                        stock_info = yf.Ticker(ticker).info
                        currency = stock_info.get('currency', 'USD')
                        prices_series = data['Close']
                        prices_converted = self._convert_to_usd(ticker_upper, prices_series, currency)
                        prices_dict[ticker_upper] = prices_converted
                    except Exception as e:
                        logging.error(f"Error procesando moneda para {ticker}: {e}")
                        prices_dict[ticker_upper] = data['Close']
            else:
                for ticker in tickers:
                    ticker_upper = ticker.upper()
                    try:
                        if ticker in data.columns.levels[0]:
                            if 'Close' in data[ticker].columns:
                                stock_info = yf.Ticker(ticker).info
                                currency = stock_info.get('currency', 'USD')
                                prices_series = data[ticker]['Close']
                                prices_converted = self._convert_to_usd(ticker_upper, prices_series, currency)
                                prices_dict[ticker_upper] = prices_converted
                    except Exception as e:
                        logging.error(f"Error procesando {ticker}: {e}")
                        try:
                            if ticker in data.columns.levels[0] and 'Close' in data[ticker].columns:
                                prices_dict[ticker_upper] = data[ticker]['Close']
                        except Exception:
                            pass

            if not prices_dict:
                return pd.DataFrame()

            prices = pd.DataFrame(prices_dict)
            prices = prices.sort_index(ascending=False)
            prices = prices.reset_index()
            if 'Date' in prices.columns:
                prices = prices.rename(columns={'Date': 'date'})
            elif 'index' in prices.columns:
                prices = prices.rename(columns={'index': 'date'})

            return prices

        except Exception as e:
            logging.error(f"Error en descarga batch: {e}")
            return pd.DataFrame()

    def _fetch_yfinance_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Descarga precios históricos individuales y los normaliza a USD"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='max')

            if hist.empty:
                return None

            info = stock.info
            currency = info.get('currency', 'USD')

            prices_series = hist['Close']
            prices_converted = self._convert_to_usd(ticker.upper(), prices_series, currency)

            df = pd.DataFrame({'date': hist.index, 'price': prices_converted.values})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False)
            df = df[['date', 'price']].copy()

            return df

        except Exception as e:
            logging.error(f"Error fetching yfinance data for {ticker}: {e}")
            return None

    def _get_ticker_info(self, ticker: str) -> Dict:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            company_name = info.get('longName') or info.get('shortName', ticker)
            country = info.get('country', 'Unknown')
            currency = info.get('currency', 'USD')

            return {
                'sector': sector,
                'industry': industry,
                'company_name': company_name,
                'country': country,
                'currency': currency
            }

        except Exception as e:
            logging.error(f"Error getting info for {ticker}: {e}")
            return {
                'sector': 'Unknown',
                'industry': 'Unknown',
                'company_name': ticker,
                'country': 'Unknown',
                'currency': 'USD'
            }

    def _fetch_yfinance_financials(self, ticker: str) -> Optional[Dict[str, Dict[str, pd.DataFrame]]]:
        try:
            stock = yf.Ticker(ticker)

            income_map = {
                'Total Revenue': 'totalRevenue',
                'Gross Profit': 'grossProfit',
                'Operating Income': 'operatingIncome',
                'Net Income': 'netIncome',
                'EBIT': 'ebit',
                'Ebit': 'ebit',
                'Pretax Income': 'incomeBeforeTax',
                'Interest Expense': 'interestExpense'
            }

            balance_map = {
                'Total Assets': 'totalAssets',
                'Total Liab': 'totalLiabilities',
                'Total Liabilities Net Minority Interest': 'totalLiabilities',
                'Total Stockholder Equity': 'totalShareholderEquity',
                "Stockholders' Equity": 'totalShareholderEquity',
                'Total Current Assets': 'totalCurrentAssets',
                'Total Current Liabilities': 'totalCurrentLiabilities',
                'Inventory': 'inventory',
                'Cash And Short Term Investments': 'cashAndShortTermInvestments',
                'Common Stock Shares Outstanding': 'commonStockSharesOutstanding',
                'Ordinary Shares Number': 'commonStockSharesOutstanding'
            }

            cashflow_map = {
                'Operating Cash Flow': 'operatingCashflow',
                'Total Cash From Operating Activities': 'operatingCashflow',
                'Dividends Paid': 'dividendPayoutCommonStock',
                'Capital Expenditure': 'capitalExpenditure',
                'Capital Expenditures': 'capitalExpenditure'
            }

            income_annual = _normalizar_df_yf(stock.financials, income_map)
            income_quarterly = _normalizar_df_yf(stock.quarterly_financials, income_map)

            balance_annual = _normalizar_df_yf(stock.balance_sheet, balance_map)
            balance_quarterly = _normalizar_df_yf(stock.quarterly_balance_sheet, balance_map)

            cashflow_annual = _normalizar_df_yf(stock.cashflow, cashflow_map)
            cashflow_quarterly = _normalizar_df_yf(stock.quarterly_cashflow, cashflow_map)

            if all(df is None for df in [income_annual, income_quarterly, balance_annual,
                                          balance_quarterly, cashflow_annual, cashflow_quarterly]):
                return None

            result = {
                'income_statement': {},
                'balance_sheet': {},
                'cash_flow': {}
            }

            if income_annual is not None:
                result['income_statement']['annual'] = income_annual
            if income_quarterly is not None:
                result['income_statement']['quarterly'] = income_quarterly

            if balance_annual is not None:
                result['balance_sheet']['annual'] = balance_annual
            if balance_quarterly is not None:
                result['balance_sheet']['quarterly'] = balance_quarterly

            if cashflow_annual is not None:
                result['cash_flow']['annual'] = cashflow_annual
            if cashflow_quarterly is not None:
                result['cash_flow']['quarterly'] = cashflow_quarterly

            return result

        except Exception as e:
            logging.error(f"Error fetching yfinance financials for {ticker}: {e}")
            return None

    def _fetch_fred_series(self, series_id: str, series_name: str) -> Optional[pd.DataFrame]:
        fred_api_key = get_fred_api_key()
        if not fred_api_key:
            logging.error(
                f"FRED_API_KEY is not configured; skipping FRED series {series_id}."
            )
            return None

        try:
            params = {
                'series_id': series_id,
                'api_key': fred_api_key,
                'file_type': 'json'
            }

            response = self.session.get(
                f"{FRED_BASE_URL}/series/observations",
                params=params,
                timeout=30
            )

            if response.status_code != 200:
                return None

            data = response.json()

            if 'observations' not in data:
                return None

            df = pd.DataFrame(data['observations'])

            if df.empty:
                return None

            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df[['date', 'value']].dropna()
            df = df.rename(columns={'value': series_name})
            df = df.sort_values('date', ascending=False)

            return df

        except Exception as e:
            logging.error(f"Error fetching FRED series {series_id}: {_redact_secret(str(e), fred_api_key)}")
            return None

    def _fetch_macro_data(self) -> Dict[str, pd.DataFrame]:
        series_mapping = {
            'GDP': 'GDP',
            'UNRATE': 'unemployment_rate',
            'CPIAUCSL': 'inflation',
            'FEDFUNDS': 'fed_rate',
            'DGS10': 'treasury_10y',
            'DEXUSEU': 'usd_eur'
        }

        macro_data = {}

        for series_id, series_name in series_mapping.items():
            df = self._fetch_fred_series(series_id, series_name)
            if df is not None:
                macro_data[series_name] = df

        return macro_data

    def ingest_all(self) -> IngestedData:
        print(f"\n🔄 Descargando datos financieros para {len(self.tickers)} activos...")

        prices_batch = self._fetch_prices_batch(self.tickers)
        if not prices_batch.empty:
            self.data.prices = prices_batch
            num_activos = len([c for c in prices_batch.columns if c not in ['date', 'Date', 'index']])
            print(f"✅ Precios descargados en batch para {num_activos} activos")

        for ticker in tqdm(self.tickers, desc="Procesando metadatos y financials"):
            try:
                financials_yf = self._fetch_yfinance_financials(ticker)
                if financials_yf:
                    self.data.financials[ticker] = financials_yf

                if prices_batch.empty or ticker.upper() not in prices_batch.columns:
                    prices = self._fetch_yfinance_data(ticker)
                    if prices is not None:
                        if self.data.prices.empty:
                            self.data.prices = prices[['date', 'price']].copy()
                            self.data.prices = self.data.prices.rename(columns={'price': ticker})
                        else:
                            temp = prices[['date', 'price']].copy()
                            temp = temp.rename(columns={'price': ticker})
                            self.data.prices = pd.merge(
                                self.data.prices,
                                temp,
                                on='date',
                                how='outer'
                            )

                metadata = self._get_ticker_info(ticker)
                self.data.metadata[ticker] = metadata

            except Exception as e:
                logging.error(f"Error ingesting data for {ticker}: {e}")
                continue

        if not self.data.prices.empty:
            if 'date' in self.data.prices.columns:
                self.data.prices = self.data.prices.sort_values('date', ascending=False)
            else:
                self.data.prices = self.data.prices.sort_index(ascending=False)
            self.data.prices = self.data.prices.ffill().bfill()

        macro_data = self._fetch_macro_data()
        self.data.economic = macro_data

        print("\n✅ Ingesta completada:")
        print(f"  • Activos con datos financieros: {len(self.data.financials)}")
        if 'date' in self.data.prices.columns:
            print(f"  • Activos con precios: {len([c for c in self.data.prices.columns if c != 'date'])}")
        else:
            print(f"  • Activos con precios: {len(self.data.prices.columns)}")
        print(f"  • Series económicas: {len(self.data.economic)}")

        return self.data


class PortfolioProcessor:

    def __init__(self, ingested_data: IngestedData):
        self.ingested_data = ingested_data
        self.processed_data = {}

    def _calculate_financial_ratios(self, ticker: str, financials: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        try:
            inc_annual = financials.get('income_statement', {}).get('annual')
            bal_annual = financials.get('balance_sheet', {}).get('annual')
            cf_annual = financials.get('cash_flow', {}).get('annual')

            if inc_annual is None or bal_annual is None:
                return pd.DataFrame()

            merged = inc_annual.copy()

            if bal_annual is not None:
                merged = pd.merge(
                    merged,
                    bal_annual,
                    on='fiscalDateEnding',
                    how='outer',
                    suffixes=('', '_bal')
                )

            if cf_annual is not None:
                merged = pd.merge(
                    merged,
                    cf_annual,
                    on='fiscalDateEnding',
                    how='outer',
                    suffixes=('', '_cf')
                )

            merged = merged.sort_values('fiscalDateEnding', ascending=False)

            ratios = pd.DataFrame()
            ratios['fiscalDateEnding'] = merged['fiscalDateEnding']

            revenue = merged.get('totalRevenue')
            net_income = merged.get('netIncome')
            total_assets = merged.get('totalAssets')
            total_equity = merged.get('totalShareholderEquity')
            total_liabilities = merged.get('totalLiabilities')
            current_assets = merged.get('totalCurrentAssets')
            current_liabilities = merged.get('totalCurrentLiabilities')
            operating_cashflow = merged.get('operatingCashflow')
            capex = merged.get('capitalExpenditure')

            if revenue is not None and (revenue != 0).any():
                if net_income is not None:
                    ratios['profit_margin'] = net_income / revenue

            if total_assets is not None and (total_assets != 0).any():
                if net_income is not None:
                    ratios['roa'] = net_income / total_assets
                if revenue is not None:
                    ratios['asset_turnover'] = revenue / total_assets

            if total_equity is not None and (total_equity != 0).any():
                if net_income is not None:
                    ratios['roe'] = net_income / total_equity

            if total_liabilities is not None and total_equity is not None and (total_equity != 0).any():
                ratios['debt_to_equity'] = total_liabilities / total_equity

            if current_assets is not None and current_liabilities is not None and (current_liabilities != 0).any():
                ratios['current_ratio'] = current_assets / current_liabilities

            if operating_cashflow is not None:
                if capex is not None:
                    ratios['fcf'] = operating_cashflow + capex

            return ratios.replace([np.inf, -np.inf], np.nan)

        except Exception as e:
            logging.error(f"Error calculating ratios for {ticker}: {e}")
            return pd.DataFrame()

    def _merge_with_prices(self, ticker: str, ratios: pd.DataFrame) -> pd.DataFrame:
        """
        Fusiona los datos fundamentales con los precios históricos, corrigiendo el Look-Ahead Bias.

        IMPORTANTE: Los reportes financieros no están disponibles el día del cierre fiscal.
        Se aplica un lag conservador de 90 días para simular el retraso real de publicación.

        Args:
            ticker: Símbolo del activo
            ratios: DataFrame con ratios financieros y columna 'fiscalDateEnding'

        Returns:
            DataFrame con precios y ratios alineados temporalmente sin Look-Ahead Bias
        """
        try:
            if self.ingested_data.prices.empty or ticker not in self.ingested_data.prices.columns:
                return pd.DataFrame()

            if 'date' in self.ingested_data.prices.columns:
                prices = self.ingested_data.prices[['date', ticker]].copy()
                prices = prices.rename(columns={ticker: 'price'})
                prices['date'] = pd.to_datetime(prices['date'])
                prices = prices.sort_values('date', ascending=True)
            else:
                prices = self.ingested_data.prices[[ticker]].copy()
                prices = prices.rename(columns={ticker: 'price'})
                prices = prices.reset_index()
                prices = prices.rename(columns={'index': 'date'})
                prices['date'] = pd.to_datetime(prices['date'])
                prices = prices.sort_values('date', ascending=True)

            if ratios.empty:
                return prices.sort_values('date', ascending=False)

            ratios = ratios.copy()
            ratios['fiscalDateEnding'] = pd.to_datetime(ratios['fiscalDateEnding'])

            ratios['reporting_date'] = ratios['fiscalDateEnding'] + pd.Timedelta(days=90)

            ratios = ratios.sort_values('reporting_date', ascending=True)

            ratios_for_merge = ratios.drop(columns=['fiscalDateEnding'])

            merged = pd.merge_asof(
                prices,
                ratios_for_merge,
                left_on='date',
                right_on='reporting_date',
                direction='backward'
            )

            merged = merged.drop(columns=['reporting_date'], errors='ignore')

            merged = merged.sort_values('date', ascending=False)

            return merged

        except Exception as e:
            logging.error(f"Error merging prices for {ticker}: {e}")
            return pd.DataFrame()

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if df.empty or 'price' not in df.columns:
                return df

            df = df.sort_values('date')

            df['sma_50'] = df['price'].rolling(window=50).mean()
            df['sma_200'] = df['price'].rolling(window=200).mean()

            df['price_to_sma50'] = df['price'] / df['sma_50']
            df['price_to_sma200'] = df['price'] / df['sma_200']
            df['sma50_to_sma200'] = df['sma_50'] / df['sma_200']

            df['log_return_1m'] = np.log(df['price'] / df['price'].shift(20))
            df['log_return_3m'] = np.log(df['price'] / df['price'].shift(60))
            df['log_return_6m'] = np.log(df['price'] / df['price'].shift(120))
            df['log_return_1y'] = np.log(df['price'] / df['price'].shift(252))

            df['volatility_30d'] = df['price'].pct_change().rolling(window=30).std()
            df['volatility_90d'] = df['price'].pct_change().rolling(window=90).std()

            df = df.sort_values('date', ascending=False)

            return df

        except Exception as e:
            logging.error(f"Error adding technical indicators: {e}")
            return df

    def _merge_macro_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if df.empty or 'date' not in df.columns:
                return df

            df = df.sort_values('date')

            for series_name, series_df in self.ingested_data.economic.items():
                if series_df.empty:
                    continue

                series_df = series_df.copy()
                series_df['date'] = pd.to_datetime(series_df['date'])
                series_df = series_df.sort_values('date')

                df = pd.merge_asof(
                    df,
                    series_df,
                    on='date',
                    direction='backward'
                )

            df = df.sort_values('date', ascending=False)

            return df

        except Exception as e:
            logging.error(f"Error merging macro data: {e}")
            return df

    def _add_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if df.empty or 'price' not in df.columns:
                return df

            df = df.sort_values('date')

            df['future_price_1q'] = df['price'].shift(-63)
            df['target_next_return'] = (df['future_price_1q'] / df['price']) - 1

            df['future_price_5y'] = df['price'].shift(-252 * 5)
            df['target_5y_return'] = np.log(df['future_price_5y'] / df['price'])

            df = df.sort_values('date', ascending=False)

            return df

        except Exception as e:
            logging.error(f"Error adding target variables: {e}")
            return df

    def process_all(self) -> Dict[str, pd.DataFrame]:
        print("\n🔧 Procesando datos financieros y agregando features...")

        for ticker in tqdm(self.ingested_data.financials.keys(), desc="Features ML"):
            try:
                financials = self.ingested_data.financials.get(ticker)
                if not financials:
                    continue

                ratios = self._calculate_financial_ratios(ticker, financials)
                df = self._merge_with_prices(ticker, ratios)

                if df.empty:
                    continue

                df = self._add_technical_indicators(df)
                df = self._merge_macro_data(df)
                df = self._add_target_variables(df)

                self.processed_data[ticker] = df

            except Exception as e:
                logging.error(f"Error processing {ticker}: {e}")
                continue

        print(f"\n✅ Procesamiento completado: {len(self.processed_data)} activos listos para ML")

        return self.processed_data


class PortfolioPredictor:

    def __init__(self, processed_data: Dict[str, pd.DataFrame]):
        self.processed_data = processed_data
        self.models = {}
        self.models_5y = {}
        self.feature_importance = {}
        self.feature_importance_5y = {}

    def _train_xgb(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X, y, verbose=False)

        return model

    def train_short_term_model(self, ticker: str) -> Optional[Dict]:
        if ticker not in self.processed_data:
            return None

        try:
            df = self.processed_data[ticker].copy()

            if len(df) < 12:
                return None

            exclude_cols = ["target_next_return", "target_5y_return", "price", "future_price_1q", "future_price_5y", "date"]
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            df_train = df.dropna(subset=["target_next_return"])

            if df_train.empty:
                return None

            X = df_train[feature_cols].fillna(0)
            y = df_train["target_next_return"]

            model = self._train_xgb(X, y)
            self.models[ticker] = model

            importances = model.feature_importances_
            sorted_importance = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
            self.feature_importance[ticker] = sorted_importance

            last_row = df.iloc[-1:][feature_cols].fillna(0)
            current_price = float(df.iloc[-1]["price"])
            predicted_return = float(model.predict(last_row)[0])
            predicted_price = current_price * (1 + predicted_return)

            return {
                'ticker': ticker,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_return': predicted_return,
                'pct_change': predicted_return * 100,
                'top_features': sorted_importance[:5],
                'n_features': len(feature_cols),
                'n_samples': len(df_train),
                'prediction_method': "XGBoost Only"
            }

        except Exception as e:
            logging.error(f"Error al predecir 1Q {ticker}: {e}")
            return None

    def predict_all(self) -> Dict[str, Dict]:
        results = {}
        for ticker in self.processed_data.keys():
            result = self.train_short_term_model(ticker)
            if result is not None:
                results[ticker] = result
        return results

    def train_long_term_model(self, ticker: str) -> Optional[Dict]:
        if ticker not in self.processed_data:
            return None

        try:
            df = self.processed_data[ticker].copy()
            if len(df) < 12:
                return None

            exclude_cols = ["target_next_return", "target_5y_return", "price", "future_price_1q", "future_price_5y", "date"]
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            df_train = df.dropna(subset=["target_5y_return"])
            if df_train.empty:
                return None

            X = df_train[feature_cols].fillna(0)
            y = df_train["target_5y_return"]
            model = self._train_xgb(X, y)
            self.models_5y[ticker] = model

            importances = model.feature_importances_
            sorted_importance = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
            self.feature_importance_5y[ticker] = sorted_importance

            last_row = df.iloc[-1:][feature_cols].fillna(0)
            current_price = float(df.iloc[-1]["price"])
            base_return = float(model.predict(last_row)[0])

            scenarios = {
                'base': {
                    'predicted_return': base_return,
                    'predicted_price': current_price * np.exp(base_return),
                    'description': 'Escenario Base'
                },
                'bull': {
                    'predicted_return': base_return * 1.15,
                    'predicted_price': current_price * np.exp(base_return * 1.15),
                    'description': 'Escenario Bull'
                },
                'bear': {
                    'predicted_return': base_return * 0.85,
                    'predicted_price': current_price * np.exp(base_return * 0.85),
                    'description': 'Escenario Bear'
                }
            }

            return {
                'ticker': ticker,
                'current_price': current_price,
                'scenarios': scenarios,
                'top_features': sorted_importance[:5],
                'n_features': len(feature_cols),
                'n_samples': len(df_train)
            }

        except Exception as e:
            logging.error(f"Error al predecir 5Y {ticker}: {e}")
            return None

    def predict_all_long_term(self) -> Dict[str, Dict]:
        results = {}
        for ticker in self.processed_data.keys():
            result = self.train_long_term_model(ticker)
            if result is not None:
                results[ticker] = result
        return results


# =============================================================================
# Cell 16: Robust TimeSeriesSplit XGBoost monkey-patches on PortfolioPredictor
# =============================================================================

def _train_xgb_robust(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 4):
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    import xgboost as xgb

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    best_iterations = []

    X_values = X.values if hasattr(X, "values") else X
    y_values = y.values if hasattr(y, "values") else y

    for train_idx, val_idx in tscv.split(X_values):
        X_train = X_values[train_idx]
        X_val = X_values[val_idx]
        y_train = y_values[train_idx]
        y_val = y_values[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror"
        )

        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=25
            )
        except TypeError:
            try:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            except TypeError:
                model.fit(
                    X_train,
                    y_train,
                    verbose=False
                )

        preds = model.predict(X_val)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        mae_scores.append(mean_absolute_error(y_val, preds))
        r2_scores.append(r2_score(y_val, preds))

        if hasattr(model, "best_iteration") and model.best_iteration is not None:
            best_iterations.append(model.best_iteration + 1)

    best_n_estimators = int(np.mean(best_iterations)) if best_iterations else 500

    final_model = xgb.XGBRegressor(
        n_estimators=best_n_estimators,
        max_depth=3,
        learning_rate=0.05,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror"
    )

    try:
        final_model.fit(X_values, y_values, verbose=False)
    except TypeError:
        final_model.fit(X_values, y_values)

    metrics = {
        "rmse_mean": float(np.mean(rmse_scores)) if rmse_scores else None,
        "rmse_std": float(np.std(rmse_scores)) if rmse_scores else None,
        "mae_mean": float(np.mean(mae_scores)) if mae_scores else None,
        "mae_std": float(np.std(mae_scores)) if mae_scores else None,
        "r2_mean": float(np.mean(r2_scores)) if r2_scores else None,
        "r2_std": float(np.std(r2_scores)) if r2_scores else None,
        "best_n_estimators": int(best_n_estimators)
    }

    return final_model, metrics


def _train_short_term_model_robust(self, ticker: str) -> Optional[Dict]:
    if ticker not in self.processed_data:
        return None

    try:
        df = self.processed_data[ticker].copy()

        if len(df) < 12:
            return None

        exclude_cols = ["target_next_return", "target_5y_return", "price", "future_price_1q", "future_price_5y", "date"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        df_train = df.dropna(subset=["target_next_return"])

        if df_train.empty:
            return None

        X = df_train[feature_cols].fillna(0)
        y = df_train["target_next_return"]

        model, metrics = self._train_xgb(X, y)
        self.models[ticker] = model

        importances = model.feature_importances_
        sorted_importance = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        self.feature_importance[ticker] = sorted_importance

        df_latest = df.sort_values("date") if "date" in df.columns else df
        last_row = df_latest.iloc[-1:][feature_cols].fillna(0)
        current_price = float(df_latest.iloc[-1]["price"])
        predicted_return = float(model.predict(last_row)[0])
        predicted_price = current_price * (1 + predicted_return)

        return {
            'ticker': ticker,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_return': predicted_return,
            'pct_change': predicted_return * 100,
            'top_features': sorted_importance[:5],
            'n_features': len(feature_cols),
            'n_samples': len(df_train),
            'prediction_method': "XGBoost Only",
            'metrics': metrics
        }

    except Exception as e:
        logging.error(f"Error al predecir 1Q {ticker}: {e}")
        return None


def _train_long_term_model_robust(self, ticker: str) -> Optional[Dict]:
    if ticker not in self.processed_data:
        return None

    try:
        df = self.processed_data[ticker].copy()
        if len(df) < 12:
            return None

        exclude_cols = ["target_next_return", "target_5y_return", "price", "future_price_1q", "future_price_5y", "date"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        df_train = df.dropna(subset=["target_5y_return"])
        if df_train.empty:
            return None

        X = df_train[feature_cols].fillna(0)
        y = df_train["target_5y_return"]
        model, metrics = self._train_xgb(X, y)
        self.models_5y[ticker] = model

        importances = model.feature_importances_
        sorted_importance = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        self.feature_importance_5y[ticker] = sorted_importance

        df_latest = df.sort_values("date") if "date" in df.columns else df
        last_row = df_latest.iloc[-1:][feature_cols].fillna(0)
        current_price = float(df_latest.iloc[-1]["price"])
        base_return = float(model.predict(last_row)[0])

        scenarios = {
            'base': {
                'predicted_return': base_return,
                'predicted_price': current_price * np.exp(base_return),
                'description': 'Escenario Base'
            },
            'bull': {
                'predicted_return': base_return * 1.15,
                'predicted_price': current_price * np.exp(base_return * 1.15),
                'description': 'Escenario Bull'
            },
            'bear': {
                'predicted_return': base_return * 0.85,
                'predicted_price': current_price * np.exp(base_return * 0.85),
                'description': 'Escenario Bear'
            }
        }

        return {
            'ticker': ticker,
            'current_price': current_price,
            'scenarios': scenarios,
            'top_features': sorted_importance[:5],
            'n_features': len(feature_cols),
            'n_samples': len(df_train),
            'metrics': metrics
        }

    except Exception as e:
        logging.error(f"Error al predecir 5Y {ticker}: {e}")
        return None


PortfolioPredictor._train_xgb = _train_xgb_robust
PortfolioPredictor.train_short_term_model = _train_short_term_model_robust
PortfolioPredictor.train_long_term_model = _train_long_term_model_robust


# =============================================================================
# Cell 17: PortfolioProcessor._process_macro monkey-patch
# =============================================================================

def _process_macro_with_lag(self, sector_code: str) -> pd.DataFrame:
    return pd.DataFrame()


PortfolioProcessor._process_macro = _process_macro_with_lag

__all__ = [
    "ALPHA_VANTAGE_BASE_URL",
    "FRED_BASE_URL",
    "GOOGLE_NEWS_RSS_URL",
    "SECTOR_MAPPING",
    "IngestedData",
    "_normalizar_df_yf",
    "DataIngestor",
    "PortfolioProcessor",
    "PortfolioPredictor",
    "_train_xgb_robust",
    "_train_short_term_model_robust",
    "_train_long_term_model_robust",
    "_process_macro_with_lag",
]
