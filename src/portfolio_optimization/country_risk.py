"""Country risk data acquisition and relative ETF risk premia.

Migrated faithfully from Cells 03 and 04 of the source notebook.
Network calls are encapsulated in callable functions rather than running on import.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from . import state
from ._display import HTML, display

# Cell 04: Country-to-ETF mapping (preserved verbatim)
country_etf_mapping: Dict[str, str] = {
    'Australia': 'EWA',
    'Austria': 'EWO',
    'Belgium': 'EWK',
    'Brazil': 'EWZ',
    'Canada': 'EWC',
    'Chile': 'ECH',
    'China': 'MCHI',
    'Colombia': 'GXG',
    'Denmark': 'EDEN',
    'Finland': 'EFNL',
    'France': 'EWQ',
    'Germany': 'EWG',
    'Hong Kong': 'EWH',
    'India': 'INDA',
    'Indonesia': 'EIDO',
    'Ireland': 'EIRL',
    'Israel': 'EIS',
    'Italy': 'EWI',
    'Japan': 'EWJ',
    'Malaysia': 'EWM',
    'Mexico': 'EWW',
    'Netherlands': 'EWN',
    'New Zealand': 'ENZL',
    'Norway': 'ENOR',
    'Peru': 'EPU',
    'Philippines': 'EPHE',
    'Portugal': 'PGAL',
    'Russia': 'ERUS',
    'Singapore': 'EWS',
    'South Africa': 'EZA',
    'South Korea': 'EWY',
    'Spain': 'EWP',
    'Sweden': 'EWD',
    'Switzerland': 'EWL',
    'Taiwan': 'EWT',
    'Thailand': 'THD',
    'Turkey': 'TUR',
    'United Kingdom': 'EWU',
}


def obtener_riesgo_pais_argentina(
    allow_manual_input: bool = False,
    render_html: bool = True,
) -> Dict[str, Optional[object]]:
    """Fetch Argentina country risk premium (Cell 03 logic).

    Attempts Ámbito Financiero JSON API first, then Rava Bursátil HTML scraping,
    and optionally interactive stdin input when ``allow_manual_input=True``.
    Updates ``state.prima_riesgo_mas_reciente``, ``state.fecha_actualizacion``,
    and ``state.fuente_datos``.
    """
    prima_riesgo_mas_reciente = None
    fecha_actualizacion = None
    fuente_datos = None

    hoy = datetime.now()
    hace_dos_anios = hoy - timedelta(days=730)
    fecha_inicio = hace_dos_anios.strftime('%d-%m-%Y')
    fecha_fin = hoy.strftime('%d-%m-%Y')

    url_ambito = f"https://mercados.ambito.com/riesgopais/historico-general/{fecha_inicio}/{fecha_fin}"

    headers_completos = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://mercados.ambito.com/',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin'
    }

    try:
        session = requests.Session()
        response = session.get(url_ambito, headers=headers_completos, timeout=15)

        if response.status_code == 200:
            datos = response.json()
            df = pd.DataFrame(datos[1:], columns=datos[0])
            df['Puntos'] = df['Puntos'].astype(str).str.replace(',', '.')
            df['Puntos'] = pd.to_numeric(df['Puntos'])
            df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d-%m-%Y')
            df = df.sort_values('Fecha', ascending=False).reset_index(drop=True)

            prima_riesgo_mas_reciente = df.iloc[0]['Puntos'] / 100
            fecha_actualizacion = df.iloc[0]['Fecha'].strftime('%d/%m/%Y')
            fuente_datos = "Ámbito Financiero"
        else:
            print(f"⚠️ Ámbito respondió con código {response.status_code}. Intentando Rava...")
    except Exception as e:
        print(f"⚠️ Error en Ámbito: {str(e)}. Intentando Rava...")

    if prima_riesgo_mas_reciente is None:
        try:
            print("🔄 Intentando Rava...")
            url_rava = "https://www.rava.com/perfil/riesgo%20pais"

            headers_rava = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'es-AR,es;q=0.9,en;q=0.8',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }

            session_rava = requests.Session()
            response_rava = session_rava.get(url_rava, headers=headers_rava, timeout=15, allow_redirects=True)

            if response_rava.status_code == 200:
                soup = BeautifulSoup(response_rava.content, 'html.parser')
                all_text = soup.get_text()
                valor_encontrado = None

                price_elem = soup.find('div', class_='price')
                if price_elem:
                    texto_price = price_elem.get_text(strip=True)
                    numeros_solo = re.sub(r'[^\d,\.]', '', texto_price)
                    match = re.search(r'(\d{3,4})[,\.]?(\d{2})?', numeros_solo)
                    if match:
                        valor_encontrado = match.group(1) + '.' + (match.group(2) if match.group(2) else '00')

                if not valor_encontrado:
                    matches = re.findall(r'(\d{3,4})[,\.](\d{2})', all_text)
                    if matches:
                        for match in matches:
                            valor_num = float(match[0] + '.' + match[1])
                            if 300 <= valor_num <= 3000:
                                valor_encontrado = match[0] + '.' + match[1]
                                break

                if valor_encontrado:
                    prima_riesgo_mas_reciente = float(valor_encontrado) / 100
                    fecha_actualizacion = datetime.now().strftime('%d/%m/%Y')
                    fuente_datos = "Rava Bursátil"
            else:
                print(f"⚠️ Rava respondió con código {response_rava.status_code}")
        except Exception as e:
            print(f"❌ Error en Rava: {str(e)}")

    if prima_riesgo_mas_reciente is None and allow_manual_input:
        print("\n" + "="*70)
        print("❌ ERROR: No se pudo obtener el Riesgo País automáticamente.")
        print("="*70)
        print("\n💡 SOLUCIÓN: Ingresa el valor manualmente desde alguna de estas fuentes:")
        print("   📍 https://www.ambito.com/contenidos/riesgo-pais.html")
        print("   📍 https://www.rava.com/perfil/riesgo%20pais")
        print("\n")

        try:
            valor_manual = input("Ingresa el Riesgo País en puntos básicos (ej: 650): ")
            if valor_manual and valor_manual.strip():
                valor_limpio = re.sub(r'[^\d\.]', '', valor_manual)
                if valor_limpio:
                    valor_puntos = float(valor_limpio)
                    # Convertir puntos básicos a porcentaje: 650 puntos = 6.5%
                    prima_riesgo_mas_reciente = valor_puntos / 100

                    fecha_actualizacion = datetime.now().strftime('%d/%m/%Y')
                    fuente_datos = "Ingreso Manual"
        except Exception as e:
            print(f"Error al procesar input: {str(e)}")

    if prima_riesgo_mas_reciente is None:
        raise RuntimeError("❌ ERROR CRÍTICO: No se pudo obtener el Riesgo País. Por favor, ejecuta la celda nuevamente e ingresa el valor manualmente.")

    state.prima_riesgo_mas_reciente = prima_riesgo_mas_reciente
    state.fecha_actualizacion = fecha_actualizacion
    state.fuente_datos = fuente_datos

    if prima_riesgo_mas_reciente is not None:
        if render_html:
            html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        text-align: center;
        max-width: 500px;
        margin: 30px auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    ">
        <h2 style="
            color: white;
            font-size: 24px;
            margin-bottom: 10px;
            font-weight: 300;
            letter-spacing: 1px;
        ">Prima de Riesgo País</h2>

        <div style="
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
        ">
            <div style="
                font-size: 64px;
                font-weight: bold;
                color: #667eea;
                margin-bottom: 10px;
            ">{prima_riesgo_mas_reciente:.2f}%</div>

            <div style="
                font-size: 14px;
                color: #666;
                margin-top: 15px;
            ">Actualizado: {fecha_actualizacion}</div>
        </div>

        <div style="
            color: rgba(255,255,255,0.8);
            font-size: 12px;
            margin-top: 15px;
        ">Fuente: {fuente_datos}</div>
    </div>
    """

            display(HTML(html))
    else:
        print("❌ Error crítico: No se pudo obtener el riesgo país de ninguna fuente.")

    return {
        "prima_riesgo_mas_reciente": prima_riesgo_mas_reciente,
        "fecha_actualizacion": fecha_actualizacion,
        "fuente_datos": fuente_datos,
    }


def calcular_primas_riesgo_pais(
    etf_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Compute relative country risk scores ``sigma(ETF) / sigma(URTH)`` (Cell 04 logic).

    Updates ``state.country_risk_premia`` and returns the resulting dictionary.
    """
    mapping = country_etf_mapping if etf_mapping is None else etf_mapping

    print("\n" + "="*70)
    print("🌍 CÁLCULO DE PRIMAS DE RIESGO PAÍS")
    print("="*70)
    print("Método: Country Risk Score = σ(ETF país) / σ(URTH)")
    print("Período: Últimos 3 años | Ventana: 252 días de trading")
    print("="*70 + "\n")

    country_risk_premia: Dict[str, float] = {}

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)

        urth_ticker = yf.Ticker('URTH')
        urth_hist = urth_ticker.history(start=start_date, end=end_date)

        if urth_hist.empty or len(urth_hist) < 200:
            print("⚠️ No se pudo obtener suficiente historial para URTH")
            urth_vol = None
        else:
            urth_returns = np.log(urth_hist['Close'] / urth_hist['Close'].shift(1)).dropna()
            urth_vol = urth_returns.std() * np.sqrt(252)
            print(f"✅ URTH (MSCI World): σ anualizada = {urth_vol*100:.2f}%")

        if urth_vol is not None and urth_vol > 0:
            success_count = 0
            failed_count = 0

            for country, etf_ticker in mapping.items():
                try:
                    etf = yf.Ticker(etf_ticker)
                    etf_hist = etf.history(start=start_date, end=end_date)

                    if etf_hist.empty or len(etf_hist) < 200:
                        failed_count += 1
                        continue

                    etf_returns = np.log(etf_hist['Close'] / etf_hist['Close'].shift(1)).dropna()
                    etf_vol = etf_returns.std() * np.sqrt(252)

                    country_risk_score = etf_vol / urth_vol
                    country_risk_premia[country] = country_risk_score

                    print(f"  {country:20s} ({etf_ticker:5s}): σ={etf_vol*100:5.2f}% | Score={country_risk_score:.4f}")
                    success_count += 1

                except Exception:
                    failed_count += 1
                    continue

            print(f"\n{'='*70}")
            print(f"✅ Procesados: {success_count} países")
            if failed_count > 0:
                print(f"⚠️ Excluidos: {failed_count} países (sin datos suficientes)")
            print(f"{'='*70}\n")
        else:
            print("❌ No se pudo calcular volatilidad de URTH. Usando valores por defecto.\n")

    except Exception as e:
        print(f"❌ Error al calcular primas de riesgo país: {str(e)}\n")

    if not country_risk_premia:
        print("⚠️ No se pudieron calcular primas de riesgo país. El modelo continuará sin ajustes adicionales.\n")

    state.country_risk_premia = country_risk_premia
    return country_risk_premia
