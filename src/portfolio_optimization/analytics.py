"""Portfolio analytics, charts, drawdowns, MCR, risk profile, and capital allocation.

Migrated faithfully from Cells 22, 24, 26, 30, 32, 34, and 36 of the source notebook.
All routines are callable functions with zero execution on module import.
"""

from __future__ import annotations

import traceback
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from . import state
from ._display import HTML, display
from .market_data import (
    calcular_cvar_historico,
    descargar_precios_historicos,
    metricas_portfolio,
    obtener_tasa_libre_riesgo,
)


# =============================================================================
# Cell 22: Efficient frontier simulation and plot
# =============================================================================

def generar_frontera_eficiente(n_samples: int = 1000000):
    resultado_analisis = state.resultado_analisis
    ret_esp = state.ret_esp
    cov = state.cov

    if resultado_analisis is not None and ret_esp is not None and cov is not None:
        try:
            display(HTML("""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2);
                    border-radius: 15px; padding: 25px; margin: 30px 0; text-align: center;">
            <h2 style="color: white; margin: 0 0 10px 0;">📊 Frontera Eficiente</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 14px;">
                Visualización de todos los portfolios posibles
            </p>
        </div>
        """))

            rf = obtener_tasa_libre_riesgo()
            precios = resultado_analisis.prices_hist

            if precios is not None and not precios.empty:
                retornos_diarios = precios.pct_change(fill_method=None).dropna()

                n_activos = len(ret_esp)

                print(f"🔄 Generando {n_samples:,} portfolios simulados...")

                pesos_simulados = np.random.dirichlet(np.ones(n_activos), size=n_samples)

                retornos_port = pesos_simulados @ ret_esp.values
                vol_port = np.sqrt(np.einsum("ij,jk,ik->i", pesos_simulados, cov.values, pesos_simulados))
                sharpe_port = (retornos_port - rf) / vol_port

                mejor_sharpe = int(np.nanargmax(sharpe_port))
                mejor_minvar = int(np.nanargmin(vol_port))

                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(10, 6.5), facecolor='#0b1221')
                ax.set_facecolor('#111827')

                sc = ax.scatter(vol_port * 100, retornos_port * 100, c=sharpe_port, cmap="viridis",
                               s=15, alpha=0.6, edgecolors='none')

                ax.scatter(vol_port[mejor_sharpe] * 100, retornos_port[mejor_sharpe] * 100,
                          color="#f97316", s=120, edgecolor="white", linewidth=1.5,
                          label="Max Sharpe", zorder=5)
                ax.scatter(vol_port[mejor_minvar] * 100, retornos_port[mejor_minvar] * 100,
                          color="#22c55e", s=90, edgecolor="white", linewidth=1.2,
                          label="Min Var", zorder=5)

                ax.set_xlabel("Volatilidad Anual (%)", fontsize=11, color="#e5e7eb", fontweight='normal')
                ax.set_ylabel("Retorno Anual Esperado (%)", fontsize=11, color="#e5e7eb", fontweight='normal')
                ax.set_title("Frontera Eficiente Simulada", fontsize=14, color="#e5e7eb",
                            fontweight='bold', pad=15)

                cbar = plt.colorbar(sc, ax=ax, pad=0.02)
                cbar.set_label("Ratio de Sharpe", fontsize=10, color="#e5e7eb")
                cbar.ax.tick_params(colors="#e5e7eb", labelsize=9)

                legend = ax.legend(loc='best', fontsize=10, framealpha=0.9,
                                  edgecolor='#38bdf8', facecolor='#1f2937')
                for text in legend.get_texts():
                    text.set_color('#e5e7eb')

                ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, color='#94a3b8')
                ax.tick_params(colors="#e5e7eb", labelsize=9)

                for spine in ax.spines.values():
                    spine.set_color('#1f2937')
                    spine.set_linewidth(1.5)

                plt.subplots_adjust(left=0.10, right=0.92, top=0.94, bottom=0.10)

                display(fig)
                plt.close()

                print(f"\n✅ Frontera eficiente generada con {n_samples:,} portfolios simulados")
                print(f"📊 Rango de volatilidad: {vol_port.min()*100:.2f}% - {vol_port.max()*100:.2f}%")
                print(f"📈 Rango de retorno esperado: {retornos_port.min()*100:.2f}% - {retornos_port.max()*100:.2f}%")
                print(f"⭐ Mejor Sharpe Ratio: {sharpe_port[mejor_sharpe]:.2f}")
            else:
                print("❌ No hay precios históricos disponibles para generar la frontera eficiente.")

        except Exception as e:
            print(f"❌ Error al generar frontera eficiente: {e}")
            traceback.print_exc()
    else:
        display(HTML("""
    <div style="background: #fff3cd; border-left: 5px solid #ffc107; padding: 20px;
                margin: 20px 0; border-radius: 8px;">
        <h3 style="color: #856404; margin: 0 0 10px 0;">⚠️ Análisis no disponible</h3>
        <p style="color: #856404; margin: 0; font-size: 14px;">
            Debes ejecutar primero el análisis completo de portfolio usando el botón
            <strong>🚀 EJECUTAR ANÁLISIS COMPLETO</strong> en la celda anterior.
        </p>
    </div>
    """))


# =============================================================================
# Cell 24: Historical 10-year performance analysis
# =============================================================================

def analizar_rendimiento_historico():
    activos_portfolio = state.activos_portfolio
    if not activos_portfolio:
        print("❌ No hay activos en el portfolio.")
        return

    print("⏳ Descargando datos históricos de los últimos 10 años...")

    tickers = [a['ticker'] for a in activos_portfolio]
    datos_5y = yf.download(tickers, period='10y', progress=False, threads=True)

    if isinstance(datos_5y.columns, pd.MultiIndex):
        precios_5y = datos_5y['Adj Close'] if 'Adj Close' in datos_5y else datos_5y['Close']
    else:
        precios_5y = datos_5y['Adj Close'] if 'Adj Close' in datos_5y else datos_5y['Close']

    if isinstance(precios_5y, pd.Series):
        precios_5y = precios_5y.to_frame(name=tickers[0])

    precios_5y = precios_5y.ffill().bfill().dropna()

    if precios_5y.empty:
        print("❌ No se pudieron descargar datos históricos suficientes.")
        return

    precios_norm = (precios_5y / precios_5y.iloc[0]) * 100

    rendimientos_acum = ((precios_5y / precios_5y.iloc[0]) - 1) * 100

    retornos_diarios = precios_5y.pct_change().dropna()
    retornos_anuales = ((precios_5y.iloc[-1] / precios_5y.iloc[0]) ** (1/10) - 1) * 100
    volatilidad_anual = retornos_diarios.std() * np.sqrt(252) * 100

    rf = obtener_tasa_libre_riesgo()

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0b1221')
    ax.set_facecolor('#111827')

    for ticker in precios_5y.columns:
        ax.plot(precios_norm.index, precios_norm[ticker], label=ticker, linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Fecha", fontsize=11, color="#e5e7eb", fontweight='normal')
    ax.set_ylabel("Valor Normalizado (Base 100)", fontsize=11, color="#e5e7eb", fontweight='normal')
    ax.set_title("Rendimiento Histórico del Portfolio (10 años)", fontsize=14, color="#e5e7eb",
                fontweight='bold', pad=15)

    legend = ax.legend(loc='upper left', fontsize=9, framealpha=0.9,
                      edgecolor='#38bdf8', facecolor='#1f2937')
    for text in legend.get_texts():
        text.set_color('#e5e7eb')

    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, color='#94a3b8')
    ax.tick_params(colors="#e5e7eb", labelsize=9)

    for spine in ax.spines.values():
        spine.set_color('#1f2937')
        spine.set_linewidth(1.5)

    plt.subplots_adjust(left=0.10, right=0.95, top=0.94, bottom=0.10)

    display(fig)
    plt.close()

    html_stats = "<div style='margin-top:30px;'><h3 style='text-align:center; color:#2C3E50; margin-bottom:20px;'>📊 Estadísticas de Rendimiento (10 años)</h3>"
    html_stats += "<table style='width:100%; border-collapse:collapse; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>"
    html_stats += "<thead><tr style='background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:white;'>"
    html_stats += "<th style='padding:15px; text-align:left;'>Activo</th>"
    html_stats += "<th style='padding:15px; text-align:center;'>Rendimiento Anualizado</th>"
    html_stats += "<th style='padding:15px; text-align:center;'>Rendimiento Total</th>"
    html_stats += "<th style='padding:15px; text-align:center;'>Volatilidad Anual</th>"
    html_stats += f"<th style='padding:15px; text-align:center;'>Sharpe Ratio (rf={rf:.1%})</th>"
    html_stats += "</tr></thead><tbody>"

    for i, ticker in enumerate(precios_5y.columns):
        ret_anual = retornos_anuales[ticker]
        ret_total = rendimientos_acum[ticker].iloc[-1]
        vol_anual = volatilidad_anual[ticker]
        sharpe = (ret_anual/100 - rf) / (vol_anual/100) if vol_anual > 0 else 0

        color_ret = '#006400' if ret_anual > 0 else '#8B0000'
        bg_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'

        html_stats += f"<tr style='background:{bg_color}; border-bottom:1px solid #dee2e6;'>"
        html_stats += f"<td style='padding:12px; font-weight:bold; color:#2C3E50;'>{ticker}</td>"
        html_stats += f"<td style='padding:12px; text-align:center; color:{color_ret}; font-weight:bold;'>{ret_anual:.2f}%</td>"
        html_stats += f"<td style='padding:12px; text-align:center; color:{color_ret};'>{ret_total:.2f}%</td>"
        html_stats += f"<td style='padding:12px; text-align:center; color:#FF6B6B;'>{vol_anual:.2f}%</td>"
        html_stats += f"<td style='padding:12px; text-align:center; color:#667eea;'>{sharpe:.2f}</td>"
        html_stats += "</tr>"

    html_stats += "</tbody></table>"
    html_stats += "</tbody></table></div>"

    display(HTML(html_stats))
    pesos_eq = np.array([1/len(tickers)] * len(tickers))
    ret_portfolio = retornos_diarios.dot(pesos_eq).ffill().bfill().fillna(0)
    valor_portfolio = (1 + ret_portfolio).cumprod().replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(1) * 100

    ret_total_pf = ((valor_portfolio.iloc[-1] / 100) - 1) * 100
    ret_anual_pf = ((valor_portfolio.iloc[-1] / 100) ** (1/10) - 1) * 100
    vol_anual_pf = ret_portfolio.std() * np.sqrt(252) * 100
    sharpe_pf = (ret_anual_pf/100 - rf) / (vol_anual_pf/100)

    display(HTML(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 30px;
        margin: 30px auto;
        max-width: 700px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    ">
        <h2 style="color: white; text-align: center; margin-bottom: 25px;">
            📈 Portfolio Equiponderado (10 años)
        </h2>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
            <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 10px;">
                <div style="color: #666; font-size: 14px; margin-bottom: 5px;">Rendimiento Total</div>
                <div style="color: {'#006400' if ret_total_pf > 0 else '#8B0000'}; font-size: 32px; font-weight: bold;">
                    {ret_total_pf:+.2f}%
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 10px;">
                <div style="color: #666; font-size: 14px; margin-bottom: 5px;">Rendimiento Anualizado</div>
                <div style="color: {'#006400' if ret_anual_pf > 0 else '#8B0000'}; font-size: 32px; font-weight: bold;">
                    {ret_anual_pf:+.2f}%
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 10px;">
                <div style="color: #666; font-size: 14px; margin-bottom: 5px;">Volatilidad Anual</div>
                <div style="color: #FF6B6B; font-size: 32px; font-weight: bold;">
                    {vol_anual_pf:.2f}%
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 10px;">
                <div style="color: #666; font-size: 14px; margin-bottom: 5px;">Sharpe Ratio</div>
                <div style="color: #667eea; font-size: 32px; font-weight: bold;">
                    {sharpe_pf:.2f}
                </div>
            </div>
        </div>
        <div style="color: rgba(255,255,255,0.8); text-align: center; margin-top: 20px; font-size: 12px;">
            Ponderación: {', '.join([f'{t} ({1/len(tickers):.1%})' for t in tickers])}
        </div>
    </div>
    """))


# =============================================================================
# Cell 26: Correlation heatmap
# =============================================================================

def generar_heatmap_correlaciones():
    activos_portfolio = state.activos_portfolio
    if not activos_portfolio:
        print("❌ No hay activos en el portfolio.")
        return

    print("⏳ Calculando matriz de correlaciones...")

    tickers = [a['ticker'] for a in activos_portfolio]
    precios = descargar_precios_historicos(tickers, periodo='10y')

    if precios.empty:
        print("❌ No se pudieron descargar los datos históricos.")
        return

    precios = precios.ffill().bfill()

    retornos_log = np.log(precios / precios.shift(1)).dropna()

    matriz_corr = retornos_log.corr()
    if matriz_corr.empty:
        print("❌ Correlaciones no disponibles.")
        return

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0b1221')
    ax.set_facecolor('#111827')

    im = ax.imshow(matriz_corr.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(range(len(matriz_corr.columns)))
    ax.set_yticks(range(len(matriz_corr.index)))
    ax.set_xticklabels(matriz_corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(matriz_corr.index)

    for i in range(len(matriz_corr.index)):
        for j in range(len(matriz_corr.columns)):
            ax.text(j, i, f'{matriz_corr.values[i, j]:.2f}',
                    ha='center', va='center', color='white', fontsize=9)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Correlación", fontsize=10, color="#e5e7eb")
    cbar.ax.tick_params(colors="#e5e7eb", labelsize=9)

    ax.set_xlabel("Activos", fontsize=11, color="#e5e7eb", fontweight='normal')
    ax.set_ylabel("Activos", fontsize=11, color="#e5e7eb", fontweight='normal')
    ax.set_title("Matriz de Correlación de Retornos (10 años)", fontsize=14, color="#e5e7eb",
                fontweight='bold', pad=15)

    ax.tick_params(colors="#e5e7eb", labelsize=9)

    for spine in ax.spines.values():
        spine.set_color('#1f2937')
        spine.set_linewidth(1.5)

    plt.subplots_adjust(left=0.15, right=0.95, top=0.94, bottom=0.15)

    display(fig)
    plt.close()

    corr_promedio = matriz_corr.values[np.triu_indices_from(matriz_corr.values, k=1)].mean()
    corr_max = matriz_corr.values[np.triu_indices_from(matriz_corr.values, k=1)].max()
    corr_min = matriz_corr.values[np.triu_indices_from(matriz_corr.values, k=1)].min()

    mask = np.triu(np.ones_like(matriz_corr, dtype=bool), k=1)
    corr_masked = matriz_corr.where(mask)
    if corr_masked.isnull().all().all():
        print("⚠️ Correlaciones no disponibles (todo NaN).")
        return

    max_corr_idx = np.unravel_index(np.nanargmax(corr_masked.values), corr_masked.shape)
    min_corr_idx = np.unravel_index(np.nanargmin(corr_masked.values), corr_masked.shape)

    par_max = f"{matriz_corr.index[max_corr_idx[0]]} - {matriz_corr.columns[max_corr_idx[1]]}"
    par_min = f"{matriz_corr.index[min_corr_idx[0]]} - {matriz_corr.columns[min_corr_idx[1]]}"

    display(HTML(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 30px;
        margin: 30px auto;
        max-width: 800px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    ">
        <h2 style="color: white; text-align: center; margin-bottom: 25px;">
            📊 Análisis de Correlaciones
        </h2>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
            <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 10px;">
                <div style="color: #666; font-size: 14px; margin-bottom: 5px;">Correlación Promedio</div>
                <div style="color: #667eea; font-size: 32px; font-weight: bold;">
                    {corr_promedio:.3f}
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 10px;">
                <div style="color: #666; font-size: 14px; margin-bottom: 5px;">Rango de Correlación</div>
                <div style="color: #667eea; font-size: 32px; font-weight: bold;">
                    {corr_min:.3f} a {corr_max:.3f}
                </div>
            </div>
        </div>
        <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 10px; margin-top: 20px;">
            <div style="color: #666; font-size: 14px; margin-bottom: 10px;">Mayor Correlación</div>
            <div style="color: #FF0000; font-size: 20px; font-weight: bold;">
                {par_max}: {corr_max:.3f}
            </div>
        </div>
        <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 10px; margin-top: 20px;">
            <div style="color: #666; font-size: 14px; margin-bottom: 10px;">Menor Correlación (Mejor Diversificación)</div>
            <div style="color: #0000FF; font-size: 20px; font-weight: bold;">
                {par_min}: {corr_min:.3f}
            </div>
        </div>
        <div style="color: rgba(255,255,255,0.9); text-align: center; margin-top: 20px; font-size: 13px; line-height: 1.6;">
            💡 <b>Interpretación:</b><br>
            Valores cercanos a <span style="color: #FF6B6B;">+1</span> indican activos que se mueven juntos (menor diversificación)<br>
            Valores cercanos a <span style="color: #4ECDC4;">-1</span> o <span style="color: #4ECDC4;">0</span> indican mayor potencial de diversificación
        </div>
    </div>
    """))


# =============================================================================
# Cell 30: Underwater drawdown plot
# =============================================================================

def generar_underwater_plot():
    activos_portfolio = state.activos_portfolio
    if not activos_portfolio:
        print("❌ No hay activos en el portfolio.")
        return

    print("⏳ Calculando drawdowns históricos...")

    tickers = [a['ticker'] for a in activos_portfolio]
    precios = descargar_precios_historicos(tickers, periodo='10y')

    if precios.empty:
        print("❌ No se pudieron descargar los datos históricos.")
        return

    precios = precios.ffill().bfill()

    drawdowns = pd.DataFrame(index=precios.index)

    for ticker in precios.columns:
        max_acumulado = precios[ticker].cummax()
        drawdowns[ticker] = (precios[ticker] / max_acumulado - 1) * 100

    retornos_diarios = precios.pct_change().dropna()
    pesos_eq = np.array([1/len(tickers)] * len(tickers))
    ret_portfolio = retornos_diarios.dot(pesos_eq).ffill().bfill().fillna(0)
    valor_portfolio = (1 + ret_portfolio).cumprod().replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(1)
    valor_portfolio = pd.concat([pd.Series([1], index=[precios.index[0]]), valor_portfolio])

    max_acum_pf = valor_portfolio.cummax()
    drawdown_portfolio = (valor_portfolio / max_acum_pf - 1) * 100

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#0b1221')
    ax.set_facecolor('#111827')

    for ticker in drawdowns.columns:
        ax.fill_between(drawdowns.index, drawdowns[ticker], 0, alpha=0.4, label=ticker)
        ax.plot(drawdowns.index, drawdowns[ticker], linewidth=1.5, alpha=0.9)

    ax.fill_between(drawdown_portfolio.index, drawdown_portfolio.values, 0,
                    color='gray', alpha=0.5, label='Portfolio Equiponderado')
    ax.plot(drawdown_portfolio.index, drawdown_portfolio.values,
           color='white', linewidth=2.5, alpha=0.9)

    ax.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.7)

    ax.set_xlabel("Fecha", fontsize=11, color="#e5e7eb", fontweight='normal')
    ax.set_ylabel("Drawdown (%)", fontsize=11, color="#e5e7eb", fontweight='normal')
    ax.set_title("🌊 Profundidad de las Caídas (Drawdown Histórico - 10 años)",
                fontsize=14, color="#e5e7eb", fontweight='bold', pad=15)

    legend = ax.legend(loc='lower right', fontsize=9, framealpha=0.9,
                      edgecolor='#38bdf8', facecolor='#1f2937')
    for text in legend.get_texts():
        text.set_color('#e5e7eb')

    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, color='#94a3b8')
    ax.tick_params(colors="#e5e7eb", labelsize=9)

    for spine in ax.spines.values():
        spine.set_color('#1f2937')
        spine.set_linewidth(1.5)

    plt.subplots_adjust(left=0.10, right=0.95, top=0.94, bottom=0.10)

    display(fig)
    plt.close()

    max_dd_por_activo = drawdowns.min()
    max_dd_portfolio = drawdown_portfolio.min()

    duraciones = {}
    for ticker in drawdowns.columns:
        dd = drawdowns[ticker]
        en_drawdown = dd < -0.1
        if en_drawdown.any():
            periodos = (en_drawdown != en_drawdown.shift()).cumsum()
            duraciones[ticker] = en_drawdown.groupby(periodos).sum().max()
        else:
            duraciones[ticker] = 0

    dd_pf = drawdown_portfolio
    en_drawdown_pf = dd_pf < -0.1
    if en_drawdown_pf.any():
        periodos_pf = (en_drawdown_pf != en_drawdown_pf.shift()).cumsum()
        duracion_pf = en_drawdown_pf.groupby(periodos_pf).sum().max()
    else:
        duracion_pf = 0

    html_stats = "<div style='margin-top:30px;'><h3 style='text-align:center; color:#2C3E50; margin-bottom:20px;'>📉 Estadísticas de Drawdown</h3>"
    html_stats += "<table style='width:100%; border-collapse:collapse; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>"
    html_stats += "<thead><tr style='background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:white;'>"
    html_stats += "<th style='padding:15px; text-align:left;'>Activo</th>"
    html_stats += "<th style='padding:15px; text-align:center;'>Peor Drawdown</th>"
    html_stats += "<th style='padding:15px; text-align:center;'>Mayor Duración (días)</th>"
    html_stats += "</tr></thead><tbody>"

    for i, ticker in enumerate(drawdowns.columns):
        bg_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        html_stats += f"<tr style='background:{bg_color}; border-bottom:1px solid #dee2e6;'>"
        html_stats += f"<td style='padding:12px; font-weight:bold; color:#2C3E50;'>{ticker}</td>"
        html_stats += f"<td style='padding:12px; text-align:center; color:#8B0000; font-weight:bold;'>{max_dd_por_activo[ticker]:.2f}%</td>"
        html_stats += f"<td style='padding:12px; text-align:center; color:#FF6B6B;'>{int(duraciones[ticker])} días</td>"
        html_stats += "</tr>"

    html_stats += "<tr style='background:#e3f2fd; border-top:3px solid #667eea; font-weight:bold;'>"
    html_stats += "<td style='padding:12px; color:#1565c0;'>📊 Portfolio Equiponderado</td>"
    html_stats += f"<td style='padding:12px; text-align:center; color:#8B0000; font-weight:bold;'>{max_dd_portfolio:.2f}%</td>"
    html_stats += f"<td style='padding:12px; text-align:center; color:#FF6B6B;'>{int(duracion_pf)} días</td>"
    html_stats += "</tr>"

    html_stats += "</tbody></table></div>"

    display(HTML(html_stats))

    mejoria_dd = ((max_dd_portfolio - max_dd_por_activo.mean()) / abs(max_dd_por_activo.mean())) * 100

    display(HTML(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 30px;
        margin: 30px auto;
        max-width: 800px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    ">
        <h2 style="color: white; text-align: center; margin-bottom: 25px;">
            🛡️ Efecto Diversificación
        </h2>
        <div style="background: rgba(255,255,255,0.95); padding: 25px; border-radius: 10px;">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="color: #666; font-size: 14px; margin-bottom: 10px;">
                    Drawdown Promedio Individual vs Portfolio
                </div>
                <div style="display: flex; justify-content: center; gap: 30px; align-items: center;">
                    <div>
                        <div style="color: #999; font-size: 12px;">Promedio Individual</div>
                        <div style="color: #8B0000; font-size: 28px; font-weight: bold;">
                            {max_dd_por_activo.mean():.2f}%
                        </div>
                    </div>
                    <div style="font-size: 32px; color: #667eea;">→</div>
                    <div>
                        <div style="color: #999; font-size: 12px;">Portfolio Diversificado</div>
                        <div style="color: #006400; font-size: 28px; font-weight: bold;">
                            {max_dd_portfolio:.2f}%
                        </div>
                    </div>
                </div>
            </div>
            <div style="text-align: center; padding: 15px; background: #f0f4ff; border-radius: 8px;">
                <div style="color: #667eea; font-size: 16px; font-weight: bold;">
                    {'✅ Mejora de ' + f'{abs(mejoria_dd):.1f}%' if mejoria_dd > 0 else '⚠️ El portfolio no mejora el drawdown promedio'}
                </div>
                <div style="color: #666; font-size: 13px; margin-top: 8px;">
                    La diversificación {'reduce' if mejoria_dd > 0 else 'no reduce significativamente'} la profundidad de las caídas
                </div>
            </div>
        </div>
        <div style="color: rgba(255,255,255,0.9); text-align: center; margin-top: 20px; font-size: 13px; line-height: 1.6;">
            💡 <b>Interpretación:</b><br>
            El Underwater Plot muestra cuánto pierdes desde el máximo histórico en cada momento.<br>
            Un portfolio diversificado suele tener drawdowns menos pronunciados que activos individuales.
        </div>
    </div>
    """))


# =============================================================================
# Cell 32: Marginal Contribution to Risk (MCR) analysis
# =============================================================================

def analizar_contribucion_marginal_riesgo():
    activos_portfolio = state.activos_portfolio
    if not activos_portfolio:
        print("❌ No hay activos en el portfolio.")
        return

    cov = state.cov
    pesos_sharpe = state.pesos_sharpe
    pesos_minvar = state.pesos_minvar
    pesos_cvar = state.pesos_cvar
    tickers = state.tickers

    if cov is None or pesos_sharpe is None or pesos_minvar is None or pesos_cvar is None or tickers is None:
        print("❌ Datos insuficientes para calcular MCR.")
        return
    if hasattr(cov, "empty") and cov.empty:
        print("❌ Matriz de covarianza vacía.")
        return
    if len(tickers) == 0:
        print("❌ No hay tickers válidos.")
        return
    if len(pesos_sharpe) != len(tickers) or len(pesos_minvar) != len(tickers) or len(pesos_cvar) != len(tickers):
        print("❌ Dimensiones incompatibles entre pesos y tickers.")
        return

    print("⏳ Calculando Contribución Marginal al Riesgo (MCR)...")

    def calcular_mcr_y_contribucion(pesos, cov_matrix):
        """
        Calcula la Contribución Marginal al Riesgo (MCR) y la contribución porcentual
        de cada activo al riesgo total del portfolio.

        MCR_i = (Σ · w)_i / √(w^T · Σ · w)
        Contribución_i = w_i × MCR_i
        Contribución_%_i = Contribución_i / Volatilidad_Portfolio × 100
        """
        cov_matrix = np.asarray(cov_matrix, dtype=float)
        pesos = np.asarray(pesos, dtype=float)
        if cov_matrix.size == 0 or pesos.size == 0:
            return np.array([]), np.array([]), np.array([]), float("nan")
        if cov_matrix.shape[0] != cov_matrix.shape[1] or cov_matrix.shape[0] != pesos.shape[0]:
            return np.array([]), np.array([]), np.array([]), float("nan")

        cov_dot_w = np.dot(cov_matrix, pesos)
        vol_portfolio = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))
        if not np.isfinite(vol_portfolio) or vol_portfolio == 0:
            return np.array([]), np.array([]), np.array([]), float("nan")

        mcr = cov_dot_w / vol_portfolio
        contribucion_absoluta = pesos * mcr
        contribucion_porcentual = (contribucion_absoluta / vol_portfolio) * 100
        return mcr, contribucion_absoluta, contribucion_porcentual, vol_portfolio

    mcr_sharpe, contrib_abs_sharpe, contrib_pct_sharpe, vol_sharpe = calcular_mcr_y_contribucion(pesos_sharpe, cov)
    mcr_minvar, contrib_abs_minvar, contrib_pct_minvar, vol_minvar = calcular_mcr_y_contribucion(pesos_minvar, cov)
    mcr_cvar, contrib_abs_cvar, contrib_pct_cvar, vol_cvar = calcular_mcr_y_contribucion(pesos_cvar, cov)

    if len(mcr_sharpe) == 0 or len(mcr_minvar) == 0 or len(mcr_cvar) == 0:
        print("❌ No se pudo calcular MCR con los datos actuales.")
        return

    df_mcr = pd.DataFrame({
        'Activo': tickers,
        'Peso Sharpe': pesos_sharpe * 100,
        'MCR Sharpe': mcr_sharpe * 100,
        'Contrib% Sharpe': contrib_pct_sharpe,
        'Peso MinVar': pesos_minvar * 100,
        'MCR MinVar': mcr_minvar * 100,
        'Contrib% MinVar': contrib_pct_minvar,
        'Peso CVaR': pesos_cvar * 100,
        'MCR CVaR': mcr_cvar * 100,
        'Contrib% CVaR': contrib_pct_cvar
    })

    np.random.seed(42)
    colores_rgb = [(np.random.randint(50,255)/255, np.random.randint(50,255)/255, np.random.randint(50,255)/255)
                   for _ in range(len(tickers))]

    portfolios = ['Máximo Sharpe', 'Mínima Varianza', 'Mínimo CVaR']
    contrib_data = [contrib_pct_sharpe, contrib_pct_minvar, contrib_pct_cvar]
    vols = [vol_sharpe * 100, vol_minvar * 100, vol_cvar * 100]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#0b1221')
    ax.set_facecolor('#111827')

    bottom = np.zeros(3)
    for i, ticker in enumerate(tickers):
        contribuciones = [contrib_data[j][i] for j in range(3)]
        ax.bar(portfolios, contribuciones, bottom=bottom, label=ticker,
              color=colores_rgb[i], alpha=0.9, edgecolor='white', linewidth=0.5)

        for j, (contrib, bot) in enumerate(zip(contribuciones, bottom)):
            if contrib > 3:
                ax.text(j, bot + contrib/2, f'{contrib:.1f}%',
                       ha='center', va='center', color='white', fontsize=9, fontweight='bold')

        bottom += contribuciones

    for i, (portfolio, vol) in enumerate(zip(portfolios, vols)):
        ax.text(i, 103, f'Vol: {vol:.2f}%', ha='center', fontsize=10,
               color='#667eea', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#667eea', linewidth=1.5, alpha=0.9))

    ax.set_xlabel("Portfolio Óptimo", fontsize=11, color="#e5e7eb", fontweight='normal')
    ax.set_ylabel("Contribución al Riesgo Total (%)", fontsize=11, color="#e5e7eb", fontweight='normal')
    ax.set_title("📊 Contribución Marginal al Riesgo por Activo\nCada barra suma 100% del riesgo total del portfolio",
                fontsize=13, color="#e5e7eb", fontweight='bold', pad=15)

    ax.set_ylim(0, 110)

    legend = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9,
                      framealpha=0.9, edgecolor='#38bdf8', facecolor='#1f2937')
    for text in legend.get_texts():
        text.set_color('#e5e7eb')

    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, color='#94a3b8', axis='y')
    ax.tick_params(colors="#e5e7eb", labelsize=9)

    for spine in ax.spines.values():
        spine.set_color('#1f2937')
        spine.set_linewidth(1.5)

    plt.subplots_adjust(left=0.10, right=0.82, top=0.92, bottom=0.10)

    display(fig)
    plt.close()

    html_table = "<div style='margin-top:30px;'><h3 style='text-align:center; color:#2C3E50; margin-bottom:20px;'>📈 Análisis Detallado de Contribución al Riesgo</h3>"

    html_table += "<h4 style='color:#667eea; margin-top:30px;'>⭐ Portfolio Máximo Sharpe</h4>"
    html_table += "<table style='width:100%; border-collapse:collapse; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom:30px;'>"
    html_table += "<thead><tr style='background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:white;'>"
    html_table += "<th style='padding:12px; text-align:left;'>Activo</th>"
    html_table += "<th style='padding:12px; text-align:center;'>Peso (%)</th>"
    html_table += "<th style='padding:12px; text-align:center;'>MCR (%)</th>"
    html_table += "<th style='padding:12px; text-align:center;'>Contribución al Riesgo (%)</th>"
    html_table += "<th style='padding:12px; text-align:center;'>Ratio Riesgo/Peso</th>"
    html_table += "</tr></thead><tbody>"

    for i, row in df_mcr.iterrows():
        bg_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        ratio = row['Contrib% Sharpe'] / row['Peso Sharpe'] if row['Peso Sharpe'] > 0.01 else 0
        color_ratio = '#006400' if ratio < 1.2 else ('#FF8C00' if ratio < 1.5 else '#8B0000')

        html_table += f"<tr style='background:{bg_color}; border-bottom:1px solid #dee2e6;'>"
        html_table += f"<td style='padding:10px; font-weight:bold; color:#2C3E50;'>{row['Activo']}</td>"
        html_table += f"<td style='padding:10px; text-align:center; color:#667eea;'>{row['Peso Sharpe']:.2f}%</td>"
        html_table += f"<td style='padding:10px; text-align:center; color:#FF6B6B;'>{row['MCR Sharpe']:.2f}%</td>"
        html_table += f"<td style='padding:10px; text-align:center; color:#8B0000; font-weight:bold;'>{row['Contrib% Sharpe']:.2f}%</td>"
        html_table += f"<td style='padding:10px; text-align:center; color:{color_ratio}; font-weight:bold;'>{ratio:.2f}x</td>"
        html_table += "</tr>"

    html_table += "</tbody></table>"

    html_table += "<h4 style='color:#00bcd4; margin-top:30px;'>💎 Portfolio Mínima Varianza</h4>"
    html_table += "<table style='width:100%; border-collapse:collapse; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom:30px;'>"
    html_table += "<thead><tr style='background:linear-gradient(135deg, #00bcd4 0%, #0097a7 100%); color:white;'>"
    html_table += "<th style='padding:12px; text-align:left;'>Activo</th>"
    html_table += "<th style='padding:12px; text-align:center;'>Peso (%)</th>"
    html_table += "<th style='padding:12px; text-align:center;'>MCR (%)</th>"
    html_table += "<th style='padding:12px; text-align:center;'>Contribución al Riesgo (%)</th>"
    html_table += "<th style='padding:12px; text-align:center;'>Ratio Riesgo/Peso</th>"
    html_table += "</tr></thead><tbody>"

    for i, row in df_mcr.iterrows():
        bg_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        ratio = row['Contrib% MinVar'] / row['Peso MinVar'] if row['Peso MinVar'] > 0.01 else 0
        color_ratio = '#006400' if ratio < 1.2 else ('#FF8C00' if ratio < 1.5 else '#8B0000')

        html_table += f"<tr style='background:{bg_color}; border-bottom:1px solid #dee2e6;'>"
        html_table += f"<td style='padding:10px; font-weight:bold; color:#2C3E50;'>{row['Activo']}</td>"
        html_table += f"<td style='padding:10px; text-align:center; color:#00bcd4;'>{row['Peso MinVar']:.2f}%</td>"
        html_table += f"<td style='padding:10px; text-align:center; color:#FF6B6B;'>{row['MCR MinVar']:.2f}%</td>"
        html_table += f"<td style='padding:10px; text-align:center; color:#8B0000; font-weight:bold;'>{row['Contrib% MinVar']:.2f}%</td>"
        html_table += f"<td style='padding:10px; text-align:center; color:{color_ratio}; font-weight:bold;'>{ratio:.2f}x</td>"
        html_table += "</tr>"

    html_table += "</tbody></table>"

    html_table += "<h4 style='color:#9c27b0; margin-top:30px;'>🛡️ Portfolio Mínimo CVaR</h4>"
    html_table += "<table style='width:100%; border-collapse:collapse; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom:30px;'>"
    html_table += "<thead><tr style='background:linear-gradient(135deg, #9c27b0 0%, #7b1fa2 100%); color:white;'>"
    html_table += "<th style='padding:12px; text-align:left;'>Activo</th>"
    html_table += "<th style='padding:12px; text-align:center;'>Peso (%)</th>"
    html_table += "<th style='padding:12px; text-align:center;'>MCR (%)</th>"
    html_table += "<th style='padding:12px; text-align:center;'>Contribución al Riesgo (%)</th>"
    html_table += "<th style='padding:12px; text-align:center;'>Ratio Riesgo/Peso</th>"
    html_table += "</tr></thead><tbody>"

    for i, row in df_mcr.iterrows():
        bg_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        ratio = row['Contrib% CVaR'] / row['Peso CVaR'] if row['Peso CVaR'] > 0.01 else 0
        color_ratio = '#006400' if ratio < 1.2 else ('#FF8C00' if ratio < 1.5 else '#8B0000')

        html_table += f"<tr style='background:{bg_color}; border-bottom:1px solid #dee2e6;'>"
        html_table += f"<td style='padding:10px; font-weight:bold; color:#2C3E50;'>{row['Activo']}</td>"
        html_table += f"<td style='padding:10px; text-align:center; color:#9c27b0;'>{row['Peso CVaR']:.2f}%</td>"
        html_table += f"<td style='padding:10px; text-align:center; color:#FF6B6B;'>{row['MCR CVaR']:.2f}%</td>"
        html_table += f"<td style='padding:10px; text-align:center; color:#8B0000; font-weight:bold;'>{row['Contrib% CVaR']:.2f}%</td>"
        html_table += f"<td style='padding:10px; text-align:center; color:{color_ratio}; font-weight:bold;'>{ratio:.2f}x</td>"
        html_table += "</tr>"

    html_table += "</tbody></table></div>"

    display(HTML(html_table))

    display(HTML("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 30px;
        margin: 30px auto;
        max-width: 900px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    ">
        <h2 style="color: white; text-align: center; margin-bottom: 25px;">
            🎯 Interpretación del Análisis MCR
        </h2>
        <div style="background: rgba(255,255,255,0.95); padding: 25px; border-radius: 10px;">
            <h4 style="color: #667eea; margin-top: 0;">📌 ¿Qué es la Contribución Marginal al Riesgo (MCR)?</h4>
            <p style="color: #333; line-height: 1.8; margin: 15px 0;">
                El <b>MCR</b> mide cuánto riesgo adicional aporta cada activo al portfolio total,
                considerando las correlaciones entre activos. No es lo mismo que la volatilidad individual.
            </p>

            <h4 style="color: #667eea; margin-top: 20px;">📊 Ratio Riesgo/Peso</h4>
            <ul style="color: #333; line-height: 1.8;">
                <li><b style="color: #006400;">Ratio < 1.2x:</b> Activo bien diversificado, aporta menos riesgo que su peso</li>
                <li><b style="color: #FF8C00;">Ratio 1.2x - 1.5x:</b> Activo con riesgo proporcional, requiere monitoreo</li>
                <li><b style="color: #8B0000;">Ratio > 1.5x:</b> Activo concentra riesgo, considerar reducir exposición</li>
            </ul>

            <h4 style="color: #667eea; margin-top: 20px;">💡 Insights Clave</h4>
            <p style="color: #333; line-height: 1.8; margin: 15px 0;">
                • Un activo con <b>peso pequeño</b> pero <b>alta contribución</b> puede desestabilizar el portfolio<br>
                • El portfolio de <b>Mínima Varianza</b> busca que todos los activos tengan MCR similares<br>
                • El portfolio de <b>Máximo Sharpe</b> tolera mayor concentración de riesgo si mejora el retorno<br>
                • El portfolio de <b>Mínimo CVaR</b> minimiza pérdidas extremas en la cola de distribución
            </p>
        </div>
    </div>
    """))

    print("\n✅ Análisis de Contribución Marginal al Riesgo completado.")


# =============================================================================
# Cell 34: Volatility vs CVaR risk chart
# =============================================================================

def generar_grafico_riesgo():
    activos_portfolio = state.activos_portfolio
    if not activos_portfolio:
        print("❌ No hay activos en el portfolio.")
        return

    cov = state.cov
    pesos_sharpe = state.pesos_sharpe
    pesos_minvar = state.pesos_minvar
    pesos_cvar = state.pesos_cvar
    ret_esp = state.ret_esp
    tickers = state.tickers

    if cov is None or pesos_sharpe is None or pesos_minvar is None or pesos_cvar is None or ret_esp is None or tickers is None:
        print("❌ Error: Debes ejecutar primero la celda de 'FRONTERA EFICIENTE' para calcular los portfolios óptimos.")
        return

    rf = obtener_tasa_libre_riesgo()

    r_s, v_s, s_s = metricas_portfolio(pesos_sharpe, ret_esp, cov, rf)
    r_m, v_m, s_m = metricas_portfolio(pesos_minvar, ret_esp, cov, rf)
    r_c, v_c, s_c = metricas_portfolio(pesos_cvar, ret_esp, cov, rf)

    precios = descargar_precios_historicos(tickers, periodo='10y')
    retornos_diarios = precios.pct_change().dropna()

    cvar_s = calcular_cvar_historico(pesos_sharpe, retornos_diarios, 0.95)
    cvar_m = calcular_cvar_historico(pesos_minvar, retornos_diarios, 0.95)
    cvar_c = calcular_cvar_historico(pesos_cvar, retornos_diarios, 0.95)

    pesos_eq = np.array([1/len(tickers)] * len(tickers))
    r_eq, v_eq, s_eq = metricas_portfolio(pesos_eq, ret_esp, cov, rf)
    cvar_eq = calcular_cvar_historico(pesos_eq, retornos_diarios, 0.95)

    portfolios = ['Mínima Varianza', 'Mínimo CVaR', 'Máximo Sharpe', 'Equiponderado']
    volatilidades = [v_m * 100, v_c * 100, v_s * 100, v_eq * 100]
    cvars = [cvar_m * 100, cvar_c * 100, cvar_s * 100, cvar_eq * 100]

    colores = ['#00C853', '#9C27B0', '#FFD700', '#757575']

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#0b1221')
    ax.set_facecolor('#111827')

    for i, (vol, cvar, portfolio, color) in enumerate(zip(volatilidades, cvars, portfolios, colores)):
        ax.scatter(vol, cvar, color=color, s=400, edgecolor='white', linewidth=2,
                  label=portfolio, zorder=5, alpha=0.9)
        ax.text(vol, cvar + (max(cvars) - min(cvars)) * 0.05, portfolio,
               ha='center', fontsize=10, color='#e5e7eb', fontweight='bold')

    x_range = max(volatilidades) - min(volatilidades)
    y_range = max(cvars) - min(cvars)
    ax.set_xlim(min(volatilidades) - x_range * 0.15, max(volatilidades) + x_range * 0.15)
    ax.set_ylim(min(cvars) - y_range * 0.15, max(cvars) + y_range * 0.15)

    ax.set_xlabel("📊 Volatilidad Anualizada (%)", fontsize=11, color="#e5e7eb", fontweight='normal')
    ax.set_ylabel("⚠️ CVaR al 95% (%)", fontsize=11, color="#e5e7eb", fontweight='normal')
    ax.set_title("🎯 Perfil de Riesgo de Portfolios Óptimos\nVolatilidad vs CVaR: Relación entre métricas de riesgo",
                fontsize=13, color="#e5e7eb", fontweight='bold', pad=15)

    legend = ax.legend(loc='best', fontsize=9, framealpha=0.9,
                      edgecolor='#38bdf8', facecolor='#1f2937')
    for text in legend.get_texts():
        text.set_color('#e5e7eb')

    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, color='#94a3b8')
    ax.tick_params(colors="#e5e7eb", labelsize=9)

    for spine in ax.spines.values():
        spine.set_color('#1f2937')
        spine.set_linewidth(1.5)

    plt.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.10)

    display(fig)
    plt.close()

    html_resumen = """
    <div style='margin-top:30px;'>
        <h3 style='text-align:center; color:#2C3E50; margin-bottom:20px;'>📋 Resumen Comparativo de Riesgo-Retorno</h3>
        <table style='width:100%; border-collapse:collapse; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <thead>
                <tr style='background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:white;'>
                    <th style='padding:15px; text-align:left;'>Portfolio</th>
                    <th style='padding:15px; text-align:center;'>Retorno Anual (%)</th>
                    <th style='padding:15px; text-align:center;'>Volatilidad (%)</th>
                    <th style='padding:15px; text-align:center;'>CVaR 95% (%)</th>
                    <th style='padding:15px; text-align:center;'>Sharpe Ratio</th>
                    <th style='padding:15px; text-align:center;'>Nivel de Riesgo</th>
                </tr>
            </thead>
            <tbody>
    """

    datos = [
        ('💎 Mínima Varianza', r_m * 100, v_m * 100, cvar_m * 100, s_m, '#00C853'),
        ('🛡️ Mínimo CVaR', r_c * 100, v_c * 100, cvar_c * 100, s_c, '#9C27B0'),
        ('⭐ Máximo Sharpe', r_s * 100, v_s * 100, cvar_s * 100, s_s, '#FFD700'),
        ('📊 Equiponderado', r_eq * 100, v_eq * 100, cvar_eq * 100, s_eq, '#757575')
    ]

    for i, (nombre, ret, vol, cvar, sharpe, color) in enumerate(datos):
        if vol < max(volatilidades) * 0.4:
            nivel = '🟢 Bajo'
        elif vol < max(volatilidades) * 0.7:
            nivel = '🟡 Medio'
        else:
            nivel = '🔴 Alto'

        bg_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'

        html_resumen += f"""
        <tr style='background:{bg_color}; border-bottom:1px solid #dee2e6;'>
            <td style='padding:12px; font-weight:bold; color:{color};'>{nombre}</td>
            <td style='padding:12px; text-align:center; color:#006400; font-weight:bold;'>{ret:.2f}%</td>
            <td style='padding:12px; text-align:center; color:#FF6B6B;'>{vol:.2f}%</td>
            <td style='padding:12px; text-align:center; color:#8B0000;'>{cvar:.2f}%</td>
            <td style='padding:12px; text-align:center; color:#667eea; font-weight:bold;'>{sharpe:.2f}</td>
            <td style='padding:12px; text-align:center; font-weight:bold;'>{nivel}</td>
        </tr>
        """

    html_resumen += """
            </tbody>
        </table>
    </div>
    """

    display(HTML(html_resumen))

    display(HTML("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 30px;
        margin: 30px auto;
        max-width: 900px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    ">
        <h2 style="color: white; text-align: center; margin-bottom: 25px;">
            🎯 Guía de Selección de Portfolio
        </h2>
        <div style="background: rgba(255,255,255,0.95); padding: 25px; border-radius: 10px;">
            <div style="margin-bottom: 20px; padding: 15px; background: #e8f5e9; border-left: 5px solid #00C853; border-radius: 5px;">
                <h4 style="color: #00C853; margin-top: 0;">💎 Mínima Varianza</h4>
                <p style="color: #333; margin: 5px 0;">
                    <b>Para:</b> Inversores conservadores que priorizan estabilidad<br>
                    <b>Ventaja:</b> Menor volatilidad día a día<br>
                    <b>Desventaja:</b> Puede sacrificar retorno potencial
                </p>
            </div>

            <div style="margin-bottom: 20px; padding: 15px; background: #f3e5f5; border-left: 5px solid #9C27B0; border-radius: 5px;">
                <h4 style="color: #9C27B0; margin-top: 0;">🛡️ Mínimo CVaR</h4>
                <p style="color: #333; margin: 5px 0;">
                    <b>Para:</b> Inversores que temen pérdidas extremas<br>
                    <b>Ventaja:</b> Protección ante crisis y eventos de cola<br>
                    <b>Desventaja:</b> Puede ser muy conservador en mercados alcistas
                </p>
            </div>

            <div style="margin-bottom: 20px; padding: 15px; background: #fff8e1; border-left: 5px solid #FFD700; border-radius: 5px;">
                <h4 style="color: #FF8F00; margin-top: 0;">⭐ Máximo Sharpe</h4>
                <p style="color: #333; margin: 5px 0;">
                    <b>Para:</b> Inversores que buscan el mejor balance riesgo-retorno<br>
                    <b>Ventaja:</b> Maximiza retorno ajustado por riesgo<br>
                    <b>Desventaja:</b> Puede tener volatilidad moderada-alta
                </p>
            </div>

            <div style="padding: 15px; background: #f5f5f5; border-left: 5px solid #757575; border-radius: 5px;">
                <h4 style="color: #424242; margin-top: 0;">📊 Equiponderado</h4>
                <p style="color: #333; margin: 5px 0;">
                    <b>Para:</b> Referencia simple y diversificada<br>
                    <b>Ventaja:</b> Fácil de implementar y rebalancear<br>
                    <b>Desventaja:</b> No optimiza ninguna métrica específica
                </p>
            </div>
        </div>

        <div style="color: rgba(255,255,255,0.95); text-align: center; margin-top: 25px; font-size: 14px; line-height: 1.8;">
            <b>📌 Recomendación:</b><br>
            El portfolio <b>Máximo Sharpe</b> suele ser el más equilibrado para la mayoría de inversores,<br>
            pero la decisión final debe alinearse con tu tolerancia al riesgo y horizonte temporal.
        </div>
    </div>
    """))


# =============================================================================
# Cell 36: Capital allocation / portfolio construction
# =============================================================================

def armar_cartera(capital_disponible: Optional[float] = None):
    if capital_disponible is None:
        capital_disponible = float(input("💰 Ingrese su capital disponible para invertir (USD): "))
    else:
        capital_disponible = float(capital_disponible)

    pesos_sharpe = state.pesos_sharpe
    pesos_minvar = state.pesos_minvar
    pesos_cvar = state.pesos_cvar
    tickers = state.tickers

    if pesos_sharpe is None or pesos_minvar is None or pesos_cvar is None or tickers is None:
        print("❌ Error: Primero debes ejecutar las celdas anteriores para calcular los portfolios óptimos.")
        raise RuntimeError("Portfolios óptimos no calculados aún.")

    pesos_eq = np.array([1/len(tickers)] * len(tickers))

    portfolios_data = {
        '💎 Mínima Varianza': pesos_minvar,
        '🛡️ Mínimo CVaR': pesos_cvar,
        '⭐ Máximo Sharpe': pesos_sharpe,
        '📊 Equiponderado': pesos_eq
    }

    resultados = {}

    for nombre_portfolio, pesos in portfolios_data.items():
        print(f"\n{'='*80}")
        print(f"{nombre_portfolio}")
        print(f"{'='*80}")

        distribuciones = []
        for ticker, peso in zip(tickers, pesos):
            monto = capital_disponible * peso
            distribuciones.append({
                'Activo': ticker,
                'Peso (%)': peso * 100,
                'Monto (USD)': monto
            })

        df_dist = pd.DataFrame(distribuciones)
        df_dist = df_dist[df_dist['Peso (%)'] > 0.01]
        df_dist = df_dist.sort_values('Monto (USD)', ascending=False)
        resultados[nombre_portfolio] = df_dist

        print(f"\n📊 Distribución de Capital: ${capital_disponible:,.2f} USD\n")
        for _, row in df_dist.iterrows():
            print(f"  {row['Activo']:8} → {row['Peso (%)']:6.2f}%  =  ${row['Monto (USD)']:>12,.2f} USD")

        total_asignado = df_dist['Monto (USD)'].sum()
        print(f"\n  {'TOTAL':8}    {100.0:6.2f}%  =  ${total_asignado:>12,.2f} USD")

        html_table = f"""
        <div style='margin:20px 0;'>
            <h3 style='color:#2C3E50; margin-bottom:15px;'>{nombre_portfolio}</h3>
            <table style='width:100%; border-collapse:collapse; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <thead>
                    <tr style='background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:white;'>
                        <th style='padding:12px; text-align:left;'>Activo</th>
                        <th style='padding:12px; text-align:center;'>Peso (%)</th>
                        <th style='padding:12px; text-align:right;'>Monto (USD)</th>
                    </tr>
                </thead>
                <tbody>
        """

        for i, row in df_dist.iterrows():
            bg_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
            html_table += f"""
            <tr style='background:{bg_color}; border-bottom:1px solid #dee2e6;'>
                <td style='padding:10px; font-weight:bold; color:#2C3E50;'>{row['Activo']}</td>
                <td style='padding:10px; text-align:center; color:#667eea;'>{row['Peso (%)']:.2f}%</td>
                <td style='padding:10px; text-align:right; color:#006400; font-weight:bold;'>${row['Monto (USD)']:,.2f}</td>
            </tr>
            """

        html_table += f"""
            <tr style='background:#e8f5e9; border-top:2px solid #667eea; font-weight:bold;'>
                <td style='padding:12px; color:#2C3E50;'>TOTAL</td>
                <td style='padding:12px; text-align:center; color:#667eea;'>100.00%</td>
                <td style='padding:12px; text-align:right; color:#006400;'>${total_asignado:,.2f}</td>
            </tr>
        """

        html_table += """
                </tbody>
            </table>
        </div>
        """

        display(HTML(html_table))

    display(HTML(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 30px auto;
        max-width: 900px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        color: white;
        text-align: center;
    ">
        <h2 style="margin-bottom: 15px;">✅ Distribución de Capital Completada</h2>
        <p style="font-size: 18px; margin: 10px 0;">
            Capital Total: <b>${capital_disponible:,.2f} USD</b>
        </p>
        <p style="font-size: 14px; margin-top: 15px; opacity: 0.9;">
            💡 Ahora puedes implementar el portfolio que mejor se ajuste a tu perfil de riesgo
        </p>
    </div>
    """))

    return resultados
