"""Main execution pipeline and portfolio rebalancing.

Migrated faithfully from Cell 19 of the source notebook.
"""

from __future__ import annotations

import contextlib
import io
import traceback
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import state
from ._display import HTML, Markdown, display, suppress_display
from .market_data import obtener_tasa_libre_riesgo
from .ml import DataIngestor, PortfolioPredictor, PortfolioProcessor
from .optimization import AnalysisResult, analyze_portfolio


def _calcular_rebalanceo(assets_list: List[dict], tickers_list: List[str], weights: np.ndarray, prices_last: pd.Series) -> List[dict]:
    rebalanceo = []
    if prices_last is None or prices_last.empty:
        return rebalanceo

    weight_map = {t: float(w) for t, w in zip(tickers_list, weights)}

    holdings = []
    for asset in assets_list:
        qty = asset.get('quantity', 0) or 0
        buy_price = asset.get('buy_price', 0) or 0
        ticker = asset.get('ticker', '').upper()
        if qty > 0 and ticker in prices_last.index:
            current_price = float(prices_last[ticker])
            current_value = qty * current_price
            holdings.append((ticker, qty, buy_price, current_price, current_value))

    if not holdings:
        return rebalanceo

    total_value = sum(h[4] for h in holdings)

    for ticker, qty, buy_price, current_price, current_value in holdings:
        if ticker not in weight_map:
            continue

        target_weight = weight_map[ticker]
        current_weight = current_value / total_value
        target_value = total_value * target_weight
        diff_value = target_value - current_value
        diff_qty = diff_value / current_price
        gain_pct = (current_price / buy_price - 1) * 100 if buy_price > 0 else 0

        rebalanceo.append({
            'ticker': ticker,
            'peso_actual': current_weight,
            'peso_objetivo': target_weight,
            'cantidad_actual': qty,
            'cantidad_objetivo': qty + diff_qty,
            'diferencia_cantidad': diff_qty,
            'diferencia_valor': diff_value,
            'precio_compra': buy_price,
            'precio_actual': current_price,
            'ganancia_perdida_pct': gain_pct
        })

    return rebalanceo


def ejecutar_analisis_completo(tickers_list, activos_dict=None, ejecutar_ml=True):
    display(HTML("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px; padding: 25px; margin: 20px 0; text-align: center;">
        <h1 style="color: white; margin: 0;">🚀 ANÁLISIS COMPLETO DE PORTFOLIO</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0;">Optimización Moderna + Machine Learning</p>
    </div>
    """))

    display(HTML("""
    <div style="background: #e3f2fd; border-left: 5px solid #2196F3; padding: 15px; margin: 15px 0; border-radius: 5px;">
        <h3 style="color: #1976D2; margin: 0 0 10px 0;">📊 FASE 1: Ingesta y Predicción ML (modo silencioso)</h3>
        <p style="color: #555; margin: 0;">Descargando datos, entrenando modelos y generando predicciones...</p>
    </div>
    """))

    predicciones_5y_local = None
    datos_procesados_ml_local = None

    if ejecutar_ml:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                ingestor = DataIngestor(tickers_list)
                raw_data = ingestor.ingest_all()
                processor = PortfolioProcessor(raw_data)
                datos_procesados_ml_local = processor.process_all()

                if datos_procesados_ml_local:
                    predictor = PortfolioPredictor(datos_procesados_ml_local)
                    predicciones_5y_local = predictor.predict_all_long_term()

            state.datos_procesados_ml = datos_procesados_ml_local
            state.predicciones_5y = predicciones_5y_local

            display(HTML("""
            <div style="background: #d4edda; border-left: 5px solid #28a745; padding: 15px; margin: 15px 0; border-radius: 5px;">
                <h4 style="color: #155724; margin: 0;">✅ Predicciones ML obtenidas en modo silencioso</h4>
            </div>
            """))
        except Exception as e:
            display(HTML(f"""
            <div style="background: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 5px;">
                <h4 style="color: #856404; margin: 0;">⚠️ Error en proyección ML (se continuará sin ML)</h4>
                <p style="color: #856404; margin: 10px 0 0 0;">{str(e)}</p>
            </div>
            """))

    display(HTML("""
    <div style="background: #e3f2fd; border-left: 5px solid #2196F3; padding: 15px; margin: 15px 0; border-radius: 5px;">
        <h3 style="color: #1976D2; margin: 0 0 10px 0;">📊 FASE 2: Pipeline μ (DCF → Black-Litterman) + Optimización robusta</h3>
        <p style="color: #555; margin: 0;">μ_DCF sagrado | BL para resto | Σ Ledoit-Wolf + régimen + VIX | Stress test corr=1</p>
    </div>
    """))

    try:
        assets = []
        dcf_upsides = {}
        if activos_dict:
            for ticker_data in activos_dict:
                t = ticker_data.get('ticker', '').upper()
                asset = {
                    'ticker': t,
                    'mode': 'existing' if ticker_data.get('cantidad', 0) > 0 else 'new',
                    'quantity': ticker_data.get('cantidad', 0),
                    'buy_price': ticker_data.get('precio_compra', 0),
                    'buy_date': ticker_data.get('fecha_compra', None)
                }
                assets.append(asset)
                upside = ticker_data.get('upside_esperado', 0) or 0
                if upside > 0:
                    dcf_upsides[t] = upside / 100.0
        else:
            assets = [{'ticker': t, 'mode': 'new'} for t in tickers_list]

        ml_preds_for_bl = predicciones_5y_local or state.predicciones_5y or None

        state.resultado_analisis = analyze_portfolio(
            assets=assets,
            rf_rate=obtener_tasa_libre_riesgo(),
            samples=2000,
            dcf_upsides=dcf_upsides if dcf_upsides else None,
            ml_predictions_5y=ml_preds_for_bl,
            investment_horizon=5,
            sales=[]
        )

        resultado_analisis = state.resultado_analisis
        state.tickers = resultado_analisis.tickers
        state.cov = resultado_analisis.cov_matrix
        state.ret_esp = resultado_analisis.expected_returns
        state.pesos_sharpe = resultado_analisis.weights_sharpe
        state.pesos_minvar = resultado_analisis.weights_minvar
        state.pesos_cvar = resultado_analisis.weights_cvar
        state.mcr_sharpe = resultado_analisis.mcr_sharpe
        state.mcr_minvar = resultado_analisis.mcr_minvar
        state.mcr_cvar = resultado_analisis.mcr_cvar

        tickers = state.tickers
        cov = state.cov
        ret_esp = state.ret_esp
        pesos_sharpe = state.pesos_sharpe
        pesos_minvar = state.pesos_minvar
        pesos_cvar = state.pesos_cvar

        if ret_esp is None or cov is None or (hasattr(ret_esp, "empty") and ret_esp.empty) or (hasattr(cov, "empty") and cov.empty):
            raise ValueError("No hay datos válidos. Verifique los tickers")

        prices_last = resultado_analisis.prices_hist.ffill().iloc[-1] if resultado_analisis.prices_hist is not None and not resultado_analisis.prices_hist.empty else None
        state.rebalanceo_sharpe = _calcular_rebalanceo(assets, tickers, pesos_sharpe, prices_last)
        state.rebalanceo_minvar = _calcular_rebalanceo(assets, tickers, pesos_minvar, prices_last)
        state.rebalanceo_cvar = _calcular_rebalanceo(assets, tickers, pesos_cvar, prices_last)

        df_resumen = pd.DataFrame({
            'Portfolio': ['💎 Máximo Sharpe', '🛡️ Mínima Varianza', '⚠️ Mínimo CVaR'],
            'Retorno Anual': [resultado_analisis.stats_sharpe['ret'],
                             resultado_analisis.stats_minvar['ret'],
                             resultado_analisis.stats_cvar['ret']],
            'Volatilidad': [resultado_analisis.stats_sharpe['vol'],
                           resultado_analisis.stats_minvar['vol'],
                           resultado_analisis.stats_cvar['vol']],
            'Sharpe Ratio': [resultado_analisis.stats_sharpe['sharpe'],
                            resultado_analisis.stats_minvar['sharpe'],
                            resultado_analisis.stats_cvar['sharpe']],
            'CVaR 95%': [resultado_analisis.stats_sharpe['cvar'],
                        resultado_analisis.stats_minvar['cvar'],
                        resultado_analisis.stats_cvar['cvar']]
        })

        display(HTML("""
        <div style="background: #d4edda; border-left: 5px solid #28a745; padding: 15px; margin: 15px 0; border-radius: 5px;">
            <h4 style="color: #155724; margin: 0;">✅ Optimización completada exitosamente</h4>
        </div>
        """))

        display(HTML("<h3 style='color: #2C3E50; margin-top: 30px;'>📋 Resumen de Estrategias Óptimas</h3>"))

        styled_df = df_resumen.style\
            .format({
                'Retorno Anual': '{:.2%}',
                'Volatilidad': '{:.2%}',
                'Sharpe Ratio': '{:.2f}',
                'CVaR 95%': '{:.2%}'
            })\
            .background_gradient(subset=['Sharpe Ratio'], cmap='RdYlGn', vmin=-1, vmax=3)\
            .background_gradient(subset=['Volatilidad'], cmap='RdYlGn_r')\
            .set_properties(**{
                'text-align': 'center',
                'padding': '10px',
                'border': '1px solid #ddd'
            })\
            .set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#667eea'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('padding', '12px')
                ]},
                {'selector': 'td', 'props': [('border', '1px solid #ddd')]},
                {'selector': '', 'props': [
                    ('border-collapse', 'collapse'),
                    ('width', '100%'),
                    ('margin', '20px 0'),
                    ('box-shadow', '0 2px 8px rgba(0,0,0,0.1)')
                ]}
            ])

        display(styled_df)

        display(HTML("<h3 style='color: #2C3E50; margin-top: 40px;'>📊 Composición de Portfolios</h3>"))

        portfolios_pesos = {
            '💎 Sharpe': pesos_sharpe,
            '🛡️ MinVar': pesos_minvar,
            '⚠️ CVaR': pesos_cvar
        }

        for nombre_portfolio, pesos in portfolios_pesos.items():
            df_pesos = pd.DataFrame({
                'Activo': tickers,
                'Peso': pesos * 100
            })
            df_pesos = df_pesos[df_pesos['Peso'] > 0.5].sort_values('Peso', ascending=False)

            styled_pesos = df_pesos.style\
                .format({'Peso': '{:.1f}%'})\
                .background_gradient(subset=['Peso'], cmap='Blues', vmin=0, vmax=50)\
                .set_properties(**{
                    'text-align': 'center',
                    'padding': '8px',
                    'border': '1px solid #ddd'
                })\
                .set_table_styles([
                    {'selector': 'th', 'props': [
                        ('background-color', '#667eea'),
                        ('color', 'white'),
                        ('font-weight', 'bold'),
                        ('text-align', 'center'),
                        ('padding', '10px')
                    ]},
                    {'selector': 'td', 'props': [('border', '1px solid #ddd')]},
                    {'selector': '', 'props': [
                        ('border-collapse', 'collapse'),
                        ('width', '100%'),
                        ('margin', '10px 0'),
                        ('box-shadow', '0 2px 8px rgba(0,0,0,0.1)')
                    ]}
                ])\
                .hide(axis='index')

            display(Markdown(f"**{nombre_portfolio}**"))
            display(styled_pesos)

    except Exception as e:
        display(HTML(f"""
        <div style="background: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; margin: 15px 0; border-radius: 5px;">
            <h4 style="color: #721c24; margin: 0;">❌ Error en análisis de optimización</h4>
            <p style="color: #721c24; margin: 10px 0 0 0;">{str(e)}</p>
        </div>
        """))
        traceback.print_exc()
        return None

    if ejecutar_ml and state.predicciones_5y:
        display(HTML("<h3 style='color: #2C3E50; margin-top: 40px;'>🎯 Proyecciones Estratégicas (5 Años)</h3>"))

        df_5y = pd.DataFrame([
            {
                'Activo': ticker,
                'Precio Actual': pred['current_price'],
                'Base': pred['scenarios']['base']['predicted_price'],
                'Bull': pred['scenarios']['bull']['predicted_price'],
                'Bear': pred['scenarios']['bear']['predicted_price']
            }
            for ticker, pred in state.predicciones_5y.items()
        ])

        styled_5y = df_5y.style\
            .format({
                'Precio Actual': '${:.2f}',
                'Base': '${:.2f}',
                'Bull': '${:.2f}',
                'Bear': '${:.2f}'
            })\
            .background_gradient(subset=['Bull'], cmap='Greens')\
            .background_gradient(subset=['Bear'], cmap='Reds_r')\
            .set_properties(**{'text-align': 'center', 'padding': '10px'})\
            .set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#667eea'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('padding', '12px')
                ]},
                {'selector': '', 'props': [
                    ('border-collapse', 'collapse'),
                    ('width', '100%'),
                    ('margin', '20px 0'),
                    ('box-shadow', '0 2px 8px rgba(0,0,0,0.1)')
                ]}
            ])

        display(styled_5y)

    if state.resultado_analisis and hasattr(state.resultado_analisis, 'mu_sources') and state.resultado_analisis.mu_sources:
        fuentes = []
        for t, src in state.resultado_analisis.mu_sources.items():
            mu_val = state.resultado_analisis.expected_returns.get(t, 0)
            fuentes.append(f"<li><b>{t}</b>: {mu_val*100:.2f}% — {src}</li>")
        display(HTML(f"""
        <div style="background: #e8eaf6; border-left: 5px solid #3f51b5; padding: 15px; margin: 15px 0; border-radius: 5px;">
            <h4 style="color: #283593; margin: 0 0 10px 0;">📌 Retornos esperados (μ) inyectados al optimizador</h4>
            <ul style="color: #333; margin: 0; line-height: 1.8;">{''.join(fuentes)}</ul>
        </div>
        """))

    if state.resultado_analisis and hasattr(state.resultado_analisis, 'stress_test_results') and state.resultado_analisis.stress_test_results:
        stress = state.resultado_analisis.stress_test_results
        stress_html = ""
        for sname, slabel in [('sharpe', '⭐ Máximo Sharpe'), ('minvar', '💎 Mínima Varianza'), ('cvar', '🛡️ Mínimo CVaR')]:
            st = stress.get(sname, {})
            if st:
                div_r = st.get('diversification_ratio', 0)
                inc = st.get('vol_increase_pct', 0)
                color = '#FF5252' if st.get('false_diversification') else '#00C853'
                recs = '<br>'.join(st.get('recommendations', []))
                rec_html = f"<div style='color:#FF5252; font-size:11px; margin-top:3px;'>{recs}</div>" if recs else ""
                stress_html += f"""
                <tr style="border-bottom:1px solid #ddd;">
                    <td style="padding:10px; font-weight:bold;">{slabel}</td>
                    <td style="padding:10px; text-align:center;">{st['normal_vol']*100:.2f}%</td>
                    <td style="padding:10px; text-align:center; color:#FF5252; font-weight:bold;">{st['stress_vol']*100:.2f}%</td>
                    <td style="padding:10px; text-align:center;">+{inc:.1f}%</td>
                    <td style="padding:10px; text-align:center; color:{color}; font-weight:bold;">{div_r:.3f}{rec_html}</td>
                </tr>"""
        display(HTML(f"""
        <div style="margin: 20px 0;">
            <h3 style="color: #c62828;">🔴 Stress Test: Correlaciones = 1</h3>
            <p style="color:#666; font-size:13px;">Evalúa el portfolio bajo el supuesto extremo de que todos los activos colapsan juntos.</p>
            <table style="width:100%; border-collapse:collapse; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <thead>
                    <tr style="background:linear-gradient(135deg, #e53935, #c62828); color:white;">
                        <th style="padding:12px;">Portfolio</th>
                        <th style="padding:12px;">Vol Normal</th>
                        <th style="padding:12px;">Vol Estrés</th>
                        <th style="padding:12px;">Incremento</th>
                        <th style="padding:12px;">Ratio Diversificación</th>
                    </tr>
                </thead>
                <tbody>{stress_html}</tbody>
            </table>
            <p style="color: #666; font-size: 11px; margin-top: 8px;">
                Ratio &lt; 0.85 = diversificación efectiva | &gt; 0.85 = falsa diversificación ⚠️
            </p>
        </div>
        """))

    display(HTML("""
    <div style="background: linear-gradient(135deg, #00c853 0%, #00b248 100%);
                border-radius: 15px; padding: 25px; margin: 30px 0; text-align: center;">
        <h2 style="color: white; margin: 0;">✅ ANÁLISIS COMPLETO FINALIZADO</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0;">
            Pipeline: μ (DCF sagrado → BL) | Σ (Ledoit-Wolf + régimen + VIX) | Stress test corr=1
        </p>
    </div>
    """))

    return state.resultado_analisis


def run_portfolio_analysis(
    tickers_list: Iterable[str],
    activos_dict: Optional[List[Dict[str, Any]]] = None,
    ejecutar_ml: bool = True,
    render_ui: bool = False,
) -> Optional[AnalysisResult]:
    """Programmatic backend entrypoint wrapping ``ejecutar_analisis_completo``.

    Runs the exact same canonical engine without changing formulas, defaults,
    result structures, or global state updates. When ``render_ui=False``,
    IPython display calls are suppressed so headless backends can invoke it cleanly.
    """
    tickers_seq = list(tickers_list)
    if render_ui:
        return ejecutar_analisis_completo(
            tickers_list=tickers_seq,
            activos_dict=activos_dict,
            ejecutar_ml=ejecutar_ml,
        )
    with suppress_display():
        return ejecutar_analisis_completo(
            tickers_list=tickers_seq,
            activos_dict=activos_dict,
            ejecutar_ml=ejecutar_ml,
        )
