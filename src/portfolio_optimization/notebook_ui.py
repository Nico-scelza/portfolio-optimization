"""Interactive Jupyter/Colab notebook UI adapter.

Migrated faithfully from Cells 06, 20, 21, and 28 of the source notebook.
``ipywidgets`` is imported lazily inside UI constructors so headless backends
can import ``portfolio_optimization`` without Jupyter or GUI dependencies.
"""

from __future__ import annotations

from typing import Dict, List

import yfinance as yf

from . import state
from ._display import HTML, clear_output, display
from .pipeline import ejecutar_analisis_completo

# Cell 06 constants (preserved verbatim)
banderas_paises: Dict[str, str] = {
    'United States': '🇺🇸', 'Argentina': '🇦🇷', 'Brazil': '🇧🇷', 'Mexico': '🇲🇽',
    'Chile': '🇨🇱', 'Colombia': '🇨🇴', 'Peru': '🇵🇪', 'United Kingdom': '🇬🇧',
    'Germany': '🇩🇪', 'France': '🇫🇷', 'Spain': '🇪🇸', 'Italy': '🇮🇹',
    'Japan': '🇯🇵', 'China': '🇨🇳', 'South Korea': '🇰🇷', 'Canada': '🇨🇦',
    'Australia': '🇦🇺', 'India': '🇮🇳', 'Netherlands': '🇳🇱', 'Switzerland': '🇨🇭',
    'US': '🇺🇸', 'AR': '🇦🇷', 'BR': '🇧🇷', 'MX': '🇲🇽', 'CL': '🇨🇱',
    'CO': '🇨🇴', 'PE': '🇵🇪', 'GB': '🇬🇧', 'DE': '🇩🇪', 'FR': '🇫🇷',
    'ES': '🇪🇸', 'IT': '🇮🇹', 'JP': '🇯🇵', 'CN': '🇨🇳', 'KR': '🇰🇷',
    'CA': '🇨🇦', 'AU': '🇦🇺', 'IN': '🇮🇳', 'NL': '🇳🇱', 'CH': '🇨🇭',
    'Belgium': '🇧🇪', 'Sweden': '🇸🇪', 'Norway': '🇳🇴', 'Denmark': '🇩🇰',
    'Finland': '🇫🇮', 'Ireland': '🇮🇪', 'Portugal': '🇵🇹', 'Austria': '🇦🇹',
    'Israel': '🇮🇱', 'Singapore': '🇸🇬', 'Hong Kong': '🇭🇰', 'Taiwan': '🇹🇼',
    'Thailand': '🇹🇭', 'Malaysia': '🇲🇾', 'Indonesia': '🇮🇩', 'Philippines': '🇵🇭',
    'New Zealand': '🇳🇿', 'South Africa': '🇿🇦', 'Russia': '🇷🇺', 'Turkey': '🇹🇷'
}

paises_disponibles: List[str] = [
    'Argentina', 'Australia', 'Austria', 'Belgium', 'Brazil', 'Canada', 'Chile',
    'China', 'Colombia', 'Denmark', 'Finland', 'France', 'Germany', 'Hong Kong',
    'India', 'Indonesia', 'Ireland', 'Israel', 'Italy', 'Japan', 'Malaysia',
    'Mexico', 'Netherlands', 'New Zealand', 'Norway', 'Peru', 'Philippines',
    'Portugal', 'Russia', 'Singapore', 'South Africa', 'South Korea', 'Spain',
    'Sweden', 'Switzerland', 'Taiwan', 'Thailand', 'Turkey', 'United Kingdom',
    'United States', 'Uruguay'
]


def obtener_info_activo(ticker: str) -> dict:
    """Validate ticker and retrieve metadata/latest price via yfinance (Cell 06)."""
    try:
        activo = yf.Ticker(ticker)
        info = activo.info
        hist = activo.history(period='5d')
        precio_actual = hist['Close'].iloc[-1] if not hist.empty else None

        nombre = info.get('longName') or info.get('shortName') or ticker
        pais = info.get('country', 'United States')
        emoji_bandera = banderas_paises.get(pais, '🌐')

        return {
            'ticker': ticker.upper(),
            'nombre': nombre,
            'pais': pais,
            'bandera': emoji_bandera,
            'precio_actual': precio_actual,
            'valido': True
        }
    except Exception:
        return {'valido': False}


# =============================================================================
# Cell 06: Interactive portfolio builder widgets
# =============================================================================

def crear_interfaz_portfolio(reset: bool = True):
    """Render the interactive Cell 06 ipywidgets portfolio editor."""
    import ipywidgets as widgets

    if reset:
        state.activos_portfolio = []
        state.indice_editando = None

    output_header = widgets.Output()
    output_validacion = widgets.Output()
    output_lista = widgets.Output()

    input_ticker = widgets.Text(
        placeholder='Ej: AAPL, GGAL, TSLA',
        description='',
        style={'description_width': '0px'},
        layout=widgets.Layout(width='250px', height='45px')
    )

    input_cantidad = widgets.FloatText(
        value=0,
        description='',
        step=0.01,
        style={'description_width': '0px'},
        layout=widgets.Layout(width='120px', height='45px')
    )

    input_precio_compra = widgets.FloatText(
        value=0,
        description='',
        step=0.01,
        style={'description_width': '0px'},
        layout=widgets.Layout(width='120px', height='45px')
    )

    input_upside_esperado = widgets.FloatText(
        value=0,
        description='',
        step=0.1,
        style={'description_width': '0px'},
        layout=widgets.Layout(width='120px', height='45px')
    )

    input_pais_riesgo = widgets.Dropdown(
        options=paises_disponibles,
        value='United States',
        description='',
        style={'description_width': '0px'},
        layout=widgets.Layout(width='150px', height='45px')
    )

    btn_agregar = widgets.Button(
        description='Agregar',
        button_style='primary',
        layout=widgets.Layout(width='120px', height='45px'),
        style={'button_color': '#667eea', 'font_weight': 'bold'}
    )

    btn_finalizar = widgets.Button(
        description='Finalizar Portfolio',
        button_style='success',
        layout=widgets.Layout(width='300px', height='50px', margin='20px 0 0 0'),
        style={'font_weight': 'bold'}
    )

    def editar_activo(index):
        state.indice_editando = index
        activo = state.activos_portfolio[index]

        input_ticker.value = activo['ticker']
        input_ticker.disabled = True
        input_cantidad.value = activo.get('cantidad', 0)
        input_precio_compra.value = activo.get('precio_compra', 0)
        input_upside_esperado.value = activo.get('upside_esperado', 0)
        input_pais_riesgo.value = activo.get('pais_riesgo', activo.get('pais', 'United States'))

        btn_agregar.description = 'Actualizar'
        btn_agregar.button_style = 'warning'

        with output_validacion:
            clear_output()
            display(HTML(f"""
            <div style="
                background: #fff3cd;
                color: #856404;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                border: 1px solid #ffeaa7;
            ">
                ✏️ Editando <strong>{activo['ticker']}</strong>. Modifica los campos y presiona "Actualizar"
            </div>
            """))

    def eliminar_activo(index):
        activo = state.activos_portfolio[index]
        state.activos_portfolio.pop(index)

        with output_validacion:
            clear_output()
            display(HTML(f"""
            <div style="
                background: #d4edda;
                color: #155724;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                border: 1px solid #c3e6cb;
            ">
                🗑️ <strong>{activo['ticker']}</strong> eliminado del portfolio
            </div>
            """))

        actualizar_lista()

    def cancelar_edicion():
        state.indice_editando = None
        input_ticker.value = ''
        input_ticker.disabled = False
        input_cantidad.value = 0
        input_precio_compra.value = 0
        input_upside_esperado.value = 0
        input_pais_riesgo.value = 'United States'
        btn_agregar.description = 'Agregar'
        btn_agregar.button_style = 'primary'

        with output_validacion:
            clear_output()

    def crear_boton_editar(index):
        btn = widgets.Button(
            description='✏️',
            button_style='warning',
            layout=widgets.Layout(width='35px', height='35px'),
            tooltip='Editar'
        )
        btn.on_click(lambda b: editar_activo(index))
        return btn

    def crear_boton_eliminar(index):
        btn = widgets.Button(
            description='🗑️',
            button_style='danger',
            layout=widgets.Layout(width='35px', height='35px'),
            tooltip='Eliminar'
        )
        btn.on_click(lambda b: eliminar_activo(index))
        return btn

    def actualizar_lista():
        with output_lista:
            clear_output()
            if state.activos_portfolio:
                items_widgets = []
                for i, activo in enumerate(state.activos_portfolio):
                    cantidad = activo.get('cantidad', 0)
                    precio_compra = activo.get('precio_compra', 0)
                    precio_actual = activo.get('precio_actual', 0) if activo.get('precio_actual') is not None else 0
                    upside_esperado = activo.get('upside_esperado', 0)
                    pais_riesgo = activo.get('pais_riesgo', activo.get('pais', 'United States'))

                    tiene_portfolio = cantidad > 0 and precio_compra > 0

                    pais_display = activo['pais']
                    if pais_riesgo != activo['pais']:
                        bandera_riesgo = banderas_paises.get(pais_riesgo, '🌐')
                        pais_display = f"{activo['pais']} | <span style='color: #FF5722;'>Riesgo: {bandera_riesgo} {pais_riesgo}</span>"

                    if tiene_portfolio:
                        valor_invertido = cantidad * precio_compra
                        valor_actual = cantidad * precio_actual if precio_actual else 0
                        ganancia_perdida = valor_actual - valor_invertido
                        pct_cambio = (ganancia_perdida / valor_invertido * 100) if valor_invertido > 0 else 0

                        color_performance = '#00C853' if pct_cambio >= 0 else '#FF5252'
                        signo = '+' if pct_cambio >= 0 else ''

                        upside_html = ""
                        if upside_esperado > 0:
                            upside_html = f"""
                            <div>
                                <div style="color: #999;">Upside Esperado</div>
                                <div style="font-weight: bold; color: #9C27B0;">+{upside_esperado:.1f}%</div>
                            </div>
                            """

                        html_items = f"""
                        <div style="
                            background: white;
                            border-left: 4px solid #667eea;
                            padding: 15px 20px;
                            margin: 10px 0;
                            border-radius: 8px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        ">
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                                <div style="display: flex; align-items: center; gap: 15px;">
                                    <span style="font-size: 32px;">{activo['bandera']}</span>
                                    <div>
                                        <div style="font-weight: bold; font-size: 18px; color: #333;">
                                            {activo['ticker']}
                                        </div>
                                        <div style="color: #666; font-size: 14px;">
                                            {activo['nombre']}
                                        </div>
                                        <div style="color: #999; font-size: 12px; margin-top: 3px;">
                                            {pais_display}
                                        </div>
                                    </div>
                                </div>
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <div style="
                                        background: #667eea;
                                        color: white;
                                        padding: 5px 12px;
                                        border-radius: 20px;
                                        font-size: 12px;
                                        font-weight: bold;
                                    ">#{i+1}</div>
                                </div>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size: 13px; padding-top: 10px; border-top: 1px solid #eee;">
                                <div>
                                    <div style="color: #999;">Cantidad</div>
                                    <div style="font-weight: bold; color: #333;">{cantidad:.4f}</div>
                                </div>
                                <div>
                                    <div style="color: #999;">Precio Compra</div>
                                    <div style="font-weight: bold; color: #333;">${precio_compra:.2f}</div>
                                </div>
                                <div>
                                    <div style="color: #999;">Precio Actual</div>
                                    <div style="font-weight: bold; color: #333;">${precio_actual:.2f}</div>
                                </div>
                                <div>
                                    <div style="color: #999;">Valor Invertido</div>
                                    <div style="font-weight: bold; color: #333;">${valor_invertido:.2f}</div>
                                </div>
                                <div>
                                    <div style="color: #999;">Valor Actual</div>
                                    <div style="font-weight: bold; color: #333;">${valor_actual:.2f}</div>
                                </div>
                                <div>
                                    <div style="color: #999;">Performance</div>
                                    <div style="font-weight: bold; color: {color_performance};">{signo}{pct_cambio:.2f}%</div>
                                </div>
                                {upside_html}
                            </div>
                        </div>
                        """
                    else:
                        upside_html = ""
                        if upside_esperado > 0:
                            upside_html = f"""
                            <div style="color: #9C27B0; margin-top: 5px; font-weight: bold;">
                                Upside Esperado: +{upside_esperado:.1f}%
                            </div>
                            """

                        html_items = f"""
                        <div style="
                            background: white;
                            border-left: 4px solid #9E9E9E;
                            padding: 15px 20px;
                            margin: 10px 0;
                            border-radius: 8px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        ">
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                                <div style="display: flex; align-items: center; gap: 15px;">
                                    <span style="font-size: 32px;">{activo['bandera']}</span>
                                    <div>
                                        <div style="font-weight: bold; font-size: 18px; color: #333;">
                                            {activo['ticker']}
                                        </div>
                                        <div style="color: #666; font-size: 14px;">
                                            {activo['nombre']}
                                        </div>
                                        <div style="color: #999; font-size: 12px; margin-top: 3px;">
                                            {pais_display}
                                        </div>
                                    </div>
                                </div>
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <div style="
                                        background: #9E9E9E;
                                        color: white;
                                        padding: 5px 12px;
                                        border-radius: 20px;
                                        font-size: 12px;
                                        font-weight: bold;
                                    ">#{i+1}</div>
                                </div>
                            </div>
                            <div style="font-size: 13px; padding-top: 10px; border-top: 1px solid #eee;">
                                <div style="color: #999;">Precio Actual: <span style="font-weight: bold; color: #333;">${precio_actual:.2f}</span></div>
                                <div style="color: #FF9800; margin-top: 5px; font-style: italic;">Sin posición actual</div>
                                {upside_html}
                            </div>
                        </div>
                        """

                    btn_editar = crear_boton_editar(i)
                    btn_eliminar = crear_boton_eliminar(i)

                    item_container = widgets.VBox([
                        widgets.HTML(html_items),
                        widgets.HBox([btn_editar, btn_eliminar], layout=widgets.Layout(justify_content='center', margin='5px 0 0 0'))
                    ])

                    items_widgets.append(item_container)

                header_html = widgets.HTML(f"""
                <div style="max-width: 800px; margin: 20px auto;">
                    <h3 style="color: #667eea; text-align: center; margin-bottom: 20px;">
                        📊 Activos en el Portfolio ({len(state.activos_portfolio)})
                    </h3>
                </div>
                """)

                display(widgets.VBox([header_html] + items_widgets))

    def on_agregar_click(b):
        ticker = input_ticker.value.strip().upper()
        cantidad = input_cantidad.value
        precio_compra = input_precio_compra.value
        upside_esperado = input_upside_esperado.value
        pais_riesgo = input_pais_riesgo.value

        with output_validacion:
            clear_output()

            if not ticker:
                display(HTML("""
                <div style="color: #e74c3c; padding: 10px; text-align: center;">
                    ⚠️ Por favor ingresa un ticker
                </div>
                """))
                return

            if state.indice_editando is not None:
                activo = state.activos_portfolio[state.indice_editando]
                activo['cantidad'] = cantidad if cantidad else 0
                activo['precio_compra'] = precio_compra if precio_compra else 0
                activo['upside_esperado'] = upside_esperado if upside_esperado else 0
                activo['pais_riesgo'] = pais_riesgo

                display(HTML(f"""
                <div style="
                    background: #d4edda;
                    color: #155724;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    border: 1px solid #c3e6cb;
                ">
                    ✅ <strong>{activo['ticker']}</strong> actualizado exitosamente
                </div>
                """))

                cancelar_edicion()
                actualizar_lista()
                return

            if any(a['ticker'] == ticker for a in state.activos_portfolio):
                display(HTML(f"""
                <div style="color: #e74c3c; padding: 10px; text-align: center;">
                    ⚠️ {ticker} ya está en el portfolio
                </div>
                """))
                return

            display(HTML("""
            <div style="color: #667eea; padding: 10px; text-align: center;">
                🔍 Validando ticker y obteniendo precio actual...
            </div>
            """))

            info = obtener_info_activo(ticker)

            clear_output()

            if info['valido']:
                input_pais_riesgo.value = info['pais'] if info['pais'] in paises_disponibles else 'United States'

                info['cantidad'] = cantidad if cantidad else 0
                info['precio_compra'] = precio_compra if precio_compra else 0
                info['upside_esperado'] = upside_esperado if upside_esperado else 0
                info['pais_riesgo'] = pais_riesgo
                state.activos_portfolio.append(info)

                tiene_portfolio = cantidad > 0 and precio_compra > 0
                mensaje_extra = "" if tiene_portfolio else " (sin posición actual)"

                upside_msg = f" con upside esperado de +{upside_esperado:.1f}%" if upside_esperado > 0 else ""

                pais_msg = f" | País de riesgo: {pais_riesgo}" if pais_riesgo != info['pais'] else ""

                display(HTML(f"""
                <div style="
                    background: #d4edda;
                    color: #155724;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    border: 1px solid #c3e6cb;
                ">
                    ✅ {info['bandera']} <strong>{info['ticker']}</strong> - {info['nombre']} agregado exitosamente{mensaje_extra}{upside_msg}{pais_msg}
                </div>
                """))
                input_ticker.value = ''
                input_cantidad.value = 0
                input_precio_compra.value = 0
                input_upside_esperado.value = 0
                input_pais_riesgo.value = 'United States'
                actualizar_lista()
            else:
                display(HTML(f"""
                <div style="
                    background: #f8d7da;
                    color: #721c24;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    border: 1px solid #f5c6cb;
                ">
                    ❌ No se pudo validar el ticker <strong>{ticker}</strong>. Verifica que sea correcto.
                </div>
                """))

    def on_finalizar_click(b):
        with output_validacion:
            clear_output()
            if len(state.activos_portfolio) < 2:
                display(HTML("""
                <div style="
                    background: #fff3cd;
                    color: #856404;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    border: 1px solid #ffeaa7;
                ">
                    ⚠️ Necesitas al menos 2 activos para crear un portfolio
                </div>
                """))
            else:
                activos_con_portfolio = [a for a in state.activos_portfolio if a.get('cantidad', 0) > 0 and a.get('precio_compra', 0) > 0]

                if activos_con_portfolio:
                    total_invertido = sum(a['cantidad'] * a['precio_compra'] for a in activos_con_portfolio)
                    total_actual = sum(a['cantidad'] * a['precio_actual'] for a in activos_con_portfolio if a.get('precio_actual'))
                    performance_total = ((total_actual - total_invertido) / total_invertido * 100) if total_invertido > 0 else 0
                    color_perf = '#00C853' if performance_total >= 0 else '#FF5252'
                    signo_perf = '+' if performance_total >= 0 else ''

                    info_portfolio = f"""
                    <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin-top: 20px;">
                        <div style="font-size: 14px; margin-bottom: 5px;">Total Invertido: ${total_invertido:,.2f}</div>
                        <div style="font-size: 14px; margin-bottom: 5px;">Valor Actual: ${total_actual:,.2f}</div>
                        <div style="font-size: 18px; font-weight: bold; margin-top: 10px;">
                            Performance: <span style="color: {color_perf};">{signo_perf}{performance_total:.2f}%</span>
                        </div>
                    </div>
                    """
                else:
                    info_portfolio = """
                    <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin-top: 20px;">
                        <div style="font-size: 14px;">Portfolio nuevo - Sin posiciones actuales</div>
                    </div>
                    """

                display(HTML(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                ">
                    <h2 style="margin: 0 0 10px 0;">🎉 Portfolio Configurado</h2>
                    <p style="font-size: 18px; margin: 10px 0;">{len(state.activos_portfolio)} activos listos para análisis</p>
                    {info_portfolio}
                </div>
                """))

    btn_agregar.on_click(on_agregar_click)
    btn_finalizar.on_click(on_finalizar_click)

    with output_header:
        display(HTML("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        ">
            <h1 style="color: white; margin: 0 0 10px 0; font-size: 32px;">
                📈 Configuración de Portfolio
            </h1>
            <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 16px;">
                Ingresa los activos. Cantidad y precio son opcionales (dejar en 0 para portfolio nuevo)
            </p>
        </div>
        """))

    label_ticker = widgets.HTML("<div style='text-align:center; font-weight:bold; font-size:14px; margin-bottom:5px;'>Ticker</div>", layout=widgets.Layout(width='250px'))
    label_cantidad = widgets.HTML("<div style='text-align:center; font-weight:bold; font-size:14px; margin-bottom:5px;'>Cantidad (opcional)</div>", layout=widgets.Layout(width='120px'))
    label_precio = widgets.HTML("<div style='text-align:center; font-weight:bold; font-size:14px; margin-bottom:5px;'>Precio Compra (opcional)</div>", layout=widgets.Layout(width='120px'))
    label_upside = widgets.HTML("<div style='text-align:center; font-weight:bold; font-size:14px; margin-bottom:5px;'>Upside Esperado % (opcional)</div>", layout=widgets.Layout(width='120px'))
    label_pais = widgets.HTML("<div style='text-align:center; font-weight:bold; font-size:14px; margin-bottom:5px;'>País de Riesgo</div>", layout=widgets.Layout(width='150px'))
    label_boton = widgets.HTML("<div style='text-align:center; font-weight:bold; font-size:14px; margin-bottom:5px; color: transparent;'>.</div>", layout=widgets.Layout(width='120px'))

    labels_box = widgets.HBox(
        [label_ticker, label_cantidad, label_precio, label_upside, label_pais, label_boton],
        layout=widgets.Layout(
            justify_content='center',
            margin='10px 0 0 0',
            gap='10px'
        )
    )

    input_box = widgets.HBox(
        [input_ticker, input_cantidad, input_precio_compra, input_upside_esperado, input_pais_riesgo, btn_agregar],
        layout=widgets.Layout(
            justify_content='center',
            margin='0 0 20px 0',
            gap='10px'
        )
    )

    container = widgets.VBox([
        output_header,
        labels_box,
        input_box,
        output_validacion,
        output_lista,
        widgets.HBox([btn_finalizar], layout=widgets.Layout(justify_content='center'))
    ])

    display(container)
    return container


# =============================================================================
# Cell 20: Integrated execution panel
# =============================================================================

def ejecutar_interfaz_analisis(activar_ml: bool = True):
    """Render Cell 20 panel and run ``ejecutar_analisis_completo``."""
    import ipywidgets as widgets

    output_ejecucion = widgets.Output()

    checkbox_activar_ml = widgets.Checkbox(
        value=activar_ml,
        description='Activar Proyección ML (más lento pero incluye predicciones)',
        style={'description_width': 'initial'}
    )

    display(HTML('''
    <div style="background: linear-gradient(135deg, #667eea, #764ba2);
                padding: 30px; border-radius: 15px; margin: 20px 0;">
        <h2 style="color: white; text-align: center; margin-bottom: 10px;">
            🎯 ANÁLISIS INTEGRADO DE PORTFOLIO
        </h2>
        <p style="color: rgba(255,255,255,0.9); text-align: center; font-size: 14px;">
            Combina optimización clásica y proyección ML
        </p>
    </div>
    '''))

    display(checkbox_activar_ml)
    display(output_ejecucion)

    with output_ejecucion:
        clear_output()

        if not state.activos_portfolio:
            display(HTML('''
            <div style="color: #e74c3c; padding: 15px; text-align: center; background: #f8d7da; border-radius: 8px;">
                ⚠️ Debes agregar al menos 2 activos al portfolio antes de ejecutar el análisis
            </div>
            '''))
        else:
            tickers_local = [a['ticker'] for a in state.activos_portfolio]

            resultado = ejecutar_analisis_completo(
                tickers_list=tickers_local,
                activos_dict=state.activos_portfolio,
                ejecutar_ml=checkbox_activar_ml.value
            )

            if resultado:
                display(HTML('''
                <div style="background: linear-gradient(135deg, #667eea, #764ba2);
                            color: white; padding: 20px; border-radius: 10px; text-align: center;
                            margin-top: 20px;">
                    <h2>✅ Análisis Completado</h2>
                    <p>Los resultados están almacenados en las variables globales:</p>
                    <ul style="list-style: none; padding: 0;">
                        <li>📊 <code>resultado_analisis</code> - Optimización clásica</li>
                        <li>🤖 <code>predicciones_ml</code> - Predicciones 1 trimestre</li>
                        <li>🎯 <code>predicciones_5y</code> - Proyecciones 5 años</li>
                    </ul>
                    <p style="margin-top: 15px; font-size: 14px;">
                        Ejecuta las celdas de visualización para ver gráficos interactivos
                    </p>
                </div>
                '''))
            return resultado
    return None


# =============================================================================
# Cell 21: ML projections table and button UI
# =============================================================================

def mostrar_proyecciones_ml():
    """Muestra tabla interactiva con predicciones de corto y largo plazo"""
    predicciones_ml = state.predicciones_ml
    predicciones_5y = state.predicciones_5y

    if predicciones_ml is None and predicciones_5y is None:
        display(HTML('''
        <div style="color: #e74c3c; padding: 15px; text-align: center; background: #f8d7da; border-radius: 8px;">
            ⚠️ No hay predicciones ML disponibles. Ejecuta primero el análisis completo con ML activado.
        </div>
        '''))
        return

    html = '''
    <div style="margin: 20px 0;">
        <h2 style="text-align: center; color: #2C3E50; margin-bottom: 25px;">
            🤖 PROYECCIONES DE MACHINE LEARNING
        </h2>
    '''

    if predicciones_ml:
        html += '''
        <div style="margin-bottom: 40px;">
            <h3 style="color: #667eea; margin-bottom: 15px;">📈 Predicciones Corto Plazo (1 Trimestre)</h3>
            <table style="width:100%; border-collapse:collapse; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <thead>
                    <tr style="background:linear-gradient(135deg, #667eea, #764ba2); color:white;">
                        <th style="padding:15px; text-align:left;">Ticker</th>
                        <th style="padding:15px; text-align:center;">Precio Actual</th>
                        <th style="padding:15px; text-align:center;">Precio Predicho</th>
                        <th style="padding:15px; text-align:center;">Variación</th>
                        <th style="padding:15px; text-align:center;">Top Feature</th>
                        <th style="padding:15px; text-align:center;">Muestras</th>
                    </tr>
                </thead>
                <tbody>
        '''

        for i, (ticker, pred) in enumerate(predicciones_ml.items()):
            bg_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
            color_var = '#00C853' if pred['pct_change'] >= 0 else '#FF5252'
            signo = '+' if pred['pct_change'] >= 0 else ''

            top_feature = pred['top_features'][0][0] if pred['top_features'] else 'N/A'

            html += f'''
            <tr style="background:{bg_color}; border-bottom:1px solid #dee2e6;">
                <td style="padding:12px; font-weight:bold; color:#667eea;">{ticker}</td>
                <td style="padding:12px; text-align:center;">${pred['current_price']:.2f}</td>
                <td style="padding:12px; text-align:center; font-weight:bold;">${pred['predicted_price']:.2f}</td>
                <td style="padding:12px; text-align:center; color:{color_var}; font-weight:bold;">{signo}{pred['pct_change']:.2f}%</td>
                <td style="padding:12px; text-align:center; font-size:12px;">{top_feature[:20]}</td>
                <td style="padding:12px; text-align:center;">{pred['n_samples']}</td>
            </tr>
            '''

        html += '''
                </tbody>
            </table>
        </div>
        '''

    if predicciones_5y:
        html += '''
        <div>
            <h3 style="color: #667eea; margin-bottom: 15px;">🎯 Proyecciones Estratégicas (5 Años)</h3>
            <table style="width:100%; border-collapse:collapse; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <thead>
                    <tr style="background:linear-gradient(135deg, #667eea, #764ba2); color:white;">
                        <th style="padding:15px; text-align:left;">Ticker</th>
                        <th style="padding:15px; text-align:center;">Precio Actual</th>
                        <th style="padding:15px; text-align:center;">Base (2031)</th>
                        <th style="padding:15px; text-align:center;">🐂 Bull</th>
                        <th style="padding:15px; text-align:center;">🐻 Bear</th>
                        <th style="padding:15px; text-align:center;">Diferencial</th>
                    </tr>
                </thead>
                <tbody>
        '''

        for i, (ticker, pred) in enumerate(predicciones_5y.items()):
            bg_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
            scenarios = pred['scenarios']

            base_price = scenarios['base']['predicted_price']
            bull_price = scenarios['bull']['predicted_price']
            bear_price = scenarios['bear']['predicted_price']
            current = pred['current_price']

            diferencial = ((bull_price - bear_price) / current) * 100

            base_pct = ((base_price - current) / current) * 100
            bull_pct = ((bull_price - current) / current) * 100
            bear_pct = ((bear_price - current) / current) * 100

            html += f'''
            <tr style="background:{bg_color}; border-bottom:1px solid #dee2e6;">
                <td style="padding:12px; font-weight:bold; color:#667eea;">{ticker}</td>
                <td style="padding:12px; text-align:center;">${current:.2f}</td>
                <td style="padding:12px; text-align:center;">
                    <div style="font-weight:bold;">${base_price:.2f}</div>
                    <div style="font-size:11px; color:{'#00C853' if base_pct >= 0 else '#FF5252'};">
                        {'+' if base_pct >= 0 else ''}{base_pct:.1f}%
                    </div>
                </td>
                <td style="padding:12px; text-align:center; background:#e8f5e9;">
                    <div style="font-weight:bold;">${bull_price:.2f}</div>
                    <div style="font-size:11px; color:#00C853;">+{bull_pct:.1f}%</div>
                </td>
                <td style="padding:12px; text-align:center; background:#ffebee;">
                    <div style="font-weight:bold;">${bear_price:.2f}</div>
                    <div style="font-size:11px; color:#FF5252;">{bear_pct:.1f}%</div>
                </td>
                <td style="padding:12px; text-align:center; font-weight:bold;">
                    {diferencial:.1f}%
                </td>
            </tr>
            '''

        html += '''
                </tbody>
            </table>

            <div style="margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 8px; font-size: 13px;">
                <b>📌 Nota:</b> El diferencial muestra la sensibilidad del precio a cambios macroeconómicos.
                Un diferencial alto indica mayor incertidumbre y dependencia del entorno económico.
            </div>
        </div>
        '''

    html += '''
    </div>
    '''

    display(HTML(html))


def crear_interfaz_proyecciones_ml():
    """Render Cell 21 button and output panel."""
    import ipywidgets as widgets

    btn_mostrar_proyecciones = widgets.Button(
        description='📊 Mostrar Proyecciones ML',
        button_style='info',
        layout=widgets.Layout(width='300px', height='50px')
    )

    output_proyecciones = widgets.Output()

    def on_mostrar_proyecciones_click(b):
        with output_proyecciones:
            clear_output()
            mostrar_proyecciones_ml()

    btn_mostrar_proyecciones.on_click(on_mostrar_proyecciones_click)

    display(btn_mostrar_proyecciones)
    display(output_proyecciones)
    return btn_mostrar_proyecciones, output_proyecciones


# =============================================================================
# Cell 28: Interactive rebalance selector
# =============================================================================

def mostrar_rebalanceo():
    """Render Cell 28 interactive rebalance strategy dropdown and table."""
    import ipywidgets as widgets

    rebalanceo_sharpe = state.rebalanceo_sharpe
    rebalanceo_minvar = state.rebalanceo_minvar
    rebalanceo_cvar = state.rebalanceo_cvar

    estrategia_selector = widgets.Dropdown(
        options=[
            ('⭐ Máximo Sharpe', 'sharpe'),
            ('💎 Mínima Varianza', 'minvar'),
            ('🛡️ Mínimo CVaR', 'cvar')
        ],
        value='sharpe',
        description='Estrategia:',
        style={'description_width': 'initial'}
    )

    output_rebalanceo = widgets.Output()

    def actualizar_rebalanceo(change):
        with output_rebalanceo:
            clear_output()

            estrategia = estrategia_selector.value

            if estrategia == 'sharpe':
                datos = rebalanceo_sharpe
                titulo = '⭐ Rebalanceo para Máximo Sharpe'
                color = '#FFD700'
            elif estrategia == 'minvar':
                datos = rebalanceo_minvar
                titulo = '💎 Rebalanceo para Mínima Varianza'
                color = '#00bcd4'
            else:
                datos = rebalanceo_cvar
                titulo = '🛡️ Rebalanceo para Mínimo CVaR'
                color = '#9c27b0'

            html = f"""
            <div style="max-width: 1200px; margin: 20px auto;">
                <h2 style="text-align: center; color: {color}; margin-bottom: 30px;">{titulo}</h2>
                <table style="width: 100%; border-collapse: collapse; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                    <thead>
                        <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                            <th style="padding: 15px; text-align: left;">Activo</th>
                            <th style="padding: 15px; text-align: center;">Peso Actual</th>
                            <th style="padding: 15px; text-align: center;">Peso Objetivo</th>
                            <th style="padding: 15px; text-align: center;">Cantidad Actual</th>
                            <th style="padding: 15px; text-align: center;">Cantidad Objetivo</th>
                            <th style="padding: 15px; text-align: center;">Acción</th>
                            <th style="padding: 15px; text-align: center;">Precio Compra</th>
                            <th style="padding: 15px; text-align: center;">Precio Actual</th>
                            <th style="padding: 15px; text-align: center;">Performance</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for i, row in enumerate(datos):
                bg_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'

                if row['diferencia_cantidad'] > 0:
                    accion = f"<span style='color: #00C853; font-weight: bold;'>COMPRAR {row['diferencia_cantidad']:.4f}</span>"
                    accion_color = '#e8f5e9'
                elif row['diferencia_cantidad'] < 0:
                    if row['ganancia_perdida_pct'] < 0:
                        accion = f"<span style='color: #FF9800; font-weight: bold;'>VENDER/MANTENER {abs(row['diferencia_cantidad']):.4f}<br><small style='font-weight: normal;'>(En pérdida: considerar esperar repunte)</small></span>"
                        accion_color = '#fff3e0'
                    else:
                        accion = f"<span style='color: #FF5252; font-weight: bold;'>VENDER {abs(row['diferencia_cantidad']):.4f}</span>"
                        accion_color = '#ffebee'
                else:
                    accion = "<span style='color: #757575;'>MANTENER</span>"
                    accion_color = bg_color

                perf_color = '#00C853' if row['ganancia_perdida_pct'] >= 0 else '#FF5252'
                perf_signo = '+' if row['ganancia_perdida_pct'] >= 0 else ''

                html += f"""
                <tr style="background: {bg_color}; border-bottom: 1px solid #dee2e6;">
                    <td style="padding: 12px; font-weight: bold; color: #2C3E50;">{row['ticker']}</td>
                    <td style="padding: 12px; text-align: center;">{row['peso_actual']:.2%}</td>
                    <td style="padding: 12px; text-align: center; font-weight: bold; color: {color};">{row['peso_objetivo']:.2%}</td>
                    <td style="padding: 12px; text-align: center;">{row['cantidad_actual']:.4f}</td>
                    <td style="padding: 12px; text-align: center; font-weight: bold;">{row['cantidad_objetivo']:.4f}</td>
                    <td style="padding: 12px; text-align: center; background: {accion_color};">{accion}</td>
                    <td style="padding: 12px; text-align: center;">${row['precio_compra']:.2f}</td>
                    <td style="padding: 12px; text-align: center;">${row['precio_actual']:.2f}</td>
                    <td style="padding: 12px; text-align: center; font-weight: bold; color: {perf_color};">{perf_signo}{row['ganancia_perdida_pct']:.2f}%</td>
                </tr>
                """

            html += """
                    </tbody>
                </table>
            """

            total_comprar = sum(row['diferencia_valor'] for row in datos if row['diferencia_valor'] > 0)
            total_vender = abs(sum(row['diferencia_valor'] for row in datos if row['diferencia_valor'] < 0))
            diferencia = total_comprar - total_vender

            html += f"""
                <div style="margin-top: 30px; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
                    <h3 style="text-align: center; margin-bottom: 20px;">💰 Resumen de Rebalanceo</h3>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px; text-align: center;">
                            <div style="font-size: 14px; margin-bottom: 5px;">Total a Comprar</div>
                            <div style="font-size: 24px; font-weight: bold; color: #00C853;">${total_comprar:,.2f}</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px; text-align: center;">
                            <div style="font-size: 14px; margin-bottom: 5px;">Total a Vender</div>
                            <div style="font-size: 24px; font-weight: bold; color: #FF5252;">${total_vender:,.2f}</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px; text-align: center;">
                            <div style="font-size: 14px; margin-bottom: 5px;">Capital Adicional Requerido</div>
                            <div style="font-size: 24px; font-weight: bold; color: {'#FFD700' if diferencia > 0 else '#00C853'};">${abs(diferencia):,.2f}</div>
                        </div>
                    </div>
                    <div style="text-align: center; margin-top: 20px; font-size: 14px; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                        {'⚠️ Necesitarás aportar capital adicional para completar el rebalanceo.' if diferencia > 0 else '✅ Puedes realizar el rebalanceo con el capital actual (vendiendo y recomprando).'}
                    </div>
                </div>
            </div>
            """

            display(HTML(html))

    estrategia_selector.observe(actualizar_rebalanceo, names='value')

    display(estrategia_selector)
    display(output_rebalanceo)

    actualizar_rebalanceo(None)
    return estrategia_selector, output_rebalanceo
