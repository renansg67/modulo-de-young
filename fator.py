import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO

# --- Funções de Carregamento e Pré-Processamento ---
@st.cache_data
def carregar_e_limpar_dados(uploaded_file, header_row=6):
    if uploaded_file is None:
        return None
        
    try:
        string_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(string_data, sep=',', header=header_row, decimal='.') 
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

    if len(df) > 0:
        df = df.iloc[1:].copy()

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=df.columns[[1, 2, 3]]).reset_index(drop=True)

    df = df.rename(columns={
        df.columns[1]: 'Forca_kN',
        df.columns[2]: 'Desl_LVDT_mm',
        df.columns[3]: 'Desl_Pistao_mm'
    })
    
    df['Forca_Abs'] = df['Forca_kN'] - df['Forca_kN'].iloc[0]
    df['Desl_LVDT_Abs'] = abs(df['Desl_LVDT_mm'] - df['Desl_LVDT_mm'].iloc[0])
    df['Desl_Pistao_Abs'] = abs(df['Desl_Pistao_mm'] - df['Desl_Pistao_mm'].iloc[0])

    return df

# --- Interface Streamlit ---
st.set_page_config(layout="wide")
st.title("📊 Visualização com Regressão e Fator de Correção do Pistão")

# Slider único na sidebar para faixa de força
F_range_pct = st.sidebar.slider(
    "Selecione a faixa de força (%)",
    0.0, 100.0, (10.0, 40.0), 1.0
)
F_min_pct, F_max_pct = F_range_pct[0] / 100, F_range_pct[1] / 100

uploaded_file_calibracao = st.file_uploader(
    "Carregue o Arquivo de Calibração (4 colunas)", type=["csv"]
)

if uploaded_file_calibracao:
    df_calibracao = carregar_e_limpar_dados(uploaded_file_calibracao)
    
    if df_calibracao is not None:
        Forca_max_abs = df_calibracao['Forca_Abs'].max()
        F_limite_min = Forca_max_abs * F_min_pct
        F_limite_max = Forca_max_abs * F_max_pct
        
        st.info(f"Força máxima: {Forca_max_abs:.3f} kN\n"
                f"Faixa de exibição: {F_limite_min:.3f} – {F_limite_max:.3f} kN")
        
        # --- Filtrar apenas os pontos dentro da faixa ---
        df_filtrado = df_calibracao[
            (df_calibracao['Forca_Abs'] >= F_limite_min) & 
            (df_calibracao['Forca_Abs'] <= F_limite_max)
        ]
        
        # --- Gráfico ---
        fig = go.Figure()
        
        # Linhas auxiliares fora da faixa (mesma cor dos pontos, finas)
        # LVDT fora da faixa
        df_out_lvd = df_calibracao[~df_calibracao.index.isin(df_filtrado.index)]
        fig.add_trace(go.Scatter(
            x=df_out_lvd['Desl_LVDT_Abs'], y=df_out_lvd['Forca_Abs'],
            mode='lines', line=dict(color='blue', width=1), name='LVDT Fora da Faixa'
        ))
        # Pistão fora da faixa
        fig.add_trace(go.Scatter(
            x=df_out_lvd['Desl_Pistao_Abs'], y=df_out_lvd['Forca_Abs'],
            mode='lines', line=dict(color='red', width=1), name='Pistão Fora da Faixa'
        ))
        
        # Pontos filtrados coloridos
        fig.add_trace(go.Scatter(
            x=df_filtrado['Desl_LVDT_Abs'], y=df_filtrado['Forca_Abs'],
            mode='markers', marker=dict(color='blue', size=6), name='LVDT (Faixa)'
        ))
        fig.add_trace(go.Scatter(
            x=df_filtrado['Desl_Pistao_Abs'], y=df_filtrado['Forca_Abs'],
            mode='markers', marker=dict(color='red', size=6), name='Pistão (Faixa)'
        ))
        
        # Linhas horizontais de limite destacadas com rótulo
        for pct, f_lim in zip([F_min_pct, F_max_pct], [F_limite_min, F_limite_max]):
            fig.add_shape(type="line",
                          x0=0, x1=max(df_calibracao['Desl_LVDT_Abs'].max(), df_calibracao['Desl_Pistao_Abs'].max())*1.05,
                          y0=f_lim, y1=f_lim,
                          line=dict(color="magenta", width=4, dash="dash"))
            fig.add_annotation(x=0.5, y=f_lim,
                               text=f"{int(pct*100)}% da Fmáx",
                               showarrow=False,
                               font=dict(color="magenta", size=12),
                               xanchor="left", yanchor="bottom")
        
        # --- Regressão linear (somente pontos filtrados) ---
        if len(df_filtrado) >= 2:
            # LVDT
            slope_lvd, intercept_lvd = np.polyfit(df_filtrado['Desl_LVDT_Abs'], df_filtrado['Forca_Abs'], 1)
            x_lvd = np.array([df_filtrado['Desl_LVDT_Abs'].min(), df_filtrado['Desl_LVDT_Abs'].max()])
            y_lvd = slope_lvd * x_lvd + intercept_lvd
            fig.add_trace(go.Scatter(
                x=x_lvd, y=y_lvd, mode='lines',
                line=dict(color='yellow', width=3, dash='dash'),
                name=f'Reg. LVDT (slp={slope_lvd:.2f})'
            ))
            
            # Pistão
            slope_pis, intercept_pis = np.polyfit(df_filtrado['Desl_Pistao_Abs'], df_filtrado['Forca_Abs'], 1)
            x_pis = np.array([df_filtrado['Desl_Pistao_Abs'].min(), df_filtrado['Desl_Pistao_Abs'].max()])
            y_pis = slope_pis * x_pis + intercept_pis
            fig.add_trace(go.Scatter(
                x=x_pis, y=y_pis, mode='lines',
                line=dict(color='orange', width=3, dash='dash'),
                name=f'Reg. Pistão (slp={slope_pis:.2f})'
            ))
            
            # --- Cálculo do fator de correção do módulo do pistão ---
            fator_correcao = slope_lvd / slope_pis
            st.success(f"✅ Fator de Correção do Módulo (Pistão) = {fator_correcao:.3f}")
        
        fig.update_layout(
            title="Carga vs Deslocamento com Regressão na Faixa Selecionada",
            xaxis_title="Deslocamento Absoluto (mm)",
            yaxis_title="Força (kN)",
            legend_title="Sensor",
            template="simple_white",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Erro no processamento do arquivo.")
else:
    st.info("Aguardando o carregamento do arquivo.")
