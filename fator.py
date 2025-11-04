import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
import os

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
st.title("📊 Análise de Regressão e Fator de Correção do Pistão")

# Slider de força
F_range_pct = st.sidebar.slider(
    "Selecione a faixa de força (%)",
    0.0, 100.0, (10.0, 40.0), 1.0
)
F_min_pct, F_max_pct = F_range_pct[0] / 100, F_range_pct[1] / 100

# Upload do arquivo
uploaded_file_calibracao = st.file_uploader(
    "Carregue o Arquivo de Calibração (4 colunas)", type=["csv"]
)

if uploaded_file_calibracao:
    df_calibracao = carregar_e_limpar_dados(uploaded_file_calibracao)
    
    if df_calibracao is not None:
        # Nome do arquivo sem extensão
        nome_amostra = os.path.splitext(uploaded_file_calibracao.name)[0]

        # Cálculo dos limites de força
        Forca_max_abs = df_calibracao['Forca_Abs'].max()
        F_limite_min = Forca_max_abs * F_min_pct
        F_limite_max = Forca_max_abs * F_max_pct
        
        # Slider de deslocamento
        desl_min, desl_max = (
            float(df_calibracao[['Desl_LVDT_Abs', 'Desl_Pistao_Abs']].min().min()),
            float(df_calibracao[['Desl_LVDT_Abs', 'Desl_Pistao_Abs']].max().max())
        )
        desl_range = st.sidebar.slider(
            "Selecione a faixa de deslocamento (mm)",
            desl_min, desl_max, (desl_min, desl_max), 0.01
        )
        D_limite_min, D_limite_max = desl_range
        
        st.info(f"**Amostra:** `{nome_amostra}`  \n"
                f"**Força máxima:** {Forca_max_abs:.3f} kN  \n"
                f"**Faixa de Força:** {F_limite_min:.3f} – {F_limite_max:.3f} kN  \n"
                f"**Faixa de Deslocamento:** {D_limite_min:.3f} – {D_limite_max:.3f} mm")
        
        # --- Filtragem: dentro dos limites ---
        df_filtrado = df_calibracao[
            (df_calibracao['Forca_Abs'] >= F_limite_min) &
            (df_calibracao['Forca_Abs'] <= F_limite_max) &
            (df_calibracao['Desl_LVDT_Abs'] >= D_limite_min) &
            (df_calibracao['Desl_LVDT_Abs'] <= D_limite_max) &
            (df_calibracao['Desl_Pistao_Abs'] >= D_limite_min) &
            (df_calibracao['Desl_Pistao_Abs'] <= D_limite_max)
        ]
        
        # --- Gráfico ---
        fig = go.Figure()
        
        df_out = df_calibracao[~df_calibracao.index.isin(df_filtrado.index)]
        fig.add_trace(go.Scatter(
            x=df_out['Desl_LVDT_Abs'], y=df_out['Forca_Abs'],
            mode='lines', line=dict(color='blue', width=1, dash='dot'),
            name='LVDT Fora da Faixa'
        ))
        fig.add_trace(go.Scatter(
            x=df_out['Desl_Pistao_Abs'], y=df_out['Forca_Abs'],
            mode='lines', line=dict(color='red', width=1, dash='dot'),
            name='Pistão Fora da Faixa'
        ))
        fig.add_trace(go.Scatter(
            x=df_filtrado['Desl_LVDT_Abs'], y=df_filtrado['Forca_Abs'],
            mode='markers', marker=dict(color='blue', size=6),
            name='LVDT (Faixa)'
        ))
        fig.add_trace(go.Scatter(
            x=df_filtrado['Desl_Pistao_Abs'], y=df_filtrado['Forca_Abs'],
            mode='markers', marker=dict(color='red', size=6),
            name='Pistão (Faixa)'
        ))
        
        # Linhas horizontais dos limites de força
        for pct, f_lim in zip([F_min_pct, F_max_pct], [F_limite_min, F_limite_max]):
            fig.add_shape(type="line",
                          x0=0, x1=desl_max*1.05,
                          y0=f_lim, y1=f_lim,
                          line=dict(color="green", width=4, dash="dot"))
            fig.add_annotation(
                x=0.5, y=f_lim,
                text=f"{int(pct*100)}% da Fmáx",
                showarrow=False, font=dict(color="green", size=12),
                xanchor="left", yanchor="bottom"
            )
        
        # Linhas verticais dos limites de deslocamento
        for d_lim in [D_limite_min, D_limite_max]:
            fig.add_shape(type="line",
                          x0=d_lim, x1=d_lim,
                          y0=0, y1=Forca_max_abs*1.05,
                          line=dict(color="purple", width=3, dash="dot"))
            fig.add_annotation(
                x=d_lim, y=Forca_max_abs*1.02,
                text=f"{d_lim:.2f} mm",
                showarrow=False, font=dict(color="purple", size=11),
                xanchor="center"
            )
        
        # --- Regressão e Fator de Correção ---
        slope_lvd = slope_pis = np.nan
        if len(df_filtrado) >= 2:
            slope_lvd, intercept_lvd = np.polyfit(df_filtrado['Desl_LVDT_Abs'], df_filtrado['Forca_Abs'], 1)
            slope_pis, intercept_pis = np.polyfit(df_filtrado['Desl_Pistao_Abs'], df_filtrado['Forca_Abs'], 1)
            
            # Retas de regressão
            x_lvd = np.array([df_filtrado['Desl_LVDT_Abs'].min(), df_filtrado['Desl_LVDT_Abs'].max()])
            y_lvd = slope_lvd * x_lvd + intercept_lvd
            x_pis = np.array([df_filtrado['Desl_Pistao_Abs'].min(), df_filtrado['Desl_Pistao_Abs'].max()])
            y_pis = slope_pis * x_pis + intercept_pis

            fig.add_trace(go.Scatter(x=x_lvd, y=y_lvd, mode='lines',
                                     line=dict(color='yellow', width=3, dash='dash'),
                                     name=f'Regressão LVDT (slope={slope_lvd:.4f})'))
            fig.add_trace(go.Scatter(x=x_pis, y=y_pis, mode='lines',
                                     line=dict(color='orange', width=3, dash='dash'),
                                     name=f'Regressão Pistão (slope={slope_pis:.4f})'))

            fator_correcao = slope_lvd / slope_pis
            st.success(f"✅ **Fator de Correção do Módulo (Pistão)** = {fator_correcao:.3f}")

            # --- DataFrame de saída em linha ---
            df_resultado = pd.DataFrame([{
                "Identificação": nome_amostra,
                "LVDT": round(slope_lvd, 4),
                "Pistão": round(slope_pis, 4),
                "Fmín": int(F_min_pct * 100),
                "Fmáx": int(F_max_pct * 100)
            }])
            st.subheader("📋 Resultado consolidado")
            st.dataframe(df_resultado, use_container_width=True, hide_index=True)

        fig.update_layout(
            title="Carga vs Deslocamento com Filtros, Regressão e Fator de Correção",
            xaxis_title="Deslocamento Absoluto (mm)",
            yaxis_title="Força (kN)",
            legend_title="Sensor",
            template="simple_white",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
