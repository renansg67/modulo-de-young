import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress

st.set_page_config(page_title="Analisador de Tração", layout="wide")

st.title("📈 Analisador de Ensaios de Tração (NBR 7190)")
st.write("Faça upload de um ou mais arquivos CSV (separador `;`, vírgula decimal).")

# --- Upload ---
uploaded_files = st.file_uploader(
    "Selecione os arquivos CSV",
    type=["csv"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("👆 Carregue pelo menos um arquivo para começar.")
    st.stop()

# --- Leitura dos arquivos ---
dados = []
for file in uploaded_files:
    try:
        df = pd.read_csv(
            file,
            sep=";",
            decimal=",",
            header=None,
            skiprows=1,
            usecols=[0, 1, 2],
            names=["tempo_s", "deformacao_mm", "forca_n"],
            encoding="latin1"
        )
        df["deformacao_mm"] = -df["deformacao_mm"]  # inverte o sinal
        df["arquivo"] = file.name
        dados.append(df)
    except Exception as e:
        st.warning(f"Erro ao ler '{file.name}': {e}")

if not dados:
    st.error("Nenhum arquivo pôde ser lido.")
    st.stop()

# --- Compilação ---
df_compilado = pd.concat(dados, ignore_index=True)

# --- Seleção do arquivo ---
arquivos = df_compilado["arquivo"].unique()
arquivo_sel = st.selectbox("Escolha o arquivo para visualizar:", arquivos)
df_sel = df_compilado[df_compilado["arquivo"] == arquivo_sel].copy()

# --- Cálculo da força máxima ---
fmax = df_sel["forca_n"].max()

# --- Sliders ---
st.sidebar.header("🔧 Filtros de regressão")

# Força (% da Fmáx)
y_perc_lim = st.sidebar.slider(
    "Intervalo da força (em % da Fmáx)",
    0.0, 100.0, (10.0, 50.0), step=1.0,
)
fmin_y, fmax_y = [p / 100 * fmax for p in y_perc_lim]

# Deformação (mm)
x_min, x_max = df_sel["deformacao_mm"].min(), df_sel["deformacao_mm"].max()
x_lim = st.sidebar.slider(
    "Intervalo de deformação (mm)",
    float(x_min), float(x_max),
    (float(x_min), float(x_max)),
    step=(x_max - x_min) / 100
)

# --- Filtra os pontos usados na regressão ---
df_reg = df_sel[
    (df_sel["forca_n"] >= fmin_y)
    & (df_sel["forca_n"] <= fmax_y)
    & (df_sel["deformacao_mm"] >= x_lim[0])
    & (df_sel["deformacao_mm"] <= x_lim[1])
]

# --- Regressão linear ---
if len(df_reg) >= 2:
    slope, intercept, r_value, _, _ = linregress(df_reg["deformacao_mm"], df_reg["forca_n"])
    rigidez = slope
    r2 = r_value**2
else:
    slope = intercept = rigidez = r2 = np.nan

# --- DataFrame de resultados ---
df_resultados = pd.DataFrame({
    "Arquivo": [arquivo_sel],
    "Força Máxima (N)": [fmax],
    "ΔF/Δe (N/mm)": [rigidez],
    "R²": [r2],
    "Força mínima (%Fmáx)": [y_perc_lim[0]],
    "Força máxima (%Fmáx)": [y_perc_lim[1]],
    "Deformação mínima (mm)": [x_lim[0]],
    "Deformação máxima (mm)": [x_lim[1]]
})

st.subheader("📊 Resultados do trecho filtrado")
st.dataframe(df_resultados.style.format({
    "Força Máxima (N)": "{:.2f}",
    "ΔF/Δe (N/mm)": "{:.2f}",
    "R²": "{:.4f}",
    "Força mínima (%Fmáx)": "{:.0f}",
    "Força máxima (%Fmáx)": "{:.0f}",
    "Deformação mínima (mm)": "{:.4f}",
    "Deformação máxima (mm)": "{:.4f}"
}), use_container_width=True)

# --- Gráfico principal ---
fig = go.Figure()

# Curva completa
fig.add_trace(go.Scatter(
    x=df_sel["deformacao_mm"],
    y=df_sel["forca_n"],
    mode="lines",
    name="Curva completa",
    line=dict(color="royalblue")
))

# Pontos da regressão
fig.add_trace(go.Scatter(
    x=df_reg["deformacao_mm"],
    y=df_reg["forca_n"],
    mode="markers",
    name="Pontos da regressão",
    marker=dict(color="orange", size=6)
))

# Reta da regressão
if not np.isnan(slope):
    x_fit = np.linspace(df_reg["deformacao_mm"].min(), df_reg["deformacao_mm"].max(), 100)
    y_fit = slope * x_fit + intercept
    fig.add_trace(go.Scatter(
        x=x_fit,
        y=y_fit,
        mode="lines",
        name=f"Regressão Linear (R²={r2:.4f})",
        line=dict(color="red", dash="dot")
    ))

# --- Linhas de referência e área sombreada ---
fig.add_hline(
    y=fmin_y,
    line=dict(color="green", dash="dash"),
    annotation_text=f"{y_perc_lim[0]:.0f}% Fmáx ({fmin_y:.1f} N)",
    annotation_position="top left"
)
fig.add_hline(
    y=fmax_y,
    line=dict(color="red", dash="dash"),
    annotation_text=f"{y_perc_lim[1]:.0f}% Fmáx ({fmax_y:.1f} N)",
    annotation_position="bottom left"
)
fig.add_vline(
    x=x_lim[0],
    line=dict(color="gray", dash="dot"),
    annotation_text=f"{x_lim[0]:.3f} mm",
    annotation_position="top right"
)
fig.add_vline(
    x=x_lim[1],
    line=dict(color="gray", dash="dot"),
    annotation_text=f"{x_lim[1]:.3f} mm",
    annotation_position="top left"
)

# --- Área sombreada verde translúcida ---
fig.add_shape(
    type="rect",
    x0=x_lim[0],
    x1=x_lim[1],
    y0=fmin_y,
    y1=fmax_y,
    fillcolor="rgba(0, 200, 0, 0.3)",
    line=dict(width=0),
    layer="below"
)

fig.update_layout(
    title=f"Força × Deformação ({arquivo_sel})",
    xaxis_title="Deformação (mm)",
    yaxis_title="Força (N)",
    template="plotly_white",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# --- Dados compilados ---
st.subheader("📋 Dados compilados")
st.dataframe(df_compilado, use_container_width=True)
