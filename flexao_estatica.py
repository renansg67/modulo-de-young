import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from io import StringIO

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Módulo de Flexão (E₀) - Simplificado", layout="wide")
st.title("🧱 Módulo de Elasticidade na Flexão Estática (E₀) - NBR 7190")
st.caption("Calcula automaticamente o módulo de elasticidade na flexão com base no trecho linear (10%–40% de Fmax).")

# --- SIDEBAR ---
st.sidebar.header("📄 Arquivo e parâmetros")
uploaded_file = st.sidebar.file_uploader("Selecione o arquivo CSV", type=["csv"])
L = st.sidebar.number_input("Vão (L) [mm]", value=1200.0, min_value=1.0)
h = st.sidebar.number_input("Altura (h) [mm]", value=100.0, min_value=1.0)
b = st.sidebar.number_input("Largura (b) [mm]", value=50.0, min_value=1.0)

if not uploaded_file:
    st.info("Aguardando o upload do arquivo de ensaio (.csv)...")
    st.stop()

# --- LEITURA SIMPLIFICADA ---
try:
    text = uploaded_file.read().decode('latin-1')
    text = text.replace('","', '\t').replace('"', '').replace(',', '.')
    df = pd.read_csv(StringIO(text), sep='\t', header=None, skiprows=50)
    df = df.iloc[:, [1, 2]]
    df.columns = ['Forca_kN', 'Deformacao_mm']
    df = df.dropna().reset_index(drop=True)
except Exception as e:
    st.error(f"Erro ao ler o arquivo: {e}")
    st.stop()

# --- FILTRO DE DADOS VÁLIDOS ---
df = df[df["Deformacao_mm"] > 0].reset_index(drop=True)
if df.empty:
    st.error("Nenhuma deformação positiva encontrada. Verifique a coluna de dados.")
    st.stop()

# --- DEFINIÇÃO DE LIMITES (10%–40% Fmax) ---
Fmax = df["Forca_kN"].max()
Fmin = 0.10 * Fmax
Flim = 0.40 * Fmax
df_reg = df[(df["Forca_kN"] >= Fmin) & (df["Forca_kN"] <= Flim)]
if len(df_reg) < 2:
    st.error("Poucos pontos entre 10% e 40% de Fmax para regressão.")
    st.stop()

# --- REGRESSÃO LINEAR ---
slope, intercept, r_value, _, _ = stats.linregress(df_reg["Deformacao_mm"], df_reg["Forca_kN"])
r2 = r_value**2

# --- CÁLCULO DE E₀ ---
# Fórmula: E0 = (23/108) * (L/h)^3 * (ΔF/Δe) * (1/b)
E0_Pa = (23/108) * ((L/h)**3) * (slope * 1e6) * (1/(b/1000))
E0_GPa = E0_Pa / 1e9

# --- RESULTADOS ---
st.markdown("## 🏁 Resultados Principais")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Fmáx (kN)", f"{Fmax:.3f}")
col2.metric("ΔF/Δe (kN/mm)", f"{slope:.4f}")
col3.metric("R²", f"{r2:.4f}")
col4.metric("E₀ (GPa)", f"{E0_GPa:.2f}")

# --- GRÁFICO INTERATIVO ---
st.markdown("## 📈 Curva Força × Deformação com Trecho de Linearização")

fig = go.Figure()

# Curva completa
fig.add_trace(go.Scatter(
    x=df["Deformacao_mm"], y=df["Forca_kN"],
    mode='lines', name='Curva completa', line=dict(color='gray', width=1.5)
))

# Trecho usado para regressão
fig.add_trace(go.Scatter(
    x=df_reg["Deformacao_mm"], y=df_reg["Forca_kN"],
    mode='markers', name='Trecho 10–40% Fmax',
    marker=dict(size=6, color='blue', symbol='circle')
))

# Reta da regressão
x_line = np.linspace(df_reg["Deformacao_mm"].min(), df_reg["Deformacao_mm"].max(), 50)
fig.add_trace(go.Scatter(
    x=x_line, y=slope*x_line + intercept,
    mode='lines', name='Regressão Linear',
    line=dict(color='red', width=3, dash='dash')
))

# Linhas de referência
fig.add_hline(y=Fmin, line_color="green", line_dash="dot", annotation_text="10% Fmax", annotation_position="top left")
fig.add_hline(y=Flim, line_color="orange", line_dash="dot", annotation_text="40% Fmax", annotation_position="bottom left")

fig.update_layout(
    xaxis_title="Deformação (mm)",
    yaxis_title="Força (kN)",
    template="plotly_white",
    hovermode="x unified",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# --- MOSTRA TABELA DE REGRESSÃO ---
with st.expander("📊 Ver dados usados na regressão (10–40% Fmax):"):
    st.dataframe(df_reg)
