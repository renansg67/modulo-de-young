import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from io import StringIO

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Módulo de Flexão (E₀) - Comparativo", layout="wide")
st.title("🧱 Módulo de Elasticidade na Flexão Estática (E₀) - NBR 7190")
st.caption("Analisa múltiplos arquivos de ensaio de flexão, calcula o módulo E₀ e gera um resumo consolidado.")

# --- SIDEBAR ---
st.sidebar.header("📄 Upload e parâmetros")

uploaded_files = st.sidebar.file_uploader(
    "Selecione um ou mais arquivos CSV", 
    type=["csv"], 
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Aguardando o upload de arquivos de ensaio (.csv)...")
    st.stop()

# Parâmetros geométricos
L = st.sidebar.number_input("Vão (L) [mm]", value=1200.0, min_value=1.0)
h = st.sidebar.number_input("Altura (h) [mm]", value=100.0, min_value=1.0)
b = st.sidebar.number_input("Largura (b) [mm]", value=50.0, min_value=1.0)

# Slider de faixa de linearização
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Intervalo de linearização (% da Fmáx)")
faixa = st.sidebar.slider("Selecione a faixa (mín - máx)", 0, 100, (10, 40), 1)

# --- FUNÇÃO DE PROCESSAMENTO ---
def processar_arquivo(file, L, h, b, faixa):
    try:
        text = file.read().decode('latin-1')
        text = text.replace('","', '\t').replace('"', '').replace(',', '.')
        df = pd.read_csv(StringIO(text), sep='\t', header=None, skiprows=50)
        df = df.iloc[:, [1, 2]]
        df.columns = ['Forca_kN', 'Deformacao_mm']
        df = df.dropna().reset_index(drop=True)
        df = df[df["Deformacao_mm"] > 0].reset_index(drop=True)
        if df.empty:
            return None

        # Determina o intervalo
        Fmax = df["Forca_kN"].max()
        Fmin = (faixa[0] / 100) * Fmax
        Flim = (faixa[1] / 100) * Fmax
        df_reg = df[(df["Forca_kN"] >= Fmin) & (df["Forca_kN"] <= Flim)]
        if len(df_reg) < 2:
            return None

        # Regressão linear
        slope, intercept, r_value, _, _ = stats.linregress(df_reg["Deformacao_mm"], df_reg["Forca_kN"])
        r2 = r_value**2

        # Cálculo de E₀ (GPa)
        E0_Pa = (23/108) * ((L/h)**3) * (slope * 1e6) * (1/(b/1000))
        E0_GPa = E0_Pa / 1e9

        return {
            "Arquivo": file.name,
            "Fmax (kN)": Fmax,
            "ΔF/Δe (kN/mm)": slope,
            "R²": r2,
            "E₀ (GPa)": E0_GPa
        }
    except Exception:
        return None

# --- PROCESSA TODOS OS ARQUIVOS ---
resultados = []
for file in uploaded_files:
    resultado = processar_arquivo(file, L, h, b, faixa)
    if resultado:
        resultados.append(resultado)

if not resultados:
    st.error("Nenhum arquivo pôde ser processado corretamente.")
    st.stop()

# --- TABELA DE RESULTADOS ---
df_resultados = pd.DataFrame(resultados)
media = df_resultados["E₀ (GPa)"].mean()
desvio = df_resultados["E₀ (GPa)"].std()

# --- MOSTRA O ARQUIVO SELECIONADO ---
selected_name = st.selectbox("🔍 Escolha o arquivo para análise individual:", df_resultados["Arquivo"])
file = next(f for f in uploaded_files if f.name == selected_name)

# Reprocessa o arquivo selecionado (para gerar gráfico)
_ = file.seek(0)
resultado = processar_arquivo(file, L, h, b, faixa)

# --- RESULTADOS INDIVIDUAIS ---
st.markdown(f"## 🏁 Resultados - {selected_name}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Fmáx (kN)", f"{resultado['Fmax (kN)']:.3f}")
col2.metric("ΔF/Δe (kN/mm)", f"{resultado['ΔF/Δe (kN/mm)']:.4f}")
col3.metric("R²", f"{resultado['R²']:.4f}")
col4.metric("E₀ (GPa)", f"{resultado['E₀ (GPa)']:.2f}")

# --- GRÁFICO ---
_ = file.seek(0)
text = file.read().decode('latin-1')
text = text.replace('","', '\t').replace('"', '').replace(',', '.')
df = pd.read_csv(StringIO(text), sep='\t', header=None, skiprows=50)
df = df.iloc[:, [1, 2]]
df.columns = ['Forca_kN', 'Deformacao_mm']
df = df.dropna().reset_index(drop=True)
df = df[df["Deformacao_mm"] > 0].reset_index(drop=True)
Fmax = df["Forca_kN"].max()
Fmin = (faixa[0] / 100) * Fmax
Flim = (faixa[1] / 100) * Fmax
df_reg = df[(df["Forca_kN"] >= Fmin) & (df["Forca_kN"] <= Flim)]
slope = resultado["ΔF/Δe (kN/mm)"]
intercept = df_reg["Forca_kN"].mean() - slope * df_reg["Deformacao_mm"].mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Deformacao_mm"], y=df["Forca_kN"], mode='lines',
                         name='Curva completa', line=dict(color='gray', width=1.5)))
fig.add_trace(go.Scatter(x=df_reg["Deformacao_mm"], y=df_reg["Forca_kN"], mode='markers',
                         name=f'Trecho {faixa[0]}–{faixa[1]}% Fmax',
                         marker=dict(size=6, color='blue')))
x_line = np.linspace(df_reg["Deformacao_mm"].min(), df_reg["Deformacao_mm"].max(), 50)
fig.add_trace(go.Scatter(x=x_line, y=slope*x_line + intercept, mode='lines',
                         name='Regressão Linear', line=dict(color='red', width=3, dash='dash')))
fig.add_hline(y=Fmin, line_color="green", line_dash="dot",
              annotation_text=f"{faixa[0]}% Fmax", annotation_position="top left")
fig.add_hline(y=Flim, line_color="orange", line_dash="dot",
              annotation_text=f"{faixa[1]}% Fmax", annotation_position="bottom left")
fig.update_layout(
    xaxis_title="Deformação (mm)",
    yaxis_title="Força (kN)",
    template="plotly_white",
    hovermode="x unified",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# --- TABELA COMPILADA ---
st.markdown("## 📊 Resultados Consolidados")
st.dataframe(df_resultados.style.format({
    "Fmax (kN)": "{:.3f}",
    "ΔF/Δe (kN/mm)": "{:.4f}",
    "R²": "{:.4f}",
    "E₀ (GPa)": "{:.2f}"
}))

st.markdown(f"**Média de E₀:** {media:.2f} GPa  **Desvio Padrão:** {desvio:.2f} GPa")

# --- DOWNLOAD DO CSV ---
csv = df_resultados.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="⬇️ Baixar Resultados Consolidados (.csv)",
    data=csv,
    file_name="resultados_flexao.csv",
    mime="text/csv"
)
