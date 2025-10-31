import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Módulo de Flexão (E₀) - Comparativo", layout="wide")
st.title("🧱 Módulo de Elasticidade na Flexão Estática (E₀) - NBR 7190")
st.caption("Lê arquivos CSV de ensaio de flexão. Calcula E₀ e gera um resumo consolidado.")

# --- ESCOLHA DO TIPO DE ENSAIO ---
st.sidebar.subheader("⚙️ Tipo de Ensaio")
tipo_ensaio = st.sidebar.radio("Selecione o tipo de ensaio:", ["3 Pontos", "4 Pontos"])

st.markdown("### Fórmula utilizada (mude o tipo de ensaio na barra lateral)")
if tipo_ensaio == "3 Pontos":
    st.latex(r"E_0 = \frac{1}{4}\left(\frac{L}{b}\right)^{3}\left(\frac{\Delta F}{\Delta e}\right)\left(\frac{1}{h}\right)")
else:
    st.latex(r"E_0 = \frac{23}{108}\left(\frac{L}{h}\right)^{3}\left(\frac{\Delta F}{\Delta e}\right)\left(\frac{1}{b}\right)")

# --- SIDEBAR: Upload e parâmetros ---
st.sidebar.header("📄 Upload e parâmetros")

uploaded_files = st.sidebar.file_uploader(
    "Selecione os arquivos CSV (uma aba por arquivo)", 
    type=["csv"], 
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Aguardando upload de arquivos CSV...")
    st.stop()

# Parâmetros geométricos
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Parâmetros geométricos")
L = st.sidebar.number_input("Vão (L) [mm]", value=1200.0, min_value=1.0)
h = st.sidebar.number_input("Altura (h) [mm]", value=100.0, min_value=1.0)
b = st.sidebar.number_input("Largura (b) [mm]", value=50.0, min_value=1.0)

# Slider de faixa de linearização
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Intervalo de linearização (% da Fmáx)")
faixa = st.sidebar.slider("Selecione a faixa (mín - máx)", 0, 100, (10, 40), 1)

# --- FUNÇÃO DE PROCESSAMENTO ---
def processar_csv(nome_arquivo, df, L, h, b, faixa, tipo_ensaio):
    try:
        df = df.iloc[50:, :].reset_index(drop=True)  # remove cabeçalho
        df = df.iloc[:, [1, 2]].copy()  # mantém apenas força e deformação
        df.columns = ['Forca_kN', 'Deformacao_mm']
        df["Forca_kN"] = pd.to_numeric(df["Forca_kN"], errors='coerce')
        df["Deformacao_mm"] = pd.to_numeric(df["Deformacao_mm"], errors='coerce')
        df = df.dropna()
        df = df[df["Deformacao_mm"] > 0].reset_index(drop=True)

        if df.empty:
            return None, None, None, None, None, None, None

        Fmax = df["Forca_kN"].max()
        Fmin = (faixa[0] / 100) * Fmax
        Flim = (faixa[1] / 100) * Fmax
        df_reg = df[(df["Forca_kN"] >= Fmin) & (df["Forca_kN"] <= Flim)]

        if len(df_reg) < 2:
            return None, None, None, None, None, None, None

        slope, intercept, r_value, _, _ = stats.linregress(df_reg["Deformacao_mm"], df_reg["Forca_kN"])
        r2 = r_value**2
        delta_F_delta_e_SI = slope * 1e6
        L_SI, h_SI, b_SI = L / 1000, h / 1000, b / 1000

        if tipo_ensaio == "3 Pontos":
            E0_pa = (1/4) * ((L_SI/b_SI)**3) * delta_F_delta_e_SI * (1/h_SI)
        else:
            E0_pa = (23/108) * ((L_SI/h_SI)**3) * delta_F_delta_e_SI * (1/b_SI)

        E0_gpa = E0_pa / 1e9

        return {
            "Ensaio": nome_arquivo,
            "Tipo de Ensaio": tipo_ensaio,
            "Fmax (kN)": Fmax,
            "ΔF/Δe (kN/mm)": slope,
            "R²": r2,
            "E₀ (GPa)": E0_gpa
        }, df, df_reg, slope, intercept, Fmin, Flim
    except Exception:
        return None, None, None, None, None, None, None

# --- PROCESSAMENTO DE TODOS OS CSVs ---
resultados = []
dados_abas = {}

for file in uploaded_files:
    df = pd.read_csv(file)
    nome_arquivo = file.name
    resultado, df_all, df_reg, slope, intercept, Fmin, Flim = processar_csv(nome_arquivo, df, L, h, b, faixa, tipo_ensaio)
    if resultado:
        resultados.append(resultado)
        dados_abas[nome_arquivo] = (df_all, df_reg, slope, intercept, Fmin, Flim)
    else:
        resultados.append({
            "Ensaio": nome_arquivo,
            "Tipo de Ensaio": None,
            "Fmax (kN)": None,
            "ΔF/Δe (kN/mm)": None,
            "R²": None,
            "E₀ (GPa)": None
        })

# --- TABELA DE RESULTADOS ---
df_resultados = pd.DataFrame(resultados)
media = df_resultados["E₀ (GPa)"].mean(skipna=True)
desvio = df_resultados["E₀ (GPa)"].std(skipna=True)

# --- SELEÇÃO DE ENSAIO ---
ensaios_validos = df_resultados[df_resultados["E₀ (GPa)"].notna()]["Ensaio"]
if not ensaios_validos.empty:
    selected_name = st.selectbox("🔍 Escolha o ensaio para análise individual:", ensaios_validos)
    df_all, df_reg, slope, intercept, Fmin, Flim = dados_abas[selected_name]

    res_sel = df_resultados[df_resultados["Ensaio"] == selected_name].iloc[0]
    st.markdown(f"## 🏁 Resultados - {selected_name}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tipo de Ensaio", res_sel["Tipo de Ensaio"])
    col2.metric("Fmáx (kN)", f"{res_sel['Fmax (kN)']:.3f}")
    col3.metric("ΔF/Δe (kN/mm)", f"{res_sel['ΔF/Δe (kN/mm)']:.4f}")
    col4.metric("R²", f"{res_sel['R²']:.4f}")
    st.markdown(f"### E₀ = **{res_sel['E₀ (GPa)']:.2f} GPa**")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_all["Deformacao_mm"], y=df_all["Forca_kN"], mode='lines', name='Curva completa', line=dict(color='gray', width=1.5)))
    fig.add_trace(go.Scatter(x=df_reg["Deformacao_mm"], y=df_reg["Forca_kN"], mode='markers', name=f'Trecho {faixa[0]}–{faixa[1]}% Fmax', marker=dict(size=6, color='blue')))
    x_line = np.linspace(df_reg["Deformacao_mm"].min(), df_reg["Deformacao_mm"].max(), 50)
    fig.add_trace(go.Scatter(x=x_line, y=slope*x_line + intercept, mode='lines', name='Regressão Linear', line=dict(color='red', width=3, dash='dash')))
    fig.add_hline(y=(faixa[0]/100)*res_sel['Fmax (kN)'], line_color="green", line_dash="dot", annotation_text=f"{faixa[0]}% Fmax", annotation_position="top left")
    fig.add_hline(y=(faixa[1]/100)*res_sel['Fmax (kN)'], line_color="orange", line_dash="dot", annotation_text=f"{faixa[1]}% Fmax", annotation_position="bottom left")
    fig.update_layout(xaxis_title="Deformação (mm)", yaxis_title="Força (kN)", template="plotly_white", hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Nenhum ensaio válido para análise individual.")

# --- RESULTADOS COMPILADOS ---
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
st.sidebar.download_button(label="⬇️ Baixar Resultados Consolidados (.csv)", data=csv, file_name="resultados_flexao.csv", mime="text/csv")
