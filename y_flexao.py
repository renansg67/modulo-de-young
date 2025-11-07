import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from io import StringIO

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Módulo de Flexão (E₀) - Comparativo", layout="wide")
st.title("🧱 Módulo de Elasticidade na Flexão Estática (E₀) - NBR 7190")
st.caption("Busca automática do melhor intervalo com R² ≥ 0.95 dentro da janela de 10–40% da Fmáx.")

# --- TIPO DE ENSAIO ---
st.sidebar.subheader("⚙️ Tipo de Ensaio")
tipo_ensaio = st.sidebar.radio("Selecione o tipo de ensaio:", ["3 Pontos", "4 Pontos"])

if tipo_ensaio == "3 Pontos":
    st.latex(r"E_0 = \frac{1}{4}\left(\frac{L}{b}\right)^{3}\left(\frac{\Delta F}{\Delta e}\right)\left(\frac{1}{h}\right)")
else:
    st.latex(r"E_0 = \frac{23}{108}\left(\frac{L}{h}\right)^{3}\left(\frac{\Delta F}{\Delta e}\right)\left(\frac{1}{b}\right)")

# --- SIDEBAR: Upload e parâmetros ---
st.sidebar.header("📄 Upload e parâmetros")
uploaded_files = st.sidebar.file_uploader("Selecione os arquivos CSV", type=["csv"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Aguardando upload de arquivos CSV...")
    st.stop()

# --- GEOMETRIA ---
st.sidebar.markdown("---")
L = st.sidebar.number_input("Vão (L) [mm]", value=1200.0)
h = st.sidebar.number_input("Altura (h) [mm]", value=100.0)
b = st.sidebar.number_input("Largura (b) [mm]", value=50.0)

# --- INTERVALOS ---
st.sidebar.markdown("---")
faixa = st.sidebar.slider("Faixa inicial (% da Fmáx)", 0, 100, (10, 40))
deform_min, deform_max = st.sidebar.slider("Limite de deformação (mm)", 0.0, 200.0, (0.0, 200.0), 0.1)

# --- FUNÇÃO DE LEITURA ---
@st.cache_data
def carregar_dados(uploaded_file, header_row=50):
    try:
        string_data = StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore"))
        df = pd.read_csv(string_data, sep=",", header=None, decimal=".")
        df = df.dropna(how="all")
        df = df.iloc[header_row:, :].reset_index(drop=True)
        df = df.iloc[:, [1, 2]].copy()
        df.columns = ["Forca_kN", "Deformacao_mm"]
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        return df
    except Exception as e:
        st.error(f"Erro ao ler {uploaded_file.name}: {e}")
        return pd.DataFrame()

# --- FUNÇÃO PARA BUSCAR MELHOR INTERVALO ---
def buscar_intervalo_otimo(df, Fmin, Flim, deform_min, deform_max, r2_min=0.95, reducao_passo=0.1):
    df_faixa = df[(df["Forca_kN"] >= Fmin) & (df["Forca_kN"] <= Flim) &
                  (df["Deformacao_mm"] >= deform_min) & (df["Deformacao_mm"] <= deform_max)].copy()
    if len(df_faixa) < 5:
        return None, None, None, None, False

    n_pontos = len(df_faixa)
    melhor_r2 = -np.inf
    melhor_slope, melhor_intercept = None, None
    melhor_df = df_faixa.copy()
    automatico = False

    for frac in np.arange(1.0, 0.4, -reducao_passo):
        n_sub = max(3, int(n_pontos * frac))
        sub_df = df_faixa.iloc[:n_sub]
        slope, intercept, r_value, _, _ = stats.linregress(sub_df["Deformacao_mm"], sub_df["Forca_kN"])
        r2 = r_value**2
        if r2 > melhor_r2:
            melhor_r2 = r2
            melhor_slope, melhor_intercept = slope, intercept
            melhor_df = sub_df.copy()
        if r2 >= r2_min:
            automatico = True
            break

    return melhor_slope, melhor_intercept, melhor_df, melhor_r2, automatico

# --- FUNÇÃO DE PROCESSAMENTO ---
def processar_ensaio(nome_arquivo, df, L, h, b, faixa, tipo_ensaio, deform_min, deform_max):
    if df.empty:
        return None, df, df, None, None, None, None, None

    Fmax = df["Forca_kN"].max()
    Fmin = (faixa[0] / 100) * Fmax
    Flim = (faixa[1] / 100) * Fmax

    slope, intercept, df_reg, r2, automatico = buscar_intervalo_otimo(df, Fmin, Flim, deform_min, deform_max)

    if df_reg is None or slope is None:
        return None, df, df, None, None, Fmin, Flim, None

    delta_F_delta_e_SI = slope * 1e6
    L_SI, h_SI, b_SI = L/1000, h/1000, b/1000

    if tipo_ensaio == "3 Pontos":
        E0_pa = (1/4) * ((L_SI/b_SI)**3) * delta_F_delta_e_SI * (1/h_SI)
    else:
        E0_pa = (23/108) * ((L_SI/h_SI)**3) * delta_F_delta_e_SI * (1/b_SI)

    E0_gpa = E0_pa / 1e9

    return {
        "Ensaio": nome_arquivo,
        "Tipo": tipo_ensaio,
        "Fmax (kN)": Fmax,
        "ΔF/Δe (kN/mm)": slope,
        "R²": r2,
        "E₀ (GPa)": E0_gpa,
        "Ajuste": "Automático" if automatico else "Manual"
    }, df, df_reg, slope, intercept, Fmin, Flim, automatico

# --- LOOP DE PROCESSAMENTO ---
resultados, dados_abas = [], {}

for f in uploaded_files:
    df = carregar_dados(f)
    nome = f.name
    resultado, df_all, df_reg, slope, intercept, Fmin, Flim, automatico = processar_ensaio(
        nome, df, L, h, b, faixa, tipo_ensaio, deform_min, deform_max
    )
    if resultado:
        resultados.append(resultado)
        dados_abas[nome] = (df_all, df_reg, slope, intercept, Fmin, Flim)
    else:
        resultados.append({"Ensaio": nome, "E₀ (GPa)": None})

# --- TABELA DE RESULTADOS ---
df_result = pd.DataFrame(resultados)
st.dataframe(df_result.style.format({
    "Fmax (kN)": "{:.3f}",
    "ΔF/Δe (kN/mm)": "{:.4f}",
    "R²": "{:.4f}",
    "E₀ (GPa)": "{:.2f}"
}, na_rep="-"), use_container_width=True)

# --- SELEÇÃO E PLOTAGEM ---
ensaios_validos = df_result[df_result["E₀ (GPa)"].notna()]["Ensaio"]

if not ensaios_validos.empty:
    escolha = st.selectbox("🔍 Escolha o ensaio:", ensaios_validos)
    df_all, df_reg, slope, intercept, Fmin, Flim = dados_abas[escolha]

    fig = go.Figure()

    # Curva completa
    fig.add_trace(go.Scatter(
        x=df_all["Deformacao_mm"], y=df_all["Forca_kN"],
        mode='lines', name='Curva completa',
        line=dict(color='lightgray', width=1.5)
    ))

    # Pontos usados na regressão
    fig.add_trace(go.Scatter(
        x=df_reg["Deformacao_mm"], y=df_reg["Forca_kN"],
        mode='markers', name='Usado na regressão',
        marker=dict(size=6, color='blue')
    ))

    # Linha de regressão
    if slope is not None:
        x_fit = np.linspace(df_reg["Deformacao_mm"].min(), df_reg["Deformacao_mm"].max(), 50)
        fig.add_trace(go.Scatter(
            x=x_fit, y=slope * x_fit + intercept,
            mode='lines', name='Regressão Linear',
            line=dict(color='red', width=3, dash='dash')
        ))

    fig.add_hline(y=Fmin, line_color="green", line_dash="dot", annotation_text=f"{faixa[0]}% Fmáx")
    fig.add_hline(y=Flim, line_color="orange", line_dash="dot", annotation_text=f"{faixa[1]}% Fmáx")

    fig.update_layout(
        xaxis_title="Deformação (mm)",
        yaxis_title="Força (kN)",
        template="plotly_white",
        hovermode="x unified",
        height=550
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Nenhum ensaio válido encontrado.")

# --- MÉDIA E DESVIO ---
if "E₀ (GPa)" in df_result.columns and df_result["E₀ (GPa)"].notna().any():
    media = df_result["E₀ (GPa)"].mean()
    desvio = df_result["E₀ (GPa)"].std()
    st.markdown(f"**Média E₀:** {media:.2f} GPa  **Desvio padrão:** {desvio:.2f} GPa")
