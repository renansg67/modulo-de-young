import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from scipy import stats

# Constantes de conversão
MEGA_TO_SI = 1e6 # 1 MPa = 10^6 Pa
MM_TO_SI = 1e-3 # 1 mm = 10^-3 m
MM2_TO_SI = 1e-6 # 1 mm² = 10^-6 m² 

# Constantes da Norma (Novo)
PERCENTUAL_MIN_MODULO = 0.10 # 10% da tensão máxima
PERCENTUAL_MAX_MODULO = 0.40 # 40% da tensão máxima

# ------------------------------------------------------------
# CONFIGURAÇÕES INICIAIS
# ------------------------------------------------------------
st.set_page_config(page_title="Análise Módulo de Elasticidade (Norma)", layout="wide")
st.markdown("### 🧱 Módulo de Elasticidade (10% - 40% da Tensão Máxima)")
st.markdown(f"#### **Cálculo baseado na norma:** O Módulo $E$ é obtido pela regressão linear na região entre **{PERCENTUAL_MIN_MODULO*100:.0f}%** e **{PERCENTUAL_MAX_MODULO*100:.0f}%** da Tensão Máxima do CP.")

# ------------------------------------------------------------
# CONSTANTES E DEFINIÇÕES
# ------------------------------------------------------------
L0_MM = 50.0 # comprimento inicial do extensômetro (mm)
L0_SI = L0_MM * MM_TO_SI # L0 em metros
pasta_ensaios = "ensaios" # Assumindo que os arquivos CSV dos ensaios estão nesta pasta
COLUNAS_GERAL = [
    "cp", "nome", "largura", "espessura", 
    "ret_ext", "area", "forca_max", "tensao_max", "energia"
]

# ------------------------------------------------------------
# CARREGAR ARQUIVO GERAL (Para obter as dimensões)
# ------------------------------------------------------------
try:
    geral = pd.read_csv(
        "geral.csv", 
        sep=";", 
        decimal=",", 
        encoding_errors="ignore",
        header=None,       
        skiprows=1, # Corrigido para pular apenas a primeira linha
        names=COLUNAS_GERAL 
    )
    
    # GARANTINDO O TIPO NUMÉRICO
    geral["area"] = pd.to_numeric(geral["area"], errors='coerce')
    geral["largura"] = pd.to_numeric(geral["largura"], errors='coerce')
    geral["espessura"] = pd.to_numeric(geral["espessura"], errors='coerce')
    
    # Extrai o número do CP para auxiliar na filtragem/ordenacao
    geral["cp_num"] = geral["cp"].str.extract(r"(\d+)").astype(str)
except FileNotFoundError:
    st.error("❌ Arquivo 'geral.csv' não encontrado no diretório do projeto.")
    st.stop()
except Exception as e:
    st.error(f"❌ Erro na leitura do arquivo general.csv: {e}")
    st.stop()

if geral.empty or not {"cp", "area", "cp_num"}.issubset(geral.columns):
    st.error("❌ O arquivo general.csv não pôde ser lido corretamente.")
    st.stop()

# ------------------------------------------------------------
# LISTA DE CPs DISPONÍVEIS E ORDENAÇÃO
# ------------------------------------------------------------
cp_disponiveis = geral[geral['area'].notna() & (geral['area'] > 0)]["cp"].tolist()
if not cp_disponiveis:
    st.error("❌ Nenhum Corpo de Prova com área válida encontrado.")
    st.stop()

# Ordenação Numérica dos CPs
cp_opcoes = sorted(
    cp_disponiveis, 
    key=lambda cp: int(cp.split()[-1]) if cp.split()[-1].isdigit() else 0
)
cp_opcoes.insert(0, "Selecione um CP")

# ------------------------------------------------------------
# SELEÇÃO DO CP
# ------------------------------------------------------------
st.sidebar.header('Seleção de Corpo de Prova')
cp_selecionado = st.sidebar.selectbox("Selecione o CP:", cp_opcoes)

if cp_selecionado == "Selecione um CP":
    st.info("Selecione um Corpo de Prova na barra lateral.")
    st.stop()

cp_info = geral[geral["cp"] == cp_selecionado].iloc[0]
cp_num = cp_info["cp_num"]
area = cp_info["area"] # Área em mm²

# ------------------------------------------------------------
# CARREGAMENTO E PROCESSAMENTO DOS DADOS DO ENSAIO
# ------------------------------------------------------------
caminho_csv = os.path.join(pasta_ensaios, f"{cp_num}.csv")
if not os.path.exists(caminho_csv):
    st.warning(f"⚠️ Arquivo do ensaio '{cp_num}.csv' não encontrado. Verifique a pasta '{pasta_ensaios}'.")
    st.stop()

try:
    df_ensaio = pd.read_csv(caminho_csv, sep=";", decimal=",", encoding_errors="ignore")
    df_ensaio.columns = ["tempo_s", "deformacao_mm", "forca_n"]
    df_ensaio["tempo_s"] = pd.to_numeric(df_ensaio["tempo_s"], errors="coerce")
    df_ensaio["deformacao_mm"] = pd.to_numeric(df_ensaio["deformacao_mm"], errors="coerce")
    df_ensaio["forca_n"] = pd.to_numeric(df_ensaio["forca_n"], errors="coerce")
    df_ensaio = df_ensaio.dropna(subset=["tempo_s", "deformacao_mm", "forca_n"]).sort_values(by="tempo_s").reset_index(drop=True)
    
except Exception as e:
    st.error(f"Erro ao ler ou processar o arquivo {caminho_csv}: {e}")
    st.stop()

if df_ensaio.empty:
    st.error(f"O arquivo {caminho_csv} não contém dados válidos para plotagem.")
    st.stop()

# ------------------------------------------------------------
# CÁLCULOS (Tensão em Pascal e Deformação Específica)
# ------------------------------------------------------------
if area <= 0 or pd.isna(area):
      st.error(f"❌ Área do CP {cp_selecionado} inválida ou zero ({area} mm²).")
      st.stop()

# Tensão (N/mm²) * 10^6 = Pa.
df_ensaio["tensao_pa"] = (df_ensaio["forca_n"] / area) * MEGA_TO_SI 
# Deformação Específica (mm/mm)
df_ensaio["deformacao_especifica"] = -df_ensaio["deformacao_mm"] / L0_MM 

# ------------------------------------------------------------
# CÁLCULO E FILTRAGEM BASEADOS NA NORMA (NOVA LÓGICA)
# ------------------------------------------------------------

# 1. Determinar a Tensão Máxima do Ensaio
tensao_max_pa = df_ensaio["tensao_pa"].max()

if tensao_max_pa <= 0:
    st.warning("⚠️ Tensão máxima nula ou negativa. Não é possível aplicar o filtro da norma.")
    st.stop()

# 2. Definir os Limites de Tensão da Região Elástica
sigma_inferior_pa = PERCENTUAL_MIN_MODULO * tensao_max_pa
sigma_superior_pa = PERCENTUAL_MAX_MODULO * tensao_max_pa

# 3. Filtrar os Dados
df_filtrado = df_ensaio[
    (df_ensaio["tensao_pa"] >= sigma_inferior_pa) & 
    (df_ensaio["tensao_pa"] <= sigma_superior_pa)
].copy()

# ------------------------------------------------------------
# CÁLCULO DO MÓDULO DE ELASTICIDADE (E) E R²
# ------------------------------------------------------------
E_modulo = np.nan
r_squared = np.nan
intercept = np.nan
regressao_info = "Não foi possível calcular. Intervalo de tensão (10%-40%) não contém dados."

if len(df_filtrado) >= 2:
    try:
        # A regressão usa Deformação (adimensional) e Tensão (Pa) -> E_modulo sai em Pa
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_filtrado["deformacao_especifica"], df_filtrado["tensao_pa"]
        )
        E_modulo = slope # E_modulo agora está em Pa
        r_squared = r_value**2
        
        # Apenas para mostrar a equação no Streamlit
        regressao_info = f"\\sigma\\text{{(Pa)}} = ({E_modulo:,.2e}) \\cdot \\varepsilon + ({intercept:,.2e})"
    except ValueError:
        regressao_info = "Erro ao calcular regressão: dados inválidos no intervalo filtrado."
        
# ------------------------------------------------------------
# EXIBIÇÃO DOS RESULTADOS NA SIDEBAR
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Resultado do Módulo pela Norma")

st.sidebar.metric(
    label="Tensão Máxima ($\sigma_{máx}$)",
    value=f"{tensao_max_pa:,.2e} Pa",
    help="Tensão máxima atingida no ensaio (em Pascal)."
)

st.sidebar.metric(
    label="Módulo de Elasticidade ($E$)",
    value=f"{E_modulo:,.5e} Pa" if not np.isnan(E_modulo) else "N/A",
    help="Inclinação da regressão linear na faixa de 10% a 40% da tensão máxima (em Pascal)."
)
st.sidebar.metric(
    label="Coeficiente de Determinação ($R^2$)",
    value=f"{r_squared:,.4f}" if not np.isnan(r_squared) else "N/A",
    help="Mede o quão bem a regressão linear se ajusta aos dados filtrados."
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Filtro Automático Aplicado:")
st.sidebar.write(f"$\sigma$ Inferior (10%): **{sigma_inferior_pa:,.2e} Pa**")
st.sidebar.write(f"$\sigma$ Superior (40%): **{sigma_superior_pa:,.2e} Pa**")
st.sidebar.write(f"Pontos Filtrados: **{len(df_filtrado)}**")

st.markdown("---")

# ------------------------------------------------------------
# INFORMAÇÕES DO CP (AGORA EM SI)
# ------------------------------------------------------------
st.markdown("### 📄 Informações do Corpo de Prova Selecionado (Unidades SI)")

# Conversão das dimensões para SI
cp_dados_si = {
    "CP": [cp_info["cp"]],
    "Nome": [cp_info["nome"]],
    "Largura (m)": [cp_info["largura"] * MM_TO_SI],
    "Espessura (m)": [cp_info["espessura"] * MM_TO_SI],
    "Área (m²)": [cp_info["area"] * MM2_TO_SI],
}
df_cp_si = pd.DataFrame(cp_dados_si)

# Formatando para exibição em notação científica com 5 casas
for col in ["Largura (m)", "Espessura (m)", "Área (m²)"]:
    df_cp_si[col] = df_cp_si[col].apply(lambda x: f"{x:,.5e}" if not pd.isna(x) else "N/A")

st.dataframe(df_cp_si, use_container_width=True, hide_index=True)

# L0 em metros
st.sidebar.metric(
    label="Comprimento Inicial ($L_0$)", 
    value=f"{L0_SI:,.5e} m", 
    help="Utilizado para calcular a Deformação Específica. (50 mm)"
)

# ------------------------------------------------------------
# PLOTAGEM DOS GRÁFICOS
# ------------------------------------------------------------
# --- Gráfico 1: TENSÃO x TEMPO ---
# (Manteremos os gráficos de tempo para referência, apesar de não serem mais usados para filtro)
fig_tensao = px.line(df_ensaio, x="tempo_s", y="tensao_pa", title="Tensão (Pa) Aplicada ao Longo do Tempo", labels={"tempo_s": "Tempo (s)", "tensao_pa": "Tensão (Pa)"})
fig_tensao.add_hline(y=0, line_dash="dash", line_color="gray")
fig_tensao.update_traces(line=dict(width=2))
fig_tensao.update_layout(template="plotly_white")
st.plotly_chart(fig_tensao, use_container_width=True)
st.markdown("---") 

# --- Gráfico 3: TENSÃO x DEFORMAÇÃO ESPECÍFICA (COM FILTRO DA NORMA) ---
st.subheader("2. Tensão ($\sigma$) x Deformação Específica ($\\varepsilon$)")
st.caption(f"Regressão na região de 10% a 40% da Tensão Máxima.")

fig_tensao_deformacao = go.Figure()

# 1. Curva Completa (Linha)
fig_tensao_deformacao.add_trace(go.Scatter(
    x=df_ensaio["deformacao_especifica"],
    y=df_ensaio["tensao_pa"],
    mode='lines',
    line=dict(color='gray', width=1.5),
    name='Curva Completa'
))

# 2. Pontos Filtrados (Scatter)
fig_tensao_deformacao.add_trace(go.Scatter(
    x=df_filtrado["deformacao_especifica"],
    y=df_filtrado["tensao_pa"],
    mode='markers',
    marker=dict(size=5, color='blue'),
    name=f'Pontos Filtrados (10% a 40%)'
))

# 3. Adiciona a Trendline CALCULADA MANUALMENTE
if not np.isnan(E_modulo):
    # Usamos os limites de tensão para encontrar os limites de deformação para plotar a reta
    eps_min_plot = df_filtrado["deformacao_especifica"].min()
    eps_max_plot = df_filtrado["deformacao_especifica"].max()
    
    x_line = np.array([eps_min_plot, eps_max_plot])
    y_line = E_modulo * x_line + intercept

    fig_tensao_deformacao.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        line=dict(color='red', width=3, dash='dash'),
        name=f'Regressão Linear (E = {E_modulo:,.2e} Pa, R² = {r_squared:,.4f})'
    ))
    
    st.info(f"Fórmula da Regressão na Região Filtrada: ${regressao_info}$")

# Adiciona linhas de referência para os limites (NOVO)
fig_tensao_deformacao.add_hline(y=sigma_inferior_pa, line_dash="dot", line_color="green", name="10% $\sigma_{máx}$", annotation_text="10% $\sigma_{máx}$")
fig_tensao_deformacao.add_hline(y=sigma_superior_pa, line_dash="dot", line_color="orange", name="40% $\sigma_{máx}$", annotation_text="40% $\sigma_{máx}$")


# Configurações do Layout
fig_tensao_deformacao.update_layout(
    title="Curva Tensão x Deformação Específica",
    xaxis_title="Deformação Específica - (mm/mm)",
    yaxis_title="Tensão (Pa)", 
    template="plotly_white",
    hovermode="x unified"
)

# Adiciona linhas no zero
fig_tensao_deformacao.add_hline(y=0, line_dash="dash", line_color="gray")
fig_tensao_deformacao.add_vline(x=0, line_dash="dash", line_color="gray")

st.plotly_chart(
    fig_tensao_deformacao, 
    use_container_width=True,
    key="interactive_plot_g3"
)


# ------------------------------------------------------------
# DOWNLOAD DOS DADOS PROCESSADOS E RESULTADOS
# ------------------------------------------------------------
st.markdown("---")
st.markdown("### ⬇️ Exportar Dados e Resultados de Análise")

# 1. Exportação dos Resultados da Análise de Regressão
if not np.isnan(E_modulo):
    
    data_resultados = {
        "CP": [cp_selecionado],
        "Módulo de Elasticidade (E) [Pa]": [E_modulo], 
        "Coeficiente de Determinação (R2)": [r_squared],
        "Intercepto da Regressão (b) [Pa]": [intercept], 
        "Tensão Máxima (sigma_max) [Pa]": [tensao_max_pa],
        "Limite Inferior Tensão (sigma_min) [Pa]": [sigma_inferior_pa], 
        "Limite Superior Tensão (sigma_max) [Pa]": [sigma_superior_pa], 
    }
    df_resultados = pd.DataFrame(data_resultados)
    
    csv_resultados = df_resultados.to_csv(
        index=False, 
        sep=';', 
        decimal='.', 
        encoding='latin-1',
        float_format='%.5e' # Notação científica com 5 casas após o ponto
    ).encode('latin-1')

    # Botão para download dos Resultados
    st.download_button(
        label=f"✅ Baixar Resultados da Análise de Regressão ({cp_selecionado}) - SI",
        data=csv_resultados,
        file_name=f'resultados_regressao_norma_{cp_selecionado}_SI_5e.csv',
        mime='text/csv',
        help="Exporta os resultados da regressão (10%-40% da Tensão Máxima) em Pascal (SI)."
    )
else:
    st.warning("⚠️ O Módulo de Elasticidade não pôde ser calculado. Verifique se o CP tem tensão máxima positiva.")

# 2. Exportação dos Dados Completos
df_export_completo = df_ensaio[["tempo_s", "deformacao_mm", "forca_n", "tensao_pa", "deformacao_especifica"]].copy()
df_export_completo = df_export_completo.rename(columns={"tensao_pa": "tensao_pa"})

csv_export = df_export_completo.to_csv(
    index=False, sep=';', decimal=',', encoding='latin-1'
).encode('latin-1')

# Botão para download dos Dados Completos
st.download_button(
    label=f"📥 Baixar CSV de Dados Completos ({cp_selecionado})",
    data=csv_export,
    file_name=f'dados_completos_ordem_temporal_{cp_selecionado}.csv',
    mime='text/csv',
    help="Exporta todas as colunas de tempo, deformação, força, tensão (Pa) e deformação específica, na ordem temporal."
)