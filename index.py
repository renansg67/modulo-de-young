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

# ------------------------------------------------------------
# CONFIGURAÇÕES INICIAIS
# ------------------------------------------------------------
st.set_page_config(page_title="Análise com Módulo de Elasticidade", layout="wide")
st.title("📊 Análise de Ensaio com Regressão na Região Elástica (Unidades SI)")
st.markdown("---")

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
    # Assumindo que o arquivo geral.csv contém a lista de CPs e suas dimensões.
    geral = pd.read_csv(
        "geral.csv", 
        sep=";", 
        decimal=",", 
        encoding_errors="ignore",
        header=None,       
        skiprows=1,        
        names=COLUNAS_GERAL 
    )
    
    # GARANTINDO O TIPO NUMÉRICO (Correção para o TypeError)
    geral["area"] = pd.to_numeric(geral["area"], errors='coerce')
    geral["largura"] = pd.to_numeric(geral["largura"], errors='coerce')
    geral["espessura"] = pd.to_numeric(geral["espessura"], errors='coerce')
    
    # Extrai o número do CP para auxiliar na filtragem/ordenacao
    geral["cp_num"] = geral["cp"].str.extract(r"(\d+)").astype(str)
    st.write(geral)
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
# Filtra apenas CPs com área válida
cp_disponiveis = geral[geral['area'].notna() & (geral['area'] > 0)]["cp"].tolist()
if not cp_disponiveis:
    st.error("❌ Nenhum Corpo de Prova com área válida encontrado.")
    st.stop()

# Ordenação Numérica dos CPs (Correção para o problema de ordenação)
# A função lambda extrai o número do final da string "CP X" e o converte para int
cp_opcoes = sorted(
    cp_disponiveis, 
    key=lambda cp: int(cp.split()[-1]) if cp.split()[-1].isdigit() else 0
)

# Adiciona a opção padrão (placeholder) na primeira posição (índice 0)
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

# Tensão (N/mm²) * 10^6 = Pa. (F/A em N/mm² é o mesmo que MPa)
df_ensaio["tensao_pa"] = (df_ensaio["forca_n"] / area) * MEGA_TO_SI 
# Deformação Específica (mm/mm é adimensional, usamos L0 em mm)
df_ensaio["deformacao_especifica"] = -df_ensaio["deformacao_mm"] / L0_MM 

# ------------------------------------------------------------
# PREPARAÇÃO DOS SLIDERS E FILTRO
# ------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Filtro para Módulo de Elasticidade")

### 1. Slider para Deformação Específica (X)
all_eps_values = df_ensaio["deformacao_especifica"].round(6).unique()
all_eps_values.sort()

if all_eps_values.size < 2:
    st.error("Não há dados de deformação suficientes para a regressão.")
    st.stop()

# Valores padrão
default_min_idx_eps = 0
default_max_idx_eps = len(all_eps_values) - 1
if len(all_eps_values) > 10: 
    default_min_idx_eps = int(len(all_eps_values) * 0.10)
    default_max_idx_eps = int(len(all_eps_values) * 0.90)

default_min_eps = all_eps_values[default_min_idx_eps]
default_max_eps = all_eps_values[default_max_idx_eps]

limites_filtro_eps = st.sidebar.select_slider(
    '1. Limite X (Deformação Específica $\\varepsilon$):',
    options=all_eps_values,
    value=(default_min_eps, default_max_eps),
    format_func=lambda x: f"{x:,.6f}" # Formato para exibir mais casas decimais
)
eps_min_filtro, eps_max_filtro = limites_filtro_eps


### 2. Slider para Tensão (Y)
# Usando tensão em Pa no slider
all_tensao_pa_values = df_ensaio["tensao_pa"].round(0).unique() 
all_tensao_pa_values.sort()

if all_tensao_pa_values.size < 2:
    st.error("Não há dados de tensão suficientes para a regressão.")
    st.stop()

# Valores padrão (em Pa)
default_min_idx_tensao = 0
default_max_idx_tensao = len(all_tensao_pa_values) - 1
if len(all_tensao_pa_values) > 10: 
    default_min_idx_tensao = int(len(all_tensao_pa_values) * 0.10)
    default_max_idx_tensao = int(len(all_tensao_pa_values) * 0.90)

default_min_tensao_pa = all_tensao_pa_values[default_min_idx_tensao]
default_max_tensao_pa = all_tensao_pa_values[default_max_idx_tensao]

limites_filtro_tensao = st.sidebar.select_slider(
    '2. Limite Y (Tensão $\\sigma$):',
    options=all_tensao_pa_values,
    value=(default_min_tensao_pa, default_max_tensao_pa),
    format_func=lambda x: f"{x:,.0f} Pa" # Formato para exibir a unidade em Pa
)
tensao_min_filtro_pa, tensao_max_filtro_pa = limites_filtro_tensao


### 3. Aplica o Filtro Combinado (Usando a coluna tensao_pa)
df_filtrado = df_ensaio[
    (df_ensaio["deformacao_especifica"] >= eps_min_filtro) & 
    (df_ensaio["deformacao_especifica"] <= eps_max_filtro) &
    (df_ensaio["tensao_pa"] >= tensao_min_filtro_pa) & 
    (df_ensaio["tensao_pa"] <= tensao_max_filtro_pa)
].copy()


# ------------------------------------------------------------
# CÁLCULO DO MÓDULO DE ELASTICIDADE (E) E R² (E agora está em Pa)
# ------------------------------------------------------------
E_modulo = np.nan
r_squared = np.nan
intercept = np.nan
regressao_info = "Selecione um intervalo válido com dados suficientes para regressão."

if len(df_filtrado) > 1:
    try:
        # A regressão usa Deformação (adimensional) e Tensão (Pa) -> E_modulo sai em Pa
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_filtrado["deformacao_especifica"], df_filtrado["tensao_pa"]
        )
        E_modulo = slope # E_modulo agora está em Pa
        r_squared = r_value**2
        regressao_info = f"\\text{{Tensão (Pa)}} = {E_modulo:,.2e} \\times \\text{{Deformação}} + {intercept:,.2e}"
    except ValueError:
        regressao_info = "Erro ao calcular regressão: dados inválidos no intervalo."
        
# ------------------------------------------------------------
# EXIBIÇÃO DOS RESULTADOS NA SIDEBAR
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.metric(
    label="Módulo de Elasticidade ($E$)",
    # Exibindo em notação científica para números grandes
    value=f"{E_modulo:,.5e} Pa" if not np.isnan(E_modulo) else "N/A", # 5e para consistência
    help="Inclinação da linha de regressão (trendline) no intervalo filtrado (em Pascal)."
)
st.sidebar.metric(
    label="Coeficiente de Determinação ($R^2$)",
    value=f"{r_squared:,.4f}" if not np.isnan(r_squared) else "N/A",
    help="Mede o quão bem a regressão linear se ajusta aos dados filtrados."
)

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

# Formatando para exibição em notação científica com 5 casas, removendo a unidade da string.
for col in ["Largura (m)", "Espessura (m)", "Área (m²)"]:
    df_cp_si[col] = df_cp_si[col].apply(lambda x: f"{x:,.5e}" if not pd.isna(x) else "N/A")

st.dataframe(df_cp_si, use_container_width=True, hide_index=True)

# L0 em metros
st.metric(
    label="Comprimento Inicial ($L_0$)", 
    value=f"{L0_SI:,.5e} m", 
    help="Utilizado para calcular a Deformação Específica. (50 mm)"
)
st.markdown("---")

# ------------------------------------------------------------
# PLOTAGEM DOS TRÊS GRÁFICOS (VERTICAL)
# ------------------------------------------------------------
# --- Gráfico 1: TENSÃO x TEMPO ---
st.subheader("1. Tensão ($\sigma$) x Tempo (t)")
# Tensão em Pa
fig_tensao = px.line(df_ensaio, x="tempo_s", y="tensao_pa", title="Tensão (Pa) Aplicada ao Longo do Tempo", labels={"tempo_s": "Tempo (s)", "tensao_pa": "Tensão ($\sigma$) (Pa)"})
fig_tensao.add_hline(y=0, line_dash="dash", line_color="gray")
fig_tensao.update_traces(line=dict(width=2))
fig_tensao.update_layout(template="plotly_white")
st.plotly_chart(fig_tensao, use_container_width=True)
st.markdown("---") 

# --- Gráfico 2: DEFORMAÇÃO x TEMPO ---
st.subheader("2. Deformação ($\Delta L$) x Tempo (t)")
fig_deformacao = px.line(df_ensaio, x="tempo_s", y="deformacao_mm", title="Deformação (mm) ao Longo do Tempo", labels={"tempo_s": "Tempo (s)", "deformacao_mm": "Deformação ($\Delta L$) (mm)"})
fig_deformacao.add_hline(y=0, line_dash="dash", line_color="gray")
fig_deformacao.update_traces(line=dict(width=2))
fig_deformacao.update_layout(template="plotly_white")
st.plotly_chart(fig_deformacao, use_container_width=True)
st.markdown("---") 


# --- Gráfico 3: TENSÃO x DEFORMAÇÃO ESPECÍFICA (COM FILTRO E REGRESSÃO) ---
st.subheader("3. Tensão ($\sigma$) x Deformação Específica ($\varepsilon$)")
st.caption(f"Ajuste os limites $\varepsilon$ e $\sigma$ na barra lateral. O Módulo $E$ é a inclinação da reta vermelha.")

fig_tensao_deformacao = go.Figure()

# 1. Curva Completa (Linha)
fig_tensao_deformacao.add_trace(go.Scatter(
    x=df_ensaio["deformacao_especifica"],
    y=df_ensaio["tensao_pa"], # Usando Pa
    mode='lines',
    line=dict(color='gray', width=1.5),
    name='Curva Completa (Ordem Temporal)'
))

# 2. Pontos Filtrados (Scatter)
fig_tensao_deformacao.add_trace(go.Scatter(
    x=df_filtrado["deformacao_especifica"],
    y=df_filtrado["tensao_pa"], # Usando Pa
    mode='markers',
    marker=dict(size=5, color='blue'),
    name='Pontos Filtrados para Regressão'
))

# 3. Adiciona a Trendline CALCULADA MANUALMENTE
if not np.isnan(E_modulo):
    x_line = np.array([eps_min_filtro, eps_max_filtro])
    y_line = E_modulo * x_line + intercept

    fig_tensao_deformacao.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        line=dict(color='red', width=3, dash='dash'),
        name=f'Regressão Linear (E = {E_modulo:,.2e} Pa, R² = {r_squared:,.4f})'
    ))
    
    st.info(f"Fórmula da Regressão na Região Filtrada: ${regressao_info}$")


# Configurações do Layout
fig_tensao_deformacao.update_layout(
    title="Curva Tensão ($\sigma$) x Deformação Específica ($\varepsilon$)",
    xaxis_title="Deformação Específica ($\varepsilon$) - (mm/mm)",
    yaxis_title="Tensão ($\sigma$) (Pa)", # Unidade atualizada para Pa
    template="plotly_white",
    hovermode="x unified"
)

# APLICAÇÃO DO ESTADO DO ZOOM SALVO (se existir)
if st.session_state.get('layout_g3'):
    fig_tensao_deformacao.update_layout(st.session_state['layout_g3'])
else:
    # Define o range inicial do eixo X
    min_eps_plot = df_ensaio["deformacao_especifica"].min()
    max_eps_plot = df_ensaio["deformacao_especifica"].max()
    buffer = (max_eps_plot - min_eps_plot) * 0.05 
    x_range_plot = [min_eps_plot - buffer, max_eps_plot + buffer]
    fig_tensao_deformacao.update_xaxes(range=x_range_plot)


# Adiciona linhas no zero
fig_tensao_deformacao.add_hline(y=0, line_dash="dash", line_color="gray")
fig_tensao_deformacao.add_vline(x=0, line_dash="dash", line_color="gray")

# st.plotly_chart renderiza o gráfico e, por ter uma 'key', o Streamlit 
# tenta preservar automaticamente o estado de zoom/pan.
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

# 1. Exportação dos Resultados da Análise de Regressão (Notação Científica)
if not np.isnan(E_modulo):
    
    # Cria o DataFrame de resultados 
    data_resultados = {
        "CP": [cp_selecionado],
        "Módulo de Elasticidade (E) [Pa]": [E_modulo], 
        "Coeficiente de Determinação (R2)": [r_squared],
        "Intercepto da Regressão (b) [Pa]": [intercept], 
        "Limite Inferior Deformação Específica (eps_min)": [eps_min_filtro],
        "Limite Superior Deformação Específica (eps_max)": [eps_max_filtro],
        "Limite Inferior Tensão (sigma_min) [Pa]": [tensao_min_filtro_pa], 
        "Limite Superior Tensão (sigma_max) [Pa]": [tensao_max_filtro_pa], 
    }
    df_resultados = pd.DataFrame(data_resultados)
    
    # Formatando para notação científica com 5 casas de precisão e ponto como separador decimal
    csv_resultados = df_resultados.to_csv(
        index=False, 
        sep=';', 
        decimal='.', 
        encoding='latin-1',
        float_format='%.5e' # Notação científica com 5 casas após o ponto
    ).encode('latin-1')

    # Botão para download dos Resultados
    st.download_button(
        label=f"✅ Baixar Resultados da Análise de Regressão (CP {cp_selecionado}) - SI",
        data=csv_resultados,
        file_name=f'resultados_regressao_{cp_selecionado}_SI_5e.csv',
        mime='text/csv',
        help="Exporta os resultados da regressão em Pascal (SI) em notação científica (5 casas)."
    )
else:
    st.warning("⚠️ O Módulo de Elasticidade não pôde ser calculado. Selecione um intervalo válido.")

# 2. Exportação dos Dados Completos
# Alterando o nome da coluna de tensão no CSV completo para refletir Pa
df_export_completo = df_ensaio[["tempo_s", "deformacao_mm", "forca_n", "tensao_pa", "deformacao_especifica"]].copy()
df_export_completo = df_export_completo.rename(columns={"tensao_pa": "tensao_pa"})

csv_export = df_export_completo.to_csv(
    index=False, sep=';', decimal=',', encoding='latin-1'
).encode('latin-1')

# Botão para download dos Dados Completos
st.download_button(
    label=f"📥 Baixar CSV de Dados Completos (CP {cp_selecionado})",
    data=csv_export,
    file_name=f'dados_completos_ordem_temporal_{cp_selecionado}.csv',
    mime='text/csv',
    help="Exporta todas as colunas de tempo, deformação, força, tensão (Pa) e deformação específica, na ordem temporal."
)