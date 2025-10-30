import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

# --- CONSTANTES GLOBAIS ---
MEGA_TO_SI = 1e6   # 1 MPa = 10^6 Pa
L0_MM = 50.0       # comprimento inicial do extensômetro (mm)

# Configurações da Norma (Método Padrão: 10% - 40%)
PERCENTUAL_MIN_MODULO = 0.10 
PERCENTUAL_MAX_MODULO = 0.40 

# Configurações do R² Hunting (Método Otimizado)
MIN_R2_WINDOW_POINTS = 1500 # Novo: Mínimo de 1500 pontos
R2_WINDOW_SIZES = [1500, 2000, 3000, 4000] # Novo: Testar tamanhos comuns de janelas
R2_STEP_SIZE = 50           # Deslocamento da janela (aumentado para otimizar o tempo)

# Configurações de Arquivos
PASTA_ENSAIOS = "ensaios" 
ARQUIVO_GERAL = "geral.csv"
ARQUIVO_SAIDA = "resultados_norma_e_otimizado_v2.csv"
COLUNAS_GERAL = [
    "cp", "nome", "largura", "espessura", 
    "ret_ext", "area", "forca_max", "tensao_max", "energia"
]

# --- FUNÇÕES DE CÁLCULO ---

def hunting_E_best_R2(df_ensaio, max_sigma_index):
    """
    Procura o segmento de reta com o maior R² (melhor ajuste) 
    testando múltiplos tamanhos de janela na região elástica (0 a sigma_max).
    """
    best_r2_global = -1.0
    best_E_global = np.nan
    best_intercept_global = np.nan
    best_start_idx_global = -1
    best_end_idx_global = -1
    best_window_size_global = 0

    # 1. Definir o range de busca e verificar o mínimo de dados
    df_search = df_ensaio.iloc[:max_sigma_index].reset_index(drop=True)
    n_points = len(df_search)
    
    if n_points < MIN_R2_WINDOW_POINTS:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # 2. Iterar sobre todos os tamanhos de janela predefinidos
    for window_size in R2_WINDOW_SIZES:
        
        if window_size > n_points:
            continue # O tamanho da janela é maior que os dados disponíveis

        # 3. Iteração da Janela Deslizante
        for i in range(0, n_points - window_size + 1, R2_STEP_SIZE):
            
            start_idx = i
            end_idx = i + window_size
            
            df_window = df_search.iloc[start_idx:end_idx]
            
            try:
                # Realiza a regressão
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df_window["deformacao_especifica"], df_window["tensao_pa"]
                )
                r_squared = r_value**2
                
                # Se o R² for o melhor já encontrado em qualquer tamanho de janela, atualiza
                if r_squared > best_r2_global:
                    best_r2_global = r_squared
                    best_E_global = slope
                    best_intercept_global = intercept
                    # Usamos os índices originais do DataFrame de ensaio
                    best_start_idx_global = df_ensaio.index[start_idx] 
                    best_end_idx_global = df_ensaio.index[end_idx - 1] 
                    best_window_size_global = window_size
                    
            except ValueError:
                # Caso a regressão falhe, apenas ignora
                continue
            
    return (best_E_global, best_r2_global, best_intercept_global, 
            best_start_idx_global, best_end_idx_global, best_window_size_global)

def processar_cp(cp_nome, area_mm2):
    """
    Processa os dados de um único CP, calculando E pela Norma e por Otimização (R² Hunting).
    """
    cp_num = cp_nome.split()[-1]
    caminho_csv = os.path.join(PASTA_ENSAIOS, f"{cp_num}.csv")
    
    if not os.path.exists(caminho_csv):
        #print(f"⚠️ Aviso: Arquivo de ensaio '{caminho_csv}' não encontrado.")
        return None

    try:
        # 1. Carregar Dados do Ensaio
        df_ensaio = pd.read_csv(caminho_csv, sep=";", decimal=",", encoding_errors="ignore")
        df_ensaio.columns = ["tempo_s", "deformacao_mm", "forca_n"]
        df_ensaio["deformacao_mm"] = pd.to_numeric(df_ensaio["deformacao_mm"], errors="coerce")
        df_ensaio["forca_n"] = pd.to_numeric(df_ensaio["forca_n"], errors="coerce")
        df_ensaio = df_ensaio.dropna(subset=["deformacao_mm", "forca_n"]).reset_index(drop=True)

    except Exception:
        #print(f"❌ Erro ao ler ou processar o arquivo {caminho_csv}: {e}")
        return None

    if df_ensaio.empty:
        return None

    # 2. Cálculos de Tensão (Pa) e Deformação Específica
    df_ensaio["tensao_pa"] = (df_ensaio["forca_n"] / area_mm2) * MEGA_TO_SI 
    df_ensaio["deformacao_especifica"] = -df_ensaio["deformacao_mm"] / L0_MM 
    
    tensao_max_pa = df_ensaio["tensao_pa"].max()

    if tensao_max_pa <= 0:
        return None
        
    # Encontra o índice da tensão máxima
    max_sigma_index = df_ensaio["tensao_pa"].idxmax()
    
    # --- RESULTADOS DA NORMA (10% - 40%) ---
    sigma_inferior_norma = PERCENTUAL_MIN_MODULO * tensao_max_pa
    sigma_superior_norma = PERCENTUAL_MAX_MODULO * tensao_max_pa
    
    df_norma = df_ensaio[
        (df_ensaio["tensao_pa"] >= sigma_inferior_norma) & 
        (df_ensaio["tensao_pa"] <= sigma_superior_norma)
    ].copy()
    
    E_norma, R2_norma, Intercept_norma = np.nan, np.nan, np.nan
    if len(df_norma) >= MIN_R2_WINDOW_POINTS: # Aumentando a segurança também para a norma
        try:
            slope, intercept, r_value, _, _ = stats.linregress(df_norma["deformacao_especifica"], df_norma["tensao_pa"])
            E_norma, R2_norma, Intercept_norma = slope, r_value**2, intercept
        except ValueError:
            pass
    
    # --- RESULTADOS OTIMIZADOS (R² HUNTING) ---
    (E_otimizado, R2_otimizado, Intercept_otimizado, Start_Idx_Otimizado, End_Idx_Otimizado, Window_Size_Otimizado) = hunting_E_best_R2(df_ensaio, max_sigma_index)

    # 3. Retorno dos Resultados
    return {
        "CP": cp_nome,
        "Tensão_Máxima [Pa]": tensao_max_pa,
        
        "E_Norma [Pa]": E_norma,
        "R2_Norma": R2_norma,
        "Intercepto_Norma [Pa]": Intercept_norma,
        "Limite_Inf_Norma [Pa]": sigma_inferior_norma,
        "Limite_Sup_Norma [Pa]": sigma_superior_norma,
        "Pontos_Norma": len(df_norma) if not df_norma.empty else 0,

        "E_Otimizado [Pa]": E_otimizado,
        "R2_Otimizado": R2_otimizado,
        "Intercepto_Otimizado [Pa]": Intercept_otimizado,
        "Pontos_Otimizado": Window_Size_Otimizado,
        "Start_Idx_Otimizado": Start_Idx_Otimizado,
        "End_Idx_Otimizado": End_Idx_Otimizado,
    }

# --- EXECUÇÃO DO BATCH ---

def main():
    print(f"Iniciando processamento em lote (v2) com R² Hunting (Janelas: {R2_WINDOW_SIZES})...")

    # 1. Carregar Arquivo Geral
    if not os.path.exists(ARQUIVO_GERAL):
        print(f"❌ Erro fatal: Arquivo '{ARQUIVO_GERAL}' não encontrado.")
        sys.exit(1)

    try:
        geral = pd.read_csv(
            ARQUIVO_GERAL, 
            sep=";", 
            decimal=",", 
            encoding_errors="ignore",
            header=None,       
            skiprows=1, 
            names=COLUNAS_GERAL 
        )
        geral["area"] = pd.to_numeric(geral["area"], errors='coerce')
    except Exception as e:
        print(f"❌ Erro ao ler o arquivo {ARQUIVO_GERAL}: {e}")
        sys.exit(1)

    # 2. Filtrar CPs Válidos
    cp_validos = geral[geral['area'].notna() & (geral['area'] > 0)].copy()
    
    if cp_validos.empty:
        print("❌ Nenhum Corpo de Prova com área válida encontrado no geral.csv.")
        sys.exit(1)

    cp_validos["cp_num"] = cp_validos["cp"].str.extract(r"(\d+)").astype(int, errors='ignore')
    cp_validos = cp_validos.sort_values(by="cp_num").reset_index(drop=True)
    
    print(f"Total de CPs válidos a processar: {len(cp_validos)}")
    
    # 3. Processar em Loop
    resultados_list = []
    
    for index, row in cp_validos.iterrows():
        cp_nome = row["cp"]
        area = row["area"]
        
        sys.stdout.write(f"\r-> Processando {cp_nome} ({index + 1}/{len(cp_validos)})...")
        sys.stdout.flush()
        
        resultado = processar_cp(cp_nome, area)
        
        if resultado:
            resultados_list.append(resultado)

    # 4. Consolidar e Exportar
    if not resultados_list:
        print("\nProcessamento concluído. 😔 Nenhum CP foi processado com sucesso.")
        return

    df_final = pd.DataFrame(resultados_list)
    
    # Formato do CSV (ponto como decimal, notação científica com 5 casas)
    df_final.to_csv(
        ARQUIVO_SAIDA, 
        sep=';', 
        decimal='.', 
        index=False, 
        encoding='latin-1',
        float_format='%.5e'
    )
    
    print(f"\n✅ Sucesso! {len(df_final)} CPs processados.")
    print(f"Arquivo de resultados salvo em: {ARQUIVO_SAIDA}")

if __name__ == "__main__":
    main()