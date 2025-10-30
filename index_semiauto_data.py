import pandas as pd
import numpy as np
from scipy import stats
import os
import glob
import sys

# --- CONSTANTES GLOBAIS ---
# Constantes de conversão
MEGA_TO_SI = 1e6   # 1 MPa = 10^6 Pa
MM_TO_SI = 1e-3    # 1 mm = 10^-3 m
L0_MM = 50.0       # comprimento inicial do extensômetro (mm)

# Constantes da Norma
PERCENTUAL_MIN_MODULO = 0.10 # 10% da tensão máxima
PERCENTUAL_MAX_MODULO = 0.40 # 40% da tensão máxima

# Configurações de Arquivos
PASTA_ENSAIOS = "ensaios" 
ARQUIVO_GERAL = "geral.csv"
ARQUIVO_SAIDA = "resultados_norma_batch.csv"
COLUNAS_GERAL = [
    "cp", "nome", "largura", "espessura", 
    "ret_ext", "area", "forca_max", "tensao_max", "energia"
]

# --- FUNÇÃO PRINCIPAL DE PROCESSAMENTO ---

def processar_cp(cp_nome, area_mm2):
    """
    Processa os dados de um único CP, calcula o Módulo de Elasticidade (E) 
    usando a regra da norma (10%-40% da tensão máxima) e retorna os resultados.
    """
    cp_num = cp_nome.split()[-1]
    caminho_csv = os.path.join(PASTA_ENSAIOS, f"{cp_num}.csv")
    
    # Validação inicial do arquivo de ensaio
    if not os.path.exists(caminho_csv):
        print(f"⚠️ Aviso: Arquivo de ensaio '{caminho_csv}' não encontrado.")
        return None

    try:
        # 1. Carregar Dados do Ensaio
        df_ensaio = pd.read_csv(caminho_csv, sep=";", decimal=",", encoding_errors="ignore")
        df_ensaio.columns = ["tempo_s", "deformacao_mm", "forca_n"]
        # Garante que os dados são numéricos
        df_ensaio["deformacao_mm"] = pd.to_numeric(df_ensaio["deformacao_mm"], errors="coerce")
        df_ensaio["forca_n"] = pd.to_numeric(df_ensaio["forca_n"], errors="coerce")
        df_ensaio = df_ensaio.dropna(subset=["deformacao_mm", "forca_n"]).reset_index(drop=True)

    except Exception as e:
        print(f"❌ Erro ao ler ou processar o arquivo {caminho_csv}: {e}")
        return None

    if df_ensaio.empty:
        return None

    # 2. Cálculos de Tensão (Pa) e Deformação Específica
    # Tensão (N/mm²) * 10^6 = Pa. (F/A em N/mm² é o mesmo que MPa)
    df_ensaio["tensao_pa"] = (df_ensaio["forca_n"] / area_mm2) * MEGA_TO_SI 
    # Deformação Específica (mm/mm é adimensional)
    df_ensaio["deformacao_especifica"] = -df_ensaio["deformacao_mm"] / L0_MM 
    
    # 3. Determinar Limites da Norma
    tensao_max_pa = df_ensaio["tensao_pa"].max()

    if tensao_max_pa <= 0:
        print(f"⚠️ Aviso: Tensão máxima do CP {cp_nome} é zero ou negativa. Impossível calcular Módulo.")
        return None

    sigma_inferior_pa = PERCENTUAL_MIN_MODULO * tensao_max_pa
    sigma_superior_pa = PERCENTUAL_MAX_MODULO * tensao_max_pa
    
    # 4. Filtrar Dados
    df_filtrado = df_ensaio[
        (df_ensaio["tensao_pa"] >= sigma_inferior_pa) & 
        (df_ensaio["tensao_pa"] <= sigma_superior_pa)
    ].copy()

    if len(df_filtrado) < 2:
        print(f"⚠️ Aviso: CP {cp_nome} tem dados insuficientes no intervalo de 10%-40%.")
        return None

    # 5. Regressão Linear
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_filtrado["deformacao_especifica"], df_filtrado["tensao_pa"]
        )
        E_modulo = slope 
        r_squared = r_value**2

    except ValueError:
        print(f"❌ Erro de Valor: Não foi possível calcular a regressão para o CP {cp_nome}.")
        return None

    # 6. Retorno dos Resultados
    return {
        "CP": cp_nome,
        "Módulo de Elasticidade (E) [Pa]": E_modulo,
        "Coeficiente de Determinação (R2)": r_squared,
        "Intercepto da Regressão (b) [Pa]": intercept,
        "Tensão Máxima (sigma_max) [Pa]": tensao_max_pa,
        "Limite Inferior Tensão (sigma_min) [Pa]": sigma_inferior_pa,
        "Limite Superior Tensão (sigma_max) [Pa]": sigma_superior_pa,
    }


# --- EXECUÇÃO DO BATCH ---

def main():
    print(f"Iniciando processamento em lote...")

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
            skiprows=1, # Assume que a primeira linha é o cabeçalho/metadado
            names=COLUNAS_GERAL 
        )
        # Conversão de tipo necessária para filtragem
        geral["area"] = pd.to_numeric(geral["area"], errors='coerce')
    except Exception as e:
        print(f"❌ Erro ao ler o arquivo {ARQUIVO_GERAL}: {e}")
        sys.exit(1)

    # 2. Filtrar CPs Válidos
    cp_validos = geral[geral['area'].notna() & (geral['area'] > 0)].copy()
    
    if cp_validos.empty:
        print("❌ Nenhum Corpo de Prova com área válida encontrado no geral.csv.")
        sys.exit(1)

    # Ordenação (Apenas para garantir a ordem da saída)
    cp_validos["cp_num"] = cp_validos["cp"].str.extract(r"(\d+)").astype(int, errors='ignore')
    cp_validos = cp_validos.sort_values(by="cp_num").reset_index(drop=True)
    
    print(f"Total de CPs válidos a processar: {len(cp_validos)}")
    
    # 3. Processar em Loop
    resultados_list = []
    
    for index, row in cp_validos.iterrows():
        cp_nome = row["cp"]
        area = row["area"]
        
        print(f"-> Processando {cp_nome}...")
        resultado = processar_cp(cp_nome, area)
        
        if resultado:
            resultados_list.append(resultado)

    # 4. Consolidar e Exportar
    if not resultados_list:
        print("\nProcessamento concluído. 😔 Nenhum CP foi processado com sucesso.")
        return

    df_final = pd.DataFrame(resultados_list)

    # Ajuste o nome da coluna de limite superior para corresponder ao formato solicitado
    df_final = df_final.rename(columns={"Limite Superior Tensão (sigma_max) [Pa]": "Limite Superior Tensão (sigma_sup) [Pa]"})
    
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