import pandas as pd
import os

# --- CONFIGURAÇÃO ---
xlsx_file = "ensaios_ruptura.xlsx"  # <-- Substitua pelo caminho do seu arquivo Excel
output_folder = "ensaios_flexao"      # Pasta onde os CSVs serão salvos

# Cria a pasta de saída se não existir
os.makedirs(output_folder, exist_ok=True)

# Lê o Excel
xls = pd.ExcelFile(xlsx_file)
print(f"Abas encontradas: {xls.sheet_names}")

# Converte cada aba para CSV
for aba in xls.sheet_names:
    df = pd.read_excel(xlsx_file, sheet_name=aba)
    # Substitui caracteres inválidos no nome do arquivo
    nome_csv = "".join([c if c.isalnum() or c in "_-" else "_" for c in aba])
    csv_path = os.path.join(output_folder, f"{nome_csv}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Aba '{aba}' salva como '{csv_path}'")

print("Conversão concluída!")
