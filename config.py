"""
Arquivo de configuração do projeto
"""


from pathlib import Path

# ======================
# PATHS
# ======================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

# =====================
# DATASET FILE NAMES
# =====================

TRAIN_FILE = RAW_DATA_DIR / "caged_curitiba_consolidado_train.csv"
PREDICTION_FILE = RAW_DATA_DIR / "caged_curitiba_consolidado_test.csv"


# =====================
# EDA - MAPPINGS
# =====================

MAP_REGIAO = {
    1: "Norte",
    2: "Nordeste",
    3: "Sudeste",
    4: "Sul",
    5: "Centro-Oeste",
}

MAP_SEXO = {
    0: "?",
    1: "Masculino",
    3: "Feminino",
    2: "Ignorado",
}

MAP_RACA_COR = {
    0: "?",
    1: "Branca",
    2: "Preta",
    3: "Parda",
    4: "Amarela",
    5: "Indígena",
    6: "Não informada",
    9: "Não informada",
}

MAP_GRAU_INSTRUCAO = {
    0: "?", 
    1: "Analfabeto", 
    2: "Até 5ª inc.", 
    3: "5ª completa",
    4: "6ª-9ª inc.", 
    5: "Fund. Completo", 
    6: "Médio Inc.",
    7: "Médio Completo", 
    8: "Superior Inc.", 
    9: "Superior Completo",
    10: "Mestrado", 
    11: "Doutorado",
}

MAP_TIPO_MOVIMENTACAO = {
    1: "Admissão",
    -1: "Desligamento",
}

MAP_TIPO_ESTABELECIMENTO = {
    0: "?",
    1: "CNPJ",
    2: "CPF",
    3: "CAEPF",
    4: "CNO",
}

MAP_TIPO_EMPREGADOR = {
    0: "?",
    1: "Pessoa Jurídica",
    2: "Pessoa Física",
    3: "CEI",
    4: "CAEPF",
}

MAP_DEFICIENCIA = {
    0: "Sem deficiência",
    1: "Física",
    2: "Auditiva",
    3: "Visual",
    4: "Intelectual",
    5: "Múltipla",
    9: "Não informado",
}

MAP_APRENDIZ = {
    0: "Não Aprendiz",
    1: "Aprendiz",
}