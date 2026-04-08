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
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "caged_curitiba_consolidado_train_processed.csv"

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

# =====================
# Processamento - ajustes de colunas
# =====================

COLUMN_TYPES_MAP = [
    ("competenciamov", "Int64"),
    ("regiao", "Int64"),
    ("uf", "Int64"),
    ("municipio", "Int64"),
    ("secao", "str"),
    ("subclasse", "str"),
    ("saldomovimentacao", "Int64"),
    ("cbo2002ocupacao", "Int64"),
    ("categoria", "Int64"),
    ("graudeinstrucao", "Int64"),
    ("idade", "Int64"),
    ("horascontratuais", "float"),
    ("racacor", "Int64"),
    ("sexo", "Int64"),
    ("tipoempregador", "Int64"),
    ("tipoestabelecimento", "Int64"),
    ("tipodedeficiencia", "Int64"),
    ("indtrabintermitente", "Int64"),
    ("indtrabparcial", "Int64"),
    ("salario", "float"),
    ("tamestabjan", "Int64"),
    ("indicadoraprendiz", "Int64"),
    ("origemdainformacao", "Int64"),
    ("competenciadec", "Int64"),
    ("unidadesalariocodigo", "Int64"),
    ("valorsalariofixo", "float"),
]

COLUMNS_TO_KEEP = [
    # "competenciamov",
    # "regiao",
    # "uf",
    "municipio",
    "secao",
    "subclasse",
    "saldomovimentacao",
    "cbo2002ocupacao",
    "categoria",
    "graudeinstrucao",
    "idade",
    "horascontratuais",
    "racacor",
    "sexo",
    "tipoempregador",
    "tipoestabelecimento",
    "tipodedeficiencia",
    "indtrabintermitente",
    "indtrabparcial",
    "salario",
    "tamestabjan",
    "indicadoraprendiz",
    # "origemdainformacao",
    # "competenciadec",
    # "unidadesalariocodigo",
    # "valorsalariofixo",
    "ano",
    "mes",
]

# =====================
# Processamento - anos e meses para filtrar
# =====================

MESES_RELEVANTES = {11, 12, 1}
ANOS_RELEVANTES = {2024, 2025}

# =====================
# Modelo - configurações
# =====================

TARGET_COL = "saldomovimentacao"

# Features numéricas (contínuas ou inteiros que XGBoost trata como numéricos via splits)
NUMERIC_FEATURES = (
    "graudeinstrucao",
    "idade",
    "horascontratuais",
    "racacor",
    "sexo",
    "tipoempregador",
    "tipoestabelecimento",
    "tipodedeficiencia",
    "indtrabintermitente",
    "indtrabparcial",
    "salario",
    "tamestabjan",
    "indicadoraprendiz",
    "ano",
    "mes",
)

# Features categóricas (string, precisam de encoding)
CATEGORICAL_FEATURES = (
    "secao",
    "subclasse",
    "cbo2002ocupacao",
    "categoria",
    "municipio",
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10

XGBOOST_PARAM_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "min_child_weight": [1, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

TRAINED_MODEL_PATH = MODEL_DIR / "best_xgb_pipeline.pkl"
OUTPUT_PREDICTIONS_FILE = OUTPUT_DATA_DIR / "caged_curitiba_consolidado_test_predictions.csv"

