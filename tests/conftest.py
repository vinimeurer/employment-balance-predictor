"""
Fixtures compartilhadas para os testes do projeto.

Centraliza a criação de DataFrames de exemplo, arquivos temporários
e objetos reutilizáveis entre os módulos de teste.
"""

import pandas as pd
import pytest
from pathlib import Path


# ======================================================================
# Dados de exemplo compatíveis com COLUMN_TYPES_MAP e COLUMNS_TO_KEEP
# ======================================================================

SAMPLE_RAW_COLUMNS = [
    "competenciamov", "regiao", "uf", "municipio", "secao", "subclasse",
    "saldomovimentacao", "cbo2002ocupacao", "categoria", "graudeinstrucao",
    "idade", "horascontratuais", "racacor", "sexo", "tipoempregador",
    "tipoestabelecimento", "tipodedeficiencia", "indtrabintermitente",
    "indtrabparcial", "salario", "tamestabjan", "indicadoraprendiz",
    "origemdainformacao", "competenciadec", "unidadesalariocodigo",
    "valorsalariofixo",
]

SAMPLE_RAW_ROW = {
    "competenciamov": "202411",
    "regiao": "4",
    "uf": "41",
    "municipio": "4106902",
    "secao": "C",
    "subclasse": "1011201",
    "saldomovimentacao": "1",
    "cbo2002ocupacao": "784205",
    "categoria": "101",
    "graudeinstrucao": "7",
    "idade": "30",
    "horascontratuais": "44,00",
    "racacor": "1",
    "sexo": "1",
    "tipoempregador": "0",
    "tipoestabelecimento": "1",
    "tipodedeficiencia": "0",
    "indtrabintermitente": "0",
    "indtrabparcial": "0",
    "salario": "2500,50",
    "tamestabjan": "5",
    "indicadoraprendiz": "0",
    "origemdainformacao": "1",
    "competenciadec": "202411",
    "unidadesalariocodigo": "5",
    "valorsalariofixo": "2500,50",
}


def _make_raw_row(**overrides) -> dict:
    """Retorna uma cópia de SAMPLE_RAW_ROW com overrides aplicados."""
    row = SAMPLE_RAW_ROW.copy()
    row.update(overrides)
    return row


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """DataFrame bruto (strings) com 4 registros, simulando dados raw do CAGED."""
    rows = [
        _make_raw_row(saldomovimentacao="1", competenciamov="202411"),
        _make_raw_row(saldomovimentacao="-1", competenciamov="202412"),
        _make_raw_row(saldomovimentacao="1", competenciamov="202501"),
        _make_raw_row(saldomovimentacao="-1", competenciamov="202511"),
    ]
    return pd.DataFrame(rows, columns=SAMPLE_RAW_COLUMNS)


@pytest.fixture
def sample_raw_df_large() -> pd.DataFrame:
    """DataFrame bruto com 20 registros para testes que precisam de mais dados."""
    rows = []
    for i in range(20):
        saldo = "1" if i % 2 == 0 else "-1"
        comp = ["202411", "202412", "202501", "202511"][i % 4]
        rows.append(_make_raw_row(
            saldomovimentacao=saldo,
            competenciamov=comp,
            idade=str(20 + i),
        ))
    return pd.DataFrame(rows, columns=SAMPLE_RAW_COLUMNS)


@pytest.fixture
def sample_processed_df() -> pd.DataFrame:
    """DataFrame já processado (tipos corretos), pronto para treino/avaliação."""
    from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL

    n = 100
    data = {}
    for col in NUMERIC_FEATURES:
        data[col] = list(range(n))
    for col in CATEGORICAL_FEATURES:
        data[col] = [f"cat_{i % 5}" for i in range(n)]
    # Target balanceado: metade 1, metade -1
    data[TARGET_COL] = [1 if i % 2 == 0 else -1 for i in range(n)]

    return pd.DataFrame(data)


@pytest.fixture
def tmp_csv(tmp_path) -> Path:
    """Cria um CSV temporário válido e retorna o caminho."""
    csv_path = tmp_path / "test_data.csv"
    rows = [
        _make_raw_row(saldomovimentacao="1"),
        _make_raw_row(saldomovimentacao="-1"),
    ]
    df = pd.DataFrame(rows, columns=SAMPLE_RAW_COLUMNS)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def tmp_empty_csv(tmp_path) -> Path:
    """Cria um CSV temporário vazio (só cabeçalho) e retorna o caminho."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text(",".join(SAMPLE_RAW_COLUMNS) + "\n")
    return csv_path


@pytest.fixture
def tmp_malformed_csv(tmp_path) -> Path:
    """Cria um CSV temporário com conteúdo inválido."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text('"col1","col2\n1,2,3,4\n"a","b","c\n')
    return csv_path
