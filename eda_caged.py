"""
Análise Exploratória de Dados – Microdados CAGED

Fonte: Novo CAGED - Ministério do Trabalho e Emprego
Objetivo: preparar dados com valores semânticos amigáveis para uso em dashboards.
"""

from pathlib import Path
import pandas as pd


# =========================
# DICIONÁRIOS DE DOMÍNIO
# =========================

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
    9: "Não informada",
}

MAP_GRAU_INSTRUCAO = {
    0: '?',
    1: "Analfabeto",
    2: "Até 5ª incompleta",
    3: "5ª completa",
    4: "6ª a 9ª incompleta",
    5: "Ensino Fundamental Completo",
    6: "Ensino Médio Incompleto",
    7: "Ensino Médio Completo",
    8: "Superior Incompleto",
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


MUNICIPIOS_RMC_IBGE = {
    410010,  # Adrianópolis
    410020,  # Agudos do Sul
    410030,  # Almirante Tamandaré
    410040,  # Araucária
    410070,  # Balsa Nova
    410060,  # Bocaiúva do Sul
    410150,  # Campina Grande do Sul
    410160,  # Campo do Tenente
    410180,  # Campo Largo
    410190,  # Campo Magro
    410200,  # Cerro Azul
    410230,  # Colombo
    410270,  # Contenda
    410315,  # Doutor Ulysses
    410765,  # Fazenda Rio Grande
    410820,  # Itaperuçu
    411070,  # Lapa
    411125,  # Mandirituba
    411620,  # Piên
    411915,  # Pinhais
    411950,  # Piraquara
    412080,  # Quatro Barras
    412220,  # Rio Branco do Sul
    412540,  # Rio Negro
    412550,  # São José dos Pinhais
    412610,  # Tijucas do Sul
    412780,  # Tunas do Paraná
    410690,  # Curitiba
    412170,  # Quitandinha
}


def filtrar_regiao_metropolitana_curitiba(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra apenas vínculos de municípios pertencentes
    à Região Metropolitana de Curitiba (RMC).
    """
    df = df.copy()
    df["municipio"] = df["municipio"].astype(int)

    return df[df["municipio"].isin(MUNICIPIOS_RMC_IBGE)]


# =========================
# FUNÇÕES DE TRANSFORMAÇÃO
# =========================

def carregar_dados(caminho_csv: Path) -> pd.DataFrame:
    """
    Carrega o CSV do CAGED e faz ajustes iniciais de tipo.
    """
    df = pd.read_csv(caminho_csv, sep=",", dtype=str)

    # Conversões numéricas relevantes
    df["idade"] = df["idade"].astype(float)
    df["horascontratuais"] = df["horascontratuais"].astype(float)

    # Salário vem com vírgula como separador decimal
    df["salario"] = (
        df["salario"]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    return df


def decodificar_campos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Substitui códigos por valores semânticos amigáveis.
    """
    df = df.copy()

    df["regiao_desc"] = df["regiao"].astype(int).map(MAP_REGIAO)
    df["sexo_desc"] = df["sexo"].astype(int).map(MAP_SEXO)
    df["raca_cor_desc"] = df["racacor"].astype(int).map(MAP_RACA_COR)
    df["grau_instrucao_desc"] = df["graudeinstrucao"].astype(int).map(MAP_GRAU_INSTRUCAO)
    df["tipo_movimentacao_desc"] = df["saldomovimentacao"].astype(int).map(MAP_TIPO_MOVIMENTACAO)
    df["tipo_estabelecimento_desc"] = df["tipoestabelecimento"].astype(int).map(MAP_TIPO_ESTABELECIMENTO)
    df["tipo_empregador_desc"] = df["tipoempregador"].astype(int).map(MAP_TIPO_EMPREGADOR)
    df["deficiencia_desc"] = df["tipodedeficiencia"].astype(int).map(MAP_DEFICIENCIA)
    df["aprendiz_desc"] = df["indicadoraprendiz"].astype(int).map(MAP_APRENDIZ)

    return df


# =========================
# ANÁLISE EXPLORATÓRIA
# =========================

import pandas as pd


def analise_exploratoria(df: pd.DataFrame) -> None:
    """
    Executa análises exploratórias básicas orientadas a dashboard.
    Garante que NUNCA seja exibida notação científica.
    """

    with pd.option_context(
        "display.float_format",
        lambda x: f"{x:,.2f}"
    ):
        print("\n=== RESUMO GERAL ===")
        print(df.describe(include="all"))

        print("\n=== MOVIMENTAÇÕES ===")
        print(df["tipo_movimentacao_desc"].value_counts())

        print("\n=== DISTRIBUIÇÃO POR SEXO (%) ===")
        print((df["sexo_desc"].value_counts(normalize=True) * 100).round(2))

        print("\n=== DISTRIBUIÇÃO POR RAÇA/COR (%) ===")
        print((df["raca_cor_desc"].value_counts(normalize=True) * 100).round(2))

        print("\n=== SALÁRIO (R$) ===")
        print(df["salario"].describe())

        print("\n=== SALÁRIO MÉDIO POR SEXO (R$) ===")
        print(df.groupby("sexo_desc")["salario"].mean().round(2))

        print("\n=== SALÁRIO MÉDIO POR RAÇA/COR (R$) ===")
        print(df.groupby("raca_cor_desc")["salario"].mean().round(2))

        print("\n=== MOVIMENTAÇÕES POR REGIÃO ===")
        print(
            df.groupby(["regiao_desc", "tipo_movimentacao_desc"])
            .size()
            .unstack(fill_value=0)
        )

# =========================
# MAIN
# =========================

def main() -> None:

    caminho_csv = Path("caged.csv")
    caminho_saida = Path("rmc.csv")

    # 1. Carga
    df_raw = carregar_dados(caminho_csv)

    # 2. Filtro geográfico – Região Metropolitana de Curitiba
    df_rmc = filtrar_regiao_metropolitana_curitiba(df_raw)

    # 3. Decodificação semântica (valores amigáveis)
    df_tratado = decodificar_campos(df_rmc)

    # 4. Análise exploratória (opcional, mas útil em dev)
    analise_exploratoria(df_tratado)

    # 5. Persistência do dataset tratado
    salvar_dataset_tratado(df_tratado, caminho_saida)

    print(f"\n✅ Dataset tratado salvo com sucesso em: {caminho_saida.resolve()}")

def salvar_dataset_tratado(df: pd.DataFrame, caminho_saida: Path) -> None:
    """
    Salva o dataset tratado da Região Metropolitana de Curitiba
    em formato CSV, pronto para consumo por ferramentas de BI.
    """
    df_ordenado = df.sort_values(
        by=["competenciamov", "uf", "municipio"]
    )

    df_ordenado.to_csv(
        caminho_saida,
        index=False,
        sep=",",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()