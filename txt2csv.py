import pandas as pd


def normalize_column_name(column_name: str) -> str:
    """
    Normaliza nomes de colunas:
    - minúsculas
    - remove acentos
    - substitui caracteres especiais por underscore
    """
    return (
        column_name
        .strip()
        .lower()
        .replace("ç", "c")
        .replace("ã", "a")
        .replace("á", "a")
        .replace("à", "a")
        .replace("â", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("õ", "o")
        .replace("ú", "u")
        .replace(" ", "_")
    )


def main() -> None:
    input_file = "caged.txt"
    output_file = "caged.csv"

    # Leitura do arquivo CAGED
    df = pd.read_csv(
        input_file,
        sep=";",
        encoding="utf-8",
        decimal=",",
        dtype=str  # evita problemas de parsing prematuro
    )

    # Normalização dos nomes das colunas
    df.columns = [normalize_column_name(col) for col in df.columns]

    # Conversão explícita de colunas numéricas relevantes
    numeric_columns = [
        "saldomovimentacao",
        "horascontratuais",
        "salario",
        "valorsalariofixo",
        "idade"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )

    # Grava CSV final padronizado
    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
        sep=","
    )


if __name__ == "__main__":
    main()