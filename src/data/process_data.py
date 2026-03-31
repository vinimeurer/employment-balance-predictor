import pandas as pd
from config import COLUMN_TYPES_MAP, COLUMNS_TO_KEEP, MESES_RELEVANTES, ANOS_RELEVANTES

class ProcessData:
    """
    Classe responsável por aplicar transformações de limpeza e tipagem no DataFrame bruto do CAGED.
    """

    FLOAT_COMMA_COLS = [
        col for col, dtype in COLUMN_TYPES_MAP if dtype == "float"
    ]

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Inicializa a classe com o DataFrame bruto.

        Parameters:
            df (pd.DataFrame): DataFrame com colunas lidas como string.

        Returns:
            None.

        Raises:
            TypeError: se df não for um pd.DataFrame.
            ValueError: se o DataFrame estiver vazio.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"df deve ser um pd.DataFrame, recebeu {type(df).__name__}."
            )
        if df.empty:
            raise ValueError("O DataFrame fornecido está vazio.")

        self.df = df.copy()

    def _validate_columns(self, columns: list[str], context: str) -> None:
        """
        Verifica se todas as colunas esperadas existem no DataFrame.

        Parameters:
            columns (list[str]): lista de nomes de colunas a verificar.
            context (str): descrição do contexto para a mensagem de erro.

        Returns:
            None.

        Raises:
            KeyError: se alguma coluna não for encontrada no DataFrame.
        """
        missing = [c for c in columns if c not in self.df.columns]
        if missing:
            raise KeyError(
                f"Colunas ausentes no DataFrame ({context}): {missing}"
            )

    def convert_column_types(self) -> None:
        """
        Converte as colunas do DataFrame conforme o mapeamento COLUMN_TYPES_MAP definido em config.py. 
        Colunas float com separador decimal são corrigidas antes da conversão. 
        Colunas do tipo 'str' são mantidas como estão.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            KeyError: se alguma coluna do mapeamento não existir no DataFrame.
        """
        col_names = [col for col, _ in COLUMN_TYPES_MAP]
        self._validate_columns(col_names, "convert_column_types")

        for col, dtype in COLUMN_TYPES_MAP:
            if dtype == "Int64":
                self.df[col] = (
                    pd.to_numeric(self.df[col], errors="coerce")
                    .astype("Int64")
                )
            elif dtype == "float":
                self.df[col] = (
                    self.df[col]
                    .str.replace(",", ".", regex=False)
                    .pipe(pd.to_numeric, errors="coerce")
                )

    def create_temporal_features(self) -> None:
        """
        Extrai 'ano' e 'mes' a partir de 'competenciamov' (formato YYYYMM).

        Parameters:
            None.

        Raises:
            KeyError: se a coluna 'competenciamov' não existir no DataFrame.

        Returns:
            None.
        """
        self._validate_columns(["competenciamov"], "create_temporal_features")

        self.df["ano"] = self.df["competenciamov"] // 100
        self.df["mes"] = self.df["competenciamov"] % 100

    def filter_relevant_periods(self) -> None:
        """
        Filtra o DataFrame mantendo apenas registros dos meses e anos relevantes.

        Parameters:
            None.

        Returns:
            None.


        Raises:
            KeyError: se as colunas 'ano' ou 'mes' não existirem no DataFrame.
            ValueError: se nenhum registro sobrar após a filtragem.
        """
        self._validate_columns(["ano", "mes"], "filter_relevant_periods")

        mask = (
            self.df["mes"].isin(MESES_RELEVANTES)
            & self.df["ano"].isin(ANOS_RELEVANTES)
        )
        self.df = self.df[mask].reset_index(drop=True)

        if self.df.empty:
            raise ValueError(
                "Nenhum registro restante após filtrar por meses "
                f"{MESES_RELEVANTES} e anos {ANOS_RELEVANTES}."
            )

    def select_columns(self) -> None:
        """
        Mantém no DataFrame apenas as colunas definidas em COLUMNS_TO_KEEP
        do config.py, descartando todas as demais.

        Parameters:
            None.

        Raises:
            KeyError: se alguma coluna de COLUMNS_TO_KEEP não existir no DataFrame.

        Returns:
            None.
        """
        self._validate_columns(COLUMNS_TO_KEEP, "select_columns")
        self.df = self.df[COLUMNS_TO_KEEP]

    def process(self) -> pd.DataFrame:
        """
        Executa o pipeline completo de processamento:
            - Conversão de tipos conforme COLUMN_TYPES_MAP.
            - Criação de features temporais.
            - Filtragem de períodos relevantes.
            - Seleção de colunas conforme COLUMNS_TO_KEEP.

        Parameters:
            None.

        Returns:
            pd.DataFrame: DataFrame processado e filtrado.

        Raises:
            KeyError: se alguma coluna necessária estiver ausente.
            ValueError: se nenhum registro sobrar após a filtragem.
        """
        self.convert_column_types()
        self.create_temporal_features()
        self.filter_relevant_periods()
        self.select_columns()
        return self.df
