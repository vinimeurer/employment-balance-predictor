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
        Apenas converte colunas que existem no DataFrame, permitindo processamento
        de dados de teste que não possuem a coluna target 'saldomovimentacao'.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            KeyError: se nenhuma coluna do mapeamento existir no DataFrame.
        """
        processed_cols = []
        skipped_cols = []
        
        for col, dtype in COLUMN_TYPES_MAP:
            # Pula colunas que não existem (como 'saldomovimentacao' em dados de teste)
            if col not in self.df.columns:
                skipped_cols.append(col)
                continue
                
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
            
            processed_cols.append(col)
        
        print(f"Conversão de tipos: {len(processed_cols)} colunas processadas")
        if skipped_cols:
            print(f"Colunas não encontradas (esperado em dados de teste): {skipped_cols}")

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
        do config.py que existem no DataFrame, descartando todas as demais.
        
        Nota: Tolera colunas ausentes (como 'saldomovimentacao' em dados de teste)
        selecionando apenas as que estão presentes.

        Parameters:
            None.

        Raises:
            KeyError: se nenhuma coluna de COLUMNS_TO_KEEP existir no DataFrame.

        Returns:
            None.
        """
        # Seleciona apenas colunas que existem
        cols_to_keep = [col for col in COLUMNS_TO_KEEP if col in self.df.columns]
        missing_cols = set(COLUMNS_TO_KEEP) - set(self.df.columns)
        
        if not cols_to_keep:
            raise KeyError(
                f"Nenhuma coluna de COLUMNS_TO_KEEP encontrada no DataFrame. "
                f"Esperadas: {COLUMNS_TO_KEEP}. Existentes: {list(self.df.columns)}"
            )
        
        if missing_cols:
            print(
                f"Colunas esperadas não preservadas (esperado em dados de teste): "
                f"{missing_cols}. Processando com {len(cols_to_keep)}/{len(COLUMNS_TO_KEEP)} colunas."
            )
        
        print(f"Selecionadas {len(cols_to_keep)} de {len(COLUMNS_TO_KEEP)} colunas esperadas")
        self.df = self.df[cols_to_keep]

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
