import pandas as pd
from pathlib import Path


class LoadData:
    """
    Classe responsável por carregar arquivos CSV em DataFrames pandas.
    """

    def __init__(self, file_path: Path) -> None:
        """
        Inicializa a classe com o caminho do arquivo.

        Parameters:
            file_path (Path): caminho completo do arquivo CSV a ser carregado.
        
        Returns:
            None.

        Raises:
            TypeError: se file_path não for do tipo Path ou str.
            FileNotFoundError: se o arquivo não existir no caminho informado.
        """
        if not isinstance(file_path, (Path, str)):
            raise TypeError(
                f"file_path deve ser Path ou str, recebeu {type(file_path).__name__}."
            )

        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {self.file_path}"
            )

    def load_csv(self) -> pd.DataFrame:
        """
        Carrega o arquivo CSV como DataFrame, sem aplicar nenhuma transformação.
        Todos os campos são lidos como string para preservar os dados brutos.

        Parameters:
            None.

        Returns:
            pd.DataFrame: DataFrame com todos os dados do CSV lidos como string.

        Raises:
            pd.errors.EmptyDataError: se o arquivo CSV estiver vazio.
            pd.errors.ParserError: se o arquivo CSV estiver mal formatado.
        """
        try:
            df = pd.read_csv(self.file_path, dtype=str)
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(
                f"O arquivo CSV está vazio: {self.file_path}"
            )
        except pd.errors.ParserError:
            raise pd.errors.ParserError(
                f"Erro ao interpretar o CSV: {self.file_path}"
            )

        if df.empty:
            raise ValueError(
                f"O arquivo foi lido mas não contém registros: {self.file_path}"
            )

        return df
