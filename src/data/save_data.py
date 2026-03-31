import pandas as pd
from pathlib import Path


class SaveData:
    """
    Classe responsável por salvar DataFrames em arquivos CSV.
    """

    def __init__(self, df: pd.DataFrame, output_path: Path) -> None:
        """
        Inicializa a classe com o DataFrame e o caminho de saída.

        Parameters:
            df (pd.DataFrame): DataFrame a ser salvo.
            output_path (Path): caminho completo do arquivo CSV de saída.

        Returns:
            None.

        Raises:
            TypeError: se df não for pd.DataFrame ou output_path não for Path/str.
            ValueError: se o DataFrame estiver vazio.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"df deve ser um pd.DataFrame, recebeu {type(df).__name__}."
            )
        if df.empty:
            raise ValueError("O DataFrame fornecido está vazio. Nada a salvar.")
        if not isinstance(output_path, (Path, str)):
            raise TypeError(
                f"output_path deve ser Path ou str, recebeu {type(output_path).__name__}."
            )

        self.df = df
        self.output_path = Path(output_path)

    def save_csv(self) -> None:
        """
        Salva o DataFrame como CSV no caminho especificado e cria o diretório de destino caso não exista.

        Parameters:
            None.
        
        Returns:
            None.

        Raises:
            PermissionError: se não houver permissão de escrita no diretório.
            OSError: se ocorrer erro de I/O ao criar diretório ou gravar arquivo.


        """
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise PermissionError(
                f"Sem permissão para criar o diretório: {self.output_path.parent}"
            )
        except OSError as e:
            raise OSError(
                f"Erro ao criar diretório {self.output_path.parent}: {e}"
            )

        try:
            self.df.to_csv(self.output_path, index=False)
        except PermissionError:
            raise PermissionError(
                f"Sem permissão para gravar o arquivo: {self.output_path}"
            )
        except OSError as e:
            raise OSError(
                f"Erro ao gravar o arquivo {self.output_path}: {e}"
            )
