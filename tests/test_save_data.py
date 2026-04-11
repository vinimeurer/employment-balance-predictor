"""
Testes unitários para src.data.save_data.SaveData.

Cobre:
    - Validação do construtor (tipos e DataFrame vazio)
    - Salvamento de CSV em diretório existente
    - Criação automática de diretório inexistente
    - Conteúdo salvo corresponde ao DataFrame original
"""

import pandas as pd
import pytest
from pathlib import Path

from src.data.save_data import SaveData


class TestSaveDataInit:
    """Testes de inicialização e validação do construtor."""

    def test_init_com_parametros_validos(self, tmp_path):
        """Deve aceitar DataFrame e Path válidos."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        saver = SaveData(df, tmp_path / "out.csv")
        assert saver.output_path == tmp_path / "out.csv"

    def test_init_aceita_string_como_path(self):
        """Deve aceitar string como output_path e converter para Path."""
        df = pd.DataFrame({"a": [1]})
        saver = SaveData(df, "/tmp/out.csv")
        assert isinstance(saver.output_path, Path)

    def test_init_df_tipo_invalido(self, tmp_path):
        """Deve lançar TypeError se df não for DataFrame."""
        with pytest.raises(TypeError, match="df deve ser um pd.DataFrame"):
            SaveData("não sou dataframe", tmp_path / "out.csv")

    def test_init_df_vazio(self, tmp_path):
        """Deve lançar ValueError se DataFrame estiver vazio."""
        with pytest.raises(ValueError, match="DataFrame fornecido está vazio"):
            SaveData(pd.DataFrame(), tmp_path / "out.csv")

    def test_init_path_tipo_invalido(self):
        """Deve lançar TypeError se output_path não for Path nem str."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(TypeError, match="output_path deve ser Path ou str"):
            SaveData(df, 12345)


class TestSaveDataSaveCsv:
    """Testes do método save_csv."""

    def test_salva_arquivo_csv(self, tmp_path):
        """Deve criar o arquivo CSV no caminho especificado."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        output = tmp_path / "resultado.csv"
        SaveData(df, output).save_csv()
        assert output.exists()

    def test_conteudo_salvo_correto(self, tmp_path):
        """O CSV salvo deve conter os mesmos dados do DataFrame."""
        df = pd.DataFrame({"x": [10, 20, 30], "y": ["a", "b", "c"]})
        output = tmp_path / "check.csv"
        SaveData(df, output).save_csv()

        df_lido = pd.read_csv(output)
        pd.testing.assert_frame_equal(df_lido, df)

    def test_cria_diretorio_inexistente(self, tmp_path):
        """Deve criar diretórios intermediários automaticamente."""
        output = tmp_path / "sub" / "dir" / "deep" / "out.csv"
        df = pd.DataFrame({"v": [1]})
        SaveData(df, output).save_csv()
        assert output.exists()

    def test_salva_sem_indice(self, tmp_path):
        """O CSV não deve conter coluna de índice."""
        df = pd.DataFrame({"a": [1, 2]})
        output = tmp_path / "no_index.csv"
        SaveData(df, output).save_csv()

        df_lido = pd.read_csv(output)
        assert list(df_lido.columns) == ["a"]

    def test_numero_linhas_preservado(self, tmp_path):
        """O número de linhas salvas deve ser igual ao do DataFrame."""
        df = pd.DataFrame({"v": list(range(50))})
        output = tmp_path / "rows.csv"
        SaveData(df, output).save_csv()

        df_lido = pd.read_csv(output)
        assert len(df_lido) == 50
