"""
Testes unitários para src.data.load_data.LoadData.

Cobre:
    - Carregamento de CSV válido
    - Validação de tipo do file_path
    - Arquivo inexistente
    - CSV vazio (só cabeçalho)
    - Todos os dados lidos como string (preservação raw)
"""

import pandas as pd
import pytest
from pathlib import Path

from src.data.load_data import LoadData


class TestLoadDataInit:
    """Testes de inicialização e validação do construtor."""

    def test_init_com_path_valido(self, tmp_csv):
        """Deve aceitar Path válido sem lançar exceção."""
        loader = LoadData(tmp_csv)
        assert loader.file_path == tmp_csv

    def test_init_com_string_valida(self, tmp_csv):
        """Deve aceitar string como caminho e converter para Path."""
        loader = LoadData(str(tmp_csv))
        assert isinstance(loader.file_path, Path)

    def test_init_tipo_invalido_int(self):
        """Deve lançar TypeError ao receber tipo numérico."""
        with pytest.raises(TypeError, match="file_path deve ser Path ou str"):
            LoadData(12345)

    def test_init_tipo_invalido_lista(self):
        """Deve lançar TypeError ao receber lista."""
        with pytest.raises(TypeError, match="file_path deve ser Path ou str"):
            LoadData(["/tmp/file.csv"])

    def test_init_arquivo_inexistente(self, tmp_path):
        """Deve lançar FileNotFoundError para arquivo inexistente."""
        fake_path = tmp_path / "nao_existe.csv"
        with pytest.raises(FileNotFoundError, match="Arquivo não encontrado"):
            LoadData(fake_path)


class TestLoadDataLoadCsv:
    """Testes do método load_csv."""

    def test_load_csv_retorna_dataframe(self, tmp_csv):
        """Deve retornar um pd.DataFrame."""
        df = LoadData(tmp_csv).load_csv()
        assert isinstance(df, pd.DataFrame)

    def test_load_csv_nao_vazio(self, tmp_csv):
        """O DataFrame retornado não pode estar vazio."""
        df = LoadData(tmp_csv).load_csv()
        assert not df.empty

    def test_load_csv_colunas_corretas(self, tmp_csv):
        """Deve conter todas as colunas do CSV original."""
        from tests.conftest import SAMPLE_RAW_COLUMNS
        df = LoadData(tmp_csv).load_csv()
        assert list(df.columns) == SAMPLE_RAW_COLUMNS

    def test_load_csv_tipos_string(self, tmp_csv):
        """Todos os campos devem ser lidos como string (dtype=str)."""
        df = LoadData(tmp_csv).load_csv()
        for col in df.columns:
            assert pd.api.types.is_string_dtype(df[col]), f"Coluna '{col}' não é string"

    def test_load_csv_vazio_lanca_excecao(self, tmp_empty_csv):
        """Deve lançar ValueError se CSV tem cabeçalho mas nenhum registro."""
        with pytest.raises(ValueError, match="não contém registros"):
            LoadData(tmp_empty_csv).load_csv()

    def test_load_csv_preserva_numero_linhas(self, tmp_csv):
        """Deve preservar o número de linhas do arquivo."""
        df = LoadData(tmp_csv).load_csv()
        assert len(df) == 2
