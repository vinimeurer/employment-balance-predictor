"""
Testes unitários para src.data.process_data.ProcessData.

Cobre:
    - Validação do construtor (tipo e DataFrame vazio)
    - Conversão de tipos (Int64, float com vírgula, str)
    - Criação de features temporais (ano, mes)
    - Filtragem de períodos relevantes
    - Seleção de colunas
    - Pipeline completo (process)
    - Tolerância a colunas ausentes (dados de teste sem target)
"""

import pandas as pd
import pytest

from src.data.process_data import ProcessData
from config import COLUMN_TYPES_MAP, COLUMNS_TO_KEEP, MESES_RELEVANTES, ANOS_RELEVANTES


class TestProcessDataInit:
    """Testes de inicialização e validação do construtor."""

    def test_init_com_dataframe_valido(self, sample_raw_df):
        """Deve aceitar DataFrame válido."""
        processor = ProcessData(sample_raw_df)
        assert not processor.df.empty

    def test_init_faz_copia_do_dataframe(self, sample_raw_df):
        """Deve criar cópia do DataFrame, sem alterar o original."""
        df_original = sample_raw_df.copy()
        ProcessData(sample_raw_df)
        pd.testing.assert_frame_equal(sample_raw_df, df_original)

    def test_init_tipo_invalido(self):
        """Deve lançar TypeError ao receber tipo inválido."""
        with pytest.raises(TypeError, match="df deve ser um pd.DataFrame"):
            ProcessData("não sou dataframe")

    def test_init_tipo_invalido_lista(self):
        """Deve lançar TypeError ao receber lista."""
        with pytest.raises(TypeError, match="df deve ser um pd.DataFrame"):
            ProcessData([1, 2, 3])

    def test_init_dataframe_vazio(self):
        """Deve lançar ValueError se DataFrame estiver vazio."""
        with pytest.raises(ValueError, match="DataFrame fornecido está vazio"):
            ProcessData(pd.DataFrame())


class TestValidateColumns:
    """Testes do método _validate_columns."""

    def test_validate_colunas_existentes(self, sample_raw_df):
        """Não deve lançar exceção para colunas existentes."""
        processor = ProcessData(sample_raw_df)
        processor._validate_columns(["competenciamov", "regiao"], "teste")

    def test_validate_colunas_ausentes(self, sample_raw_df):
        """Deve lançar KeyError para colunas inexistentes."""
        processor = ProcessData(sample_raw_df)
        with pytest.raises(KeyError, match="Colunas ausentes"):
            processor._validate_columns(["coluna_fantasma"], "teste")


class TestConvertColumnTypes:
    """Testes do método convert_column_types."""

    def test_converte_int64(self, sample_raw_df):
        """Colunas mapeadas como Int64 devem ser convertidas."""
        processor = ProcessData(sample_raw_df)
        processor.convert_column_types()

        int_cols = [col for col, dtype in COLUMN_TYPES_MAP if dtype == "Int64" and col in processor.df.columns]
        for col in int_cols:
            assert processor.df[col].dtype == "Int64", f"Coluna '{col}' não foi convertida para Int64"

    def test_converte_float_com_virgula(self, sample_raw_df):
        """Colunas float com separador vírgula devem ser convertidas corretamente."""
        processor = ProcessData(sample_raw_df)
        processor.convert_column_types()

        # horascontratuais original: "44,00" → 44.0
        assert processor.df["horascontratuais"].iloc[0] == pytest.approx(44.0)
        # salario original: "2500,50" → 2500.5
        assert processor.df["salario"].iloc[0] == pytest.approx(2500.5)

    def test_tolera_coluna_ausente(self, sample_raw_df):
        """Deve tolerar colunas do mapeamento que não existem no DataFrame."""
        df_sem_saldo = sample_raw_df.drop(columns=["saldomovimentacao"])
        processor = ProcessData(df_sem_saldo)
        # Não deve lançar exceção
        processor.convert_column_types()


class TestCreateTemporalFeatures:
    """Testes do método create_temporal_features."""

    def test_cria_colunas_ano_e_mes(self, sample_raw_df):
        """Deve criar colunas 'ano' e 'mes' a partir de 'competenciamov'."""
        processor = ProcessData(sample_raw_df)
        processor.convert_column_types()
        processor.create_temporal_features()

        assert "ano" in processor.df.columns
        assert "mes" in processor.df.columns

    def test_valores_ano_corretos(self, sample_raw_df):
        """Deve extrair ano corretamente (YYYY de YYYYMM)."""
        processor = ProcessData(sample_raw_df)
        processor.convert_column_types()
        processor.create_temporal_features()

        # competenciamov = 202411 → ano = 2024
        assert processor.df["ano"].iloc[0] == 2024

    def test_valores_mes_corretos(self, sample_raw_df):
        """Deve extrair mês corretamente (MM de YYYYMM)."""
        processor = ProcessData(sample_raw_df)
        processor.convert_column_types()
        processor.create_temporal_features()

        # competenciamov = 202411 → mes = 11
        assert processor.df["mes"].iloc[0] == 11

    def test_erro_sem_competenciamov(self, sample_raw_df):
        """Deve lançar KeyError se 'competenciamov' não existir."""
        df = sample_raw_df.drop(columns=["competenciamov"])
        processor = ProcessData(df)
        with pytest.raises(KeyError, match="Colunas ausentes"):
            processor.create_temporal_features()


class TestFilterRelevantPeriods:
    """Testes do método filter_relevant_periods."""

    def test_filtra_meses_e_anos_relevantes(self, sample_raw_df):
        """Deve manter apenas registros dos meses e anos definidos em config."""
        processor = ProcessData(sample_raw_df)
        processor.convert_column_types()
        processor.create_temporal_features()
        processor.filter_relevant_periods()

        # Todos os meses restantes devem estar em MESES_RELEVANTES
        assert set(processor.df["mes"].unique()).issubset(MESES_RELEVANTES)
        # Todos os anos restantes devem estar em ANOS_RELEVANTES
        assert set(processor.df["ano"].unique()).issubset(ANOS_RELEVANTES)

    def test_remove_registros_irrelevantes(self, sample_raw_df):
        """Registros de meses/anos fora do filtro devem ser removidos."""
        processor = ProcessData(sample_raw_df)
        processor.convert_column_types()
        processor.create_temporal_features()

        n_antes = len(processor.df)
        processor.filter_relevant_periods()
        # Pelo menos o registro com mês fora dos relevantes deve ser removido
        assert len(processor.df) <= n_antes

    def test_erro_sem_colunas_temporais(self, sample_raw_df):
        """Deve lançar KeyError se 'ano' ou 'mes' não existirem."""
        processor = ProcessData(sample_raw_df)
        with pytest.raises(KeyError, match="Colunas ausentes"):
            processor.filter_relevant_periods()

    def test_erro_nenhum_registro_restante(self):
        """Deve lançar ValueError se nenhum registro sobrar após filtragem."""
        # Cria DataFrame com competência que não está nos períodos relevantes
        from tests.conftest import _make_raw_row, SAMPLE_RAW_COLUMNS
        row = _make_raw_row(competenciamov="202306")  # Jun/2023 — fora do filtro
        df = pd.DataFrame([row], columns=SAMPLE_RAW_COLUMNS)

        processor = ProcessData(df)
        processor.convert_column_types()
        processor.create_temporal_features()

        with pytest.raises(ValueError, match="Nenhum registro restante"):
            processor.filter_relevant_periods()


class TestSelectColumns:
    """Testes do método select_columns."""

    def test_mantem_apenas_colunas_esperadas(self, sample_raw_df):
        """Após select_columns, só devem restar colunas de COLUMNS_TO_KEEP."""
        processor = ProcessData(sample_raw_df)
        processor.convert_column_types()
        processor.create_temporal_features()
        processor.select_columns()

        for col in processor.df.columns:
            assert col in COLUMNS_TO_KEEP, f"Coluna inesperada: '{col}'"

    def test_tolera_colunas_ausentes_no_teste(self, sample_raw_df):
        """Deve funcionar mesmo sem a coluna target (cenário de teste)."""
        df = sample_raw_df.drop(columns=["saldomovimentacao"])
        processor = ProcessData(df)
        processor.convert_column_types()
        processor.create_temporal_features()
        # Não deve lançar exceção
        processor.select_columns()
        assert "saldomovimentacao" not in processor.df.columns

    def test_erro_nenhuma_coluna_valida(self):
        """Deve lançar KeyError se nenhuma coluna de COLUMNS_TO_KEEP existir."""
        df = pd.DataFrame({"col_inexistente": [1, 2, 3]})
        processor = ProcessData(df)
        with pytest.raises(KeyError, match="Nenhuma coluna"):
            processor.select_columns()


class TestProcessPipeline:
    """Testes do pipeline completo (método process)."""

    def test_process_retorna_dataframe(self, sample_raw_df):
        """O método process deve retornar um pd.DataFrame."""
        result = ProcessData(sample_raw_df).process()
        assert isinstance(result, pd.DataFrame)

    def test_process_nao_vazio(self, sample_raw_df):
        """O DataFrame retornado não pode estar vazio."""
        result = ProcessData(sample_raw_df).process()
        assert not result.empty

    def test_process_contem_features_temporais(self, sample_raw_df):
        """O resultado deve conter colunas 'ano' e 'mes'."""
        result = ProcessData(sample_raw_df).process()
        assert "ano" in result.columns
        assert "mes" in result.columns

    def test_process_filtra_periodos(self, sample_raw_df):
        """Após process, só devem existir registros de períodos relevantes."""
        result = ProcessData(sample_raw_df).process()
        assert set(result["mes"].unique()).issubset(MESES_RELEVANTES)
