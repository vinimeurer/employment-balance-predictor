"""
Testes unitários para src.models.train.TrainModel.

Cobre:
    - Construção do preprocessor (ColumnTransformer)
    - Divisão de dados (split_data)
    - Treinamento do modelo final (train_final_model)
    - Salvamento e carregamento do modelo
    - Validações de tipo e dados vazios
"""

import pandas as pd
import pytest
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.models.train import TrainModel
from config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    XGBOOST_PARAMS,
)


@pytest.fixture
def trainer():
    """Instância limpa de TrainModel."""
    return TrainModel()


@pytest.fixture
def training_df():
    """DataFrame processado e balanceado para testes de treino."""
    np.random.seed(42)
    n = 200
    data = {}
    for col in NUMERIC_FEATURES:
        data[col] = np.random.rand(n) * 100
    for col in CATEGORICAL_FEATURES:
        data[col] = [f"cat_{i % 5}" for i in range(n)]
    # Target: metade 1 (admissão), metade -1 (desligamento)
    data[TARGET_COL] = [1 if i < n // 2 else -1 for i in range(n)]
    return pd.DataFrame(data)


class TestBuildPreprocessor:
    """Testes do método build_preprocessor."""

    def test_retorna_column_transformer(self, trainer):
        """Deve retornar um ColumnTransformer."""
        preprocessor = trainer.build_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_armazena_preprocessor(self, trainer):
        """Deve armazenar o preprocessor como atributo."""
        trainer.build_preprocessor()
        assert trainer.preprocessor is not None

    def test_contem_transformadores_num_e_cat(self, trainer):
        """Deve conter transformadores 'num' e 'cat'."""
        preprocessor = trainer.build_preprocessor()
        nomes = [name for name, _, _ in preprocessor.transformers]
        assert "num" in nomes
        assert "cat" in nomes


class TestSplitData:
    """Testes do método split_data."""

    def test_retorna_8_elementos(self, trainer, training_df):
        """Deve retornar tupla com 8 elementos (X/y para train, val, full, test)."""
        result = trainer.split_data(training_df)
        assert len(result) == 8

    def test_split_preserva_total_amostras(self, trainer, training_df):
        """A soma train_full + test deve igualar o total de registros."""
        X_train, X_val, X_train_full, X_test, y_train, y_val, y_train_full, y_test = (
            trainer.split_data(training_df)
        )
        assert len(X_train_full) + len(X_test) == len(training_df)

    def test_split_train_full_contem_train_e_val(self, trainer, training_df):
        """train_full deve ter mais amostras que train sozinho."""
        X_train, X_val, X_train_full, X_test, *_ = trainer.split_data(training_df)
        assert len(X_train_full) > len(X_train)
        assert len(X_train_full) == len(X_train) + len(X_val)

    def test_split_labels_mapeadas(self, trainer, training_df):
        """Labels devem ser mapeadas: -1 → 0, 1 → 1."""
        *_, y_train, y_val, y_train_full, y_test = trainer.split_data(training_df)
        for y in [y_train, y_val, y_train_full, y_test]:
            assert set(y.unique()).issubset({0, 1})

    def test_split_tipo_invalido(self, trainer):
        """Deve lançar TypeError se receber tipo inválido."""
        with pytest.raises(TypeError, match="df deve ser um pd.DataFrame"):
            trainer.split_data("string")

    def test_split_df_vazio(self, trainer):
        """Deve lançar ValueError se DataFrame estiver vazio."""
        with pytest.raises(ValueError, match="DataFrame fornecido está vazio"):
            trainer.split_data(pd.DataFrame())

    def test_split_coluna_target_ausente(self, trainer, training_df):
        """Deve lançar KeyError se coluna target estiver ausente."""
        df = training_df.drop(columns=[TARGET_COL])
        with pytest.raises(KeyError, match="Coluna target"):
            trainer.split_data(df)

    def test_split_feature_ausente(self, trainer, training_df):
        """Deve lançar KeyError se alguma feature estiver ausente."""
        col = list(NUMERIC_FEATURES)[0]
        df = training_df.drop(columns=[col])
        with pytest.raises(KeyError, match="Colunas de features ausentes"):
            trainer.split_data(df)


class TestTrainFinalModel:
    """Testes do método train_final_model."""

    def test_retorna_pipeline(self, trainer, training_df):
        """Deve retornar Pipeline sklearn treinada."""
        preprocessor = trainer.build_preprocessor()
        _, _, X_full, _, _, _, y_full, _ = trainer.split_data(training_df)
        pipe = trainer.train_final_model(preprocessor, X_full, y_full, XGBOOST_PARAMS)
        assert isinstance(pipe, Pipeline)

    def test_pipeline_pode_prever(self, trainer, training_df):
        """A pipeline treinada deve ser capaz de gerar previsões."""
        preprocessor = trainer.build_preprocessor()
        _, _, X_full, X_test, _, _, y_full, _ = trainer.split_data(training_df)
        pipe = trainer.train_final_model(preprocessor, X_full, y_full, XGBOOST_PARAMS)
        preds = pipe.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})

    def test_best_params_tipo_invalido(self, trainer, training_df):
        """Deve lançar TypeError se best_params não for dict."""
        preprocessor = trainer.build_preprocessor()
        _, _, X_full, _, _, _, y_full, _ = trainer.split_data(training_df)
        with pytest.raises(TypeError, match="best_params deve ser um dict"):
            trainer.train_final_model(preprocessor, X_full, y_full, "params")

    def test_best_params_vazio(self, trainer, training_df):
        """Deve lançar ValueError se best_params estiver vazio."""
        preprocessor = trainer.build_preprocessor()
        _, _, X_full, _, _, _, y_full, _ = trainer.split_data(training_df)
        with pytest.raises(ValueError, match="best_params não pode estar vazio"):
            trainer.train_final_model(preprocessor, X_full, y_full, {})


class TestSaveModel:
    """Testes do método save_model."""

    def test_salva_modelo_em_disco(self, trainer, training_df, tmp_path, monkeypatch):
        """Deve salvar o modelo treinado no caminho configurado."""
        import config
        model_path = tmp_path / "test_model.pkl"
        monkeypatch.setattr(config, "TRAINED_MODEL_PATH", model_path)

        # Reimporta para usar o path atualizado
        from src.models import train as train_module
        monkeypatch.setattr(train_module, "TRAINED_MODEL_PATH", model_path)

        preprocessor = trainer.build_preprocessor()
        _, _, X_full, _, _, _, y_full, _ = trainer.split_data(training_df)
        pipe = trainer.train_final_model(preprocessor, X_full, y_full, XGBOOST_PARAMS)
        trainer.save_model(pipe)
        assert model_path.exists()

    def test_save_tipo_invalido(self, trainer):
        """Deve lançar TypeError se pipeline não for Pipeline."""
        with pytest.raises(TypeError, match="pipeline deve ser sklearn"):
            trainer.save_model("nao_sou_pipeline")
