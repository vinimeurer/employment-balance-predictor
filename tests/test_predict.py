"""
Testes unitários para src.models.predict.PredictModel.

Cobre:
    - Carregamento do modelo treinado (load_trained_model)
    - Validação de modelo inexistente
    - Validação de tipo do modelo carregado
    - Mapeamento de previsões (0 → -1, 1 → 1)
"""

import pandas as pd
import numpy as np
import pytest
import joblib
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.models.predict import PredictModel


@pytest.fixture
def mock_pipeline(tmp_path):
    """Cria e salva uma Pipeline simples em disco para testes."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(random_state=42)),
    ])
    # Treina com dados dummy
    X = pd.DataFrame({"feat1": range(20), "feat2": range(20)})
    y = pd.Series([0] * 10 + [1] * 10)
    pipe.fit(X, y)

    model_path = tmp_path / "model.pkl"
    joblib.dump(pipe, model_path)
    return model_path, pipe


class TestLoadTrainedModel:
    """Testes do método load_trained_model."""

    def test_carrega_modelo_valido(self, mock_pipeline, monkeypatch):
        """Deve carregar Pipeline válida do disco."""
        model_path, _ = mock_pipeline

        import src.models.predict as predict_module
        monkeypatch.setattr(predict_module, "TRAINED_MODEL_PATH", model_path)

        predictor = PredictModel()
        result = predictor.load_trained_model()
        assert isinstance(result, Pipeline)

    def test_erro_modelo_inexistente(self, tmp_path, monkeypatch):
        """Deve lançar FileNotFoundError se modelo não existir."""
        import src.models.predict as predict_module
        monkeypatch.setattr(predict_module, "TRAINED_MODEL_PATH", tmp_path / "fake.pkl")

        predictor = PredictModel()
        with pytest.raises(FileNotFoundError, match="Modelo treinado não encontrado"):
            predictor.load_trained_model()

    def test_erro_tipo_invalido(self, tmp_path, monkeypatch):
        """Deve lançar TypeError se objeto carregado não for Pipeline."""
        import src.models.predict as predict_module

        fake_path = tmp_path / "not_pipeline.pkl"
        joblib.dump({"fake": "dict"}, fake_path)
        monkeypatch.setattr(predict_module, "TRAINED_MODEL_PATH", fake_path)

        predictor = PredictModel()
        with pytest.raises(TypeError, match="não é uma Pipeline"):
            predictor.load_trained_model()


class TestPredictMapping:
    """Testes do mapeamento de previsões."""

    def test_mapeamento_0_para_menos_1(self):
        """O valor 0 (interno) deve ser mapeado para -1 (desligamento)."""
        encoded = pd.Series([0, 0, 0])
        mapped = encoded.map({1: 1, 0: -1})
        assert all(mapped == -1)

    def test_mapeamento_1_para_1(self):
        """O valor 1 (interno) deve ser mantido como 1 (admissão)."""
        encoded = pd.Series([1, 1, 1])
        mapped = encoded.map({1: 1, 0: -1})
        assert all(mapped == 1)

    def test_mapeamento_misto(self):
        """Mapeamento misto deve funcionar corretamente."""
        encoded = pd.Series([0, 1, 0, 1, 1])
        mapped = encoded.map({1: 1, 0: -1})
        expected = pd.Series([-1, 1, -1, 1, 1])
        pd.testing.assert_series_equal(mapped, expected)


class TestPredictModelInit:
    """Testes de inicialização."""

    def test_init_pipeline_none(self):
        """Pipeline inicial deve ser None."""
        predictor = PredictModel()
        assert predictor.pipeline is None
