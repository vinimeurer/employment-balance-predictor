"""
Testes unitários para src.models.evaluate.EvaluateModel.

Cobre:
    - Validação do construtor (tipos, tamanhos, vazios)
    - Cálculo de métricas com modelo real
    - Retorno de dicionário com chaves esperadas
    - Valores de métricas dentro de faixas válidas [0, 1]
"""

import pandas as pd
import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.models.evaluate import EvaluateModel


@pytest.fixture
def trained_pipeline():
    """Pipeline simples treinada para testes de avaliação."""
    np.random.seed(42)
    X = pd.DataFrame({
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100),
    })
    y = pd.Series([0, 1] * 50)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe, X, y


class TestEvaluateModelInit:
    """Testes de inicialização e validação do construtor."""

    def test_init_valido(self, trained_pipeline):
        """Deve aceitar pipeline, X e y válidos."""
        pipe, X, y = trained_pipeline
        evaluator = EvaluateModel(pipe, X, y)
        assert evaluator.pipeline is pipe

    def test_init_pipeline_tipo_invalido(self):
        """Deve lançar TypeError se pipeline não for Pipeline."""
        X = pd.DataFrame({"a": [1]})
        y = pd.Series([0])
        with pytest.raises(TypeError, match="pipeline deve ser sklearn"):
            EvaluateModel("nao_sou_pipeline", X, y)

    def test_init_x_tipo_invalido(self, trained_pipeline):
        """Deve lançar TypeError se X não for DataFrame."""
        pipe, _, y = trained_pipeline
        with pytest.raises(TypeError, match="X deve ser um pd.DataFrame"):
            EvaluateModel(pipe, [[1, 2]], y)

    def test_init_y_tipo_invalido(self, trained_pipeline):
        """Deve lançar TypeError se y não for Series."""
        pipe, X, _ = trained_pipeline
        with pytest.raises(TypeError, match="y deve ser um pd.Series"):
            EvaluateModel(pipe, X, [0, 1])

    def test_init_x_vazio(self, trained_pipeline):
        """Deve lançar ValueError se X estiver vazio."""
        pipe, _, _ = trained_pipeline
        with pytest.raises(ValueError, match="X não pode estar vazio"):
            EvaluateModel(pipe, pd.DataFrame(), pd.Series([0]))

    def test_init_y_vazio(self, trained_pipeline):
        """Deve lançar ValueError se y estiver vazio."""
        pipe, X, _ = trained_pipeline
        with pytest.raises(ValueError, match="y não pode estar vazio"):
            EvaluateModel(pipe, X, pd.Series(dtype=int))

    def test_init_tamanhos_diferentes(self, trained_pipeline):
        """Deve lançar ValueError se X e y tiverem tamanhos diferentes."""
        pipe, _, _ = trained_pipeline
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1])
        with pytest.raises(ValueError, match="mesmo tamanho"):
            EvaluateModel(pipe, X, y)


class TestEvaluateModelEvaluate:
    """Testes do método evaluate."""

    def test_retorna_dicionario(self, trained_pipeline):
        """Deve retornar um dicionário de métricas."""
        pipe, X, y = trained_pipeline
        metrics = EvaluateModel(pipe, X, y).evaluate()
        assert isinstance(metrics, dict)

    def test_chaves_esperadas(self, trained_pipeline):
        """O dicionário deve conter todas as métricas esperadas."""
        pipe, X, y = trained_pipeline
        metrics = EvaluateModel(pipe, X, y).evaluate()

        expected_keys = {
            "f1_binary", "f1_macro", "f1_weighted",
            "accuracy", "balanced_accuracy", "mcc",
        }
        assert expected_keys == set(metrics.keys())

    def test_metricas_entre_zero_e_um(self, trained_pipeline):
        """Métricas de probabilidade devem estar entre 0 e 1."""
        pipe, X, y = trained_pipeline
        metrics = EvaluateModel(pipe, X, y).evaluate()

        for key in ["f1_binary", "f1_macro", "f1_weighted", "accuracy", "balanced_accuracy"]:
            assert 0.0 <= metrics[key] <= 1.0, f"Métrica '{key}' fora do intervalo [0, 1]"

    def test_mcc_entre_menos_um_e_um(self, trained_pipeline):
        """MCC deve estar entre -1 e 1."""
        pipe, X, y = trained_pipeline
        metrics = EvaluateModel(pipe, X, y).evaluate()
        assert -1.0 <= metrics["mcc"] <= 1.0

    def test_modelo_perfeito_f1_alto(self):
        """Modelo que decora os dados de treino deve ter F1 ~1.0."""
        np.random.seed(0)
        X = pd.DataFrame({"f1": list(range(20)), "f2": list(range(20))})
        y = pd.Series([0] * 10 + [1] * 10)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=0)),
        ])
        pipe.fit(X, y)

        metrics = EvaluateModel(pipe, X, y).evaluate()
        assert metrics["f1_binary"] > 0.9
