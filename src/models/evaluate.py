"""
Módulo de avaliação do modelo.

Calcula e exibe métricas de classificação, priorizando F1 Score
conforme critério de avaliação do trabalho.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.pipeline import Pipeline


class EvaluateModel:
    """
    Classe responsável por avaliar um pipeline sklearn treinado em um conjunto de dados.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """
        Inicializa a classe com o pipeline treinado e os dados de avaliação.

        Parameters:
            pipeline (Pipeline): pipeline sklearn treinada (preprocessor + classifier).
            X (pd.DataFrame): features do conjunto a avaliar.
            y (pd.Series): labels verdadeiros (0 = desligamento, 1 = admissão).

        Returns:
            None.

        Raises:
            TypeError: se pipeline não for Pipeline, X não for DataFrame ou y não for Series.
            ValueError: se X ou y estiverem vazios, ou se tiverem tamanhos diferentes.
        """
        if not isinstance(pipeline, Pipeline):
            raise TypeError(
                f"pipeline deve ser sklearn.pipeline.Pipeline, recebeu {type(pipeline).__name__}."
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"X deve ser um pd.DataFrame, recebeu {type(X).__name__}."
            )
        if not isinstance(y, pd.Series):
            raise TypeError(
                f"y deve ser um pd.Series, recebeu {type(y).__name__}."
            )
        if X.empty:
            raise ValueError("X não pode estar vazio.")
        if y.empty:
            raise ValueError("y não pode estar vazio.")
        if len(X) != len(y):
            raise ValueError(
                f"X e y devem ter o mesmo tamanho. X tem {len(X)}, y tem {len(y)}."
            )

        self.pipeline = pipeline
        self.X = X
        self.y = y

    def evaluate(self) -> Dict[str, float]:
        """
        Avalia o pipeline treinado em um conjunto de dados (X, y).

        Métricas reportadas:
            - F1 Score (binary) — métrica principal do trabalho
            - F1 Score (macro e weighted)
            - Accuracy e Balanced Accuracy
            - MCC (Matthews Correlation Coefficient)
            - Classification Report completo
            - Matriz de confusão

        Parameters:
            None.

        Returns:
            Dict[str, float]: dicionário com as métricas calculadas.
        """
        y_pred = self.pipeline.predict(self.X)

        metrics = {
            "f1_binary": f1_score(self.y, y_pred),
            "f1_macro": f1_score(self.y, y_pred, average="macro"),
            "f1_weighted": f1_score(self.y, y_pred, average="weighted"),
            "accuracy": accuracy_score(self.y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(self.y, y_pred),
            "mcc": matthews_corrcoef(self.y, y_pred),
        }

        # Labels legíveis para os relatórios
        target_names = ["Desligamento (-1)", "Admissão (1)"]

        print("=" * 80)
        print(f"  F1 Score (binary)    : {metrics['f1_binary']:.4f}  ← métrica principal")
        print(f"  F1 Score (macro)     : {metrics['f1_macro']:.4f}")
        print(f"  F1 Score (weighted)  : {metrics['f1_weighted']:.4f}")
        print(f"  Accuracy             : {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy    : {metrics['balanced_accuracy']:.4f}")
        print(f"  MCC                  : {metrics['mcc']:.4f}")
        print("-" * 80)
        print("Classification Report:")
        print(classification_report(self.y, y_pred, target_names=target_names, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(self.y, y_pred))
        print("=" * 80)

        return metrics
