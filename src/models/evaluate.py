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


def evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float]:
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
        pipeline: Pipeline sklearn treinada (preprocessor + classifier).
        X: Features do conjunto a avaliar.
        y: Labels verdadeiros (0 = desligamento, 1 = admissão).

    Returns:
        Dicionário com as métricas calculadas.
    """
    y_pred = pipeline.predict(X)

    metrics = {
        "f1_binary": f1_score(y, y_pred),
        "f1_macro": f1_score(y, y_pred, average="macro"),
        "f1_weighted": f1_score(y, y_pred, average="weighted"),
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "mcc": matthews_corrcoef(y, y_pred),
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
    print(classification_report(y, y_pred, target_names=target_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print("=" * 80)

    return metrics
