"""
Módulo de treinamento do modelo XGBoost para previsão de saldomovimentacao.

Pipeline:
    1. Carrega dados processados
    2. Divide em treino_completo (80%) e teste (20%)
    3. Subdivide treino_completo em treino (90%) e validação (10%)
    4. Busca melhores hiperparâmetros via ParameterGrid na validação
    5. Retreina modelo final no treino_completo com melhores params
    6. Avalia no teste holdout
    7. Salva pipeline treinada
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from config import (
    PROCESSED_TRAIN_FILE,
    TRAINED_MODEL_PATH,
    TARGET_COL,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
    XGBOOST_PARAM_GRID,
)
from src.models.evaluate import evaluate_model


def build_preprocessor() -> ColumnTransformer:
    """
    Constrói o ColumnTransformer com pipelines distintos para features
    numéricas (imputação por mediana) e categóricas (imputação + OrdinalEncoder).

    Returns:
        ColumnTransformer configurado.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="desconhecido")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("cat", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
    return preprocessor


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Divide os dados em três conjuntos estratificados:
        - treino: para ajustar o modelo durante o grid search
        - validação: para avaliar hiperparâmetros no grid search
        - treino_completo: treino + validação (para retreino final)
        - teste: avaliação final do modelo

    Parameters:
        df: DataFrame processado com features e target.

    Returns:
        X_train, X_val, X_train_full, X_test, y_train, y_val, y_train_full, y_test
    """
    feature_cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    X = df[feature_cols].copy()

    # XGBoost espera labels >= 0: mapeia -1 → 0, mantém 1 → 1
    y = df[TARGET_COL].map({1: 1, -1: 0})

    # Split 1: treino_completo (80%) + teste (20%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Split 2: treino (90% do treino_completo) + validação (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    return X_train, X_val, X_train_full, X_test, y_train, y_val, y_train_full, y_test


def score_on_validation(
    estimator: XGBClassifier,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """
    Treina pipeline em X_train e retorna F1 Score avaliado em X_val.
    Usado exclusivamente para o grid search manual (sem cross-validation).

    Returns:
        F1 Score (binary, pos_label=1) na validação.
    """
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])

    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)

    return f1_score(y_val, y_val_pred)


def grid_search_validation(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    param_grid: Dict,
) -> Tuple[Dict, float]:
    """
    Executa busca de hiperparâmetros via ParameterGrid usando
    single validation split. Avalia cada combinação pelo F1 Score.

    Returns:
        Tupla (melhores_params, melhor_f1_score).
    """
    best_score = -np.inf
    best_params = None

    grid = list(ParameterGrid(param_grid))
    total = len(grid)
    print(f"Testando {total} combinações de hiperparâmetros...")
    print("-" * 80)

    for i, params in enumerate(grid, 1):
        clf = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

        score = score_on_validation(
            estimator=clf,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        marker = " *** NOVO MELHOR" if score > best_score else ""
        print(f"[{i:>{len(str(total))}}/{total}] F1: {score:.4f}{marker} | {params}")

        if score > best_score:
            best_score = score
            best_params = params

    print("-" * 80)
    print(f"Melhores hiperparâmetros: {best_params}")
    print(f"Melhor F1 (validação): {best_score:.4f}")

    return best_params, best_score


def train_final_model(
    preprocessor: ColumnTransformer,
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    best_params: Dict,
) -> Pipeline:
    """
    Treina o pipeline final (preprocessor + XGBClassifier) no treino completo
    usando os melhores hiperparâmetros encontrados pelo grid search.

    Returns:
        Pipeline treinada pronta para predição.
    """
    clf = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    pipe.fit(X_train_full, y_train_full)
    return pipe


def save_model(pipeline: Pipeline) -> None:
    """Salva o pipeline treinado em disco via joblib."""
    TRAINED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, TRAINED_MODEL_PATH)
    print(f"Modelo salvo em: {TRAINED_MODEL_PATH}")


def run_training_pipeline() -> Pipeline:
    """
    Orquestra o pipeline completo de treinamento:
        1. Carrega dados processados
        2. Divide em treino / validação / teste
        3. Grid search na validação (F1 Score)
        4. Retreina modelo final no treino completo
        5. Avalia no teste holdout
        6. Salva pipeline

    Returns:
        Pipeline treinada.
    """
    # 1. Carregar dados
    print("Carregando dados processados...")
    df = pd.read_csv(PROCESSED_TRAIN_FILE)
    print(f"Dataset: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

    # 2. Dividir dados
    print("\nDividindo dados (estratificado)...")
    (X_train, X_val, X_train_full, X_test,
     y_train, y_val, y_train_full, y_test) = split_data(df)

    print(f"  Treino       : {X_train.shape[0]:>10,}")
    print(f"  Validação    : {X_val.shape[0]:>10,}")
    print(f"  Treino compl.: {X_train_full.shape[0]:>10,}")
    print(f"  Teste        : {X_test.shape[0]:>10,}")

    # 3. Preprocessador
    preprocessor = build_preprocessor()

    # 4. Grid search na validação
    print("\nIniciando grid search (single validation split):")
    best_params, best_val_score = grid_search_validation(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        param_grid=XGBOOST_PARAM_GRID,
    )

    # 5. Treinar modelo final no treino completo
    print("\nTreinando modelo FINAL no treino completo com melhores hiperparâmetros...")
    pipeline = train_final_model(preprocessor, X_train_full, y_train_full, best_params)

    # 6. Avaliar no teste holdout
    print("\nAvaliação no conjunto de TESTE holdout:")
    evaluate_model(pipeline, X_test, y_test)

    # 7. Salvar
    save_model(pipeline)

    return pipeline
