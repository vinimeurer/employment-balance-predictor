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

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
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
    XGBOOST_PARAMS,
)
from src.models.evaluate import EvaluateModel


class TrainModel:
    """
    Classe responsável pelo pipeline completo de treinamento do modelo XGBoost.
    """

    def __init__(self) -> None:
        """
        Inicializa a classe de treinamento.

        Parameters:
            None.

        Returns:
            None.
        """
        self.preprocessor = None
        self.pipeline = None
        self.best_params = None
        self.best_val_score = None

    def build_preprocessor(self) -> ColumnTransformer:
        """
        Constrói o ColumnTransformer com pipelines distintos para features
        numéricas (imputação por mediana) e categóricas (imputação + OrdinalEncoder).

        Parameters:
            None.

        Returns:
            ColumnTransformer: transformador configurado com pipelines numérico e categórico.

        Raises:
            ValueError: se NUMERIC_FEATURES ou CATEGORICAL_FEATURES estiverem vazias.
        """
        if not NUMERIC_FEATURES and not CATEGORICAL_FEATURES:
            raise ValueError(
                "NUMERIC_FEATURES e CATEGORICAL_FEATURES estão ambas vazias. "
                "Ao menos uma deve conter colunas."
            )

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

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, list(NUMERIC_FEATURES)),
                ("cat", categorical_pipeline, list(CATEGORICAL_FEATURES)),
            ],
            remainder="drop",
        )
        return self.preprocessor

    def split_data(
        self,
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
            df (pd.DataFrame): DataFrame processado contendo features e a coluna target.

        Returns:
            Tuple: (X_train, X_val, X_train_full, X_test,
                    y_train, y_val, y_train_full, y_test).

        Raises:
            TypeError: se df não for um pd.DataFrame.
            ValueError: se o DataFrame estiver vazio.
            KeyError: se a coluna target ou alguma feature estiver ausente no DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"df deve ser um pd.DataFrame, recebeu {type(df).__name__}."
            )
        if df.empty:
            raise ValueError("O DataFrame fornecido está vazio.")

        feature_cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
        missing_features = [c for c in feature_cols if c not in df.columns]
        if missing_features:
            raise KeyError(
                f"Colunas de features ausentes no DataFrame: {missing_features}"
            )
        if TARGET_COL not in df.columns:
            raise KeyError(
                f"Coluna target '{TARGET_COL}' ausente no DataFrame."
            )

        X = df[feature_cols].copy()

        # XGBoost espera labels >= 0: mapeia -1 → 0, mantém 1 → 1
        y = df[TARGET_COL].map({1: 1, -1: 0})

        if y.isna().any():
            raise ValueError(
                f"Valores inesperados na coluna '{TARGET_COL}'. "
                "Apenas 1 e -1 são permitidos."
            )

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



    def train_final_model(
        self,
        preprocessor: ColumnTransformer,
        X_train_full: pd.DataFrame,
        y_train_full: pd.Series,
        best_params: Dict,
    ) -> Pipeline:
        """
        Treina o pipeline final (preprocessor + XGBClassifier) no treino completo
        usando os melhores hiperparâmetros encontrados pelo grid search.

        Parameters:
            preprocessor (ColumnTransformer): transformador de features já construído.
            X_train_full (pd.DataFrame): features do treino completo (treino + validação).
            y_train_full (pd.Series): labels do treino completo (0 ou 1).
            best_params (dict): melhores hiperparâmetros encontrados no grid search.

        Returns:
            Pipeline: pipeline sklearn treinada (preprocessor + classifier).

        Raises:
            TypeError: se best_params não for um dicionário.
            ValueError: se best_params estiver vazio ou X_train_full estiver vazio.
        """
        if not isinstance(best_params, dict):
            raise TypeError(
                f"best_params deve ser um dict, recebeu {type(best_params).__name__}."
            )
        if not best_params:
            raise ValueError("best_params não pode estar vazio.")
        if X_train_full.empty:
            raise ValueError("X_train_full não pode estar vazio.")

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
        self.pipeline = pipe
        return pipe

    def save_model(self, pipeline: Pipeline) -> None:
        """
        Salva o pipeline treinado em disco via joblib.

        Parameters:
            pipeline (Pipeline): pipeline sklearn treinada a ser persistida.

        Returns:
            None.

        Raises:
            TypeError: se pipeline não for uma instância de Pipeline.
            PermissionError: se não houver permissão de escrita no diretório.
            OSError: se ocorrer erro de I/O ao criar diretório ou gravar arquivo.
        """
        if not isinstance(pipeline, Pipeline):
            raise TypeError(
                f"pipeline deve ser sklearn.pipeline.Pipeline, recebeu {type(pipeline).__name__}."
            )

        try:
            TRAINED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise PermissionError(
                f"Sem permissão para criar o diretório: {TRAINED_MODEL_PATH.parent}"
            )
        except OSError as e:
            raise OSError(
                f"Erro ao criar diretório {TRAINED_MODEL_PATH.parent}: {e}"
            )

        try:
            joblib.dump(pipeline, TRAINED_MODEL_PATH)
        except PermissionError:
            raise PermissionError(
                f"Sem permissão para gravar o arquivo: {TRAINED_MODEL_PATH}"
            )
        except OSError as e:
            raise OSError(
                f"Erro ao gravar o modelo em {TRAINED_MODEL_PATH}: {e}"
            )

        print(f"Modelo salvo em: {TRAINED_MODEL_PATH}")

    def run(self) -> Pipeline:
        """
        Orquestra o pipeline completo de treinamento:
            1. Carrega dados processados
            2. Divide em treino_completo (80%) e teste (20%)
            3. Treina modelo com hiperparâmetros definidos em config.py
            4. Avalia no teste holdout
            5. Salva pipeline

        Parameters:
            None.

        Returns:
            Pipeline: pipeline sklearn treinada e salva em disco.

        Raises:
            FileNotFoundError: se o arquivo de dados processados não existir.
            pd.errors.EmptyDataError: se o arquivo CSV estiver vazio.
            KeyError: se colunas esperadas estiverem ausentes nos dados.
            ValueError: se os dados ficarem vazios após o carregamento.
        """
        if not PROCESSED_TRAIN_FILE.exists():
            raise FileNotFoundError(
                f"Arquivo de dados processados não encontrado: {PROCESSED_TRAIN_FILE}"
            )

        # 1. Carregar dados
        print("Carregando dados processados...")
        df = pd.read_csv(PROCESSED_TRAIN_FILE)

        if df.empty:
            raise ValueError(
                f"O arquivo foi lido mas não contém registros: {PROCESSED_TRAIN_FILE}"
            )

        print(f"Dataset: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

        # 2. Dividir dados
        print("\nDividindo dados (estratificado)...")
        (X_train, X_val, X_train_full, X_test,
         y_train, y_val, y_train_full, y_test) = self.split_data(df)

        print(f"  Treino completo: {X_train_full.shape[0]:>10,}")
        print(f"  Teste holdout  : {X_test.shape[0]:>10,}")

        # 3. Preprocessador
        self.build_preprocessor()

        # 4. Treinar modelo com hiperparâmetros configurados
        print("\nTreinando modelo com hiperparâmetros definidos em config.py...")
        print(f"Hiperparâmetros: {XGBOOST_PARAMS}")
        pipeline = self.train_final_model(self.preprocessor, X_train_full, y_train_full, XGBOOST_PARAMS)

        # 5. Avaliar no teste holdout
        print("\nAvaliação no conjunto de TESTE holdout:")
        EvaluateModel(pipeline, X_test, y_test).evaluate()

        # 6. Salvar
        self.save_model(pipeline)

        return pipeline
