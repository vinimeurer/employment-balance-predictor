"""
Módulo de predição.

Carrega o arquivo de teste do professor, aplica o mesmo processamento
dos dados de treino, gera previsões com o modelo treinado e salva
o resultado no formato exigido (arquivo original + coluna de previsão).
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from config import (
    PREDICTION_FILE,
    TRAINED_MODEL_PATH,
    OUTPUT_PREDICTIONS_FILE,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.data.load_data import LoadData
from src.data.process_data import ProcessData


class PredictModel:
    """
    Classe responsável por gerar previsões usando o modelo treinado.
    """

    def __init__(self) -> None:
        """
        Inicializa a classe de predição.

        Parameters:
            None.

        Returns:
            None.
        """
        self.pipeline = None

    def load_trained_model(self) -> Pipeline:
        """
        Carrega o pipeline treinado salvo em disco via joblib.

        Parameters:
            None.

        Returns:
            Pipeline: pipeline sklearn treinada (preprocessor + classifier).

        Raises:
            FileNotFoundError: se o arquivo do modelo não existir.
            TypeError: se o objeto carregado não for uma Pipeline.
        """
        if not TRAINED_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo treinado não encontrado: {TRAINED_MODEL_PATH}. "
                "Execute o treinamento antes de gerar previsões."
            )

        model = joblib.load(TRAINED_MODEL_PATH)

        if not isinstance(model, Pipeline):
            raise TypeError(
                f"O objeto carregado não é uma Pipeline sklearn. "
                f"Tipo encontrado: {type(model).__name__}."
            )

        self.pipeline = model
        return model

    def process_test_data(self) -> pd.DataFrame:
        """
        Carrega e processa o arquivo de teste do professor usando
        o mesmo pipeline de ProcessData do treino.

        Parameters:
            None.

        Returns:
            pd.DataFrame: DataFrame processado (sem a coluna saldomovimentacao).

        Raises:
            FileNotFoundError: se o arquivo de teste não existir.
            KeyError: se colunas esperadas estiverem ausentes.
            ValueError: se o DataFrame ficar vazio após processamento.
        """
        df_raw = LoadData(PREDICTION_FILE).load_csv()
        processor = ProcessData(df_raw)
        processor.convert_column_types()
        processor.create_temporal_features()
        processor.select_columns()

        if processor.df.empty:
            raise ValueError(
                "O DataFrame de teste ficou vazio após o processamento."
            )

        return processor.df

    def run(self) -> None:
        """
        Pipeline completo de predição:
            1. Carrega modelo treinado
            2. Processa dados de teste
            3. Gera previsões (mapeia 0 → -1, 1 → 1)
            4. Salva CSV com dados originais + coluna de previsão

        Parameters:
            None.

        Returns:
            None.

        Raises:
            FileNotFoundError: se o modelo treinado ou o arquivo de teste não existirem.
            KeyError: se colunas de features estiverem ausentes nos dados de teste.
            PermissionError: se não houver permissão de escrita no diretório de saída.
            OSError: se ocorrer erro de I/O ao criar diretório ou gravar arquivo.
        """
        # 1. Carregar modelo
        print("Carregando modelo treinado...")
        self.load_trained_model()

        # 2. Processar dados de teste
        print("Processando dados de teste...")
        df_test = self.process_test_data()

        # 3. Preparar features na mesma ordem do treino
        feature_cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
        missing_cols = [c for c in feature_cols if c not in df_test.columns]
        if missing_cols:
            raise KeyError(
                f"Colunas de features ausentes nos dados de teste: {missing_cols}"
            )

        X_test = df_test[feature_cols]

        # 4. Gerar previsões e mapear de volta ao domínio original
        y_pred_encoded = self.pipeline.predict(X_test)
        # 0 → -1 (desligamento), 1 → 1 (admissão)
        y_pred = pd.Series(y_pred_encoded).map({1: 1, 0: -1}).values

        # 5. Montar DataFrame de saída: dados do teste + coluna de previsão
        df_output = df_test.copy()
        df_output["saldomovimentacao"] = y_pred

        # 6. Salvar
        try:
            OUTPUT_PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise PermissionError(
                f"Sem permissão para criar o diretório: {OUTPUT_PREDICTIONS_FILE.parent}"
            )
        except OSError as e:
            raise OSError(
                f"Erro ao criar diretório {OUTPUT_PREDICTIONS_FILE.parent}: {e}"
            )

        try:
            df_output.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)
        except PermissionError:
            raise PermissionError(
                f"Sem permissão para gravar o arquivo: {OUTPUT_PREDICTIONS_FILE}"
            )
        except OSError as e:
            raise OSError(
                f"Erro ao gravar o arquivo {OUTPUT_PREDICTIONS_FILE}: {e}"
            )

        print(f"Previsões salvas em: {OUTPUT_PREDICTIONS_FILE}")
        print(f"Total de registros: {len(df_output):,}")
