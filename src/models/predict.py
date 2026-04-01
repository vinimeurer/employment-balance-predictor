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


def load_trained_model() -> Pipeline:
    """Carrega o pipeline treinado salvo em disco."""
    return joblib.load(TRAINED_MODEL_PATH)


def process_test_data() -> pd.DataFrame:
    """
    Carrega e processa o arquivo de teste do professor usando
    o mesmo pipeline de ProcessData do treino.

    Returns:
        DataFrame processado (sem a coluna saldomovimentacao).
    """
    df_raw = LoadData(PREDICTION_FILE).load_csv()
    processor = ProcessData(df_raw)
    processor.convert_column_types()
    processor.create_temporal_features()
    processor.select_columns()
    return processor.df


def generate_predictions() -> None:
    """
    Pipeline completo de predição:
        1. Carrega modelo treinado
        2. Processa dados de teste
        3. Gera previsões (mapeia 0 → -1, 1 → 1)
        4. Salva CSV com dados originais + coluna de previsão
    """
    # 1. Carregar modelo
    print("Carregando modelo treinado...")
    pipeline = load_trained_model()

    # 2. Processar dados de teste
    print("Processando dados de teste...")
    df_test = process_test_data()

    # 3. Preparar features na mesma ordem do treino
    feature_cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    available_cols = [c for c in feature_cols if c in df_test.columns]
    X_test = df_test[available_cols]

    # 4. Gerar previsões e mapear de volta ao domínio original
    y_pred_encoded = pipeline.predict(X_test)
    # 0 → -1 (desligamento), 1 → 1 (admissão)
    y_pred = pd.Series(y_pred_encoded).map({1: 1, 0: -1}).values

    # 5. Montar DataFrame de saída: dados do teste + coluna de previsão
    df_output = df_test.copy()
    df_output["saldomovimentacao"] = y_pred

    # 6. Salvar
    OUTPUT_PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)
    print(f"Previsões salvas em: {OUTPUT_PREDICTIONS_FILE}")
    print(f"Total de registros: {len(df_output):,}")
