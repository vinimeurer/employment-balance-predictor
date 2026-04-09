from config import TRAIN_FILE, PROCESSED_TRAIN_FILE
from init_project import ProjectInitializer
from src.data.load_data import LoadData
from src.data.process_data import ProcessData
from src.data.save_data import SaveData
from src.models.train import TrainModel
from src.models.predict import PredictModel

if __name__ == "__main__":
    print("=" * 80)
    print("INICIANDO PIPELINE DE TREINAMENTO E PREDIÇÃO")
    print("=" * 80)
    
    try:
        # Garante a existência da estrutura de diretórios do projeto
        ProjectInitializer().run()

        # Carrega os dados brutos, processa e salva o resultado
        print("\n" + "=" * 80)
        print("PROCESSAMENTO DOS DADOS")
        print("=" * 80)
        print("\nProcessando os dados de treinamento...")
        df = LoadData(TRAIN_FILE).load_csv()
        print(f"Dataset carregado: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
        df = ProcessData(df).process()
        print(f"Dataset processado: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
        SaveData(df, PROCESSED_TRAIN_FILE).save_csv()
        print("\nDados de treinamento processados e salvos com sucesso!")
        print(f"Dados salvos em: {PROCESSED_TRAIN_FILE}")

        # Treina o modelo (grid search + avaliação no teste holdout)
        print("\n" + "=" * 80)
        print("TREINAMENTO DO MODELO")
        print("=" * 80)
        TrainModel().run()

        # Gera previsões para o arquivo de teste do professor
        print("\n" + "=" * 80)
        print("GERAÇÃO DE PREVISÕES")
        print("=" * 80)
        PredictModel().run()
        
        print("=" * 80)
        print("PIPELINE FINALIZADO COM SUCESSO")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Erro durante a execução do pipeline: {e}")
        raise