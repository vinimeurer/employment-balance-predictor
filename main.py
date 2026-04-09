from config import TRAIN_FILE, PROCESSED_TRAIN_FILE
from init_project import ProjectInitializer
from src.data.load_data import LoadData
from src.data.process_data import ProcessData
from src.data.save_data import SaveData
from src.models.train import TrainModel
from src.models.predict import PredictModel

if __name__ == "__main__":
    # Garante a existência da estrutura de diretórios do projeto
    ProjectInitializer().run()

    # Carrega os dados brutos, processa e salva o resultado
    print("\n" + "=" * 80)
    print("PROCESSAMENTO DOS DADOS")
    print("=" * 80)
    print("\nProcessando os dados de treinamento...")
    df = LoadData(TRAIN_FILE).load_csv()
    df = ProcessData(df).process()
    SaveData(df, PROCESSED_TRAIN_FILE).save_csv()
    print("\nDados de treinamento processados e salvos com sucesso!")

    # Treina o modelo (grid search + avaliação no teste holdout)
    print("\n" + "=" * 80)
    print("TREINAMENTO DO MODELO")
    print("=" * 80)
    TrainModel().run()

    # # Gera previsões para o arquivo de teste do professor
    # print("\n" + "=" * 80)
    # print("GERAÇÃO DE PREVISÕES")
    # print("=" * 80)
    # PredictModel().run()