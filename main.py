from config import TRAIN_FILE, PROCESSED_TRAIN_FILE
from init_project import ProjectInitializer
from src.data.load_data import LoadData
from src.data.process_data import ProcessData
from src.data.save_data import SaveData

if __name__ == "__main__":
    # Garante a existência da estrutura de diretórios do projeto
    ProjectInitializer().run()

    # Carrega os dados brutos, processa e salva o resultado
    df = LoadData(TRAIN_FILE).load_csv()
    df = ProcessData(df).process()
    SaveData(df, PROCESSED_TRAIN_FILE).save_csv()