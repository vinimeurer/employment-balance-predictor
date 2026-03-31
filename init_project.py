from pathlib import Path


class ProjectInitializer:
    """
    Classe responsável por garantir a existência da estrutura básica
    de diretórios do projeto.

    Essa inicialização evita erros relacionados a caminhos inexistentes
    durante a execução do pipeline de dados e modelagem.
    """

    def __init__(self):
        """
        Inicializa a lista de diretórios necessários para o projeto.
        """
        self.directories = [
            Path("data/raw"),
            Path("data/processed"),
            Path("data/outputs"),
            Path("models"),
            Path("notebooks"),
        ]

    def create_directories(self):
        """
        Cria todos os diretórios definidos na classe.

        Caso os diretórios já existam, nenhuma ação é tomada devido ao
        parâmetro `exist_ok=True`.
        """
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"[OK] Diretório garantido: {directory}")

    def run(self):
        """
        Executa o processo completo de inicialização do projeto,
        garantindo a criação da estrutura de diretórios.
        """
        print("Inicializando estrutura do projeto...\n")
        self.create_directories()
        print("\nEstrutura pronta!")


if __name__ == "__main__":
    initializer = ProjectInitializer()
    initializer.run()