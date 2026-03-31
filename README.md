

## Estrutura do Projeto

A estrutura do projeto foi baseada no padrão [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/), porém adaptada para visando simplificar o escopo e atender necessidades específicas do trabalho. A separação entre dados, código-fonte, notebooks e saídas foi adotada para garantir organização, reprodutibilidade e clareza no pipeline de aprendizado de máquina.

```
caged-ml-project/
│
├── data/
│   ├── raw/        # dados originais (NÃO alterar)
│   ├── processed/  # dados tratados
│   └── outputs/    # csv gerado com as predições 
│
├── notebooks/
│   └── eda.ipynb   # notebook de análise exploratória
│
├── src/
│   ├── data/
│   │   ├── load_data.py        # código de carregamento dos dados e converte para dataframes
│   │   └── preprocessing.py    # código de tratamento de dados
│   │
│   ├── models/
|   │   ├── train.py    # código de treinamento do modelo
|   │   ├── evaluate.py # código de avaliação do modelo
|   │   └── predict.py  # código para gerar previsões do modelo
│   │
│   └── config.py   # código de configuração (dicionários de mapeamento, caminhos, configurações de modelo, etc.)
│
├── models/ # diretório para salvar modelos treinados
│
├── requirements.txt
├── README.md
├── init_project.py # código para garantir que os diretórios existam
└── main.py
```