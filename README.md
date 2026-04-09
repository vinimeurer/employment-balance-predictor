

## Estrutura do Projeto

A estrutura do projeto foi baseada no padrão [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/), porém adaptada para simplificar o escopo e atender necessidades específicas do trabalho. A separação entre dados, código-fonte, notebooks e saídas foi adotada para garantir organização, reprodutibilidade e clareza no pipeline de aprendizado de máquina.

```
caged-ml-project/
│
├── data/
│   ├── raw/        # dados originais (NÃO alterar)
│   ├── processed/  # dados tratados
│   └── outputs/    # csv gerado com as predições 
│
├── notebooks/
│   ├── eda.ipynb           # notebook de análise exploratóriam
│   └── model_tuning.ipynb  # notebook para teste de hiperparâmetros
│
├── src/
│   ├── data/
│   │   ├── load_data.py        # código de carregamento dos dados e converte para dataframes
│   │   ├── process_data.py     # código de tratamento de dados
│   │   └── save_data.py        # código para salvar o dataframe em CSV.
│   │
│   └── models/
│       ├── train.py        # código de treinamento do modelo
│       ├── evaluate.py     # código de avaliação do modelo
│       └── predict.py      # código para gerar previsões do modelo
│
├── models/         # diretório para salvar modelos treinados
│
├── requirements.txt    # dependências do projeto
├── README.md           # arquivo de descrição do projeto
├── init_project.py     # código para garantir que os diretórios existam
├── config.py           # configurações do projeto (mapeamentos, caminhos, parametros, etc.)
└── main.py
```
