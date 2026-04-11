
# Documentação do Projeto - Employment Balance Predictor

## Visão Geral

Este projeto implementa um **pipeline de aprendizado de máquina** para prever o saldo de movimentação de emprego (`saldomovimentacao`) na base do CAGED (Cadastro Geral de Empregados e Desempregados) de Curitiba e região metropolitana. O sistema carrega os microdados brutos, aplica tratamento e engenharia de features, treina um classificador **XGBoost** e gera previsões para o conjunto de teste.

## Objetivos

- Prever o saldo de movimentação (admissão ou desligamento) com base nos atributos do CAGED
- Maximizar o **F1 Score** do modelo em relação ao baseline
- Tratar e preparar os microdados brutos para consumo pelo classificador
- Gerar arquivo CSV de saída com as previsões no formato exigido

## Estrutura do Projeto

A estrutura foi baseada no padrão [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/), adaptada para simplificar o escopo e atender necessidades específicas do trabalho. A separação entre dados, código-fonte, notebooks e saídas foi adotada para garantir organização, reprodutibilidade e clareza no pipeline.

```
employment-balance-predictor/
│
├── data/
│   ├── raw/            # dados originais (NÃO alterar)
│   ├── processed/      # dados tratados
│   └── outputs/        # csv gerado com as predições
│
├── notebooks/
│   ├── eda.ipynb              # análise exploratória de dados
│   └── model_tuning.ipynb     # teste de hiperparâmetros
│
├── src/
│   ├── data/
│   │   ├── load_data.py       # carregamento de CSV em DataFrames
│   │   ├── process_data.py    # tratamento e transformação de dados
│   │   └── save_data.py       # persistência de DataFrames em CSV
│   │
│   └── models/
│       ├── train.py           # treinamento do modelo XGBoost
│       ├── evaluate.py        # avaliação com métricas (F1, accuracy, MCC)
│       └── predict.py         # geração de previsões sobre dados de teste
│
├── tests/
│   ├── conftest.py            # fixtures compartilhadas
│   ├── test_load_data.py      # testes de carregamento de dados
│   ├── test_process_data.py   # testes de processamento de dados
│   ├── test_save_data.py      # testes de salvamento de dados
│   ├── test_train.py          # testes de treinamento do modelo
│   ├── test_evaluate.py       # testes de avaliação do modelo
│   ├── test_predict.py        # testes de predição
│   └── test_init_project.py   # testes de inicialização do projeto
│
├── models/             # diretório para salvar modelos treinados
│
├── requirements.txt    # dependências do projeto
├── README.md           # arquivo de descrição do projeto
├── init_project.py     # garantia de estrutura de diretórios
├── config.py           # configurações centralizadas (caminhos, parâmetros, mapeamentos)
└── main.py             # ponto de entrada do pipeline
```

## Componentes Principais

### `config.py`
Arquivo central de configuração do projeto. Contém:
- Caminhos de diretórios e arquivos (`TRAIN_FILE`, `PREDICTION_FILE`, etc.)
- Mapeamentos descritivos para análise exploratória (região, sexo, raça/cor, escolaridade)
- Tipagem de colunas (`COLUMN_TYPES_MAP`)
- Seleção de features (`COLUMNS_TO_KEEP`, `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`)
- Filtros temporais (`MESES_RELEVANTES`, `ANOS_RELEVANTES`)
- Hiperparâmetros do XGBoost (`XGBOOST_PARAMS`)

### `src/data/load_data.py`
Carrega arquivos CSV em DataFrames pandas.
- Lê todos os campos como string para preservar dados brutos
- Valida existência e tipo do caminho do arquivo
- Trata arquivos vazios ou mal formatados

### `src/data/process_data.py`
Aplica transformações de limpeza e tipagem no DataFrame bruto do CAGED.
- Converte tipos conforme `COLUMN_TYPES_MAP` (Int64, float com vírgula, str)
- Cria features temporais (`ano`, `mes`) a partir de `competenciamov`
- Filtra períodos relevantes (nov, dez, jan de 2024 e 2025)
- Seleciona colunas conforme `COLUMNS_TO_KEEP`
- Tolera colunas ausentes (cenário de dados de teste sem coluna target)

### `src/data/save_data.py`
Salva DataFrames em arquivos CSV.
- Cria diretórios intermediários automaticamente
- Valida tipo e conteúdo do DataFrame antes de salvar
- Salva sem coluna de índice

### `src/models/train.py`
Pipeline completo de treinamento do modelo XGBoost.
- Constrói `ColumnTransformer` com pipelines distintos para features numéricas (imputação por mediana) e categóricas (imputação + `OrdinalEncoder`)
- Divide dados em treino completo (80%) e teste (20%), com subdivisão treino/validação
- Treina pipeline final (`preprocessor` + `XGBClassifier`) com hiperparâmetros de `config.py`
- Avalia no teste holdout e salva modelo via `joblib`

### `src/models/evaluate.py`
Avalia um pipeline sklearn treinado em um conjunto de dados.
- Calcula F1 Score (binary, macro, weighted) — **métrica principal do trabalho**
- Accuracy, Balanced Accuracy e MCC
- Exibe Classification Report e Matriz de Confusão

### `src/models/predict.py`
Gera previsões usando o modelo treinado.
- Carrega pipeline treinada do disco
- Processa dados de teste em memória (sem alterar arquivo original)
- Mapeia previsões: `0 → -1` (desligamento), `1 → 1` (admissão)
- Salva CSV com dados **originais** + coluna de previsão (`saldomovimentacao`)

### `init_project.py`
Garante a existência da estrutura de diretórios do projeto.
- Cria `data/raw/`, `data/processed/`, `data/outputs/` e `models/`
- Idempotente: executar múltiplas vezes não causa erro

## Como Executar

1. **Crie um ambiente virtual (recomendado)**

    ```bash
    python -m venv .venv
    ```

2. **Ative o ambiente virtual**

    - Linux/Mac:
        ```bash
        source .venv/bin/activate
        ```

    - Windows:
        ```bash
        .venv\Scripts\activate
        ```

3. **Instale as dependências**

    ```bash
    pip install -r requirements.txt
    ```

4. **Coloque os dados na pasta correta**

    Os arquivos de dados devem estar em `data/raw/`:
    - `caged_curitiba_consolidado_train.csv` — dados de treino
    - `caged_curitiba_consolidado_test.csv` — dados de teste do professor

5. **Execute o pipeline principal**

    ```bash
    python main.py
    ```

6. **Verifique o output**

    Após a execução, será exibido o log completo do pipeline:

    ```
    ================================================================================
    INICIANDO PIPELINE DE TREINAMENTO E PREDIÇÃO
    ================================================================================

    PROCESSAMENTO DOS DADOS
    Dataset carregado: 500.000 linhas × 26 colunas
    Dataset processado: 350.000 linhas × 20 colunas

    TREINAMENTO DO MODELO
    F1 Score (binary)    : 0.6842  ← métrica principal

    GERAÇÃO DE PREVISÕES
    Previsões salvas em: data/outputs/caged_curitiba_consolidado_test_predictions.csv

    PIPELINE FINALIZADO COM SUCESSO
    ================================================================================
    ```

    O arquivo de previsões estará em `data/outputs/caged_curitiba_consolidado_test_predictions.csv`, contendo os mesmos dados do arquivo de teste acrescidos da coluna `saldomovimentacao` com as previsões.

### Fluxo de Execução

1. **Inicialização**: Garante existência dos diretórios do projeto
2. **Carregamento**: Lê `caged_curitiba_consolidado_train.csv` como strings
3. **Processamento**: Converte tipos, cria features temporais, filtra períodos, seleciona colunas
4. **Treinamento**: Divide dados, constrói preprocessor, treina XGBoost, avalia no teste holdout
5. **Predição**: Carrega modelo, processa dados de teste, gera previsões e salva CSV de saída

## Testes

Testes disponíveis:
- `test_load_data.py` - Carregamento e validação de CSVs
- `test_process_data.py` - Conversão de tipos, features temporais, filtragem e seleção
- `test_save_data.py` - Persistência de DataFrames em CSV
- `test_train.py` - Preprocessor, split de dados, treinamento e salvamento do modelo
- `test_evaluate.py` - Cálculo de métricas de classificação
- `test_predict.py` - Carregamento de modelo e mapeamento de previsões
- `test_init_project.py` - Criação de diretórios do projeto

Execute os testes unitários com pytest, medindo a cobertura:

```bash
pytest --cov=src --cov-report term-missing
```

Ou execute testes específicos:

```bash
pytest tests/test_load_data.py -v
pytest tests/test_process_data.py -v
pytest tests/test_train.py -v
```

**IMPORTANTE:** Para o funcionamento adequado, é necessário que todas as dependências estejam corretamente instaladas. Portanto, certifique-se de ter executado o comando `pip install -r requirements.txt` antes de rodar os testes.

## Conceitos Técnicos

### Classificador XGBoost

O modelo utiliza **XGBoost** (`XGBClassifier`) como algoritmo de classificação. A escolha foi feita por:

- **Performance**: XGBoost é consistentemente um dos melhores algoritmos para dados tabulares
- **Robustez**: Lida bem com valores faltantes e features de diferentes escalas
- **Controle de overfitting**: Parâmetros como `max_depth`, `min_child_weight` e `subsample` permitem regularização
- **Execução local**: Não requer serviços externos nem redes neurais, conforme exigido

### Preprocessamento

O pipeline de preprocessamento utiliza `ColumnTransformer` do scikit-learn com dois fluxos:

- **Features numéricas**: Imputação por mediana (`SimpleImputer`) — robusta a outliers
- **Features categóricas**: Imputação por valor constante + `OrdinalEncoder` com tratamento de categorias desconhecidas

### Hiperparâmetros

Os hiperparâmetros do XGBoost configurados em `config.py`:

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `n_estimators` | 800 | Número de árvores suficiente para convergência com learning rate baixo |
| `learning_rate` | 0.03 | Taxa de aprendizado baixa para generalização mais estável |
| `max_depth` | 13 | Profundidade que permite capturar interações complexas entre features |
| `min_child_weight` | 5 | Regularização para evitar splits em partições pequenas |
| `subsample` | 0.9 | Amostragem por árvore para reduzir overfitting |
| `colsample_bytree` | 0.6 | Amostragem de features por árvore para diversidade |

### Mapeamento da Variável Target

A coluna `saldomovimentacao` possui valores `1` (admissão) e `-1` (desligamento). Para compatibilidade com XGBoost (que espera labels ≥ 0):

- **Treino**: `-1 → 0`, `1 → 1`
- **Predição**: `0 → -1`, `1 → 1` (restaura domínio original)

### Métricas de Avaliação

- **F1 Score (binary)** — métrica principal conforme critério de avaliação
- **F1 Score (macro e weighted)** — visão balanceada entre classes
- **Accuracy e Balanced Accuracy** — métricas complementares
- **MCC (Matthews Correlation Coefficient)** — métrica robusta para classes desbalanceadas
- **Classification Report** — precisão, recall e F1 por classe
- **Matriz de Confusão** — distribuição de acertos e erros

### Seleção de Features

Colunas descartadas no processamento e a justificativa:
- `competenciamov`: redundante após extração de `ano` e `mes`
- `regiao`, `uf`: constantes para Curitiba/região metropolitana
- `origemdainformacao`, `competenciadec`: informações administrativas sem poder preditivo
- `unidadesalariocodigo`, `valorsalariofixo`: redundantes com `salario`

### Filtragem de Períodos

Apenas dados dos meses **novembro, dezembro e janeiro** dos anos **2024 e 2025** são utilizados, conforme orientação do professor. Dados de outros meses estão presentes na base mas são descartados por não serem representativos do período de previsão (janeiro/2026).

## Configuração

Para ajustar parâmetros do modelo ou do processamento, edite o arquivo `config.py`:

- **Hiperparâmetros**: Altere `XGBOOST_PARAMS`
- **Features**: Modifique `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES` ou `COLUMNS_TO_KEEP`
- **Períodos**: Ajuste `MESES_RELEVANTES` e `ANOS_RELEVANTES`
- **Split de dados**: Altere `TEST_SIZE`, `VALIDATION_SIZE` ou `RANDOM_STATE`

## Troubleshooting

**Arquivo não encontrado:**
- Certifique-se de que os CSVs de treino e teste existem em `data/raw/`

**Erro de tipos nas colunas:**
- Verifique se o formato do CSV corresponde ao esperado (separador vírgula decimal em `horascontratuais` e `salario`)

**DataFrame vazio após filtragem:**
- Confirme que o CSV contém registros dos meses 11, 12 e 1, anos 2024 e 2025

**Modelo não encontrado ao prever:**
- Execute o treinamento (`python main.py`) antes de gerar previsões isoladas

**Testes falhando:**
- Verifique se as dependências estão instaladas: `pip install -r requirements.txt`
- Execute em Python 3.10+
- Limpe cache: `pytest --cache-clear`

## Estrutura de um Teste

Exemplo de teste unitário do projeto:

```python
import pandas as pd
import pytest
from src.data.process_data import ProcessData

def test_converte_float_com_virgula(sample_raw_df):
    """Colunas float com separador vírgula devem ser convertidas corretamente."""
    processor = ProcessData(sample_raw_df)
    processor.convert_column_types()

    # horascontratuais original: "44,00" → 44.0
    assert processor.df["horascontratuais"].iloc[0] == pytest.approx(44.0)

def test_init_dataframe_vazio():
    """Deve lançar ValueError se DataFrame estiver vazio."""
    with pytest.raises(ValueError, match="DataFrame fornecido está vazio"):
        ProcessData(pd.DataFrame())
```

## Licença

Este código está sob a licença MIT. Você pode usar, copiar, modificar e distribuir este projeto livremente, desde que mantenha o aviso de copyright e a licença incluídos. Para mais detalhes, consulte o arquivo LICENSE.
