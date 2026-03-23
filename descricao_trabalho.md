# Aprendizado de máquina - atividade discente supervisionada 1

**Prof. Mozart Hasse**

---

**LEIA ATENTAMENTE TODAS AS INTRUÇÕES ATÉ O FINAL DA ÚLTIMA PÁGINA.  
CADA PALAVRA CONTA!**

Use exclusivamente os dados da base pública fornecida pelo professor. Descubra tipos, faixas de valores e distribuições através de análise exploratória. Dados a usar: base do CAGED de Curitiba e região metropolitana, dos meses novembro (2024 e 2025), dezembro (2024 e 2025) e janeiro (2024 e 2025). Há dados de outros meses, que a equipe precisa entender por que eles vieram e por que o professor recomenda ignorá-los.

---

**Ajustes adicionais poderão ser necessários dependendo do algoritmo escolhido.**

**O que o algoritmo de aprendizado de máquina implementado pela equipe deve fazer é PREVER O VALOR DA COLUNA _saldomovimentacao_, com base em todos os outros campos fornecidos, para o mês de JANEIRO/2026.**

---

SUA IMPLEMENTAÇÃO DEVE RECEBER COMO ENTRADA O ARQUIVO FORNECIDO PELO PROFESSOR E GERAR COMO SAÍDA UM ARQUIVO CONTENDO OS MESMOS DADOS DO ARQUIVO DE ENTRADA, TENDO CADA LINHA ACRESCIDA COM A PREVISÃO CORRESPONDENTE DA COLUNA _saldomovimentacao_.

---

NÃO É PERMITIDO complementar estes dados com informações de outras bases de dados, use apenas os dados fornecidos no arquivo ou valores deduzidos através de fórmulas. Se quiser e achar conveniente, pode usar os dados dos meses anteriores e montar seu classificador/regressor com base em quaisquer totalizações ou cálculos executados sobre o período fornecido.

---

Não custa avisar que a base fornecida NÃO FOI planejada nem tem como ser eficiente com algoritmos de séries temporais. O objetivo é usar um classificador e/ou regressor. Não há como esperar um classificador com grande poder preditivo pois não está incluída na base nenhuma espécie de indicador econômico ou de variação nos períodos, que obviamente são fatores relevantes para fazer a previsão mas que não fazem parte deste trabalho devido ao prazo disponível.

Você precisará tratar os dados antes de aplicar o algoritmo:

- Selecionar campos relevantes  
- Tratar nulos  
- Agrupar ou totalizar linhas  
- Se alguma conversão de dados for feita, você terá de converter também os dados fornecidos no dataset de teste para fazer as previsões. O importante é manter o leiaute proposto no arquivo de saída solicitado.

---

## 📊 Dicionário de Dados — Novo CAGED (microdados de movimentação)

| Campo              	| Tipo   | Descrição                        | Valores / Observações						|
|-----------------------|--------|----------------------------------|-------------------------------------------------------------------|
| competênciamov    	| int    | Mês da movimentação              | Formato YYYYMM       						|
| região            	| int    | Região geográfica (IBGE)         | 1 Norte, 2 Nordeste, 3 Sudeste, 4 Sul, 5 Centro-Oeste 		|
| uf                	| int    | Unidade da Federação             | Código IBGE          						|
| município         	| int    | Município                        | Código IBGE (7 dígitos) 						|
| seção             	| string | Seção CNAE                       | Letra (A–U)          						|
| subclasse         	| int    | CNAE 2.0 subclasse               | 7 dígitos            						|
| saldomovimentação 	| int    | Tipo agregado de movimentação    | 1 admissões, -1 desligamento 					|
| cbo2002ocupação   	| int    | Código da ocupação               | Tabela CBO 2002      						|
| categoria         	| int    | Categoria do vínculo             | Ex: 101 = empregado CLT 						|
| grauDeinstrução   	| int    | Escolaridade                     | 1 a 8 (ver abaixo)   						|
| idade             	| int    | Idade do trabalhador             | Em anos              						|
| horascontratuais  	| string | Jornada semanal                  | Formato com vírgula (ex: 44,00) 					|
| raçaCor           	| int    | Raça/cor                         | 1 branca, 2 preta, 3 parda, 4 amarela, 5 indígena, 9 ignorado 	|
| sexo                  | int    | Sexo                             | 1 masc, 2 fem, 3 não informado 					|
| tipoempregador        | int    | Tipo do empregador               | 0 PJ, 1 PF 							|
| tiposestabelecimento  | int    | Tipo do estabelecimento          | 1 empresa (outros possíveis) 					|
| tipodeficiencia       | int    | Tipo de deficiência              | 0 nenhuma 							|
| indtrabintermitente   | int    | Trabalho intermitente            | 0 não, 1 sim 							|
| indtrabparcial        | int    | Trabalho parcial                 | 0 não, 1 sim 							|
| salário               | float  | Salário mensal total             | Pode incluir variáveis 						|
| tamestabjan           | int    | Tamanho do estabelecimento       | Faixas (1 até 8+) 						|
| indicadoraprendiz     | int    | Indica aprendiz                  | 0 não, 1 sim 							|
| origemdainformação    | int    | Origem do registro               | 1 eSocial, 2 CAGED antigo 					|
| competênciadec        | int    | Competência da declaração        | YYYYMM 								|
| unidadesalariocodigo  | int    | Unidade do salário               | 5 mensal (mais comum) 						|
| valorSalariofixo      | float  | Salário fixo                     | Sem variáveis 							|

---

## 🎓 Escolaridade (grauDeinstrução)

| Código | Descrição                   |
|--------|-----------------------------|
| 1      | Analfabeto                  |
| 2      | Fundamental incompleto      |
| 3      | Fundamental completo        |
| 4      | Médio incompleto            |
| 5      | Médio completo              |
| 6      | Superior incompleto         |
| 7      | Superior completo           |
| 8      | Pós-graduação               |

---

## 🏢 Tamanho do estabelecimento (tamestabjan)

| Código | Faixa de empregados |
|--------|---------------------|
| 1      | Até 4               |
| 2      | 5 a 9               |
| 3      | 10 a 19             |
| 4      | 20 a 49             |
| 5      | 50 a 99             |
| 6      | 100 a 249           |
| 7      | 250 a 999           |
| 8      | 1000+               |

---

**A solução deste problema DEVE OBRIGATORIAMENTE ser buscada usando um algoritmo de aprendizado de máquina que não envolva serviços externos nem algoritmos relacionados a redes neurais artificiais (execução 100% na máquina local).**


---

# Critérios de avaliação

**Organização e clareza do código: (40% da nota)**, incluindo testes unitários automatizados e incorporados na solução segundo as melhores práticas de mercado, comentários com justificativa para as escolhas feitas na análise e na otimização. Durante a execução do código deve rodar EXCLUSIVAMENTE EM UMA MÁQUINA SEM CONEXÃO COM A INTERNET.

**Qualidade do tratamento de dados e análise exploratória: (30% da nota)**, o que inclui tudo o que for feito antes da aplicação do algoritmo.

**Score F1 do modelo em relação ao baseline do professor: (30% da nota)**. O professor usará um baseline com um algoritmo extremamente simples e comparará os resultados: o score F1 da diferença entre as previsões e os resultados corretos. A nota será proporcional ao ganho de desempenho na base de testes do PROFESSOR em relação à taxa de acertos do algoritmo de baseline.

## Instruções para entrega

O trabalho deve ser entregue em UM arquivo ZIP contendo:

- o notebook Python ou código-fonte necessário para executar o tratamento de dados, a análise exploratória e o algoritmo;
- o arquivo de saída CSV com os dados de teste fornecidos pelo professor, acrescidos de uma coluna com as respostas previstas pelo seu algoritmo para as linhas correspondentes. A única diferença entre o arquivo fornecido pelo professor e o resultado da equipe deve ser a coluna com as previsões.

## Observações gerais

O trabalho pode ser feito em equipes de até 4 alunos. **A EQUIPE TODA É IGUALMENTE RESPONSÁVEL PELO SUCESSO DO TRABALHO.**

**CUIDADO:** aqui está se avaliando tanto o resultado gerado quanto o código e algoritmo escolhidos. Apresente um código compreensível por todos os membros da equipe, especialmente quanto aos parâmetros escolhidos.

**Não use este documento com ferramentas de IA pois o professor vê isso como uma atitude, hum... contraproducente.** Ao pedir algo a uma IA, explique o que você precisa com as SUAS palavras.

**É TERMINANTEMENTE PROIBIDO compartilhar arquivos entre equipes.** Qualquer tentativa de fazer isso implicará na atribuição de nota ZERO a TODOS os membros de TODAS as equipes envolvidas. Casos suspeitos passarão por prova de autoria, portanto todos os membros da equipe devem saber como o código funciona e os motivos de cada escolha feita.

Cabe lembrar que compartilhar VERBALMENTE caminhos bem e mal sucedidos é permitido. O único cuidado é não compartilhar também os eventuais valores de parâmetros de configuração, que devem ter uma diferença de no mínimo 10% para mais ou para menos em pelo menos um parâmetro da implementação para cada equipe.
