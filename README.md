# Predição da Resistência do Concreto com Machine Learning

📌 Sobre o Projeto

Este projeto aplica técnicas de Machine Learning para prever a resistência à compressão do concreto, utilizando dados sobre a composição do material.

A resistência do concreto depende da proporção de diversos componentes, como:

- Cimento

- Água

- Escória de alto forno

- Cinzas volantes

- Agregados

- Aditivos químicos

O objetivo do projeto é treinar modelos de regressão capazes de prever a resistência do concreto a partir da composição da mistura, auxiliando engenheiros a otimizar formulações.

🧠 Tecnologias Utilizadas

- Python

- Pandas

- NumPy

- Seaborn

- Matplotlib

- Scikit-Learn

Modelos utilizados:

- Random Forest Regressor

Linear Regression

📂 Estrutura do Projeto

concrete-strength-prediction/

├── Desafio_Regressão_Aplicada_à_Engenharia_de_Materiais.ipynb

├── dados_concreto_-_Sheet1.csv

└── README.md

📊 Dataset

O dataset contém informações sobre a composição do concreto e sua resistência à compressão.

Principais variáveis:

Variável     	                     Descrição

Cement	                           Quantidade de cimento
Blast                              Furnace Slag	Escória de alto forno
Fly Ash	                           Cinzas volantes
Water	                             Quantidade de água
Superplasticizer	                 Aditivo plastificante
Coarse Aggregate	                 Agregado graúdo
Fine Aggregate	                   Agregado fino
Strength Category	                 Categoria de resistência
Concrete compressive strength	     Resistência do concreto (variável alvo)

A variável alvo é:

- Concrete compressive strength

que representa a resistência à compressão do concreto em MPa.

🔎 Etapas da Análise

1️⃣ Importação das Bibliotecas e Carregamento do Dataset

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

df = pd.read_csv("dados_concreto_-_Sheet1.csv")

O que acontece nesta etapa

São importadas bibliotecas fundamentais para análise de dados.

- Pandas

Utilizado para:

- Manipulação de dados

- Análise estatística

- Criação de DataFrames

- NumPy

Biblioteca utilizada para cálculos matemáticos e manipulação de arrays.

- Seaborn e Matplotlib

Bibliotecas responsáveis por visualização de dados.

Depois disso o dataset é carregado utilizando:

pd.read_csv()

2️⃣ Seleção de Variáveis Numéricas

numeric_df = df.select_dtypes(include=np.number)

Essa etapa seleciona apenas as colunas numéricas do dataset.

Isso é útil porque:

- Muitos modelos de Machine Learning trabalham apenas com números

- Permite análises estatísticas mais diretas

3️⃣ Análise da Relação entre Cimento e Resistência

sns.scatterplot(x="Cement", y="Concrete compressive strength", data=df)

plt.title("Cimento vs Resistência")

plt.show()

Objetivo

Investigar a relação entre:

quantidade de cimento
vs
resistência do concreto

O gráfico scatterplot permite visualizar:

- Padrões de correlação

- Dispersão dos dados

- Tendência entre variáveis

- Insight esperado

Quanto maior a quantidade de cimento, maior tende a ser a resistência do concreto.

4️⃣ Análise da Relação entre Água e Resistência

sns.scatterplot(x="Water", y="Concrete compressive strength", data=df)

Objetivo

Analisar como a quantidade de água influencia a resistência do concreto.

Na engenharia civil é conhecido que:

- Excesso de água reduz a resistência do concreto

Esse gráfico permite observar essa relação diretamente nos dados.

5️⃣ Análise da Resistência por Categoria

df.groupby("Strength Category")["Concrete compressive strength"].mean().plot(kind="barh")

O que essa análise faz

- Agrupa os dados por categoria de resistência e calcula a resistência média para cada grupo.

- Isso ajuda a entender como diferentes categorias se comportam em média.

- O gráfico utilizado é um bar chart horizontal, que facilita a comparação entre categorias.

6️⃣ Verificação de Dados Nulos

df.isnull().sum()

Objetivo

Identificar valores ausentes no dataset.

- Valores nulos podem causar problemas em modelos de machine learning, portanto precisam ser tratados.

7️⃣ Tratamento de Valores Nulos

df['Concrete compressive strength'] = df['Concrete compressive strength'].fillna(

    df['Concrete compressive strength'].mean()
    
)

Aqui os valores faltantes são substituídos pela média da variável.

Essa técnica é chamada de:

- Mean Imputation

Ela mantém a distribuição geral dos dados sem remover registros.

8️⃣ Codificação de Variáveis Categóricas

df = pd.get_dummies(df, drop_first=True)

Algumas variáveis são categóricas (texto), e modelos de machine learning não conseguem trabalhar diretamente com esse tipo de dado.

Por isso é aplicado One-Hot Encoding, que transforma categorias em colunas numéricas.

Exemplo:

Strength Category

pode virar:

Strength_Category_High

Strength_Category_Medium

Strength_Category_Low

9️⃣ Separação entre Variáveis de Entrada e Saída

X = df.drop('Concrete compressive strength', axis=1)

y = df['Concrete compressive strength']

Aqui os dados são divididos em:

X → variáveis independentes

Composição do concreto.

y → variável alvo

Resistência do concreto.

🔟 Divisão entre Treino e Teste

train_test_split()

Os dados são divididos em dois conjuntos:

Conjunto	Função
Treino	    Treinar o modelo
Teste	    Avaliar o desempenho

Essa técnica evita overfitting.

🤖 11️⃣ Treinamento do Modelo Random Forest

RandomForestRegressor(

    n_estimators=200,
    
    random_state=42
)

Random Forest é um algoritmo de ensemble learning baseado em múltiplas árvores de decisão.

Vantagens:

- Alta precisão

- Captura relações não lineares

- Robusto contra overfitting

Parâmetros usados:

n_estimators = 200

Número de árvores no modelo.

12️⃣ Avaliação do Modelo

r2_score()

mean_absolute_error()

São utilizadas duas métricas principais.

R² (Coeficiente de Determinação)

Mede quanto da variabilidade dos dados o modelo consegue explicar.

Valores próximos de 1 indicam melhor desempenho.

MAE (Mean Absolute Error)

Mede o erro médio absoluto das previsões.

Quanto menor o valor, melhor o modelo.

13️⃣ Treinamento de Regressão Linear

LinearRegression()

A regressão linear é um modelo mais simples que assume relação linear entre variáveis.

Ele é utilizado para comparar desempenho com o Random Forest.

📊 14️⃣ Importância das Variáveis

rf.feature_importances_

Essa análise mostra quais variáveis têm maior impacto na previsão da resistência do concreto.

O resultado é exibido em um gráfico:

Feature Importance

Isso ajuda a entender quais componentes da mistura mais influenciam a resistência.

🧪 15️⃣ Predição de Novo Concreto

O modelo também é utilizado para prever a resistência de uma nova composição de concreto.

Exemplo:

Cement: 550

Water: 180

Superplasticizer: 2.5

Coarse Aggregate: 1000

O modelo calcula a resistência estimada para essa mistura.

Essa funcionalidade pode ser usada para simulação de formulações de concreto.

📈 Possíveis Insights

A análise pode revelar padrões como:

- Maior quantidade de cimento tende a aumentar a resistência

- Excesso de água reduz a resistência

- Certos agregados impactam mais o resultado final

- Modelos de machine learning conseguem prever a resistência com boa precisão

🚀 Possíveis Melhorias Futuras

- Utilizar XGBoost ou Gradient Boosting

- Aplicar Cross Validation

criar dashboard interativo

otimizar hiperparâmetros com GridSearch

criar um sistema de recomendação de mistura ideal
