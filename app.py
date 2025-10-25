#%%
# ===============================================================
# 🧠 PROJETO: MACHINE LEARNING APLICADO À SAÚDE — DIABETES
# ===============================================================
# 
# 🎯 OBJETIVO GERAL:
# Desenvolver uma aplicação em Python (com interface Streamlit)
# que utilize técnicas de aprendizado supervisionado e não supervisionado
# para analisar e modelar dados clínicos de pacientes com diabetes.
# 
# O foco está em compreender os fatores que influenciam
# a progressão da doença e identificar possíveis grupos
# de pacientes com perfis semelhantes.
# 
# ---------------------------------------------------------------
# 📘 CONTEXTO DO ESTUDO:
# O conjunto de dados utilizado é o "Diabetes Dataset",
# disponível na biblioteca scikit-learn (função load_diabetes()).
# 
# Ele contém informações clínicas padronizadas de 442 pacientes
# diagnosticados com diabetes. Cada registro inclui variáveis como:
# 
# - age: idade do paciente
# - sex: sexo (masculino ou feminino)
# - bmi: índice de massa corporal
# - bp: pressão arterial média
# - s1 a s6: medidas séricas (colesterol total, LDL, HDL, triglicerídeos, etc.)
# - target: medida quantitativa da progressão da doença após 1 ano
# 
# ---------------------------------------------------------------
# 🔍 PROBLEMA A SER INVESTIGADO:
# Todos os pacientes da base já são diabéticos.
# Portanto, o objetivo do estudo não é identificar quem tem ou não diabetes,
# mas sim compreender os fatores que influenciam a progressão da doença.
# 
# A partir disso, o estudo busca responder perguntas como:
# - Quais variáveis estão mais associadas à progressão do diabetes?
# - É possível prever o grau de progressão de um paciente?
# - Existem grupos distintos de pacientes com perfis clínicos semelhantes?
# 
# ---------------------------------------------------------------
# 📊 ANÁLISE EXPLORATÓRIA (EDA):
# A etapa inicial consiste em explorar o comportamento das variáveis:
# 
# - Verificar distribuições, correlações e possíveis padrões clínicos.
# - Identificar relações importantes, como a correlação entre IMC (bmi)
#   e pressão arterial (bp), que se mostraram positivas.
# - Avaliar se fatores como idade e sexo têm impacto significativo.
# 
# Resultados observados:
# - Não há valores ausentes ou inconsistências.
# - Variáveis mais correlacionadas à progressão da doença: bmi e s5.
# - Sexo não apresentou correlação relevante, sendo mantido apenas
#   como dado demográfico.
# - IMC elevado e pressão arterial alta aparecem como principais
#   indicadores de piora na progressão do diabetes.
# 
# ---------------------------------------------------------------
# 🩺 DEFINIÇÃO DO PROBLEMA DE MACHINE LEARNING:
# Com base na análise dos dados, foram definidos três objetivos:

# 1. **Classificação (Aprendizado Supervisionado):**
#    - Classificar os pacientes entre “baixa progressão” e “alta progressão” 
#      da doença com base nas variáveis clínicas.
# 
# 1. **Regressão (Aprendizado Supervisionado):**
#    - Predizer o grau de progressão da doença (target)
#      com base nas variáveis clínicas.
# 
# 2. **Clusterização (Aprendizado Não Supervisionado):**
#    - Agrupar pacientes de acordo com semelhanças clínicas,
#      identificando perfis de risco distintos.
# 
# 3. **Visualização Interativa:**
#    - Criar uma interface intuitiva em Streamlit,
#      permitindo explorar os dados, visualizar gráficos
#      e interpretar os resultados dos modelos.
# 
# ---------------------------------------------------------------
# 📈 INSIGHTS INICIAIS:
# - O aumento do IMC e da pressão arterial tende a elevar a progressão da doença.
# - A idade apresenta variação normal, mas não influencia fortemente o target.
# - Sexo não é um fator determinante, apresentando correlação próxima de zero.
# - Os grupos identificados via clusterização podem ajudar a compreender
#   padrões de risco, apoiando decisões médicas e preventivas.
# 
# ---------------------------------------------------------------
# 🧩 CONCLUSÃO:
# Este projeto demonstra como técnicas de Machine Learning podem ser aplicadas
# para apoiar o diagnóstico e o acompanhamento de doenças crônicas como o diabetes,
# permitindo identificar padrões clínicos e prever a progressão da condição
# com base em dados objetivos.
# 
# A abordagem serve como exemplo de como ciência de dados e saúde
# podem se unir para gerar insights valiosos e contribuir para a medicina preventiva.
# ===============================================================


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve, validation_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, classification_report,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator


# %%
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

print("Formato do dataset:", df.shape)
print("\nColunas disponíveis:", df.columns.to_list())

df.head()

# %%
# %%

print(diabetes.DESCR)

# %%
print(df.info())
print("\nValores ausentes por coluna:\n", df.isnull().sum())
df.describe().T
# %%
plt.figure(figsize=(12, 10))
df.hist(bins=20, figsize=(12, 10), color="#1f77b4", edgecolor="black")
plt.suptitle("Distribuição das variáveis do dataset de Diabetes", fontsize=16)
plt.show()
# %%
# %%
plt.figure(figsize=(6, 4))
sns.scatterplot(x='bmi', y='bp', data=df, color="#2ca02c")
plt.title("Relação entre IMC e Pressão Arterial")
plt.xlabel("Índice de Massa Corporal (bmi)")
plt.ylabel("Pressão Arterial (bp)")
plt.show()

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação entre as Variáveis")
plt.show()
# %%
plt.figure(figsize=(7, 4))
sns.histplot(df['age'], bins=30, kde=True, color="#1f77b4")
plt.title("Distribuição da variável 'age' (idade normalizada)")
plt.xlabel("Idade (valor padronizado)")
plt.ylabel("Frequência")
plt.show()
# %%
df['sexo_label'] = df['sex'].apply(lambda x: 'Masculino' if x > 0 else 'Feminino')
df[['sex', 'sexo_label']].head()
plt.figure(figsize=(6,4))
sns.countplot(x='sexo_label', data=df, palette="pastel")
plt.title("Distribuição de Pacientes por Sexo")
plt.xlabel("Sexo")
plt.ylabel("Quantidade de Pacientes")
plt.show()

# %%
plt.figure(figsize=(7,5))
sns.boxplot(x='sexo_label', y='bmi', data=df, palette="Set2")
plt.title("Distribuição do IMC (bmi) por Sexo")
plt.xlabel("Sexo")
plt.ylabel("Índice de Massa Corporal (IMC)")
plt.show()

# %%
df['faixa_etaria'] = pd.cut(
    df['age'],
    bins=[-0.12, -0.06, 0.0, 0.06, 0.12],
    labels=['Jovens', 'Adultos Jovens', 'Adultos', 'Idosos']
)
plt.figure(figsize=(7,5))
sns.countplot(x='faixa_etaria', data=df, palette='crest')
plt.title("Distribuição de Pacientes por Faixa Etária", fontsize=14)
plt.xlabel("Faixa Etária (estimada)")
plt.ylabel("Quantidade de Pacientes")
plt.show()
# %%
plt.figure(figsize=(8,5))
sns.countplot(x='faixa_etaria', hue='sexo_label', data=df, palette='Set2')
plt.title("Distribuição de Pacientes por Faixa Etária e Sexo", fontsize=14)
plt.xlabel("Faixa Etária (estimada)")
plt.ylabel("Quantidade de Pacientes")
plt.legend(title="Sexo")
plt.show()

# %%
plt.figure(figsize=(6,4))
sns.boxplot(x='sexo_label', y='bmi', data=df, palette="Set2")
plt.title("Distribuição do IMC por Sexo")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='sexo_label', y=diabetes.target, data=df, palette="Set2")
plt.title("Progressão da Doença (target) por Sexo")
plt.show()
# %%
