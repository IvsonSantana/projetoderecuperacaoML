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
# 2. **Regressão (Aprendizado Supervisionado):**
#    - Predizer o grau de progressão da doença (target)
#      com base nas variáveis clínicas.
# 
# 3. **Clusterização (Aprendizado Não Supervisionado):**
#    - Agrupar pacientes de acordo com semelhanças clínicas,
#      identificando perfis de risco distintos.
# 
# 4. **Visualização Interativa:**
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

st.set_page_config(page_title="ML Saúde — Diabetes", layout="wide")


# %%
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

print("Formato do dataset:", df.shape)
print("\nColunas disponíveis:", df.columns.to_list())

df.head()
#
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
# Modelo de Classificação
df_model = df.select_dtypes(include=['float64', 'int64']).copy()
if 'sex' in df_model.columns:
    df_model.drop(columns=['sex'], inplace=True)
    print("Coluna 'sex' removida com sucesso.")
else:
    print("Coluna 'sex' já não está presente.")
df_model['target'] = diabetes.target

print("\n Colunas numéricas selecionadas para os modelos:")
print(df_model.columns.to_list())
print("\n Valores ausentes no dataset:", df_model.isnull().sum().sum())
print("\n Informações gerais do df_model:")
df_model.info()
# %%
threshold = df_model['target'].median()
df_model['progressao_alta'] = (df_model['target'] >= threshold).astype(int)

X = df_model.drop(columns=['target', 'progressao_alta'])
y = df_model['progressao_alta']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %%
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)
rf.fit(X_train, y_train)
# %%
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)

print("=== Desempenho do Modelo ===")
print(f"Acurácia : {acc:.4f}")
print(f"Precisão : {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")

print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Baixa', 'Alta'])
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão — Random Forest')
plt.show()

# Curva ROC
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title(f'Curva ROC — Random Forest (AUC={auc:.3f})')
plt.show()
# %%
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=importances.index, palette='viridis')
plt.title("Importância das Variáveis - Random Forest")
plt.xlabel("Importância")
plt.ylabel("Variável")
plt.show()

print("\nTop 5 variáveis mais importantes:")
print(importances.head())
# %%

# ===============================================================
# RELATÓRIO DE RESULTADOS — MODELO DE CLASSIFICAÇÃO (Random Forest)
# ===============================================================
#
# OBJETIVO:
# Classificar os pacientes diabéticos entre “baixa progressão” e “alta progressão”
# da doença, com base em variáveis clínicas como IMC, pressão arterial e colesterol.
# A variável-alvo (target) representa a progressão da doença após 1 ano de acompanhamento.
#
# ---------------------------------------------------------------
# MODELO UTILIZADO:
# RandomForestClassifier (n_estimators=300, random_state=42)
#
# Justificativa da escolha:
# - É um modelo robusto contra ruídos e correlações entre variáveis clínicas.
# - Captura relações não lineares entre as variáveis (por exemplo, IMC e colesterol).
# - Permite interpretar a importância das features, o que é fundamental em análises médicas.
# - Garante boa generalização sem necessidade de muitos ajustes de hiperparâmetros.
#
# ---------------------------------------------------------------
# RESULTADOS OBTIDOS:
# Acurácia : 0.7895
# Precisão : 0.7568
# Recall   : 0.8485
# F1-Score : 0.8000
# ROC AUC  : 0.8440
#
# Esses valores indicam que o modelo consegue classificar corretamente cerca de 79% dos casos.
# A alta taxa de recall (≈85%) demonstra que o modelo identifica a maioria dos pacientes
# com alta progressão da doença, o que é desejável no contexto médico.
#
# ---------------------------------------------------------------
# INTERPRETAÇÃO:
# - O modelo mostrou excelente capacidade de distinguir entre pacientes de baixa e alta progressão.
# - O valor de AUC = 0.84 indica um bom poder de separação das classes.
# - A matriz de confusão mostrou:
#     ▫️ 49 pacientes corretamente identificados como “baixa progressão”
#     ▫️ 56 pacientes corretamente identificados como “alta progressão”
#     ▫️ 18 falsos positivos e 10 falsos negativos (quantidade aceitável)
#
# Em um cenário clínico, é preferível ter falsos positivos (monitorar pacientes saudáveis)
# do que falsos negativos (não identificar casos graves). Portanto, o modelo é adequado
# para apoiar decisões médicas preventivas.
#
# ---------------------------------------------------------------
# INSIGHTS CLÍNICOS:
# As variáveis mais relevantes para o modelo foram:
# IMC (bmi) — pacientes com maior índice de massa corporal apresentam progressão mais rápida.
# Pressão Arterial (bp) — níveis elevados se associam à piora do quadro clínico.
# s5 — relacionada ao metabolismo de lipídios e colesterol.
#
# Esses fatores são consistentes com o conhecimento médico sobre complicações do diabetes.
#
# ---------------------------------------------------------------
# CONCLUSÃO:
# O modelo de Random Forest apresentou um bom desempenho e alta sensibilidade na detecção
# de pacientes com alta progressão da doença, sendo capaz de apoiar análises preditivas
# e estratégias de acompanhamento clínico.
#
# Em contextos práticos, poderia ser usado em sistemas de monitoramento preventivo,
# priorizando pacientes que exigem acompanhamento mais próximo e intervenções precoces.
# ===============================================================

# %%
# Modelo de Regressão
X_reg = df_model.drop(columns=['target', 'progressao_alta'])
y_reg = df_model['target']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

scaler_reg = StandardScaler()
Xr_train = scaler_reg.fit_transform(Xr_train)
Xr_test = scaler_reg.transform(Xr_test)

# %%
modelo_lr = LinearRegression()
modelo_lr.fit(Xr_train, yr_train)
# %%
y_pred_reg = modelo_lr.predict(Xr_test)

mae = mean_absolute_error(yr_test, y_pred_reg)
mse = mean_squared_error(yr_test, y_pred_reg)
r2 = r2_score(yr_test, y_pred_reg)

print("=== Desempenho do Modelo de Regressão ===")
print(f"Erro Absoluto Médio (MAE): {mae:.2f}")
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.3f}")
# %%
plt.figure(figsize=(7,6))
plt.scatter(yr_test, y_pred_reg, color='royalblue', alpha=0.7)
plt.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], 'r--')
plt.xlabel("Valores Reais")
plt.ylabel("Valores Previstos")
plt.title("Regressão Linear — Valores Reais vs Previstos")
plt.show()

#%%
coef = pd.Series(modelo_lr.coef_, index=X_reg.columns).sort_values(key=abs, ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=coef.values, y=coef.index, palette='crest')
plt.title("Impacto das Variáveis na Progressão (Regressão Linear)")
plt.xlabel("Peso (coeficiente)")
plt.ylabel("Variável")
plt.show()

print("\nCoeficientes do modelo:")
print(coef)
# %%
# ===============================================================
# ANÁLISE DE DESEMPENHO — MODELO DE REGRESSÃO LINEAR
# ===============================================================
#
# OBJETIVO:
# Prever o valor contínuo da progressão do diabetes (target)
# com base nas variáveis clínicas dos pacientes.
#
# ---------------------------------------------------------------
# RESULTADOS:
# Erro Absoluto Médio (MAE): 42.85
# Erro Quadrático Médio (MSE): 2870.77
# Coeficiente de Determinação (R²): 0.468
#
# O modelo explica aproximadamente 47% da variação da progressão.
# O desempenho é sólido considerando a natureza clínica dos dados,
# onde fatores externos (hábitos, genética, adesão ao tratamento)
# não são capturados diretamente.
#
# ---------------------------------------------------------------
# INTERPRETAÇÃO DOS COEFICIENTES:
# - s1 (-39.3): colesterol total tem relação inversa com a progressão.
# - s5 (+32.3): lipídios altos aumentam a progressão da doença.
# - bmi (+28.5): IMC é um dos fatores mais impactantes.
# - bp (+16.4): pressão arterial elevada agrava o quadro clínico.
# - age (-0.04): idade tem impacto mínimo e não significativo.
#
# ---------------------------------------------------------------
# CONCLUSÃO:
# O modelo de Regressão Linear apresentou bom desempenho e alta
# interpretabilidade. Mostrou que IMC, lipídios e pressão arterial
# são os fatores mais influentes na progressão do diabetes,
# fornecendo um suporte analítico útil para acompanhamento clínico.
# ===============================================================
# %%
X_cluster = df_model.drop(columns=['target', 'progressao_alta'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
# %%
sse = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(7,5))
plt.plot(range(2, 11), sse, marker='o')
plt.title("Método do Cotovelo — Definição do número ideal de Clusters")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Soma dos Erros Quadráticos (SSE)")
plt.grid(True)
plt.show()
# %%
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df_model['cluster'] = clusters
# %%
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df_model['PCA1'] = pca_result[:, 0]
df_model['PCA2'] = pca_result[:, 1]
# %%
plt.figure(figsize=(8,6))
sns.scatterplot(
    x='PCA1', y='PCA2',
    hue='cluster', data=df_model,
    palette='Set2', alpha=0.8
)
plt.title("Agrupamento de Pacientes (K-Means + PCA)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Cluster")
plt.show()
# %%
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"Média da Silhueta: {silhouette_avg:.2f}")
# %%
cluster_summary = df_model.groupby('cluster').mean().round(2)
print("\n=== Média das variáveis por cluster ===")
st.dataframe(cluster_summary)

# %%
# ===============================================================
# 🧾 RELATÓRIO DE RESULTADOS — CLUSTERIZAÇÃO (K-Means, k=4)
# ===============================================================
#
# 🎯 OBJETIVO:
# Identificar grupos de pacientes com perfis clínicos semelhantes
# com base em variáveis contínuas (idade, IMC, pressão arterial,
# e indicadores metabólicos). O intuito é reconhecer padrões
# que indiquem diferentes níveis de risco e progressão da doença.
#
# ---------------------------------------------------------------
# ⚙️ MODELO:
# - Algoritmo: K-Means
# - Número de clusters: 4
# - Padronização: StandardScaler
# - Visualização: PCA (2 componentes principais)
#
# ---------------------------------------------------------------
# DESEMPENHO DO MODELO:
# - Método do Cotovelo: Cotovelo identificado em k ≈ 4
# - Média da Silhueta: 0.22 → separação moderada entre grupos
#
# Interpretação da Silhueta:
#   ▫️ Valores entre 0.2 e 0.3 indicam sobreposição leve,
#      mas com tendência a formação de grupos distinguíveis.
#   ▫️ Em datasets clínicos contínuos como este, valores
#      nessa faixa são considerados aceitáveis.
#
# ---------------------------------------------------------------
# RESULTADOS DOS CLUSTERS:
#
# A tabela de médias indica os perfis médios de cada grupo:
#
# | Cluster | IMC (bmi) | Pressão (bp) | s5 (lipídios) | Progressão Média (target) |
# |----------|------------|----------------|----------------|----------------------------|
# | 0 | -0.03 | -0.03 | -0.04 | **106.89** |
# | 1 | 0.02 | 0.00 | 0.01 | **164.97** |
# | 2 | 0.00 | 0.00 | 0.03 | **156.42** |
# | 3 | 0.05 | 0.04 | 0.05 | **232.46** |
#
# ---------------------------------------------------------------
# INTERPRETAÇÃO CLÍNICA DOS GRUPOS:
#
# 🟢 **Cluster 0 — Pacientes de Baixo Risco**
#   - IMC e pressão arterial abaixo da média
#   - Menor progressão da doença (≈106)
#   - Representa pacientes metabolicamente estáveis
#
# 🟠 **Cluster 1 — Pacientes com Risco Moderado**
#   - IMC levemente acima da média
#   - Lipídios (s5) e pressão normais
#   - Progressão intermediária (≈165)
#
# 🔵 **Cluster 2 — Risco Metabólico Elevado**
#   - s1 e s2 mais altos (indicando variações lipídicas)
#   - Progressão um pouco abaixo do cluster 1 (≈156)
#   - Pode representar pacientes sob tratamento ou controle parcial
#
# 🟣 **Cluster 3 — Alto Risco e Progressão Acelerada**
#   - IMC e pressão arterial bem acima da média
#   - Todos os indicadores metabólicos (s4, s5, s6) elevados
#   - Maior progressão da doença (≈232)
#   - Indica pacientes com fatores clínicos de risco combinados
#
# ---------------------------------------------------------------
# ANÁLISE GERAL:
# - O modelo conseguiu separar pacientes em quatro grupos
#   com perfis clínicos coerentes.
# - O grupo 0 representa pacientes controlados.
# - Os grupos 1 e 2 mostram níveis intermediários com pequenas diferenças
#   metabólicas.
# - O grupo 3 se destaca com alto risco clínico e progressão acentuada.
#
# ---------------------------------------------------------------
# INTERPRETAÇÃO VISUAL:
# O gráfico PCA mostra quatro regiões bem distribuídas,
# com sobreposição parcial entre clusters intermediários (1 e 2),
# e maior separação do grupo 0 (baixo risco) e grupo 3 (alto risco).
#
# ---------------------------------------------------------------
# CONCLUSÃO:
# - O K-Means (k=4) apresentou desempenho razoável (Silhouette = 0.22),
#   segmentando o conjunto em perfis de progressão distintos.
# - Os grupos extremos (baixo e alto risco) são bem definidos e úteis
#   para aplicações clínicas e de monitoramento.
# - Essa segmentação pode apoiar ações de saúde preventiva e
#   personalização do acompanhamento médico de pacientes diabéticos.
# ===============================================================
# %%
# ===============================================================
# 🧠 HUB — MACHINE LEARNING APLICADO À SAÚDE (DIABETES)
# ===============================================================
#st.set_page_config(page_title="ML Saúde — Diabetes", layout="wide")
sns.set_theme(style="whitegrid")

# ---------------------------
# Utils / Cache
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["target"] = diabetes.target
    # Campos auxiliares (usuário final)
    df["sexo_label"] = df["sex"].apply(lambda x: "Masculino" if x > 0 else "Feminino")
    df["faixa_etaria"] = pd.cut(
        df["age"], bins=[-0.12, -0.06, 0.0, 0.06, 0.12],
        labels=["Jovens", "Adultos Jovens", "Adultos", "Idosos"]
    )
    return df, diabetes

@st.cache_resource(show_spinner=False)
def train_models(df):
    # Base para modelos: sem 'sex' explícito (pouca relevância)
    df_model = df.drop(columns=["sexo_label", "faixa_etaria", "sex"])

    # Alvo binário para classificação (alta progressão >= mediana)
    threshold = df_model["target"].median()
    df_model["progressao_alta"] = (df_model["target"] >= threshold).astype(int)

    # --- Classificação
    X_cls = df_model.drop(columns=["target", "progressao_alta"])
    y_cls = df_model["progressao_alta"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_cls, y_cls, test_size=0.3, random_state=42, stratify=y_cls
    )
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42))
    ])
    clf.fit(Xc_train, yc_train)

    # Métricas rápidas
    y_pred = clf.predict(Xc_test)
    y_prob = clf.predict_proba(Xc_test)[:, 1]
    cls_metrics = {
        "acc": accuracy_score(yc_test, y_pred),
        "prec": precision_score(yc_test, y_pred),
        "rec": recall_score(yc_test, y_pred),
        "f1": f1_score(yc_test, y_pred),
        "auc": roc_auc_score(yc_test, y_prob),
        "y_true": yc_test, "y_pred": y_pred, "y_prob": y_prob,
        "X_cols": list(X_cls.columns)
    }

    # Importância das features (retirada do passo final)
    rf_final = clf.named_steps["rf"]
    cls_importances = pd.Series(rf_final.feature_importances_, index=X_cls.columns)\
                        .sort_values(ascending=False)

    # --- Regressão
    X_reg = df_model.drop(columns=["target", "progressao_alta"])
    y_reg = df_model["target"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )
    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    reg.fit(Xr_train, yr_train)

    y_pred_reg = reg.predict(Xr_test)
    reg_metrics = {
        "mae": mean_absolute_error(yr_test, y_pred_reg),
        "mse": mean_squared_error(yr_test, y_pred_reg),
        "r2": r2_score(yr_test, y_pred_reg),
        "y_true": yr_test, "y_pred": y_pred_reg,
        "X_cols": list(X_reg.columns),
        "coefs": pd.Series(reg.named_steps["lr"].coef_, index=X_reg.columns)
                  .sort_values(key=np.abs, ascending=False)
    }

    return clf, cls_metrics, cls_importances, reg, reg_metrics, df_model

@st.cache_data(show_spinner=False)
def compute_clusters(df_model, k=4):
    X_cluster = df_model.drop(columns=["target", "progressao_alta"], errors="ignore")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, clusters)

    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(X_scaled)

    vis = pd.DataFrame(pca_2d, columns=["PCA1", "PCA2"])
    vis["cluster"] = clusters
    summary = pd.concat([df_model, vis], axis=1).groupby("cluster").mean(numeric_only=True).round(2)

    return vis, summary, sil

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("🧭 Navegação")
page = st.sidebar.radio(
    "Ir para",
    ["Início", "Exploração (EDA)", "Predição", "Clusterização", "Sobre"]
)

df, diabetes = load_data()
clf, cls_metrics, cls_importances, reg, reg_metrics, df_model = train_models(df)

# ---------------------------
# Páginas
# ---------------------------
if page == "Início":
    st.title("🧠 ML na Saúde — Diabetes")
    st.markdown(
        """
        Este hub interativo mostra **como dados clínicos podem apoiar decisões** sobre o cuidado com pessoas com diabetes.

        **Você pode:**
        - Explorar os dados com gráficos fáceis de entender;
        - Testar a **predição** de progressão (baixa/alta) e **prever o valor** da progressão;
        - Ver agrupamentos (clusters) de pacientes com perfis parecidos.

        **Atenção:** este app é **educacional** e não substitui avaliação médica.
        """
    )
    st.success("Dica: use o menu à esquerda para navegar entre as seções.")

elif page == "Exploração (EDA)":
    st.title("📊 Exploração dos Dados (EDA)")
    st.caption("Dados anonimizados de 442 pacientes: idade, IMC, pressão arterial e exames sanguíneos (s1...s6).")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Amostra dos dados")
        st.dataframe(df.head(15), use_container_width=True)
    with col2:
        st.metric("Pacientes", len(df))
        st.metric("Variáveis", df.shape[1])

    st.markdown("---")
    st.subheader("Distribuição das variáveis")
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    axes = axes.ravel()
    for i, col in enumerate(["age","bmi","bp","s1","s2","s3","s4","s5","s6","target","sex",""]):
        if col == "" or i >= len(axes): break
        sns.histplot(df[col], bins=20, kde=True, ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Relação entre IMC e Pressão Arterial")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x="bmi", y="bp", data=df, ax=ax)
    ax.set_xlabel("IMC (bmi)"); ax.set_ylabel("Pressão Arterial (bp)")
    st.pyplot(fig)

    st.subheader("Correlação entre variáveis")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.drop(columns=["sexo_label","faixa_etaria"]).corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.markdown(
        """
        **Como ler:**  
        - **bmi** (IMC) e **bp** (pressão) tendem a crescer juntos com a **progressão**.  
        - **s5** (perfil lipídico) também está associado à piora da doença.  
        - **sex** tem pouca correlação com a progressão.
        """
    )

elif page == "Predição":
    st.title("🔮 Predição")
    tabs = st.tabs(["Classificação (Baixa/Alta)", "Regressão (Valor contínuo)"])

    # ----- CLASSIFICAÇÃO -----
    with tabs[0]:
        st.subheader("Classificar progressão: Baixa vs Alta")
        st.caption("Modelo: RandomForest (com padronização). Alvo: acima/abaixo da mediana do target.")
        cols = cls_metrics["X_cols"]

        # Controles de entrada amigáveis
        st.markdown("**Informe os dados do paciente:**")
        c1, c2, c3 = st.columns(3)
        inputs = {}
        # Usamos o range baseado nos dados para sliders sensatos
        for i, col in enumerate(cols):
            series = df[col]
            mn, mx = float(series.min()), float(series.max())
            default = float(series.median())
            with (c1 if i % 3 == 0 else c2 if i % 3 == 1 else c3):
                inputs[col] = st.slider(col, mn, mx, default, step=(mx-mn)/100)

        X_user = pd.DataFrame([inputs], columns=cols)
        proba = clf.predict_proba(X_user)[0, 1]
        pred = "Alta" if proba >= 0.5 else "Baixa"

        st.markdown(f"### 🧾 Resultado: **{pred}** (probabilidade de alta = **{proba:.2f}**)")
        st.caption("Probabilidade ≥ 0.50 é classificada como Alta progressão.")

        # Métricas do modelo (fixas do conjunto de teste)
        st.markdown("#### Desempenho do modelo (teste)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Acurácia", f"{cls_metrics['acc']:.2f}")
        m2.metric("Precisão", f"{cls_metrics['prec']:.2f}")
        m3.metric("Recall", f"{cls_metrics['rec']:.2f}")
        m4.metric("F1", f"{cls_metrics['f1']:.2f}")
        m5.metric("ROC AUC", f"{cls_metrics['auc']:.2f}")

        # Importâncias
        st.markdown("#### O que mais pesou na decisão?")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=cls_importances.values, y=cls_importances.index, ax=ax)
        ax.set_xlabel("Importância"); ax.set_ylabel("Variável")
        st.pyplot(fig)

    # ----- REGRESSÃO -----
    with tabs[1]:
        st.subheader("Prever valor da progressão (target)")
        st.caption("Modelo: Regressão Linear (com padronização). Saída: valor contínuo da progressão.")
        cols = reg_metrics["X_cols"]

        c1, c2, c3 = st.columns(3)
        inputs = {}
        for i, col in enumerate(cols):
            series = df[col]
            mn, mx = float(series.min()), float(series.max())
            default = float(series.median())
            with (c1 if i % 3 == 0 else c2 if i % 3 == 1 else c3):
                inputs[col] = st.slider(col, mn, mx, default, step=(mx-mn)/100)

        X_user = pd.DataFrame([inputs], columns=cols)
        y_hat = reg.predict(X_user)[0]
        st.markdown(f"### 🧾 Progressão prevista: **{y_hat:.1f}**")

        st.markdown("#### Desempenho do modelo (teste)")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{reg_metrics['mae']:.2f}")
        m2.metric("MSE", f"{reg_metrics['mse']:.0f}")
        m3.metric("R²", f"{reg_metrics['r2']:.2f}")

        st.markdown("#### Variáveis com maior impacto (coeficientes absolutos)")
        fig, ax = plt.subplots(figsize=(7, 4))
        coef = reg_metrics["coefs"].head(10)
        sns.barplot(x=coef.values, y=coef.index, ax=ax)
        ax.set_xlabel("Peso (coeficiente)"); ax.set_ylabel("Variável")
        st.pyplot(fig)

elif page == "Clusterização":
    st.title("🧩 Clusterização (K-Means)")
    st.caption("Agrupa pacientes com perfis semelhantes. Recurso educacional; não substitui diagnóstico.")

    k = st.slider("Número de clusters (k)", 2, 8, 4, 1)
    vis, summary, sil = compute_clusters(df_model, k)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Visualização (PCA em 2D)")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=vis, x="PCA1", y="PCA2", hue="cluster", palette="Set2", ax=ax, alpha=0.85)
        st.pyplot(fig)
    with c2:
        st.metric("Silhouette (qualidade)", f"{sil:.2f}")
        st.caption("Quanto mais próximo de 1, melhor a separação entre grupos. 0.2–0.3 já indica tendência a grupos.")

    st.subheader("Médias por cluster (interpretação)")
    st.dataframe(summary.filter(items=["age","bmi","bp","s1","s2","s3","s4","s5","s6","target"]).style.highlight_max(color="#d1ffd1", axis=0), use_container_width=True)

    st.info(
        "Leitura rápida: IMC (bmi) e pressão (bp) maiores tendem a aparecer nos clusters com maior target (progressão)."
    )

elif page == "Sobre":
    st.title("ℹ️ Sobre o projeto")
    st.markdown(
        """
        - **Base:** `sklearn.datasets.load_diabetes` (442 pacientes).
        - **Classificação:** RandomForest (probabilidade de progressão alta).
        - **Regressão:** LinearRegression (valor contínuo de progressão).
        - **Clusterização:** K-Means + PCA (visualização em 2D).

        **Observações importantes**
        - Resultados são *estimativas* baseadas em dados padronizados.
        - Não utilizar para fins clínicos sem validação médica.
        """
    )