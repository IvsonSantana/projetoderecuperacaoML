# 🧠 Machine Learning Aplicado à Saúde — Diabetes

Este projeto apresenta uma aplicação interativa em **Python (Streamlit)** que utiliza **modelos de aprendizado supervisionado e não supervisionado** para analisar dados clínicos de pacientes diabéticos.  
O objetivo é compreender os fatores que influenciam a **progressão da doença** e identificar **grupos de pacientes com perfis clínicos semelhantes**.

---

## 🎯 Objetivo Geral

Desenvolver uma aplicação preditiva capaz de:

- Classificar pacientes com base no **risco de progressão** do diabetes (Baixo ou Alto risco);
- Estimar o **grau de progressão** (valor contínuo) através de regressão;
- Agrupar pacientes em **perfis clínicos semelhantes** via clusterização;
- Fornecer uma interface interativa e explicativa sobre os resultados.

---

## 📘 Base de Dados

- **Fonte:** `sklearn.datasets.load_diabetes`  
- **Total:** 442 pacientes  
- **Atributos:**
  - `age`: idade normalizada  
  - `sex`: sexo (masculino/feminino)  
  - `bmi`: índice de massa corporal  
  - `bp`: pressão arterial média  
  - `s1`–`s6`: medidas séricas (colesterol, LDL, HDL, triglicerídeos, glicose, etc.)  
  - `target`: medida quantitativa da progressão da doença após 1 ano

---

## 🧩 Modelos Utilizados

| Tipo | Modelo | Objetivo |
|------|---------|-----------|
| **Classificação** | `RandomForestClassifier` | Identificar pacientes com **alta ou baixa progressão** |
| **Regressão** | `LinearRegression` | Estimar o **grau contínuo** da progressão da doença |
| **Clusterização** | `KMeans` + `PCA` | Agrupar pacientes em **perfis clínicos semelhantes** |

# 🧠 Machine Learning Aplicado à Saúde — Diabetes

Este projeto apresenta uma aplicação interativa em **Python (Streamlit)** que utiliza **modelos de aprendizado supervisionado e não supervisionado** para analisar dados clínicos de pacientes diabéticos.  
O objetivo é compreender os fatores que influenciam a **progressão da doença** e identificar **grupos de pacientes com perfis clínicos semelhantes**.

---

## 🎯 Objetivo Geral

Desenvolver uma aplicação preditiva capaz de:

- Classificar pacientes com base no **risco de progressão** do diabetes (Baixo ou Alto risco);
- Estimar o **grau de progressão** (valor contínuo) através de regressão;
- Agrupar pacientes em **perfis clínicos semelhantes** via clusterização;
- Fornecer uma interface interativa e explicativa sobre os resultados.

---

## 📘 Base de Dados

- **Fonte:** `sklearn.datasets.load_diabetes`  
- **Total:** 442 pacientes  
- **Atributos:**
  - `age`: idade normalizada  
  - `sex`: sexo (masculino/feminino)  
  - `bmi`: índice de massa corporal  
  - `bp`: pressão arterial média  
  - `s1`–`s6`: medidas séricas (colesterol, LDL, HDL, triglicerídeos, glicose, etc.)  
  - `target`: medida quantitativa da progressão da doença após 1 ano

---

## 🧩 Modelos Utilizados

| Tipo | Modelo | Objetivo |
|------|---------|-----------|
| **Classificação** | `RandomForestClassifier` | Identificar pacientes com **alta ou baixa progressão** |
| **Regressão** | `LinearRegression` | Estimar o **grau contínuo** da progressão da doença |
| **Clusterização** | `KMeans` + `PCA` | Agrupar pacientes em **perfis clínicos semelhantes** |

---


---

## 🧠 Componentes da Aplicação (Streamlit)

### 🔹 1. **Análise Preditiva**
Explora os dados clínicos e permite observar padrões como:
- Correlação entre **IMC e glicose**;
- Distribuição por sexo e faixa etária;
- Variações de pressão arterial e colesterol.

### 🔹 2. **Hub Preditivo**
Área interativa de simulação com três modelos:

#### 🩺 Classificação
Classifica o risco de progressão entre **Baixa** e **Alta**, com base nos valores inseridos:
- IMC, glicose, colesterol, pressão arterial etc.
- Mostra também as variáveis mais importantes no modelo Random Forest.

#### 📈 Regressão
Prediz o valor contínuo da progressão (`target`) e retorna:
- Nível de risco (Baixo, Moderado ou Alto);
- Recomendação interpretativa.

#### 🧩 Agrupamento
Aponta a qual **grupo clínico (cluster)** o paciente pertence, com interpretação:
- Grupo 1: perfil saudável;  
- Grupo 2: leve aumento de pressão/colesterol;  
- Grupo 3: IMC e glicose altos (risco elevado);  
- Grupo 4: perfil intermediário.

### 🔹 3. **Sobre**
Apresenta informações do projeto, autor e aviso de uso educacional.

---

## 📊 Principais Resultados

### 🩺 Classificação (Random Forest)
| Métrica | Valor |
|----------|--------|
| Acurácia | 0.79 |
| Precisão | 0.76 |
| Recall | 0.85 |
| F1-Score | 0.80 |
| AUC | 0.84 |

🔹 **Variáveis mais influentes:**
- IMC (`bmi`), Glicose (`s5`), Pressão Arterial (`bp`)

### 📈 Regressão (Linear Regression)
| Métrica | Valor |
|----------|--------|
| MAE | 42.85 |
| MSE | 2870.77 |
| R² | 0.47 |

🔹 **Interpretação:**  
A regressão explica 47% da variação do `target`, destacando **IMC, glicose e pressão arterial** como principais fatores de progressão.

### 🧬 Clusterização (K-Means, k=4)
- **Silhouette Score:** 0.22  
- **Clusters identificados:**
  - Grupo 0 — Baixo risco  
  - Grupo 1 — Risco moderado  
  - Grupo 2 — Risco metabólico elevado  
  - Grupo 3 — Alta progressão  

---

## 💡 Insights Clínicos

- O **IMC elevado** e a **alta glicose** estão fortemente associados à progressão da doença.  
- **Sexo** não apresentou correlação significativa.  
- A **clusterização** revelou grupos distintos com padrões clínicos coerentes.  
- O modelo de **classificação** é adequado para apoio em monitoramento preventivo.  

---

## 🚀 Como Executar o Projeto

### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/IvsonSantana/projetoderecuperacaoML.git
cd projetoderecuperacaoML
``` 

### 2️⃣ Criar ambiente virtual e instalar dependências
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
``` 
### 3️⃣ Executar o aplicativo Streamlit
```bash
streamlit run ml_hub.py
``` 
### Principais Dependências

- Python 3.10+
- Streamlit
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

### Conclusão

O projeto demonstra o potencial do Machine Learning na área da saúde, especialmente no apoio ao diagnóstico e monitoramento de doenças crônicas.
As técnicas aplicadas permitem identificar fatores de risco, prever a progressão e agrupar perfis clínicos, oferecendo uma base sólida para decisões médicas preventivas e personalizadas.