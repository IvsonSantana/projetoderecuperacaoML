# ===============================================================
# 🧠 MACHINE LEARNING APLICADO À SAÚDE — DIABETES
# ===============================================================

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from modelo_diabetes import ModelosMLDiabetes
import numpy as np

# ---------------------------------------------------------------
# ⚙️ Configuração inicial
# ---------------------------------------------------------------
st.set_page_config(page_title="Análise Preditiva — Diabetes", layout="wide")
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------
# 📥 Carregar dados
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["target"] = diabetes.target
    df["sexo_label"] = df["sex"].apply(lambda x: "Masculino" if x > 0 else "Feminino")
    df["faixa_etaria"] = pd.cut(
        df["age"],
        bins=[-0.12, -0.06, 0.0, 0.06, 0.12],
        labels=["Jovens", "Adultos Jovens", "Adultos", "Idosos"]
    )
    return df, diabetes

# ---------------------------------------------------------------
# 🧠 Treinar modelos
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def treinar_modelos(df):
    modelos = ModelosMLDiabetes(df)
    cls_metrics, importancias = modelos.treinar_classificacao()
    reg_metrics = modelos.treinar_regressao()
    df_clusters, sil = modelos.treinar_clusterizacao(n_clusters=4)  # fixo no backend
    return modelos, cls_metrics, importancias, reg_metrics, df_clusters, sil

# ---------------------------------------------------------------
# 🧭 Sidebar
# ---------------------------------------------------------------
st.sidebar.title("🧭 Navegação")
page = st.sidebar.radio("Ir para", ["Análise Preditiva", "Hub Preditivo", "Sobre"])

# ---------------------------------------------------------------
# 🔹 Carregar dados e modelos
# ---------------------------------------------------------------
df, diabetes = get_data()
modelos, cls_metrics, importancias, reg_metrics, df_clusters, sil = treinar_modelos(df)

# ===============================================================
# 1️⃣ ANÁLISE PREDITIVA — VISÃO CLÍNICA
# ===============================================================
if page == "Análise Preditiva":
    st.title("📊 Análise Preditiva — Dados Clínicos de Diabetes")
    st.markdown("""
    Esta seção apresenta uma **análise detalhada dos principais indicadores clínicos**
    relacionados à progressão do diabetes.  
    O objetivo é apoiar decisões médicas ao identificar **padrões de risco e tendências**.
    """)

    st.divider()

    # -----------------------------------------------------------
    # 🎯 Filtros Interativos
    # -----------------------------------------------------------
    st.sidebar.subheader("🔧 Filtros para análise")
    sexo_filter = st.sidebar.multiselect("Sexo", df["sexo_label"].unique(), default=df["sexo_label"].unique())
    faixa_filter = st.sidebar.multiselect("Faixa Etária", df["faixa_etaria"].unique(), default=df["faixa_etaria"].unique())
    df_filtered = df[(df["sexo_label"].isin(sexo_filter)) & (df["faixa_etaria"].isin(faixa_filter))]

    # -----------------------------------------------------------
    # 👥 Perfil Demográfico
    # -----------------------------------------------------------
    st.markdown("### 👥 Perfil Demográfico dos Pacientes")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x="sexo_label", data=df_filtered, palette="Set2", ax=ax)
        ax.set_title("Distribuição por Sexo")
        ax.set_xlabel("")
        ax.set_ylabel("Quantidade de Pacientes")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x="faixa_etaria", data=df_filtered, palette="crest", ax=ax)
        ax.set_title("Distribuição por Faixa Etária")
        ax.set_xlabel("")
        ax.set_ylabel("Quantidade de Pacientes")
        st.pyplot(fig)

    st.info("🔹 Observa-se uma leve predominância de adultos e adultos jovens, com equilíbrio entre os sexos.")

    st.divider()

    # -----------------------------------------------------------
    # ❤️ Indicadores Metabólicos
    # -----------------------------------------------------------
    st.markdown("### ❤️ Indicadores Metabólicos e de Risco")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x="bmi", y="s5", hue="sexo_label", data=df_filtered, palette="coolwarm", ax=ax)
        ax.set_xlabel("IMC (Índice de Massa Corporal)")
        ax.set_ylabel("Nível de Glicose (mg/dL)")
        ax.set_title("Relação entre IMC e Glicose")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x="bp", y="s2", hue="faixa_etaria", data=df_filtered, palette="viridis", ax=ax)
        ax.set_xlabel("Pressão Arterial Média (mmHg)")
        ax.set_ylabel("LDL (Colesterol Ruim, mg/dL)")
        ax.set_title("Pressão Arterial vs LDL")
        st.pyplot(fig)

    st.info("💡 Pacientes com IMC mais alto e maior glicose tendem a apresentar progressão mais rápida da doença.")

    st.divider()

    # -----------------------------------------------------------
    # 🩸 Comparações por Grupo
    # -----------------------------------------------------------
    st.markdown("### 🩸 Comparações Clínicas por Grupo")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="faixa_etaria", y="bmi", data=df_filtered, palette="Set3", ax=ax)
        ax.set_xlabel("Faixa Etária")
        ax.set_ylabel("IMC")
        ax.set_title("Distribuição de IMC por Faixa Etária")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="sexo_label", y="bp", data=df_filtered, palette="Set2", ax=ax)
        ax.set_xlabel("Sexo")
        ax.set_ylabel("Pressão Arterial Média")
        ax.set_title("Pressão Arterial Média por Sexo")
        st.pyplot(fig)

    st.success("✅ Homens tendem a ter **pressão arterial ligeiramente mais alta**, enquanto o **IMC cresce com a idade**.")

# ===============================================================
# 2️⃣ HUB PREDITIVO — MODELOS DE MACHINE LEARNING
# ===============================================================
elif page == "Hub Preditivo":
    st.title("🧠 Hub de Modelagem Preditiva — Diabetes")
    st.markdown("""
    Simule o **perfil clínico de um paciente** e obtenha uma **predição automatizada** do risco,
    da progressão da doença e do **perfil de grupo clínico (cluster)** com base em 
    modelos de **classificação, regressão e clusterização (K-Means)**.
    """)

    st.divider()
    st.header("🔮 Predição Personalizada")

    # Agora temos 3 abas
    tabs = st.tabs([
        "Classificação (Baixa/Alta)",
        "Regressão (Valor Contínuo)",
        "🧩 Agrupamento (Perfil de Grupo)"
    ])

    # 🟢 CLASSIFICAÇÃO
    with tabs[0]:
        st.subheader("🩺 Classificação de Risco de Diabetes")
        st.info("Ajuste os valores abaixo para simular o perfil clínico do paciente:")

        feature_labels_real = {
            "age": ("Idade (anos)", 20.0, 80.0, 40.0),
            "bmi": ("IMC (Índice de Massa Corporal)", 15.0, 40.0, 25.0),
            "bp": ("Pressão Arterial Média (mmHg)", 60.0, 140.0, 90.0),
            "s1": ("Colesterol Total (mg/dL)", 150.0, 300.0, 200.0),
            "s2": ("LDL (Colesterol Ruim, mg/dL)", 70.0, 190.0, 120.0),
            "s3": ("HDL (Colesterol Bom, mg/dL)", 30.0, 80.0, 50.0),
            "s4": ("Triglicerídeos (mg/dL)", 50.0, 250.0, 150.0),
            "s5": ("Nível de Glicose (mg/dL)", 70.0, 250.0, 110.0),
            "s6": ("Insulina no Sangue (µU/mL)", 2.0, 25.0, 10.0)
        }

        inputs = {}
        for col, (label, vmin, vmax, vmed) in feature_labels_real.items():
            inputs[col] = st.slider(label, float(vmin), float(vmax), float(vmed), step=1.0, key=f"class_{col}")

        X_user = []
        for col in cls_metrics["X_cols"]:
            if col == "sex":
                X_user.append(0.0)
            elif col in inputs:
                val = inputs[col]
                vmin, vmax, vmed = feature_labels_real[col][1:]
                norm = (val - vmed) / (vmax - vmin)
                X_user.append(norm)
            else:
                X_user.append(0.0)

        pred, proba = modelos.prever_classificacao(X_user)

        color = "#2ecc71" if pred == "Baixa" else "#e74c3c"
        titulo = "Risco de Diabetes: Baixo" if pred == "Baixa" else "Risco de Diabetes: Alto"

        st.markdown(f"""
        <div style="padding:15px;border-radius:10px;background-color:{color};text-align:center;">
        <h3 style="color:white;">{titulo} (probabilidade {proba:.2f})</h3>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📊 Fatores de Maior Impacto no Modelo")
        feature_labels_pt = {
            "age": "Idade (anos)", "bmi": "IMC", "bp": "Pressão Arterial Média",
            "s1": "Colesterol Total", "s2": "LDL", "s3": "HDL",
            "s4": "Triglicerídeos", "s5": "Glicose", "s6": "Insulina"
        }
        importancias_leg = importancias.rename(index=feature_labels_pt)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=importancias_leg.values[:8], y=importancias_leg.index[:8], palette="rocket", ax=ax)
        ax.set_xlabel("Importância Relativa")
        ax.set_ylabel("Variáveis Clínicas")
        st.pyplot(fig)

    # 🔵 REGRESSÃO
    with tabs[1]:
        st.subheader("📈 Predição de Valor Contínuo (Progressão)")
        inputs_reg = {}
        for col, (label, vmin, vmax, vmed) in feature_labels_real.items():
            inputs_reg[col] = st.slider(label, float(vmin), float(vmax), float(vmed), step=1.0, key=f"reg_{col}")

        X_user_reg = []
        for col in reg_metrics["X_cols"]:
            if col == "sex":
                X_user_reg.append(0.0)
            elif col in inputs_reg:
                val = inputs_reg[col]
                vmin, vmax, vmed = feature_labels_real[col][1:]
                norm = (val - vmed) / (vmax - vmin)
                X_user_reg.append(norm)
            else:
                X_user_reg.append(0.0)

        pred_reg = modelos.prever_regressao(X_user_reg)
        if pred_reg <= 120:
            color, risco_txt, recomendacao = "#2ecc71", "Baixo", "✅ Continue com bons hábitos."
        elif 120 < pred_reg <= 200:
            color, risco_txt, recomendacao = "#f1c40f", "Moderado", "⚠️ Mantenha dieta equilibrada e check-ups."
        else:
            color, risco_txt, recomendacao = "#e74c3c", "Alto", "🚨 Procure acompanhamento médico frequente."

        st.markdown(f"""
        <div style="padding:15px;border-radius:10px;background-color:{color};text-align:center;">
        <h3 style="color:white;">Progressão prevista: {pred_reg:.2f} — <b>Risco {risco_txt}</b></h3>
        </div>
        """, unsafe_allow_html=True)
        st.info(recomendacao)

    # 🧩 AGRUPAMENTO
    with tabs[2]:
        st.subheader("🧩 Agrupamento — Identificação de Grupo Clínico")
        st.info("""
        O modelo de **clusterização (K-Means)** identifica a qual **grupo clínico** o paciente pertence, 
        com base nas mesmas variáveis utilizadas na classificação e regressão.
        """)

        # 🔹 Entradas do usuário (iguais às abas anteriores)
        feature_labels_real = {
            "age": ("Idade (anos)", 20.0, 80.0, 40.0),
            "bmi": ("IMC (Índice de Massa Corporal)", 15.0, 40.0, 25.0),
            "bp": ("Pressão Arterial Média (mmHg)", 60.0, 140.0, 90.0),
            "s1": ("Colesterol Total (mg/dL)", 150.0, 300.0, 200.0),
            "s2": ("LDL (Colesterol Ruim, mg/dL)", 70.0, 190.0, 120.0),
            "s3": ("HDL (Colesterol Bom, mg/dL)", 30.0, 80.0, 50.0),
            "s4": ("Triglicerídeos (mg/dL)", 50.0, 250.0, 150.0),
            "s5": ("Nível de Glicose (mg/dL)", 70.0, 250.0, 110.0),
            "s6": ("Insulina no Sangue (µU/mL)", 2.0, 25.0, 10.0)
        }

        inputs_cluster = {}
        for col, (label, vmin, vmax, vmed) in feature_labels_real.items():
            inputs_cluster[col] = st.slider(label, float(vmin), float(vmax), float(vmed), step=1.0, key=f"cluster_{col}")

        # 🔸 Normaliza os dados para compatibilidade com o modelo
        X_user_cluster = []
        for col in reg_metrics["X_cols"]:  # mesma ordem de treino
            if col == "sex":
                X_user_cluster.append(0.0)
            elif col in inputs_cluster:
                val = inputs_cluster[col]
                vmin, vmax, vmed = feature_labels_real[col][1:]
                norm = (val - vmed) / (vmax - vmin)
                X_user_cluster.append(norm)
            else:
                X_user_cluster.append(0.0)

        # 🔹 Predição do cluster
        cluster_pred = modelos.prever_cluster(X_user_cluster)

        # 🔹 Exibição simples
        st.markdown(f"""
        <div style="padding:15px;border-radius:10px;background-color:#34495e;text-align:center;">
        <h3 style="color:white;">🧬 O paciente pertence ao <b>Grupo {cluster_pred + 1}</b></h3>
        </div>
        """, unsafe_allow_html=True)

        # 💡 Insights interpretativos
        st.markdown("### 💡 Interpretação do Grupo:")
        if cluster_pred == 0:
            st.success("🩺 **Grupo 1:** Perfil equilibrado — IMC e glicose dentro de faixas saudáveis, baixo risco metabólico.")
        elif cluster_pred == 1:
            st.warning("⚠️ **Grupo 2:** Leve aumento na pressão arterial e colesterol — manter acompanhamento preventivo.")
        elif cluster_pred == 2:
            st.error("🚨 **Grupo 3:** IMC e glicose elevados — risco maior de progressão do diabetes.")
        else:
            st.info("ℹ️ **Grupo 4:** Perfil misto — valores intermediários entre glicose, colesterol e pressão.")

# ===============================================================
# ℹ️ 3️⃣ SOBRE
# ===============================================================
elif page == "Sobre":
    st.title("ℹ️ Sobre o Projeto")
    st.markdown("""
    Este aplicativo demonstra a aplicação de **Machine Learning na área da saúde**, 
    com foco em **análises preditivas sobre diabetes**.

    - **Modelos Utilizados:** Random Forest, Linear Regression, K-Means  
    - **Base de Dados:** `sklearn.datasets.load_diabetes`  
    - **Desenvolvido por:** Ivson Santana — *para fins educacionais.*  
    ⚠️ *Este aplicativo não substitui avaliação médica profissional.*
    """)
