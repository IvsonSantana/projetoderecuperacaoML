# ===============================================================
# 🧠 MODELOS DE MACHINE LEARNING — DIABETES
# ===============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class ModelosMLDiabetes:

    def __init__(self, df_model):
        self.df_model = df_model.copy()

        # Inicializa escalonadores e modelos
        self.scaler_cls = StandardScaler()
        self.scaler_reg = StandardScaler()
        self.scaler_cluster = StandardScaler()

        self.classificador = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1
        )
        self.regressor = LinearRegression()
        self.kmeans = None
        self.pca = PCA(n_components=2)

        # Variáveis auxiliares
        self.metricas_cls = {}
        self.metricas_reg = {}
        self.silhouette = None
        self.importancias = None
        self.df_clusters = None


    # ==========================================================
    # 🧩 CLASSIFICAÇÃO
    # ==========================================================
    def treinar_classificacao(self):
        threshold = self.df_model["target"].median()
        self.df_model["progressao_alta"] = (self.df_model["target"] >= threshold).astype(int)

        # 🔹 Apenas colunas numéricas
        X = self.df_model.select_dtypes(include=["int64", "float64"]).drop(
            columns=["target", "progressao_alta"], errors="ignore"
        )
        y = self.df_model["progressao_alta"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        X_train = self.scaler_cls.fit_transform(X_train)
        X_test = self.scaler_cls.transform(X_test)

        self.classificador.fit(X_train, y_train)
        y_pred = self.classificador.predict(X_test)
        y_prob = self.classificador.predict_proba(X_test)[:, 1]

        # Métricas
        self.metricas_cls = {
            "Acurácia": accuracy_score(y_test, y_pred),
            "Precisão": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "X_cols": list(X.columns)
        }

        self.importancias = pd.Series(
            self.classificador.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        return self.metricas_cls, self.importancias


    def prever_classificacao(self, entrada_usuario):
        X_user = np.array([entrada_usuario])
        X_scaled = self.scaler_cls.transform(X_user)
        prob = self.classificador.predict_proba(X_scaled)[0, 1]
        classe = "Alta" if prob >= 0.5 else "Baixa"
        return classe, float(prob)


    # ==========================================================
    # 📈 REGRESSÃO
    # ==========================================================
    def treinar_regressao(self):
        # 🔹 Apenas colunas numéricas
        X = self.df_model.select_dtypes(include=["int64", "float64"]).drop(
            columns=["target", "progressao_alta"], errors="ignore"
        )
        y = self.df_model["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        X_train = self.scaler_reg.fit_transform(X_train)
        X_test = self.scaler_reg.transform(X_test)

        self.regressor.fit(X_train, y_train)
        y_pred = self.regressor.predict(X_test)

        self.metricas_reg = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "R²": r2_score(y_test, y_pred),
            "X_cols": list(X.columns),
            "y_test": y_test,
            "y_pred": y_pred
        }

        return self.metricas_reg


    def prever_regressao(self, entrada_usuario):
        X_user = np.array([entrada_usuario])
        X_scaled = self.scaler_reg.transform(X_user)
        y_pred = self.regressor.predict(X_scaled)[0]
        return float(y_pred)


    # ==========================================================
    # 🧬 CLUSTERIZAÇÃO
    # ==========================================================
    def treinar_clusterizacao(self, n_clusters=4):
        # 🔹 Apenas colunas numéricas
        X_cluster = self.df_model.select_dtypes(include=["int64", "float64"]).drop(
            columns=["target", "progressao_alta"], errors="ignore"
        )

        X_scaled = self.scaler_cluster.fit_transform(X_cluster)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.kmeans.fit_predict(X_scaled)

        self.df_model["cluster"] = clusters
        pca_result = self.pca.fit_transform(X_scaled)
        self.df_model["PCA1"] = pca_result[:, 0]
        self.df_model["PCA2"] = pca_result[:, 1]

        self.silhouette = silhouette_score(X_scaled, clusters)
        self.df_clusters = self.df_model.copy()

        return self.df_clusters, self.silhouette


    def prever_cluster(self, entrada_usuario):
        if self.kmeans is None:
            raise ValueError("O modelo de cluster ainda não foi treinado. Chame treinar_clusterizacao() primeiro.")
        
        # 🔧 Garante que a entrada tenha formato 2D (1, n_features)
        X_user = np.atleast_2d(entrada_usuario)
        
        # 🔹 Escala e prediz o cluster
        X_scaled = self.scaler_cluster.transform(X_user)
        cluster_pred = int(self.kmeans.predict(X_scaled)[0])
        return cluster_pred
