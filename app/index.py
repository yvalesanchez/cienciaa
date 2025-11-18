import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# =========================================================
# CONFIGURACIÓN BÁSICA
# =========================================================
st.set_page_config(
    page_title="Análisis de ACV (Stroke)",
    layout="wide",
)

st.title("Análisis de ACV (Stroke)")
st.write("Dashboard basado en el notebook original (sin modificarlo).")


# =========================================================
# CARGA DE DATOS
# =========================================================
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    return df


DATA_PATH = "data/healthcare-dataset-stroke-data.csv"

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"No se encontró el archivo de datos en `{DATA_PATH}`.\n\n"
        "Revisa que exista la carpeta `data/` y que el archivo "
        "`healthcare-dataset-stroke-data.csv` esté dentro."
    )
    st.stop()
except Exception as e:
    st.error("Ocurrió un error al cargar los datos:")
    st.exception(e)
    st.stop()

# Creamos una columna de texto para el target
if "stroke" not in df.columns:
    st.error("El dataset no tiene la columna 'stroke'. Revisa que sea el archivo correcto.")
    st.write("Columnas detectadas:", list(df.columns))
    st.stop()

df_viz = df.copy()
df_viz["stroke_label"] = df_viz["stroke"].map({0: "Sin ataque", 1: "Ataque"})

# =========================================================
# TABS PRINCIPALES
# =========================================================
tab_datos, tab_graficos, tab_modelo = st.tabs(
    ["📄 Dataset", "📊 Gráficos descriptivos", "🤖 Modelo predictivo"]
)

# =========================================================
# TAB 1: DATASET
# =========================================================
with tab_datos:
    st.subheader("Vista general de los datos")

    st.write("Primeras filas del dataset:")
    st.dataframe(df.head())

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Información de columnas**")
        st.write(pd.DataFrame({
            "columna": df.columns,
            "tipo": df.dtypes.astype(str).values
        }))
    with c2:
        st.write("**Estadísticos descriptivos (numéricos)**")
        st.write(df.describe())

    st.write("Tamaño del dataset:")
    st.write(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")


# =========================================================
# TAB 2: GRÁFICOS
# =========================================================
with tab_graficos:
    st.subheader("Gráficos descriptivos")

    grafico = st.selectbox(
        "Selecciona el gráfico:",
        [
            "Distribución de edad según ACV",
            "Distribución de glucosa según ACV",
            "Frecuencia de ACV",
        ],
    )

    if grafico == "Distribución de edad según ACV":
        fig, ax = plt.subplots()
        sns.boxplot(data=df_viz, x="stroke_label", y="age", ax=ax)
        ax.set_xlabel("ACV")
        ax.set_ylabel("Edad")
        ax.set_title("Distribución de edad según presencia de ACV")
        st.pyplot(fig)

    elif grafico == "Distribución de glucosa según ACV":
        # Para que sea legible, usamos bins de glucosa
        df_glu = df_viz.copy()
        df_glu["glucose_bin"] = pd.qcut(
            df_glu["avg_glucose_level"],
            q=10,
            duplicates="drop"
        )

        grupo = (
            df_glu.groupby(["glucose_bin", "stroke_label"])
            .size()
            .reset_index(name="count")
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(
            data=grupo,
            x="glucose_bin",
            y="count",
            hue="stroke_label",
            ax=ax,
        )
        ax.set_xlabel("Rango de glucosa promedio")
        ax.set_ylabel("Cantidad de pacientes")
        ax.set_title("Glucosa vs presencia de ACV")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    elif grafico == "Frecuencia de ACV":
        fig, ax = plt.subplots()
        sns.countplot(data=df_viz, x="stroke_label", ax=ax)
        ax.set_xlabel("ACV")
        ax.set_ylabel("Cantidad de pacientes")
        ax.set_title("Frecuencia de ACV en el dataset")
        st.pyplot(fig)

        st.write("**Conteo:**")
        st.write(df_viz["stroke_label"].value_counts())

        st.write("**Porcentajes (%):**")
        st.write(df_viz["stroke_label"].value_counts(normalize=True) * 100)


# =========================================================
# TAB 3: MODELO PREDICTIVO
# =========================================================
with tab_modelo:
    st.subheader("Modelo predictivo de ACV (RandomForest)")

    st.write(
        "Se construye un modelo de clasificación usando un pipeline con "
        "preprocesamiento (escalado + one-hot) y un RandomForest."
    )

    if st.button("Entrenar modelo"):
        try:
            # Quitamos filas con NaN para simplificar
            df_model = df.dropna().copy()

            # Variables de entrada y salida
            X = df_model.drop(columns=["stroke", "id"], errors="ignore")
            y = df_model["stroke"]

            # Separar numéricas y categóricas
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns
            cat_cols = X.select_dtypes(include=["object"]).columns

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ]
            )

            clf = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
            )

            model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", clf),
                ]
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=42,
                stratify=y,
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Matriz de confusión")
                cm = confusion_matrix(y_test, y_pred)
                st.write(cm)

            with col2:
                st.markdown("### ROC-AUC")
                roc_auc = roc_auc_score(y_test, y_proba)
                st.metric("ROC-AUC", f"{roc_auc:.3f}")

            st.markdown("### Reporte de clasificación")
            st.text(classification_report(y_test, y_pred))

        except Exception as e:
            st.error("Ocurrió un error al entrenar o evaluar el modelo:")
            st.exception(e)
