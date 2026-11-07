# app.py
# Streamlit front-end: upload test data, pick a model, see metrics and a confusion matrix.
# The models are LOADED from disk. Nothing is retrained here.

import joblib
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from evaluation import score_model   # same metric function used during training

# Name of the label column in the CSV the user uploads
TARGET = "target"

# Dropdown label -> saved model file.
# These paths are relative, so run the app from the project folder.
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}

# ---- Page setup ----  (change the title and icon to make it yours)
st.set_page_config(page_title="Tumour Classifier Comparison", page_icon="🧬", layout="wide")
st.title("🧬 Tumour Classifier Comparison")
st.write(
    "Upload the **test CSV** (features plus a `target` column where 1 = malignant). "
    "The five pre-trained models are loaded from disk and scored on your file. "
    "No retraining happens in this app."
)


# cache_resource keeps a loaded model in memory, so switching models or
# interacting with the page doesn't reload the file from disk every time.
@st.cache_resource
def load_model(path):
    return joblib.load(path)


# ---- Sidebar controls ----
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("1) Upload test data (CSV)", type="csv")
choice = st.sidebar.selectbox("2) Choose a model", list(MODEL_FILES.keys()))

# Nothing to show until a file is uploaded, so stop here with a hint.
if uploaded is None:
    st.info("👈 Upload `test_data.csv` from the repo in the sidebar to begin.")
    st.stop()

# ---- Read and validate the uploaded file ----
data = pd.read_csv(uploaded)
model = load_model(MODEL_FILES[choice])

# The saved pipeline remembers which feature columns it was trained on.
needed_cols = list(model.feature_names_in_)
missing = [c for c in needed_cols + [TARGET] if c not in data.columns]
if missing:
    st.error(f"Your file is missing {len(missing)} required column(s), e.g. {missing[:3]}")
    st.stop()

X = data[needed_cols]   # only the feature columns, in the order the model expects
y = data[TARGET]        # the true labels

# AUC and MCC can't be computed if the file contains only one class.
if y.nunique() < 2:
    st.error("The target column has only one class, so metrics can't be computed. Upload the full test set.")
    st.stop()

with st.expander("Preview uploaded data"):
    st.write(f"{len(data)} rows, {len(needed_cols)} features")
    st.dataframe(data.head())

# ---- Two tabs: selected model, and all models side by side ----
tab_one, tab_all = st.tabs(["Selected model", "Compare all models"])

with tab_one:
    st.subheader(f"Results for: {choice}")

    # Six metric cards in a row
    scores = score_model(model, X, y)
    cols = st.columns(len(scores))
    for col, (name, value) in zip(cols, scores.items()):
        col.metric(name, f"{value:.4f}")

    left, right = st.columns(2)
    preds = model.predict(X)

    with left:
        st.markdown("**Confusion matrix**")
        # labels=[0, 1] fixes the row/column order, so it always reads Benign then Malignant
        cm = confusion_matrix(y, preds, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax,
            xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.markdown("**Classification report**")
        st.code(classification_report(y, preds, target_names=["Benign", "Malignant"]))

with tab_all:
    st.subheader("All five models on the uploaded file")
    rows = []
    for name, path in MODEL_FILES.items():
        m = load_model(path)
        rows.append({"Model": name, **score_model(m, data[list(m.feature_names_in_)], y)})
    st.dataframe(pd.DataFrame(rows).set_index("Model").round(4))
