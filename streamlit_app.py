import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report)

st.set_page_config(
    page_title="Heart Disease Classifier",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Heart Disease Classification")
st.write("Compare 5 ML models on the Hear Disease")

st.sidebar.header("Configuration")

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data (CSV)",
    type=["csv"]
)

REQUIRED_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target_binary']

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_cols:
        st.toast(f"Invalid Test dataset.", icon="❌", duration="long")
        st.stop()
    
    X_test = df.drop(columns=['target_binary'])
    y_test = df['target_binary']

    st.toast(f"Test Data Uploaded Successfully!", icon="✅", duration="long")
    st.dataframe(df.head())
else: 
    st.info(f"Please Upload test_data.csv from the sidebar to get started", title="Info")
    st.stop()


MODEL_PATHS = { 
   "Logistic Regression" : "models/pkl_files/logistic_regression.pkl",
   "Decision Tree": "models/pkl_files/decision_tree.pkl",
   "KNN": "models/pkl_files/knn.pkl",
   "Naive Bayes": "models/pkl_files/naive_bayes.pkl",
   "Random Forest": "models/pkl_files/random_forest.pkl"

}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATHS[selected_model_name])
st.toast(f"**Model loaded:** {selected_model_name}", duration="long")

st.subheader(f"Evaluation Metrics — {selected_model_name}")

y_predicted = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

col1.metric("🎯 Accuracy",  f"{accuracy_score(y_test, y_predicted):.4f}")
col2.metric("📐 AUC",       f"{roc_auc_score(y_test, y_probability):.4f}")
col3.metric("🔍 Precision", f"{precision_score(y_test, y_predicted):.4f}")
col4.metric("🎪 Recall",    f"{recall_score(y_test, y_predicted):.4f}")
col5.metric("⚖️ F1 Score",  f"{f1_score(y_test, y_predicted):.4f}")
col6.metric("🔮 MCC",       f"{matthews_corrcoef(y_test, y_predicted):.4f}")


st.subheader("🔲 Confusion Matrix")

cm = confusion_matrix(y_test, y_predicted)

fig, ax = plt.subplots(figsize=(5, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
ax.set_title(f'{selected_model_name} — Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()

col1, col2 = st.columns([1, 1])
with col1:
    st.pyplot(fig)


st.subheader("📋 Classification Report")

report = classification_report(y_test, y_predicted, 
                               target_names=['No Disease', 'Disease'],
                               output_dict=True)

report_df = pd.DataFrame(report).transpose().round(4)

st.dataframe(report_df, use_container_width=True)


st.subheader("🏆 All Models Comparison")

@st.cache_resource
def load_all_models():
    models = {
        "Logistic Regression": joblib.load("models/pkl_files/logistic_regression.pkl"),
        "Decision Tree":        joblib.load("models/pkl_files/decision_tree.pkl"),
        "KNN":                  joblib.load("models/pkl_files/knn.pkl"),
        "Naive Bayes":          joblib.load("models/pkl_files/naive_bayes.pkl"),
        "Random Forest":        joblib.load("models/pkl_files/random_forest.pkl")
    }
    return models

all_models = load_all_models()

results = []
for name, m in all_models.items():
    yp = m.predict(X_test)
    yb = m.predict_proba(X_test)[:, 1]
    results.append({
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, yp), 4),
        "AUC":       round(roc_auc_score(y_test, yb), 4),
        "Precision": round(precision_score(y_test, yp), 4),
        "Recall":    round(recall_score(y_test, yp), 4),
        "F1 Score":  round(f1_score(y_test, yp), 4),
        "MCC":       round(matthews_corrcoef(y_test, yp), 4)
    })

comparison_df = pd.DataFrame(results)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)