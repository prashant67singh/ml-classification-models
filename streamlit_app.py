import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report)


# Page Configuration
st.set_page_config(page_title="Heart Disease Classifier", page_icon="🫀", layout="wide")

st.markdown("""
        <style>
        .stMainBlockContainer {
            padding: 0.5rem 2rem
        }
        </style>
    """, unsafe_allow_html=True)

# Page Footer
st.markdown("""
        <style>
        .footer {
            position: fixed; bottom: 0; left: 0; width: 100%;
            background-color: #0e1117; color: #666; text-align: center;
            padding: 4px; font-size: 13px; border-top: 1px solid #333;
            z-index: 99999;
        }
        </style>
        <div class="footer">
            Designed By - Prashant Singh
        </div>
    """, unsafe_allow_html=True)


# Header
st.title("🫀 Heart Disease Classification")
st.write("Compare 5 ML models on the Heart Disease Dataset")

# SideBar
st.sidebar.header("Configuration")

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

def clear_file_uploader():
    st.session_state.uploader_key += 1

selected_model_name = st.sidebar.selectbox(
    "Select Classifier Model",
    [ "Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest"],
    on_change=clear_file_uploader,
)

uploaded_file = st.sidebar.file_uploader( "Upload Test Data (CSV)",type=["csv"], key=f"Uploader_{st.session_state.uploader_key}")

# Model Paths
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

@st.cache_resource
def load_all_models():
    return {name: load_model(path) for name, path in MODEL_PATHS.items()}


@st.dialog("Test Dataset Overview", width="large")
def show_dataset(df):
    COLUMN_DESCRIPTIONS = {
        'age':      'Age of the patient in years',
        'sex':      'Gender of the patient(1 = male, 0 = female)',
        'cp':       'Chest pain type (1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic)',
        'trestbps': 'Resting blood pressure',
        'chol':     'Serum cholesterol level in mg/dl',
        'fbs':      'Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)',
        'restecg':  'Resting electrocardiographic results (0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy)',
        'thalach':  'Maximum heart rate achieved',
        'exang':    'Exercise induced angina (1 = Yes, 0 = No)',
        'oldpeak':  'ST depression induced by exercise relative to rest',
        'slope':    'Slope of peak exercise ST segment (1 = Upsloping, 2 = Flat, 3 = Downsloping)',
        'ca':       'Number of major vessels (0-3) colored by fluoroscopy',
        'thal':     'Thalassemia blood disorder (3 = Normal, 6 = Fixed defect, 7 = Reversible defect)',
        'target_binary': 'Heart disease presence (0 = No Disease, 1 = Disease)'
    }
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    # Column descriptions
    tab1, tab2 = st.tabs(["Column Descriptions", "Sample Data"])
    with tab1:
        col_df = pd.DataFrame({
            'Column':      list(COLUMN_DESCRIPTIONS.keys()),
            'Description': list(COLUMN_DESCRIPTIONS.values())
        })
        st.dataframe(col_df, use_container_width=True, hide_index=True)

    with tab2:
        st.dataframe(df.head(10), use_container_width=True)
    

# Helper Methods
def compute_Metrics(model, X_test, y_test):
    y_predicted = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:,1]
    return y_predicted, y_probability, {
        "Accuracy":  round(accuracy_score(y_test, y_predicted), 4),
        "AUC":       round(roc_auc_score(y_test, y_probability), 4),
        "Precision": round(precision_score(y_test, y_predicted), 4),
        "Recall":    round(recall_score(y_test, y_predicted), 4),
        "F1 Score":  round(f1_score(y_test, y_predicted), 4),
        "MCC":       round(matthews_corrcoef(y_test, y_predicted), 4)
    }

# Main Content
REQUIRED_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target_binary']

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    missing_cols = set(df.columns) != set(REQUIRED_COLUMNS)

    if missing_cols:
        st.error(f"Invalid Test dataset. Please Upload Correct Test dataset.", icon="❌", title="Error")
        st.stop()
    
    X_test = df.drop(columns=['target_binary'])
    y_test = df['target_binary']

    progress = st.progress(0, text=f"Loading ...", width="stretch")
    for percent_complete in range(100):
        time.sleep(0.006)
        progress.progress(percent_complete + 1, text=f"Loading ...", width="stretch")
    progress.progress(100, text="Done!")
    progress.empty()

    st.sidebar.button("View Uploaded Dataset", width="stretch", type="primary", on_click=show_dataset, kwargs={"df": df})
        
    model = load_model(MODEL_PATHS[selected_model_name])
    all_models = load_all_models()

    # Tabs
    tab1, tab2 = st.tabs(["Single Model Analysis", "All Models Comparison"])

    # Tab 1 
    with tab1:
        st.subheader(f"{selected_model_name} - Evaluation Metrics")
        y_predicted, y_probability, metrics = compute_Metrics(model, X_test, y_test)

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("🎯 Accuracy",  metrics["Accuracy"])
        col2.metric("📐 AUC",       metrics["AUC"])
        col3.metric("🔍 Precision", metrics["Precision"])
        col4.metric("🎪 Recall",    metrics["Recall"])
        col5.metric("⚖️ F1 Score",  metrics["F1 Score"])
        col6.metric("🔮 MCC",       metrics["MCC"])

        st.divider()
        col_cm, col_cr = st.columns([1, 1])
        with col_cm:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_predicted)
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                        xticklabels=['No Disease', 'Disease'],
                        yticklabels=['No Disease', 'Disease'])
            ax.set_title(f'{selected_model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.tight_layout()
            st.pyplot(fig)

        with col_cr:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_predicted, target_names=['No Disease', 'Disease'], output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(4)
            st.dataframe(report_df, use_container_width=True)

    # Tab 2
    with tab2:
        st.subheader("All Models Comparison")
        results= []
        for name, model in all_models.items():
            y_predicted, y_probability, metrics = compute_Metrics(model, X_test, y_test)
            results.append({"Model": name, **metrics})

        comparison_df = pd.DataFrame(results)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.subheader("📊 Metrics Bar Chart")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(comparison_df))
        width = 0.13
        metrics_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
        colors = ['#0ea5e9', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#ec4899']
        for i, (metric, color) in enumerate(zip(metrics_cols, colors)):
            ax2.bar(x + i * width, comparison_df[metric], width,
                    label=metric, color=color, alpha=0.85)
        ax2.set_xticks(x + width * 2.5)
        ax2.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
        ax2.set_ylim(0, 1.15)
        ax2.set_ylabel('Score')
        ax2.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)

else:
    st.info(f"Please Upload test_data.csv from the sidebar to get started", title="Info")
    st.stop()