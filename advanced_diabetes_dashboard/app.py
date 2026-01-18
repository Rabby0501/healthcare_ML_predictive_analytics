import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import plotly.graph_objects as go
from reportlab.pdfgen import canvas
from datetime import datetime
import io

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Advanced Diabetes Prediction Dashboard")
st.markdown("A real-time decision support system using ML models.")

# -------------------------------------------------------------------
# LOAD MODELS + DATA
# -------------------------------------------------------------------
DATA_PATH = "data/diabetes.csv"
MODEL_PATH = "saved_models/"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()
feature_names = df.drop("Outcome", axis=1).columns.tolist()

models = {
    "Decision Tree": joblib.load(MODEL_PATH + "decision_tree.pkl"),
    "Random Forest": joblib.load(MODEL_PATH + "random_forest.pkl"),
    "Naive Bayes": joblib.load(MODEL_PATH + "naive_bayes.pkl"),
    "Logistic Regression": joblib.load(MODEL_PATH + "logistic_regression.pkl"),
    "XGBoost": joblib.load(MODEL_PATH + "xgboost.pkl"),
}

scaler = joblib.load(MODEL_PATH + "scaler.pkl")
eval_data = np.load(MODEL_PATH + "eval_data.npz", allow_pickle=True)

y_test = eval_data["y_test"]
saved_keys = eval_data.files


# -------------------------------------------------------------------
# AUTO-DETECT PROBABILITY KEY FIX
# -------------------------------------------------------------------
def get_prob_key(model_name):
    """
    Returns the correct probability array key for the model,
    regardless of naming style.
    """

    model_name = model_name.lower().replace(" ", "")

    possible_patterns = [
        model_name,
        model_name.replace("naivebayes", "naive_bayes"),
        model_name.replace("randomforest", "random_forest"),
        model_name.replace("decisiontree", "decision_tree"),
        model_name.replace("logisticregression", "logistic_regression"),
        model_name.replace("xgboost", "xgboost"),
        model_name.replace("xgboost", "xgb"),
    ]

    for key in saved_keys:
        key_clean = key.lower().replace("_", "").replace("prob", "")

        for pattern in possible_patterns:
            if pattern in key_clean:
                return key

    # fallback: try any key ending with _prob
    for key in saved_keys:
        if key.endswith("_prob"):
            return key

    st.error(f"Could not detect probability key for {model_name}. Saved keys: {saved_keys}")
    st.stop()


# -------------------------------------------------------------------
# INPUT FORM
# -------------------------------------------------------------------
def patient_input():
    st.sidebar.header("Patient Information")

    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.number_input("Glucose", 0, 300, 120)
    bp = st.sidebar.number_input("Blood Pressure", 0, 200, 70)
    skin = st.sidebar.number_input("Skin Thickness", 0, 99, 20)
    insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
    bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.sidebar.number_input("Age", 1, 120, 33)

    return np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])


# -------------------------------------------------------------------
# PDF GENERATION
# -------------------------------------------------------------------
def generate_pdf(pred_text, prob, inputs, model_name):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, 800, "Diabetes Prediction Report")

    c.setFont("Helvetica", 12)
    c.drawString(40, 780, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(40, 760, f"Model Used: {model_name}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, 730, "Patient Inputs:")
    y = 710
    for k, v in inputs.items():
        c.setFont("Helvetica", 11)
        c.drawString(60, y, f"{k}: {v}")
        y -= 18

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y - 10, "Prediction:")
    c.setFont("Helvetica", 11)
    c.drawString(60, y - 30, f"Result: {pred_text}")
    c.drawString(60, y - 50, f"Probability: {prob:.3f}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# -------------------------------------------------------------------
# DOCTOR RECOMMENDATION SYSTEM
# -------------------------------------------------------------------
def doctor_recommend(prob, row):
    glucose = row[0][1]
    bmi = row[0][5]
    age = row[0][7]

    if prob < 0.3:
        risk = "Low"
        msg = "Maintain a healthy lifestyle."
    elif prob < 0.6:
        risk = "Moderate"
        msg = "Monitor glucose, weight & check annually."
    else:
        risk = "High"
        msg = "Consult doctor. Consider HbA1c, OGTT, FPG."

    flags = []
    if glucose > 140:
        flags.append("High glucose detected.")
    if bmi > 30:
        flags.append("BMI indicates obesity.")
    if age > 45:
        flags.append("Older age increases diabetes risk.")

    return risk, msg, "\n".join(flags) if flags else "No major risk flags."


# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📈 Data Insights"])

# ================================================================
# 🔮 PREDICTION TAB
# ================================================================
with tab1:
    st.header("🔮 Single Patient Prediction")

    input_data = patient_input()
    selected_model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
    model = models[selected_model_name]

    scaled = scaler.transform(input_data)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    pred_text = "🟥 Diabetic" if pred == 1 else "🟩 Not Diabetic"

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", pred_text)
    c2.metric("Probability", f"{prob:.3f}")
    c3.metric("Model", selected_model_name)

    # Gauge
    gauge_color = "green" if prob < 0.3 else "orange" if prob < 0.6 else "red"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(prob * 100),
        title={"text": "Risk %"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": gauge_color}},
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Doctor Recommendation
    st.subheader("🩺 Doctor Recommendation")
    risk, msg, flags = doctor_recommend(prob, input_data)
    st.write(f"**Risk Level:** {risk}")
    st.write(msg)
    st.write("**Risk Flags:**")
    st.write(flags)

    # PDF Download
    inp_dict = {
        "Pregnancies": input_data[0][0],
        "Glucose": input_data[0][1],
        "Blood Pressure": input_data[0][2],
        "Skin Thickness": input_data[0][3],
        "Insulin": input_data[0][4],
        "BMI": input_data[0][5],
        "Diabetes Pedigree": input_data[0][6],
        "Age": input_data[0][7],
    }
    pdf = generate_pdf(pred_text, prob, inp_dict, selected_model_name)

    st.download_button(
        "📄 Download PDF Report",
        pdf,
        file_name="prediction_report.pdf",
        mime="application/pdf",
    )


# ================================================================
# 📊 MODEL PERFORMANCE TAB (WITH AUTO-KEY FIX)
# ================================================================
with tab2:
    st.header("📊 Model Performance")

    metrics = []

    for name, mdl in models.items():
        key = get_prob_key(name)  # AUTO-DETECTED
        probs = eval_data[key]
        preds = (probs >= 0.5).astype(int)

        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1-score": f1_score(y_test, preds, zero_division=0),
            "AUC": auc(*roc_curve(y_test, probs)[:2]),
        })

    df_metrics = pd.DataFrame(metrics)

    # format numeric columns only
    num_cols = df_metrics.select_dtypes(include=[float]).columns
    st.dataframe(df_metrics.style.format({col: "{:.3f}" for col in num_cols}),
                 use_container_width=True)

    st.subheader("AUC Comparison")
    fig2, ax = plt.subplots()
    sns.barplot(x="Model", y="AUC", data=df_metrics, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # MODEL-SPECIFIC CONFUSION MATRIX + ROC
    st.subheader(f"Confusion Matrix & ROC for {selected_model_name}")

    key = get_prob_key(selected_model_name)
    probs = eval_data[key]
    preds = (probs >= 0.5).astype(int)

    c1, c2 = st.columns(2)

    with c1:
        cm = confusion_matrix(y_test, preds)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

    with c2:
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.legend()
        st.pyplot(fig_roc)


# ================================================================
# 📈 DATA INSIGHTS TAB
# ================================================================
with tab3:
    st.header("📈 Data Insights")

    st.subheader("Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    c1, c2 = st.columns(2)

    with c1:
        fig_g = sns.histplot(df["Glucose"], kde=True)
        plt.title("Glucose Distribution")
        st.pyplot(plt.gcf())
        plt.clf()

    with c2:
        fig_b = sns.histplot(df["BMI"], kde=True)
        plt.title("BMI Distribution")
        st.pyplot(plt.gcf())
        plt.clf()
