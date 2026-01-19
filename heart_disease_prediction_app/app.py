import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from fpdf import FPDF
import datetime

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction Web App",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction Web App")
st.write("A real-world clinical decision support system powered by improved ML & DL models.")

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_all_models():
    models = {
        "Logistic Regression": joblib.load("models/lr_model.pkl"),
        "KNN": joblib.load("models/knn_model.pkl"),
        "Decision Tree": joblib.load("models/dt_model.pkl"),
        "ANN": load_model("models/ann_model.h5"),
        "DNN": load_model("models/dnn_model.h5"),
    }
    scaler = joblib.load("models/scaler.pkl")
    return models, scaler

models, scaler = load_all_models()

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/heart-disease-UCI.csv")

df = load_data()

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------

# 1️⃣ RISK LEVEL FUNCTION
def risk_level(prob):
    if prob < 0.35:
        return "Low Risk"
    elif prob < 0.70:
        return "Moderate Risk"
    else:
        return "High Risk"


# 2️⃣ MEDICAL RECOMMENDATION GENERATOR
def generate_recommendations(prob, user_input):
    rec = []

    if prob < 0.35:
        rec += [
            "Maintain regular physical activity.",
            "Continue balanced diet low in saturated fats.",
            "Monitor cholesterol and blood pressure occasionally."
        ]
    elif prob < 0.70:
        rec += [
            "Schedule a cardiovascular checkup within 3–6 months.",
            "Reduce sodium intake and avoid smoking.",
            "Engage in moderate-intensity exercise 30 minutes daily."
        ]
    else:
        rec += [
            "Consult a cardiologist immediately for further evaluation.",
            "Request ECG, lipid profile, and stress-test.",
            "Avoid heavy physical exertion until medically cleared."
        ]

    # Feature-based observations
    if user_input["chol"] > 250:
        rec.append("Cholesterol is high — consider lipid-lowering interventions.")
    if user_input["trestbps"] > 140:
        rec.append("High BP detected — reduce salt intake and monitor daily.")
    if user_input["oldpeak"] > 2.0:
        rec.append("Elevated ST depression — possible ischemia risk.")
    if user_input["thalach"] < 120:
        rec.append("Low maximum heart rate — cardiology review suggested.")

    return rec


# 3️⃣ PDF REPORT GENERATOR (NO EMOJIS)
def generate_pdf_report(prediction, prob, risk, user_input, model_used, recommendations):
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- HEADER ----
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Heart Disease Diagnostic Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Generated On: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(8)

    # ---- PATIENT DATA TABLE ----
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Patient Input Data", ln=True)
    pdf.set_font("Arial", "", 12)

    for key, value in user_input.items():
        pdf.cell(60, 8, f"{key}:", border=1)
        pdf.cell(0, 8, str(value), border=1, ln=True)

    # ---- RESULTS ----
    pdf.ln(8)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Prediction Results", ln=True)
    pdf.set_font("Arial", "", 12)

    pdf.cell(60, 8, "Prediction:", border=1)
    pdf.cell(0, 8, "Heart Disease" if prediction == 1 else "No Heart Disease", border=1, ln=True)
    pdf.cell(60, 8, "Probability:", border=1)
    pdf.cell(0, 8, f"{prob:.3f}", border=1, ln=True)
    pdf.cell(60, 8, "Risk Level:", border=1)
    pdf.cell(0, 8, risk, border=1, ln=True)
    pdf.cell(60, 8, "Model Used:", border=1)
    pdf.cell(0, 8, model_used, border=1, ln=True)

    # ---- RECOMMENDATIONS ----
    pdf.ln(8)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Medical Recommendations", ln=True)
    pdf.set_font("Arial", "", 12)

    for r in recommendations:
        safe_text = r.encode("latin-1", "ignore").decode("latin-1")
        pdf.multi_cell(0, 8, f"- {safe_text}")

    # ---- SIGNATURE ----
    pdf.ln(15)
    pdf.cell(0, 8, "Doctor/Reviewer Signature: ____________________________", ln=True)

    # ---- FOOTER ----
    pdf.set_y(-20)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"Page {pdf.page_no()}", align="C")

    filename = "Patient_Report.pdf"
    pdf.output(filename)
    return filename

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
section = st.sidebar.radio("Navigation", ["Make Prediction", "Model Evaluation"])

model_name = st.sidebar.selectbox("Select Model:", list(models.keys()))
model = models[model_name]

# ---------------------------------------------------------
# MAKE PREDICTION
# ---------------------------------------------------------
if section == "Make Prediction":
    st.subheader("🧪 Enter Patient Details")

    # Input layout
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 100, 50)
        sex = st.selectbox("Sex (1=M, 0=F)", [0, 1])
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])

    with col2:
        trestbps = st.number_input("Resting BP", 80, 220, 130)
        chol = st.number_input("Cholesterol", 100, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

    with col3:
        restecg = st.selectbox("Rest ECG (0–2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 70, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (1,2,3)", [1, 2, 3])

    if st.button("🔍 Predict"):
        user_input = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }

        data = np.array([list(user_input.values())])
        data_scaled = scaler.transform(data)

        # Probability
        if model_name in ["ANN", "DNN"]:
            prob = float(model.predict(data_scaled)[0])
        else:
            prob = float(model.predict_proba(data_scaled)[0][1])

        prediction = 1 if prob > 0.5 else 0
        risk = risk_level(prob)

        recommendations = generate_recommendations(prob, user_input)

        # Display result
        st.subheader("🔎 Prediction Result")
        st.write(f"**Probability:** {prob:.3f}")
        st.write(f"**Risk Level:** {risk}")

        # Gauge Chart
        risk_percentage = prob * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percentage,
            title={'text': "Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 35], 'color': "green"},
                    {'range': [35, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"},
                ],
                'bar': {'color': "black"},
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.subheader("📝 Medical Recommendations")
        for r in recommendations:
            st.write(f"- {r}")

        # Generate PDF
        pdf_file = generate_pdf_report(
            prediction, prob, risk, user_input, model_name, recommendations
        )

        with open(pdf_file, "rb") as f:
            st.download_button(
                "📄 Download Patient Report (PDF)",
                f,
                file_name="Patient_Report.pdf",
                mime="application/pdf",
            )

# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------
if section == "Model Evaluation":
    st.subheader(f"📊 Evaluation of Model: {model_name}")

    X = df.drop("target", axis=1)
    y = df["target"]
    X_scaled = scaler.transform(X)

    if model_name in ["ANN", "DNN"]:
        y_prob = model.predict(X_scaled).ravel()
    else:
        y_prob = model.predict_proba(X_scaled)[:, 1]

    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision", f"{prec:.3f}")
    col3.metric("Recall", f"{rec:.3f}")
    col4.metric("F1 Score", f"{f1:.3f}")

    st.write("---")
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
    st.pyplot(fig1)

    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc_score = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    ax2.plot([0,1],[0,1],'--')
    ax2.legend()
    st.pyplot(fig2)
