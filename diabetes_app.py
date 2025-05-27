import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")
# --------------------- Load and Preprocess Dataset ---------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes - diabetes.csv")
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    return df

df = load_data()
X = df.drop('Outcome', axis=1)
y = df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# --------------------- Train the Model ---------------------
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svc = SVC(kernel='linear', probability=True)
voting_clf = VotingClassifier(estimators=[('LogReg', lr), ('RF', rf), ('SVM', svc)], voting='soft')
voting_clf.fit(X_train, y_train)

# --------------------- Streamlit UI ---------------------
st.title("🧠 Diabetes Prediction App")
st.markdown("Enter your health details below to check your diabetes risk level. This app uses AI (machine learning) to make predictions based on real medical data.")

# --------------------- Input Fields ---------------------
with st.form("input_form"):
    st.subheader("🔧 Patient Data Input")
    col1, col2 = st.columns(2)

    with col1:
        preg = st.number_input("Pregnancies", 0, 20, step=1)
        glucose = st.slider("Glucose Level", 0, 200, 110)
        bp = st.slider("Blood Pressure", 0, 140, 70)
        skin = st.slider("Skin Thickness", 0, 100, 20)

    with col2:
        insulin = st.slider("Insulin Level", 0, 900, 100)
        bmi = st.slider("BMI", 10.0, 60.0, 28.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.slider("Age", 10, 100, 30)

    submitted = st.form_submit_button("🔍 Predict")

# --------------------- Prediction & Result ---------------------
if submitted:
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = voting_clf.predict(input_scaled)[0]
    prediction_proba = voting_clf.predict_proba(input_scaled)[0][prediction]

    st.markdown("## 🩺 Prediction Result:")
    st.success("🟢 The patient is **not diabetic**." if prediction == 0 else "🔴 The patient is **diabetic**.")
    st.info(f"Prediction Confidence: **{prediction_proba*100:.2f}%**")

    # --------------------- Optional: Model Evaluation ---------------------
    with st.expander("📊 Show Model Evaluation (Developer Info)"):
        y_pred = voting_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)

        st.metric("Model Accuracy", f"{acc*100:.2f}%")

        st.markdown("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.markdown("### Classification Report")
        cr_df = pd.DataFrame(cr).transpose()
        st.dataframe(cr_df.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}'}))
