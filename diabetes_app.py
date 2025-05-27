import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("diabetes - diabetes.csv")

# Clean data
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# Split X and y
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# Models
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svc = SVC(kernel='linear', probability=True)

# Ensemble
voting_clf = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('svc', svc)], voting='soft')
voting_clf.fit(X_train, y_train)

# --- STREAMLIT UI ---
st.title("🔍 Diabetes Prediction App")
st.markdown("Enter the patient's data below to check if they are at risk of diabetes.")

# Input fields
preg = st.number_input("Pregnancies", 0, 20, step=1)
glucose = st.slider("Glucose", 0, 200, 110)
bp = st.slider("Blood Pressure", 0, 140, 70)
skin = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin", 0, 900, 100)
bmi = st.slider("BMI", 10.0, 60.0, 28.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 10, 100, 30)

if st.button("Predict"):
    user_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    user_data_scaled = scaler.transform(user_data)
    prediction = voting_clf.predict(user_data_scaled)
    result = "🟢 Not Diabetic" if prediction[0] == 0 else "🔴 Diabetic"
    st.success(f"Result: **{result}**")
