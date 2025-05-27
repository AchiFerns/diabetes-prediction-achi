# 🧠 Diabetes Prediction App

An AI-powered, interactive web app built using **Streamlit** and **scikit-learn** that predicts a patient’s risk of diabetes based on medical input parameters like glucose, BMI, insulin level, etc. The app uses an ensemble ML model and delivers real-time predictions in a clean UI.

---

## 🚀 Live Demo

🌐 [Click here to try the app](https://diabetes-prediction-achi-eq5twq4uktfjjj7wgvbzha.streamlit.app/)

> 🔍 Just enter your health details and get an instant result — “Diabetic” or “Not Diabetic” — with prediction confidence.

---

## 📊 Features

- ✅ Real-time diabetes risk prediction
- ✅ Intuitive web UI powered by Streamlit
- ✅ Ensemble model (Logistic Regression + Random Forest + SVM)
- ✅ Clean data preprocessing (NaN handling, scaling)
- ✅ Developer-mode metrics: Accuracy, Confusion Matrix, Classification Report
- ✅ Deployable on Streamlit Cloud or Render

---

## 📦 Tech Stack

- **Frontend/UI**: Streamlit  
- **Machine Learning**: scikit-learn  
- **Data**: PIMA Indian Diabetes Dataset (UCI ML Repository)  
- **Visualization**: seaborn + matplotlib

---

## 📝 How to Run Locally

```bash
git clone https://github.com/your-username/diabetes-prediction-app.git
cd diabetes-prediction-app
pip install -r requirements.txt
streamlit run diabetes_app.py
📁 Dataset Overview
diabetes.csv: Contains 768 samples with 8 features

Features include: Pregnancies, Glucose, BloodPressure, BMI, etc.

Target column: Outcome (0 = Not Diabetic, 1 = Diabetic)
