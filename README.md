# 🧠 Diabetes Prediction App

An interactive, AI-powered web app that predicts the likelihood of diabetes based on medical input parameters. Built using **Streamlit** and **scikit-learn**, this app allows users to explore how various health metrics impact diabetes risk.

---

## 🚀 Features

- ✅ Predicts diabetes using a trained ML model
- ✅ User-friendly interface powered by **Streamlit**
- ✅ Uses an **ensemble classifier** (Logistic Regression, Random Forest, SVM)
- ✅ Real-time prediction from sliders and input fields
- ✅ Handles missing or zero values with clean preprocessing
- ✅ Fully scalable and **deployable on Streamlit Cloud**

---

## 🗂 Technologies Used

- Python
- Pandas, NumPy
- scikit-learn
- Streamlit
- Matplotlib/Seaborn (for model evaluation)

---

## 📊 Input Parameters

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age

> 📝 These are features from the PIMA Indian Diabetes Dataset

---

## 🧪 How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/diabetes-predictor-app.git
   cd diabetes-predictor-app
Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run diabetes_app.py
