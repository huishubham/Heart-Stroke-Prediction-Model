# ❤️ Heart Stroke Prediction Model

A machine learning–based web application that predicts the **risk of heart disease/stroke** using patient health parameters.  
The model is trained on clinical data and deployed using **Streamlit** to provide real-time predictions along with **simple, actionable health advice**.

> ⚠️ This project is for educational purposes only and is not a medical diagnosis tool.

---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early risk detection can help individuals take preventive steps in time.

This project:
- Analyzes patient health data
- Trains multiple machine learning models
- Deploys the best-performing model in a user-friendly web app
- Provides **easy-to-understand lifestyle tips** based on prediction results

---

## 🧠 Machine Learning Pipeline

1. Data loading and exploration  
2. Data cleaning and mean imputation  
3. One-hot encoding of categorical variables  
4. Feature scaling using `StandardScaler`  
5. Model training and evaluation  
6. Model serialization using `joblib`  
7. Deployment using Streamlit  

---

## 🤖 Models Trained

- Logistic Regression ✅ *(Final deployed model)*
- Decision Tree
- K-Nearest Neighbors
- Support Vector Machine
- Naive Bayes

Logistic Regression was selected due to:
- Stable performance
- Better interpretability
- Balanced handling of class imbalance

---

## 🧪 Evaluation Metrics

- Accuracy
- F1-Score

These metrics help assess the model’s ability to detect high-risk patients while minimizing false predictions.

---

## 🖥️ Web Application Features

- Interactive sliders and dropdowns for patient input
- Real-time heart disease risk prediction
- Probability-based output
- Simple lifestyle advice for both low-risk and high-risk users
- Medical disclaimer for safe usage

---

## 🧾 Lifestyle Advice Logic

### 🟢 Low Risk (Prediction = 0)
- Walk at least 30 minutes daily and eat less oily food  
- Limit sugar and salt intake and get yearly health check-ups  

### 🔴 High Risk (Prediction = 1)
- Avoid fried and salty food, quit smoking, and start light exercise after consulting a doctor  
- Monitor blood pressure, sugar, and cholesterol regularly and consult a heart specialist  

---

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Streamlit  
- Joblib  

---

## 🤝 Let’s Connect

If you found this project interesting or have suggestions for improvement, feel free to connect with me on LinkedIn.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/shubham-panwar-8a37a0250)
