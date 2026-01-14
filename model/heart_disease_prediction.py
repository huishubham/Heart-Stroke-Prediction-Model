# Importing required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, recall_score
import joblib

warnings.filterwarnings('ignore')

# Reading the csv file
df = pd.read_csv('heart.csv')

# Data Cleaning
df_cleaned = df.copy()

# Basic Understanding of Data
df.head()
df.tail()
df.shape
df.describe()
df.isnull().sum()
df.columns

# EDA for Numerical columns
numerical_columns = ['Age','RestingBP','Cholesterol','FastingBS','MaxHR']
for col in numerical_columns:
    plt.figure(figsize = (6,4))
    sns.histplot(df[col],kde=True)

# EDA for Categorical columns
sns.countplot(x= df['Sex'])
sns.countplot(x= df['ChestPainType'])
sns.countplot(x= df['RestingECG'])
sns.countplot(x= df['ExerciseAngina'])
sns.countplot(x= df['ST_Slope'])

# To check for any outliers present
for col in numerical_columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])

# To create a correlation heatmap
sns.heatmap(df.corr(numeric_only = True), annot=True)

# Mean Imputation
ch_mean = df.loc[df['Cholesterol'] != 0,'Cholesterol'].mean()
df_cleaned['Cholesterol'] = df_cleaned['Cholesterol'].replace(0,ch_mean)
df_cleaned['Cholesterol'] = df_cleaned['Cholesterol'].round(2)
rest_bp_mean = df_cleaned.loc[df_cleaned['RestingBP'] != 0, 'RestingBP'].mean()
df_cleaned['RestingBP'] = df_cleaned['RestingBP'].replace(0, rest_bp_mean)
df_cleaned['RestingBP'].round(2)

# One Hot Encoding
df_cleaned = pd.get_dummies(df_cleaned)
df_cleaned = df_cleaned.astype(int)

# Standard Scaling
scaling_col = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
df_cleaned[scaling_col] = scaler.fit_transform(df_cleaned[scaling_col])

# Separating Features and Target Cols
X = df_cleaned.drop('HeartDisease', axis = 1)
y = df_cleaned['HeartDisease']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and training
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Naive Bayes': GaussianNB()
}

result = []

# Model Training and Evaluation
for name, model in models.items():
  model.fit(X_train,y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  result.append({
      'Model':name,
      'Accuracy':round(accuracy,2),
      'F1 Score':round(f1,2),
  })

# To create pickle files for streamlit
joblib.dump(models['Logistic Regression'], 'LogisticRegression_Heart.pkl')
joblib.dump(scaler, 'Scaler.pkl')
joblib.dump(X.columns.tolist(), 'Columns.pkl')

