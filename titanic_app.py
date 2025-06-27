import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("🚢 Titanic Survival Prediction")

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df = df.dropna(subset=['Age', 'Embarked'])
    df['Embarked'] = df['Embarked'].astype('category').cat.codes
    return df

df = load_data()
st.write("### Sample Data", df.head())

# Feature selection
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = df['Survived']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"🎯 Logistic Regression Accuracy: {accuracy:.2f}")