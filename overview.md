import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Set the dataset path
data_path = r'C:\Users\khand\Music'

# Attempt to load the dataset
try:
    data = pd.read_csv(data_path)
    st.success("Dataset successfully loaded!")
except FileNotFoundError:
    st.error(f"Invalid dataset path: {data_path}.")
    st.stop()

# Clean column names
data.columns = data.columns.str.strip()

# Check the columns in the dataset
st.write("Columns in the dataset:", data.columns.tolist())

# Check for the 'Severity' column
if 'Severity' not in data.columns:
    st.error("'Severity' column not found. Please check the dataset.")
    st.stop()
else:
    st.write("Unique values in 'Severity' column:", data['Severity'].unique())
    if not all(value in [0, 1, 2, 3] for value in data['Severity'].unique()):
        st.error("Invalid values in 'Severity' column!")
        st.stop()

# Define symptoms and demographic columns
symptoms = ['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat',
            'None_Sympton', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea', 'None_Experiencing']
age_groups = ['Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+']
gender = ['Gender_Female', 'Gender_Male', 'Gender_Transgender']
contact = ['Contact_Dont-Know', 'Contact_No', 'Contact_Yes']

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
for column_group in [symptoms, age_groups, gender, contact]:
    if not all(col in data.columns for col in column_group):
        missing_cols = [col for col in column_group if col not in data.columns]
        st.error(f"Missing columns: {missing_cols}")
        st.stop()
    else:
        data[column_group] = imputer.fit_transform(data[column_group])

# Set features and target variable
X = data[symptoms + age_groups + gender + contact]
y = data['Severity']

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# Train models
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

# GUI setup using Streamlit
st.title("COVID-19 Severity Predictor")
st.write("Predict COVID-19 severity based on symptoms, demographics, and contact history.")

# Collect symptoms input
st.subheader("Symptom Selection")
selected_symptoms = []
for i in range(5):  # Allow selection of up to 5 symptoms
    symptom = st.selectbox(f"Select Symptom {i+1}", ["None"] + symptoms)
    selected_symptoms.append(symptom)

# Collect age group input
st.subheader("Demographics")
selected_age_group = st.selectbox("Select Age Group", age_groups)

# Collect gender input
selected_gender = st.selectbox("Select Gender", gender)

# Collect contact history input
selected_contact = st.selectbox("Select Contact History", contact)

# Function to predict severity
def predict_severity(model):
    input_features = [0] * len(X.columns)  # Initialize feature vector

    # Map symptoms to the feature vector
    for symptom in selected_symptoms:
        if symptom != "None" and symptom in symptoms:
            input_features[X.columns.tolist().index(symptom)] = 1

    # Map age, gender, and contact history to the feature vector
    input_features[X.columns.tolist().index(selected_age_group)] = 1
    input_features[X.columns.tolist().index(selected_gender)] = 1
    input_features[X.columns.tolist().index(selected_contact)] = 1

    # Predict severity
    severity = model.predict([input_features])[0]  # Get the severity level
    return severity

# Prediction button and display result
for name, model in trained_models.items():
    if st.button(f"Predict Severity ({name})"):
        severity = predict_severity(model)
        severity_labels = {0: "No Risk", 1: "Mild", 2: "Moderate", 3: "Severe"}
        st.write(f"Prediction using {name}: *{severity_labels[severity]}*")

# Display feature importance for Random Forest
if st.button("Show Random Forest Feature Importance"):
    importance = trained_models["Random Forest"].feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot the graph
    fig, ax = plt.subplots()
    importance_df.plot(kind='barh', x='Feature', y='Importance', legend=False, ax=ax, color='purple')
    ax.set_title('Feature Importances (Random Forest)')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    st.pyplot(fig)
