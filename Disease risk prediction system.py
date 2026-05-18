# Collect user input for each feature in the new_features list
print("Please enter the following details for the patient (using the updated features):")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import io

uploaded = files.upload()
df = pd.read_csv(io.BytesIO(list(uploaded.values())[0]))

print('Dataset shape:', df.shape)
print(df.head())

print('Missing Values:')
print(df.isnull().sum())

label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Corrected target column name from 'disease_risk' to 'Outcome'
target_col = 'Outcome'
X = df.drop(target_col, axis=1)
y = df[target_col]

# Save feature names for consistent input later
feature_names = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy:', round(accuracy_score(y_test, y_pred), 4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

user_input_new = {
    'Glucose': float(input('Glucose: ')),
    'BloodPressure': float(input('BloodPressure: ')),
    'Insulin': float(input('Insulin: ')),
    'BMI': float(input('BMI: ')),
    'DiabetesPedigreeFunction': float(input('DiabetesPedigreeFunction: ')),
    'Age': float(input('Age: '))
}

# Create a full feature set for prediction, filling missing columns with 0
full_user_input = {col: user_input_new.get(col, 0.0) for col in feature_names}
user_df_new = pd.DataFrame([full_user_input])[feature_names]

# Scale the user input using the fitted scaler
user_scaled_new = scaler.transform(user_df_new)

# Make prediction using the model
manual_prediction_new = model.predict(user_scaled_new)
manual_prediction_proba_new = model.predict_proba(user_scaled_new)

print(f"\nPrediction for the input data: {'at risk' if manual_prediction_new[0] == 1 else 'not at risk'}")
print(f"Probability of not at risk (Class 0): {manual_prediction_proba_new[0][0]:.4f}")
print(f"Probability of at risk (Class 1): {manual_prediction_proba_new[0][1]:.4f}")
