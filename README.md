# Diabetes Risk Prediction using Neural Network

A Machine Learning project that predicts whether a patient is at risk of diabetes based on key health indicators, using a Multilayer Perceptron (MLP) Neural Network built and deployed in Google Colab.

---

## Problem Statement

Diabetes is one of the most prevalent chronic diseases worldwide, causing serious health complications if left undetected. Early diagnosis is critical for effective treatment and management.

Manual diagnosis requires extensive medical tests and expert interpretation, which may not always be accessible. This project builds a neural network classification model to predict diabetes risk based on factors like glucose level, BMI, age, insulin, blood pressure, and family history.

---

## Dataset

| Column | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Diabetes likelihood based on family history |
| Age | Age in years |
| Outcome | Target variable — 0 = Not at risk, 1 = At risk |

### Dataset Information

- Source: Pima Indians Diabetes Database (Kaggle)
- No missing values
- Target column: Outcome

---

## Project Workflow

1. Import required libraries
2. Upload and load dataset
3. Check for missing values
4. Apply Label Encoding on categorical columns
5. Apply Standard Scaling on feature values
6. Split dataset into training and testing data
7. Train MLP Neural Network model
8. Evaluate model performance
9. Accept live user input
10. Predict and display diabetes risk with probability scores

---

## Data Preprocessing

Categorical text values were converted into numerical values using Label Encoding because Machine Learning models require numerical input.

Feature values were normalized using Standard Scaler to ensure all features contribute equally during neural network training. This transforms each feature to have a mean of 0 and a standard deviation of 1.

---

## Train Test Split

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

Training data is used to teach the model, while testing data is used to evaluate prediction performance on unseen data.

---

## Algorithm Used

### Multilayer Perceptron (MLP) Neural Network

MLP is a feedforward artificial neural network that learns complex patterns through backpropagation and gradient descent.

The model architecture is:

- Input Layer — 8 features
- Hidden Layer 1 — 16 Neurons
- Hidden Layer 2 — 8 Neurons
- Output Layer — Binary Classification (At risk / Not at risk)

The model learns the relationship between:

- Glucose level
- Blood Pressure
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

and predicts whether a patient is at risk of diabetes.

| Parameter | Value |
|---|---|
| Hidden Layers | 2 |
| Neurons | 16, 8 |
| Activation | ReLU |
| Optimizer | Adam |
| Max Iterations | 500 |
| Random State | 42 |

---

## Libraries Used

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- google.colab

---

## Evaluation Metrics

### Accuracy Score

Accuracy measures the percentage of correctly classified patients out of the total test samples. A higher accuracy indicates better overall model performance.

### Confusion Matrix

The confusion matrix shows the count of:

- True Positives — At-risk patients correctly identified
- True Negatives — Healthy patients correctly identified
- False Positives — Healthy patients incorrectly flagged as at risk
- False Negatives — At-risk patients missed by the model

A heatmap of the confusion matrix is displayed after evaluation.

---

## Results

The model successfully predicts diabetes risk based on patient health data.
Actual vs Predicted classification is visualized through the confusion matrix heatmap.
Accuracy score is displayed after evaluation.
Probability scores are provided for each prediction.

---

## Example Prediction

**Input**

- Glucose: 148
- BloodPressure: 72
- Insulin: 0
- BMI: 33.6
- DiabetesPedigreeFunction: 0.627
- Age: 50

**Output**

```
Prediction for the input data: at risk
Probability of not at risk (Class 0): 0.2134
Probability of at risk (Class 1): 0.7866
```

---

## How to Run the Project

1. Open Google Colab
2. Create a new notebook
3. Paste the project code
4. Run the notebook
5. Upload the dataset CSV file when prompted
6. Enter patient values when prompted for live prediction
7. View the prediction result and confusion matrix

---

## Future Improvements

- Use larger and more diverse real-world datasets
- Build a web-based interface using Flask or Streamlit
- Add advanced algorithms such as Random Forest and XGBoost for comparison
- Apply cross-validation for more robust model evaluation
- Include additional health features like HbA1c and cholesterol levels
- Tune hyperparameters using GridSearchCV

---

## Team Details

Department: Computer Science and Engineering

| Name | USN |
|---|---|
| Shrikant | 4MC25CS286 |
| Samprit | 4MC25CS256 |
| Shankaramurti | 4MC25CS272 |
| Sachin | 4MC25CS252 |

Subject: Introduction to Artificial Intelligence and Machine Learning

---

## Conclusion

This project successfully demonstrates the use of a Multilayer Perceptron Neural Network for predicting diabetes risk in patients. The system estimates risk based on multiple health indicators and provides probability scores for each prediction.

The model helps in early identification of diabetic patients, which can support timely medical intervention and reduce complications. With further improvements such as a larger dataset and web deployment, this system can be developed into a practical healthcare decision support tool.

