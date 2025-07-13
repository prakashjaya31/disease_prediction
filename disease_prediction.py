import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier

# STEP 1: Load Dataset
data = pd.read_csv('improved_disease_dataset.csv')

# STEP 2: Encode the Target (disease column)
encoder = LabelEncoder()
data["disease"] = encoder.fit_transform(data["disease"])

# STEP 3: Separate Features and Target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# STEP 4: Resample (handle class imbalance)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# STEP 5: Handle Categorical Columns (e.g. gender)
if 'gender' in X_resampled.columns:
    le = LabelEncoder()
    X_resampled['gender'] = le.fit_transform(X_resampled['gender'])

# STEP 6: Handle missing values (fill with 0)
X_resampled = X_resampled.fillna(0)

# STEP 7: Flatten the target array if needed
if len(y_resampled.shape) > 1:
    y_resampled = y_resampled.values.ravel()

# STEP 8: Cross-Validation with multiple models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    scores = cross_val_score(
        model, X_resampled, y_resampled, 
        cv=stratified_kfold, scoring='accuracy'
    )
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f}")

# STEP 9: Train SVM, NB, RF Models
svm_model = SVC()
svm_model.fit(X_resampled, y_resampled)
svm_preds = svm_model.predict(X_resampled)

nb_model = GaussianNB()
nb_model.fit(X_resampled, y_resampled)
nb_preds = nb_model.predict(X_resampled)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)
rf_preds = rf_model.predict(X_resampled)

# STEP 10: Combine Model Predictions (Voting)
final_preds = [mode([i, j, k]).mode[0] for i, j, k in zip(svm_preds, nb_preds, rf_preds)]

# STEP 11: Accuracy Scores
print("SVM Accuracy:", accuracy_score(y_resampled, svm_preds))
print("Naive Bayes Accuracy:", accuracy_score(y_resampled, nb_preds))
print("Random Forest Accuracy:", accuracy_score(y_resampled, rf_preds))
print("Combined Model Accuracy:", accuracy_score(y_resampled, final_preds))

# STEP 12: Function for New Predictions
symptoms = X.columns.values
symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms)}

def predict_disease(input_symptoms):
    input_symptoms = input_symptoms.split(",")
    input_data = [0] * len(symptom_index)
    
    for symptom in input_symptoms:
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1

    input_data = np.array(input_data).reshape(1, -1)

    rf_pred = encoder.classes_[rf_model.predict(input_data)[0]]
    nb_pred = encoder.classes_[nb_model.predict(input_data)[0]]
    svm_pred = encoder.classes_[svm_model.predict(input_data)[0]]
    final_pred = mode([rf_pred, nb_pred, svm_pred]).mode[0]

    return {
        "Random Forest Prediction": rf_pred,
        "Naive Bayes Prediction": nb_pred,
        "SVM Prediction": svm_pred,
        "Final Prediction": final_pred
    }

# TEST the Function
print(predict_disease("Itching,Skin Rash,Nodal Skin Eruptions"))
