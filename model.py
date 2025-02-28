import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Load dataset
df = pd.read_csv("final_earthquake_data.csv")
print(df.info())

# Separate features (X) and target (y)
X = df.drop(columns=['severe','tsunami'])
y = df['severe']

# Apply SMOTE only to the entire dataset first
smote = SMOTE(sampling_strategy={0: 1500, 1: 800}, random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Add Gaussian noise to reduce overfitting
X_smote += np.random.normal(0, 5, X_smote.shape)

# Select columns to scale (example: numerical columns)
columns_to_scale = ['significance']
scaler = StandardScaler()
X_smote[columns_to_scale] = scaler.fit_transform(X_smote[columns_to_scale])

# Save the scaler for future use
with open('earthquake_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

import numpy as np

shuffled_indices = np.random.permutation(len(X_smote))
X_smote = X_smote.iloc[shuffled_indices]
y_smote = y_smote.iloc[shuffled_indices]
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=40)

# Verify shapes
print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)


from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
models = {
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    print(f"Model: {name}")
    # Train
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    # Print results
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)
    print("-" * 50)
# Initialize the Gradient Boosting model
# GBM Model with Regularization
gbm = GradientBoostingClassifier(
    learning_rate=0.03,
    n_estimators=150,
    subsample=0.75,
    max_depth=3,
    min_samples_leaf=10,
    random_state=42
)

gbm.fit(X_train, y_train)

# Make predictions
y_pred = gbm.predict(X_test)

# Calculate and print accuracy measures
print("Final model gbm")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import pickle

# Save the trained model
with open("gradient_boosting_model.pkl", "wb") as file:
    pickle.dump(gbm, file)

print("Model saved as 'gradient_boosting_model.pkl'.")
