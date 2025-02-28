import numpy as np
import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# Load the dataset
data = pd.read_csv("final_tsunami_data.csv")

tsunami_counts = data['tsunami'].value_counts()
print(tsunami_counts)
# Features and target
X = data.drop(columns=['tsunami'])
y = data['tsunami']
columns_to_scale = ['significance','depth']
scaler = StandardScaler()
X = scaler.fit_transform(X[columns_to_scale])
# Apply SMOTE only to the entire dataset first
smote = SMOTE(sampling_strategy={0: 1000, 1: 900}, random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Select columns to scale (example: numerical columns)



# Save the scaler for future use
with open('tsunami_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

import numpy as np

shuffled_indices = np.random.permutation(len(X_smote))
X_smote = X_smote[shuffled_indices]
y_smote = y_smote.iloc[shuffled_indices]
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=40)

# Verify shapes
print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Define models with optimized hyperparameters
models = {
    "SVM": SVC(kernel="rbf", C=1, gamma="scale", random_state=42),
    "GBM": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, metric="minkowski"),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_state=42),
    "lgbmmodel" : LGBMClassifier(
    num_leaves=31, 
    max_depth=10, 
    learning_rate=0.05, 
    n_estimators=200,
    min_child_samples=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.5,  # Adjust based on class imbalance
    random_state=42
)
}

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Train and evaluate each model
for name, model in models.items():
    print(f"Evaluating {name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use 'macro' if class imbalance is severe
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("-" * 50)


"""with open("svm_model.pkl", "wb") as file:
    pickle.dump(svm_model, file)



print("SVM model and scaler saved as 'svm_model.pkl' and 'scaler.pkl'.")
# Predict on the test set
y_pred = svm_model.predict(X_test)

# Metrics
print("Final model SVC")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))"""
