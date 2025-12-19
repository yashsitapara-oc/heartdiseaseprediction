import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib 
import os
import mlflow

# Load dataset (Ensure the path is correct for your DKube mount)
df_clean = pd.read_csv("/dataset/cleaned_heart_dataset.csv")

# Prepare features + target
df_clean['target'] = (df_clean['num'] > 0).astype(int)
X = df_clean.drop(['num', 'target'], axis=1)
y = df_clean['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate and Log
accuracy = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Testing accuracy: {accuracy}")
mlflow.log_metric("accuracy", accuracy)

# Save artifacts using joblib
joblib.dump(rf_model, os.path.join(MODEL_DIR, "/model/model.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "/model/scaler.joblib"))

print("Model + scaler saved successfully.")