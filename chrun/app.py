import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import pickle
# Loading data
df = pd.read_excel("Telco_customer_churn.xlsx")

scaler = StandardScaler()

# cleaning
columns_to_drop = [
    "CustomerID",
    "Churn Score",
    "Churn Reason",
    "Lat Long",
    "Country", 
    "State", 
    "City", 
    "Zip Code", 
    "Latitude", 
    "Longitude",
    "CLTV"
]
df = df.drop(columns=columns_to_drop)

# Feature engineering
# Handling total charges (converting to numeric values -> handling NaN values)
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df["Total Charges"].fillna(df["Total Charges"].median(), inplace=True)

# X -> Features ; Y -> Target
X = df.drop(columns=["Churn Value", "Churn Label"])
y = df["Churn Value"]

# Train test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


X = pd.get_dummies(X, drop_first=True)
print(X.dtypes.unique())

# Scalling : Ellaboration on Note
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class 
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {
    0: class_weights[0],
    1: class_weights[1]
}

# Creating object of the Model Class (Standard OOP concept)
log_model = LogisticRegression(
    max_iter=1000,
    class_weight=class_weight_dict,
    solver="lbfgs"
)

# Fitting model : Training part basically
log_model.fit(X_train_scaled, y_train)

# Testing on split data
y_pred = log_model.predict(X_test_scaled) # 0/1 
y_proba = log_model.predict_proba(X_test_scaled)[:, 1] # Prob value(0-1)

# Testing 
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
model_package = {
    "model": log_model,
    "scaler": scaler,
    "columns": X.columns
}

# Saving Model
with open("logistic_churn_model.pkl", "wb") as f:
    pickle.dump(model_package, f)

print("Model saved successfully.")