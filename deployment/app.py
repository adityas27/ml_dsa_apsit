import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# -----------------------
# Load Model Bundle
# -----------------------

with open("chrun/logistic_churn_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]
training_columns = bundle["columns"]


# -----------------------
# Input Schema (RAW features)
# Must match X BEFORE get_dummies()
# -----------------------

class ChurnInput(BaseModel):
    Gender: str
    Senior_Citizen: str
    Partner: str
    Dependents: str
    Tenure_Months: int
    Phone_Service: str
    Multiple_Lines: str
    Internet_Service: str
    Online_Security: str
    Online_Backup: str
    Device_Protection: str
    Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str
    Monthly_Charges: float
    Total_Charges: float


# -----------------------
# Prediction Endpoint
# -----------------------

@app.post("/predict")
def predict(data: ChurnInput):
    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])

    # Match training column naming
    df.columns = [col.replace("_", " ") for col in df.columns]

    # Apply same encoding
    df = pd.get_dummies(df, drop_first=True)

    # Align columns with training data
    df = df.reindex(columns=training_columns, fill_value=0)

    # Scale
    X_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
