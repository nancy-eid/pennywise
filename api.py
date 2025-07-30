
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.joblib")

# Create FastAPI app
app = FastAPI()

# CORS setup (for Lovable)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace * with Lovable's domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class InputData(BaseModel):
    Income: float
    FamilySize: int
    Location: str  # 'Urban' or 'Rural'
    Rent: float
    Food: float
    Health: float
    Transport: float
    Education: float
    Other: float
    TotalSpent: float

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        # One-hot encode 'Location'
        df = pd.get_dummies(df)
        if 'Location_Urban' not in df.columns:
            df['Location_Urban'] = 0
        if 'Location_Rural' not in df.columns:
            df['Location_Rural'] = 0

        # Ensure column order
        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

        # Predict
        prediction = model.predict(df)[0]
        return {"NextMonthTotalSpent": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
