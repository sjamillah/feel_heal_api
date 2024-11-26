from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Dict, Union
import uvicorn
import pickle
import numpy as np
import pandas as pd

app = FastAPI(
    title="Mental Health Prediction API",
    description="API for predicting anxiety and depression levels based on health metrics",
    version="1.0.0"
)

class InputData(BaseModel):
    Age: int = Field(..., ge=18, le=100, description="Age of the person (18-100)")
    Sleep_Quality: float = Field(..., ge=1, le=10, description="Sleep quality rating (1-10)")
    Daily_Steps: int = Field(..., ge=0, le=11000, description="Number of daily steps (0-11000)")
    Calories_Burned: float = Field(..., ge=0, le=2900, description="Daily calories burned (0-2900)")
    Heart_Rate: int = Field(..., ge=60, le=100, description="Heart rate in bpm (60-100)")
    Sleep_Duration: float = Field(..., ge=0, le=24, description="Sleep duration in hours (0-24)")
    Physical_Activity_Level: Literal['low', 'medium', 'high'] = Field(..., description="Physical activity level (low/medium/high)")
    Medication_Usage: Literal['yes', 'no'] = Field(..., description="Medication usage status (yes/no)")
    Social_Interaction: int = Field(..., ge=0, le=5, description="Social interaction level (0-5)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Age": 25,
                "Sleep_Quality": 7.5,
                "Daily_Steps": 8000,
                "Calories_Burned": 2500,
                "Heart_Rate": 75,
                "Sleep_Duration": 7.5,
                "Physical_Activity_Level": "medium",
                "Medication_Usage": "no",
                "Social_Interaction": 3
            }
        }
    )

class PredictionDetails(BaseModel):
    anxiety_risk: Literal['low', 'medium', 'high']
    depression_risk: Literal['low', 'medium', 'high']
    confidence_score: float = Field(..., description="Model's confidence in prediction (0-1)")

class PredictionResponse(BaseModel):
    anxiety_level: float = Field(..., description="Predicted anxiety level (0-1)")
    depression_level: float = Field(..., description="Predicted depression level (0-1)")
    prediction_details: PredictionDetails

def normalize_feature(value, feature_name, min_max_values):
    """Normalize a single feature value"""
    min_val, max_val = min_max_values[feature_name]
    return (value - min_val) / (max_val - min_val)

# Load the saved model
try:
    with open('mental_health_model2.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    
    model = loaded_model['model']
    label_encoders = loaded_model['label_encoders']
    features = loaded_model['features']
    
    print("Model loaded successfully")
except FileNotFoundError:
    raise Exception("Model file not found. Please ensure 'mental_health_model2.pkl' exists.")
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

# Define min-max values for normalization
min_max_values = {
    'Age': (18, 100),
    'Sleep Quality': (1, 10),
    'Daily Steps': (0, 11000),
    'Calories Burned': (0, 2900),
    'Heart Rate': (60, 100),
    'Sleep Duration': (0, 24),
    'Social Interaction': (0, 5)
}

@app.post("/predict", 
    response_model=PredictionResponse,
    summary="Predict Mental Health Metrics",
    description="Predicts anxiety and depression levels based on input health metrics"
)
async def predict_mental_health(data: InputData):
    try:
        # Create a DataFrame from input data
        input_dict = data.model_dump()
        
        # Create mapping for column names
        column_mapping = {
            'Sleep_Quality': 'Sleep Quality',
            'Daily_Steps': 'Daily Steps',
            'Calories_Burned': 'Calories Burned',
            'Heart_Rate': 'Heart Rate',
            'Sleep_Duration': 'Sleep Duration',
            'Physical_Activity_Level': 'Physical Activity Level',
            'Medication_Usage': 'Medication Usage',
            'Social_Interaction': 'Social Interaction'
        }
        
        # Rename and normalize features
        input_dict_renamed = {}
        for key, value in input_dict.items():
            new_key = column_mapping.get(key, key)
            if new_key in min_max_values:
                input_dict_renamed[new_key] = normalize_feature(value, new_key, min_max_values)
            else:
                input_dict_renamed[new_key] = value
            
        input_data = pd.DataFrame([input_dict_renamed])
        
        # Process categorical variables
        input_data['Physical Activity Level'] = label_encoders['Physical Activity Level'].transform(
            [data.Physical_Activity_Level.lower()]
        )[0]
        input_data['Medication Usage'] = label_encoders['Medication Usage'].transform(
            [data.Medication_Usage.lower()]
        )[0]
        
        # Ensure features are in correct order
        prediction_input = input_data[features]
        
        # Make prediction
        prediction = model.predict(prediction_input)
        
        # Calculate confidence score (using prediction probabilities if available)
        confidence_score = 0.8  # Default confidence
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(prediction_input)
            confidence_score = float(np.mean(np.max(probabilities, axis=1)))
        
        return PredictionResponse(
            anxiety_level=float(prediction[0][0]),
            depression_level=float(prediction[0][1]),
            prediction_details=PredictionDetails(
                anxiety_risk="high" if prediction[0][0] > 0.7 else 
                           "medium" if prediction[0][0] > 0.3 else "low",
                depression_risk="high" if prediction[0][1] > 0.7 else 
                               "medium" if prediction[0][1] > 0.3 else "low",
                confidence_score=confidence_score
            )
        )
        
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feature name: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid value in input data: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Check if the service is healthy and model is loaded"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_available": features is not None,
        "encoders_loaded": label_encoders is not None
    }

if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )