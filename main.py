from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import pandas as pd
import subprocess


# ---------------------------------------------------
# Load the trained model and preprocessors
# ---------------------------------------------------
model = joblib.load("Winner_prediction_model.pkl")

try:
    scaler_year = joblib.load("scaler.pkl")  # StandardScaler for Year
    scaler_time = joblib.load("scaler2.pkl")  # StandardScaler for Time
    country_encoder = joblib.load("country_encoder.pkl")  # OneHotEncoder for Country
    
    # Load training data to compute team means
    df = pd.read_csv("training_data2.csv")
    team_means = df.groupby('Team')['win_flag'].mean().to_dict()
    
    # Get the default value for unknown teams (overall mean)
    default_team_value = sum(team_means.values()) / len(team_means)
    
    print("✓ All preprocessors loaded successfully")
    print(f"  - Number of teams in training: {len(team_means)}")
    print(f"  - Number of countries in training: {len(country_encoder.categories_[0])}")
    
except FileNotFoundError as e:
    print(f"Warning: Preprocessor file not found: {e}")
    print("Please make sure all required files are in the same directory")
    scaler_year = None
    scaler_time = None
    team_means = None
    country_encoder = None
    default_team_value = 0.0

app = FastAPI(
    title="Tour de France Winner Prediction API",
    description="API for predicting Tour de France winner probability using a trained ML model.",
    version="1.0.0"
)

# ---------------------------------------------------
# Enable CORS
# ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Input schema - accepting user-friendly inputs
# ---------------------------------------------------
class RiderData(BaseModel):
    year: int
    time_hours: float
    country: str
    team: str


# ---------------------------------------------------
# Preprocessing function
# ---------------------------------------------------
def preprocess_inputs(year: int, time_hours: float, country: str, team: str):
    """
    Transform raw inputs into the format expected by the model.
    Returns a numpy array with all features in the correct order.
    
    Feature order: [Year_scaled, Time_scaled, Team_encoded, Country_one_hot...]
    """
    
    if not all([scaler_year, scaler_time, team_means, country_encoder]):
        raise ValueError("Preprocessors not loaded. Please run save_preprocessors.py first.")
    
    # Step 1: Scale Year
    year_scaled = scaler_year.transform([[year]])[0][0]
    
    # Step 2: Scale Time
    time_scaled = scaler_time.transform([[time_hours]])[0][0]
    
    # Step 3: Encode Team (use mean win rate, or default if team not in training data)
    team_encoded = team_means.get(team, default_team_value)
    
    # Step 4: One-Hot Encode Country
    country_encoded = country_encoder.transform([[country]])[0]
    
    # Step 5: Combine all features in the correct order
    # Order: [Year, Time, Team_encoded, Country_features...]
    features = np.concatenate([[year_scaled, time_scaled, team_encoded], country_encoded])
    
    return features.reshape(1, -1)




@app.post("/activate_model")
def activate_model():
    try:
        # Simply test model prediction to confirm readiness
        _ = model.predict(np.zeros((1, model.n_features_in_)))  
        return {"message": "Model activated successfully (server is running)."}
    except Exception as e:
        return {"error": f"Activation failed: {e}"}


# ---------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------
@app.post("/predict")
def predict(data: RiderData):
    """
    Takes rider information and returns the model prediction.
    
    Expected input format:
    {
        "year": 2022,
        "time_hours": 94.5,
        "country": "USA",
        "team": "UAE Team Emirates"
    }
    """
    try:
        # Preprocess the inputs
        X = preprocess_inputs(
            year=data.year,
            time_hours=data.time_hours,
            country=data.country,
            team=data.team
        )

        # Predict
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        
        # Get probability for the positive class (winner = 1)
        winner_probability = prediction_proba[1]

        # Build response
        confidence = "High" if winner_probability > 0.7 else "Medium" if winner_probability > 0.4 else "Low"

        return {
            "winner_prediction": int(prediction),
            "probability": f"{winner_probability * 100:.2f}%",
            "confidence": confidence,
            "message": "Prediction successful",
            "details": {
                "year": data.year,
                "time_hours": data.time_hours,
                "country": data.country,
                "team": data.team,
                "team_encoded_value": team_means.get(data.team, default_team_value) if team_means else None
            }
        }

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


# ---------------------------------------------------
# Health check endpoint
# ---------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Tour de France Winner Prediction API is running",
        "status": "ready" if all([scaler_year, scaler_time, team_means, country_encoder]) else "missing preprocessors",
        "preprocessors_loaded": {
            "scaler_year": scaler_year is not None,
            "scaler_time": scaler_time is not None,
            "team_means": team_means is not None,
            "country_encoder": country_encoder is not None
        }
    }


# ---------------------------------------------------
# Get available teams endpoint (optional, helpful for frontend)
# ---------------------------------------------------
@app.get("/teams")
def get_teams():
    """Returns list of teams from training data"""
    if team_means:
        return {
            "teams": sorted(list(team_means.keys())),
            "count": len(team_means)
        }
    return {"error": "Team data not loaded"}


# ---------------------------------------------------
# Get available countries endpoint (optional, helpful for frontend)
# ---------------------------------------------------
@app.get("/countries")
def get_countries():
    """Returns list of countries from training data"""
    if country_encoder:
        try:
            # Convert all to strings and filter out any NaN/None values
            countries = [str(c) for c in country_encoder.categories_[0] if pd.notna(c)]
            return {
                "countries": sorted(countries),
                "count": len(countries)
            }
        except Exception as e:
            return {"error": f"Failed to get countries: {str(e)}"}
    return {"error": "Country data not loaded"}