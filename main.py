from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import joblib
import pandas as pd
import sqlite3
import bcrypt
import jwt
from datetime import datetime, timedelta
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ---------------------------------------------------
# Configuration et sécurité
# ---------------------------------------------------
security = HTTPBearer()
JWT_SECRET = "your-super-secret-jwt-key-2024"
JWT_ALGORITHM = "HS256"

# ---------------------------------------------------
# Modèles Pydantic
# ---------------------------------------------------
class RiderData(BaseModel):
    year: int
    time_hours: float
    country: str
    team: str

class UserCreate(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    role: str

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
    description="API for predicting Tour de France winner probability with authentication",
    version="1.0.0"
)

# ---------------------------------------------------
# Enable CORS
# ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Fonctions d'authentification et base de données
# ---------------------------------------------------
def init_db():
    """Initialise la base de données SQLite"""
    conn = sqlite3.connect('tdf_predictor.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'viewer'
        )
    ''')
    conn.commit()
    conn.close()

def create_test_users():
    """Crée les utilisateurs de test"""
    init_db()
    conn = sqlite3.connect('tdf_predictor.db')
    cursor = conn.cursor()
    
    # Supprimer les anciens utilisateurs
    cursor.execute('DELETE FROM users')
    
    test_users = [
        ('admin@aso.com', 'admin', 'password'),
        ('team@ineos.com', 'team_manager', 'password'),
        ('sponsor@carrefour.com', 'partner', 'password'),
        ('fan@example.com', 'viewer', 'password'),
    ]
    
    for email, role, password in test_users:
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute(
            'INSERT INTO users (email, password_hash, role) VALUES (?, ?, ?)',
            (email, password_hash, role)
        )
    
    conn.commit()
    conn.close()
    print("✅ Utilisateurs de test créés")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Vérifie le token JWT"""
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expiré")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token invalide")

def require_role(required_role: str):
    """Décorateur pour vérifier les rôles"""
    def role_checker(payload: dict = Depends(verify_token)):
        if payload.get('role') != required_role and payload.get('role') != 'admin':
            raise HTTPException(status_code=403, detail="Permissions insuffisantes")
        return payload
    return role_checker

# ---------------------------------------------------
# Preprocessing function
# ---------------------------------------------------
def preprocess_inputs(year: int, time_hours: float, country: str, team: str):
    """
    Transform raw inputs into the format expected by the model.
    Returns a numpy array with all features in the correct order.
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
    features = np.concatenate([[year_scaled, time_scaled, team_encoded], country_encoded])
    
    return features.reshape(1, -1)

# ---------------------------------------------------
# Routes d'authentification
# ---------------------------------------------------
@app.post("/api/auth/register")
async def register(user: UserCreate):
    """Inscription d'un nouvel utilisateur"""
    init_db()
    conn = sqlite3.connect('tdf_predictor.db')
    cursor = conn.cursor()
    
    # Vérifier si l'utilisateur existe déjà
    cursor.execute('SELECT * FROM users WHERE email = ?', (user.email,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Un utilisateur avec cet email existe déjà")
    
    # Créer le nouvel utilisateur
    password_hash = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    cursor.execute(
        'INSERT INTO users (email, password_hash, role) VALUES (?, ?, ?)',
        (user.email, password_hash, 'viewer')
    )
    conn.commit()
    
    # Récupérer l'utilisateur créé
    cursor.execute('SELECT id, email, role FROM users WHERE email = ?', (user.email,))
    new_user = cursor.fetchone()
    conn.close()
    
    return {
        "msg": "Utilisateur créé avec succès",
        "email": new_user[1],
        "role": new_user[2]
    }

@app.post("/api/auth/login")
async def login(user: UserLogin):
    """Connexion d'un utilisateur"""
    init_db()
    conn = sqlite3.connect('tdf_predictor.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM users WHERE email = ?', (user.email,))
    db_user = cursor.fetchone()
    conn.close()
    
    if not db_user:
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")
    
    user_id, email, password_hash, role = db_user
    
    if bcrypt.checkpw(user.password.encode('utf-8'), password_hash):
        # Créer un token JWT
        token_data = {
            "sub": email,
            "role": role,
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        token = jwt.encode(token_data, JWT_SECRET, algorithm=JWT_ALGORITHM)
        
        return {
            "access_token": token,
            "role": role,
            "email": email
        }
    else:
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")

@app.get("/api/debug/users")
async def debug_users():
    """Route de débogage pour voir tous les utilisateurs"""
    init_db()
    conn = sqlite3.connect('tdf_predictor.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, email, role FROM users')
    users = cursor.fetchall()
    conn.close()
    
    user_list = []
    for user in users:
        user_list.append({
            "id": user[0],
            "email": user[1],
            "role": user[2]
        })
    
    return {
        "total_users": len(user_list),
        "users": user_list
    }

@app.get("/api/create-test-users")
async def create_test_users_endpoint():
    """Crée les utilisateurs de test"""
    create_test_users()
    return {"message": "Utilisateurs de test créés avec succès"}

@app.get("/api/test")
async def test():
    """Route de test"""
    return {
        "message": "✅ FastAPI Backend TdF Predictor fonctionne!",
        "status": "success",
        "version": "1.0"
    }

# ---------------------------------------------------
# Routes protégées pour les prédictions
# ---------------------------------------------------
@app.post("/predict")
async def predict(data: RiderData, user: dict = Depends(verify_token)):
    """
    Prédiction protégée - nécessite une authentification
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
            "user_role": user.get('role'),
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
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/teams")
async def get_teams(user: dict = Depends(verify_token)):
    """Liste des équipes - nécessite une authentification"""
    if team_means:
        return {
            "teams": sorted(list(team_means.keys())),
            "count": len(team_means),
            "user_role": user.get('role')
        }
    raise HTTPException(status_code=500, detail="Team data not loaded")

@app.get("/countries")
async def get_countries(user: dict = Depends(verify_token)):
    """Liste des pays - nécessite une authentification"""
    if country_encoder:
        try:
            countries = [str(c) for c in country_encoder.categories_[0] if pd.notna(c)]
            return {
                "countries": sorted(countries),
                "count": len(countries),
                "user_role": user.get('role')
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get countries: {str(e)}")
    raise HTTPException(status_code=500, detail="Country data not loaded")

# ---------------------------------------------------
# Routes admin protégées
# ---------------------------------------------------
@app.get("/admin/stats")
async def admin_stats(user: dict = Depends(require_role('admin'))):
    """Statistiques admin - réservé aux administrateurs"""
    return {
        "message": "Bienvenue dans l'interface admin",
        "stats": {
            "total_teams": len(team_means) if team_means else 0,
            "total_countries": len(country_encoder.categories_[0]) if country_encoder else 0,
            "model_loaded": model is not None
        },
        "user": user
    }

# ---------------------------------------------------
# Health check endpoint
# ---------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Tour de France Winner Prediction API with Auth is running",
        "status": "ready" if all([scaler_year, scaler_time, team_means, country_encoder]) else "missing preprocessors",
        "endpoints": {
            "auth": {
                "login": "POST /api/auth/login",
                "register": "POST /api/auth/register",
                "test_users": "GET /api/create-test-users"
            },
            "protected": {
                "predict": "POST /predict",
                "teams": "GET /teams", 
                "countries": "GET /countries"
            },
            "admin": {
                "stats": "GET /admin/stats"
            }
        }
    }

# ---------------------------------------------------
# Initialisation au démarrage
# ---------------------------------------------------
@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("🚀 FASTAPI TdF PREDICTOR - DÉMARRAGE")
    print("=" * 60)
    print("📡 URL: http://localhost:8000")
    print("🔗 Documentation: http://localhost:8000/docs")
    print("👥 Créer users: GET /api/create-test-users")
    print("🔐 Login: POST /api/auth/login")
    print("📝 Register: POST /api/auth/register")
    print("🤖 Predict: POST /predict (protégé)")
    print("=" * 60)
    
    # Créer les utilisateurs de test au démarrage
    create_test_users()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)