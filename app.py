from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta, timezone
import jwt
from passlib.context import CryptContext
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title='Classifier Test API')

# JWT Configuration
ACCESS_TOKEN_KEY = os.getenv('ACCESS_TOKEN_KEY')
REFRESH_TOKEN_KEY = os.getenv('REFRESH_TOKEN_KEY')
ACCESS_TOKEN_AGE = int(os.getenv('ACCESS_TOKEN_AGE', 1800))
ALGORITHM = "HS256"

# User credentials from .env
API_USERNAME = os.getenv('API_USERNAME')
API_PASSWORD = os.getenv('API_PASSWORD')

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Allow local origins used by the frontend during development
origins = [
    "http://localhost",
    "http://localhost:5173",  # Vite default port
    "http://localhost:5500",
    "http://127.0.0.1",
    "http://127.0.0.1:5173",  # Vite default port
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'classifier.h5')

# Default expected feature order (will be overridden if the loaded model exposes feature names)
EXPECTED_FEATURES = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear', 'Type']

# Mapping for Type ordinal encoding
TYPE_MAP = {'L': 0, 'M': 1, 'H': 2}

LABEL_MAP = {
    0: 'No Failure',
    1: 'Heat Dissipation Failure',
    2: 'Power Failure',
    3: 'Overstrain Failure',
    4: 'Tool Wear Failure',
    5: 'Random Failures'
}

class InputSample(BaseModel):
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: float
    Torque: float
    Tool_wear: float
    Type: str

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

# Authentication functions
def create_access_token(data: dict) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(seconds=ACCESS_TOKEN_AGE)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, ACCESS_TOKEN_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify JWT token from Authorization header"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, ACCESS_TOKEN_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user with username and password"""
    if username != API_USERNAME:
        return False
    # For production, you should hash the password in .env
    # For now, we'll do simple comparison
    return password == API_PASSWORD

# Load model at startup
try:
    model = joblib.load(MODEL_PATH)
    # Attempt to read feature names from the model (XGBoost stores them on the booster)
    try:
        booster = model.get_booster()
        model_feature_names = booster.feature_names
    except Exception:
        model_feature_names = None
    if model_feature_names:
        # Respect the model's expected feature order
        EXPECTED_FEATURES = list(model_feature_names)
        print('Model expected feature order (from booster):', EXPECTED_FEATURES)
except Exception:
    model = None


@app.get('/')
def root():
    return {'status': 'ok', 'model_loaded': model is not None}


@app.post('/login', response_model=TokenResponse)
def login(credentials: LoginRequest):
    """Login endpoint to get JWT token"""
    if not authenticate_user(credentials.username, credentials.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": credentials.username}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_AGE
    }


@app.post('/predict')
def predict(sample: InputSample, token: dict = Depends(verify_token)):
    if model is None:
        raise HTTPException(status_code=500, detail='Model not loaded on server')

    # convert to DataFrame with expected names
    df = pd.DataFrame([{
        'Air temperature': sample.Air_temperature,
        'Process temperature': sample.Process_temperature,
        'Rotational speed': sample.Rotational_speed,
        'Torque': sample.Torque,
        'Tool wear': sample.Tool_wear,
        'Type': sample.Type
    }])

    # map Type
    if df['Type'].dtype == object:
        df['Type'] = df['Type'].map(TYPE_MAP)

    # ensure ordering and numeric types
    missing = set(EXPECTED_FEATURES) - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f'Missing features: {missing}')

    # Reorder columns to exactly match model's feature names to avoid XGBoost feature_name mismatch
    df = df[EXPECTED_FEATURES]

    # If the model expects a different internal order (e.g., 'Type' first), the EXPECTED_FEATURES will reflect that
    df = df.astype(float)

    pred = model.predict(df)
    pred_label = LABEL_MAP.get(int(pred[0]), str(pred[0]))

    result = {'prediction_numeric': int(pred[0]), 'prediction_label': pred_label}

    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(df)[0].tolist()
            result['probabilities'] = probs
        except Exception:
            result['probabilities'] = None

    return result
