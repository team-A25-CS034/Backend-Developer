from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta, timezone
import jwt
from passlib.context import CryptContext
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import Optional, List, Dict

# Import our custom modules
from mongodb_client import MongoDBClient
from forecasting import generate_forecast, prepare_forecast_data
from classification import initialize_classifier, get_classifier

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv('MONGODB_URI')
# Default database and collection updated to match the new deployment
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'machine_monitoring_db')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'machine_monitoring')

# Global MongoDB client
mongodb_client: Optional[MongoDBClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage MongoDB connection and model loading lifecycle"""
    global mongodb_client
    
    # Startup
    # Initialize MongoDB
    if MONGODB_URI:
        # Pass database and collection name into the client so it can
        # connect to the correct DB / collection layout.
        mongodb_client = MongoDBClient(MONGODB_URI, MONGODB_DATABASE, MONGODB_COLLECTION)
        print(f"✅ Connected to MongoDB: {MONGODB_DATABASE} (collection={MONGODB_COLLECTION})")
    else:
        print("⚠️ MONGODB_URI not set - running without database")
    
    # Initialize Classifier
    initialize_classifier(MODEL_PATH)
    
    yield
    
    # Shutdown
    if mongodb_client:
        await mongodb_client.close()
        print("🔌 MongoDB connection closed")


app = FastAPI(title='Classifier Test API', lifespan=lifespan)

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
    "http://127.0.0.1",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'classifier.h5')

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

class ForecastRequest(BaseModel):
    machine_id: str
    # Now using minute-based forecasting (default 300 minutes)
    forecast_minutes: int = 300

class ForecastResponse(BaseModel):
    machine_id: str
    forecast_minutes: int
    forecast_data: List[Dict]
    created_at: str

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


@app.get('/')
def root():
    classifier = get_classifier()
    return {
        'status': 'ok',
        'model_loaded': classifier is not None and classifier.is_loaded()
    }


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
    """
    Predict machine failure status from sensor data
    
    Args:
        sample: InputSample with sensor readings
        token: JWT token (authenticated user)
    
    Returns:
        Prediction result with label and probabilities
    """
    classifier = get_classifier()
    
    if not classifier or not classifier.is_loaded():
        raise HTTPException(status_code=500, detail='Model not loaded on server')
    
    try:
        result = classifier.predict(
            air_temperature=sample.Air_temperature,
            process_temperature=sample.Process_temperature,
            rotational_speed=sample.Rotational_speed,
            torque=sample.Torque,
            tool_wear=sample.Tool_wear,
            machine_type=sample.Type
        )
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction error: {str(e)}')


@app.post('/forecast', response_model=ForecastResponse)
async def forecast(request: ForecastRequest, token: dict = Depends(verify_token)):
    """
    Generate forecast for machine sensor data for N days ahead
    
    Args:
        request: ForecastRequest with machine_id and forecast_days
        token: JWT token (authenticated user)
    
    Returns:
        ForecastResponse with forecast data
    """
    if not mongodb_client:
        raise HTTPException(
            status_code=500,
            detail="Database not connected. Please configure MONGODB_URI in .env"
        )
    
    machine_id = request.machine_id
    forecast_minutes = request.forecast_minutes

    # Validate forecast_minutes (limit to 1 day of minutes by default)
    if forecast_minutes < 1 or forecast_minutes > 1440:
        raise HTTPException(
            status_code=400,
            detail="forecast_minutes must be between 1 and 1440"
        )
    
    try:
        # Get historical data (last 30 days minimum for good predictions)
        historical_readings = await mongodb_client.get_readings_range(
            machine_id=machine_id,
            days=30
        )
        
        if not historical_readings:
            # Fallback: if there are no readings in the last N days (e.g. the
            # available data is historic), fetch the last available readings
            # regardless of timestamp so forecasting can still run.
            historical_readings = await mongodb_client.get_last_readings(
                machine_id=machine_id,
                limit=1440
            )

        if not historical_readings:
            raise HTTPException(
                status_code=404,
                detail=f"No historical data found for machine_id: {machine_id}"
            )
        
        if len(historical_readings) < 7:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for forecasting. Found {len(historical_readings)} readings, need at least 7"
            )
        
        # Prepare data for forecasting
        df_historical = prepare_forecast_data(historical_readings)
        
        # Get machine type from latest reading
        machine_type = historical_readings[-1].get('machine_type', 'M')
        
        # Generate forecast
        forecast_data = generate_forecast(
            historical_data=df_historical,
            machine_id=machine_id,
            machine_type=machine_type,
            forecast_minutes=forecast_minutes
        )

        # Save forecast to database (store minutes)
        await mongodb_client.save_forecast(
            machine_id=machine_id,
            forecast_minutes=forecast_minutes,
            forecast_data=forecast_data
        )

        return ForecastResponse(
            machine_id=machine_id,
            forecast_minutes=forecast_minutes,
            forecast_data=forecast_data,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating forecast: {str(e)}"
        )


@app.get('/readings')
async def get_recent_readings(machine_id: str, limit: int = 100, token: dict = Depends(verify_token)):
    """Return the most recent sensor readings for a machine.

    Query params:
    - machine_id: machine identifier
    - limit: maximum number of records to return (default 100)
    """
    if not mongodb_client:
        raise HTTPException(
            status_code=500,
            detail="Database not connected. Please configure MONGODB_URI in .env"
        )

    try:
        cursor = mongodb_client.sensor_readings.find(
            {'machine_id': machine_id},
            sort=[('timestamp', -1)]
        ).limit(limit)

        readings = await cursor.to_list(length=limit)

        # Convert non-JSON-serializable fields (ObjectId, datetime) to strings
        safe_readings = []
        for doc in readings:
            safe = dict(doc)
            if '_id' in safe:
                try:
                    safe['_id'] = str(safe['_id'])
                except Exception:
                    safe['_id'] = repr(safe.get('_id'))
            if 'timestamp' in safe and hasattr(safe['timestamp'], 'isoformat'):
                safe['timestamp'] = safe['timestamp'].isoformat()
            safe_readings.append(safe)

        return {
            'machine_id': machine_id,
            'count': len(safe_readings),
            'readings': safe_readings,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading DB: {e}")

