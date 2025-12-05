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
    # Now using minute-based forecasting (default 300 minutes)
    forecast_minutes: int = 300

class ForecastResponse(BaseModel):
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
        )
        
        # Prepare data for forecasting
        df_historical = prepare_forecast_data(historical_readings)
        
        # Generate forecast
        forecast_data = generate_forecast(
            historical_data=df_historical,
            forecast_minutes=forecast_minutes
        )


        return ForecastResponse(
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
async def get_recent_readings(limit: int = 100, token: dict = Depends(verify_token)):
    """Return the most recent sensor readings for a machine.

    Supports both legacy schema (machine_id, machine_type, air_temperature, ...)
    and the new `forecaster_input.csv` style schema (Product ID, Type, Air
    temperature [K], Process temperature [K], Rotational speed [rpm], Torque
    [Nm], Tool wear [min], UDI).

    Query params:
    - machine_id: machine identifier (matches either machine_id or Product ID)
    - limit: maximum number of records to return (default 100)
    """
    if not mongodb_client:
        raise HTTPException(
            status_code=500,
            detail="Database not connected. Please configure MONGODB_URI in .env"
        )

    try:
        # Match either legacy or new column naming
        filter_query = {
            '$or': [
            ]
        }

        # Sort by timestamp when available, otherwise by UDI (numeric order)
        sort_fields = [('timestamp', -1), ('UDI', -1)]

        cursor = mongodb_client.sensor_readings.find(
            filter_query,
            sort=sort_fields
        ).limit(limit)

        readings = await cursor.to_list(length=limit)

        # Convert to clean response format (keep only original CSV fields)
        safe_readings = []
        for doc in readings:
            safe = {}

            # Convert ObjectId to string
            if '_id' in doc:
                try:
                    safe['_id'] = str(doc['_id'])
                except Exception:
                    safe['_id'] = repr(doc.get('_id'))

            # Keep only the forecaster_input.csv fields
            safe['UDI'] = doc.get('UDI')
            safe['Type'] = doc.get('Type')
            safe['Air temperature [K]'] = doc.get('Air temperature [K]') or doc.get('air_temperature')
            safe['Process temperature [K]'] = doc.get('Process temperature [K]') or doc.get('process_temperature')
            safe['Rotational speed [rpm]'] = doc.get('Rotational speed [rpm]') or doc.get('rotational_speed')
            safe['Torque [Nm]'] = doc.get('Torque [Nm]') or doc.get('torque')
            safe['Tool wear [min]'] = doc.get('Tool wear [min]') or doc.get('tool_wear')
            safe['Target'] = doc.get('Target')
            safe['Failure Type'] = doc.get('Failure Type')

            # Include timestamp if present
            if 'timestamp' in doc and hasattr(doc['timestamp'], 'isoformat'):
                safe['timestamp'] = doc['timestamp'].isoformat()

            safe_readings.append(safe)

        return {
            'count': len(safe_readings),
            'readings': safe_readings,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading DB: {e}")


@app.get('/machines')
async def list_machines(token: dict = Depends(verify_token)):
    """Return a list of distinct machine IDs in the readings collection.

    Supports both legacy `machine_id` and new `Product ID` column naming.
    """
    if not mongodb_client:
        raise HTTPException(
            status_code=500,
            detail="Database not connected. Please configure MONGODB_URI in .env"
        )

    try:
        return {}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading DB: {e}")

