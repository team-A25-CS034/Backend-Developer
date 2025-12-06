import os
import aiofiles
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import jwt
from passlib.context import CryptContext
from dotenv import load_dotenv
from typing import Optional, List, Dict
from services.gemini_service import explain_with_gemini

from services.mongodb_service import MongoDBClient
from services.forecasting_service import generate_forecast, prepare_forecast_data
from services.machine_failure_service import initialize_classifier, get_classifier, normalize_for_classifier
from services.pos_service import load_pos_resources, predict_pos
from services.classifier_service import load_classifier_resources, predict_prompt_type
from services.injection_service import load_injection_resources, predict_injection_status
from services.gemini_service import explain_with_gemini, configure_gemini

# Load environment variables
load_dotenv()

# --- Configurations ---
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'logs', 'user_llm_conversation.txt')
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'machine_monitoring_db')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'machine_monitoring')

# JWT Config
ACCESS_TOKEN_KEY = os.getenv('ACCESS_TOKEN_KEY', 'secret_key') # Fallback for safety
ACCESS_TOKEN_AGE = int(os.getenv('ACCESS_TOKEN_AGE', 1800))
ALGORITHM = "HS256"
API_USERNAME = os.getenv('API_USERNAME')
API_PASSWORD = os.getenv('API_PASSWORD')

# Global variables
mongodb_client: Optional[MongoDBClient] = None
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class TimelineResponse(BaseModel):
    last_readings: List[Dict]
    forecast_minutes: int
    forecast_data: List[Dict]
    created_at: str

def serialize_readings(readings: List[Dict]) -> List[Dict]:
    """Helper untuk bikin data MongoDB JSON-serializable"""
    safe_readings = []
    for doc in readings:
        safe = {}
        for key, value in doc.items():
            if key == '_id':
                safe['_id'] = str(value)
            elif hasattr(value, 'isoformat'):
                safe[key] = value.isoformat()
            else:
                safe[key] = value
        safe_readings.append(safe)
    return safe_readings

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage MongoDB connection and ML models lifecycle"""
    global mongodb_client
    
    configure_gemini()
    
    # 1. Initialize MongoDB
    if MONGODB_URI:
        mongodb_client = MongoDBClient(MONGODB_URI, MONGODB_DATABASE, MONGODB_COLLECTION)
        print(f"✅ Connected to MongoDB: {MONGODB_DATABASE}")
    else:
        print("⚠️ MONGODB_URI not set - running without database")
    
    # 2. Initialize Machine Failure Classifier (Legacy)
    # Pastikan file classifier.h5 ada di folder models
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'classifier.h5')
    initialize_classifier(MODEL_PATH)
    
    # 3. Initialize NLP Models
    BASE_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
    
    # a. POS Model
    load_pos_resources(BASE_MODELS_DIR)
    
    # b. Prompt Classifier
    load_classifier_resources(os.path.join(BASE_MODELS_DIR, 'prompt_classifier'))
    
    # c. Prompt Injection (BARU)
    load_injection_resources(os.path.join(BASE_MODELS_DIR, 'prompt_injection'))

    yield
    
    # Shutdown
    if mongodb_client:
        await mongodb_client.close()
        print("🔌 MongoDB connection closed")


app = FastAPI(title='Smart Machine Backend', lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Sesuaikan di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
# (Saran: Jika ingin lebih rapi lagi, pindahkan ini ke file schemas.py)
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
    forecast_minutes: int = 300

class ForecastResponse(BaseModel):
    machine_id: str
    forecast_minutes: int
    forecast_data: List[Dict]
    created_at: str

class TextRequest(BaseModel):
    text: str

class TextPayload(BaseModel):
    text: str

# --- Helper Functions ---
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(seconds=ACCESS_TOKEN_AGE)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, ACCESS_TOKEN_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, ACCESS_TOKEN_KEY, algorithms=[ALGORITHM])
        if payload.get("sub") is None:
            raise ValueError("No subject in token")
        return payload
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

def authenticate_user(username: str, password: str) -> bool:
    if username != API_USERNAME:
        return False
    return password == API_PASSWORD

# --- Endpoints ---

@app.get('/')
def root():
    classifier = get_classifier()
    return {
        'status': 'ok',
        'machine_model_loaded': classifier is not None and classifier.is_loaded()
    }

@app.post('/login', response_model=TokenResponse)
def login(credentials: LoginRequest):
    if not authenticate_user(credentials.username, credentials.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": credentials.username})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_AGE
    }

# --- NLP Endpoints ---

@app.post('/predict/pos')
def get_pos_tags(request: TextRequest, token: dict = Depends(verify_token)):
    """Extract entities (POS Tags) from text."""
    try:
        result = predict_pos(request.text)
        return {
            "status": "success",
            "original_text": request.text,
            "entities": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict/classifier')
async def classify_prompt(payload: TextPayload, token: dict = Depends(verify_token)):
    """Classify user intent and log conversation."""
    try:
        result = predict_prompt_type(payload.text)
        
        # Async Logging
        try:
            label_result = result.get('label', 'unknown') 
            log_entry = f"{payload.text}|{label_result}\n"
            async with aiofiles.open(LOG_FILE_PATH, mode='a', encoding='utf-8') as f:
                await f.write(log_entry)
        except Exception as log_error:
            print(f"⚠️ Log Error: {log_error}")

        return {
            "original_text": payload.text,
            "classification": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/injection")
async def predict_injection(request: Request):
    """Detect prompt injection attacks."""
    try:
        data = await request.json()
        text = data.get("prompt", "")
        
        # Panggil logic dari service
        result = predict_injection_status(text)

        return {
            "prompt": text,
            "is_malicious": result['is_malicious'],
            "classification": {
                "label": result['label'],
                "confidence": result['confidence']
            }
        }
    except ValueError as ve:
        # Error jika model belum dimuat
        raise HTTPException(status_code=503, detail=str(ve))
    except Exception as e:
        print(f"❌ Injection Error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# --- Machine Monitoring Endpoints (Legacy) ---

@app.post('/predict')
def predict_machine_failure(sample: InputSample, token: dict = Depends(verify_token)):
    classifier = get_classifier()
    if not classifier or not classifier.is_loaded():
        raise HTTPException(status_code=500, detail='Model not loaded')
    try:
        return classifier.predict(
            air_temperature=sample.Air_temperature,
            process_temperature=sample.Process_temperature,
            rotational_speed=sample.Rotational_speed,
            torque=sample.Torque,
            tool_wear=sample.Tool_wear,
            machine_type=sample.Type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post('/forecast', response_model=ForecastResponse)
async def forecast_machine(request: ForecastRequest, token: dict = Depends(verify_token)):
    if not mongodb_client:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    try:
        # Ambil data dinamis dari MongoDB
        # Logic baru: Jika data range kosong, ambil last readings sebagai fallback
        historical_readings = await mongodb_client.get_readings_range(request.machine_id, days=30)
        
        if not historical_readings:
             # Fallback: Ambil 1440 data terakhir (1 hari) tanpa filter tanggal
             historical_readings = await mongodb_client.get_last_readings(limit=1440)
        
        if not historical_readings or len(historical_readings) < 7:
            raise HTTPException(status_code=400, detail="Insufficient data for forecasting (need min 7 points)")

        # Prepare & Generate (Panggil service baru)
        df_historical = prepare_forecast_data(historical_readings)
        
        # Coba tebak tipe mesin dari data terakhir
        last_reading = historical_readings[-1]
        machine_type = last_reading.get('Type', 'M') # Default M
        
        forecast_data = generate_forecast(
            historical_data=df_historical,
            forecast_minutes=request.forecast_minutes,
            machine_id=request.machine_id,
            machine_type=machine_type
        )
        
        # Simpan ke DB
        await mongodb_client.save_forecast(request.machine_id, request.forecast_minutes, forecast_data)
        
        return ForecastResponse(
            machine_id=request.machine_id,
            forecast_minutes=request.forecast_minutes,
            forecast_data=forecast_data,
            created_at=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        print(f"❌ Forecast Error: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")

@app.get('/readings')
async def get_readings(machine_id: str, limit: int = 100, token: dict = Depends(verify_token)):
    if not mongodb_client:
        raise HTTPException(status_code=500, detail="Database not connected")
    try:
        cursor = mongodb_client.sensor_readings.find({'machine_id': machine_id}, sort=[('timestamp', -1)]).limit(limit)
        readings = await cursor.to_list(length=limit)
        # Serialization helper
        for doc in readings:
            doc['_id'] = str(doc.get('_id'))
            if 'timestamp' in doc: doc['timestamp'] = doc['timestamp'].isoformat()
        
        return {'machine_id': machine_id, 'count': len(readings), 'readings': readings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/machines')
async def get_machines_list(token: dict = Depends(verify_token)):
    if not mongodb_client:
        raise HTTPException(status_code=500, detail="Database not connected")
    try:
        ids = await mongodb_client.sensor_readings.distinct('machine_id')
        return {'count': len(ids), 'machines': [{'machine_id': mid} for mid in sorted(ids)]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/timeline', response_model=TimelineResponse)
async def get_timeline(
    limit: int = 50,
    forecast_minutes: int = 100,
    token: dict = Depends(verify_token)
):
    """Mengembalikan data sensor asli terakhir + data prediksi masa depan (untuk grafik sambung)"""
    if not mongodb_client:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    try:
        # 1. Ambil Data Asli (History)
        readings = await mongodb_client.get_last_readings(limit=limit)
        if not readings:
            raise HTTPException(status_code=404, detail="No historical data available")

        safe_readings = serialize_readings(readings)

        # 2. Generate Forecast (Future)
        df_historical = prepare_forecast_data(readings)
        forecast_data = generate_forecast(
            historical_data=df_historical,
            forecast_minutes=forecast_minutes
        )

        return TimelineResponse(
            last_readings=safe_readings,
            forecast_minutes=forecast_minutes,
            forecast_data=forecast_data,
            created_at=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating timeline: {e}")

@app.get('/machine-status')
async def get_machine_status_dashboard(token: dict = Depends(verify_token)):
    """Dashboard: Cek status kesehatan semua mesin berdasarkan data terakhir"""
    classifier = get_classifier()
    if not classifier:
        raise HTTPException(status_code=500, detail='Machine Classifier model not loaded')

    try:
        # Ambil semua ID mesin
        # Note: Pastikan mongodb_client punya akses ke collection sensor
        collection = mongodb_client.sensor_readings
        machine_ids = await collection.distinct('Machine ID')
        
        results = []
        for mid in machine_ids:
            if not mid: continue
            
            # Ambil 1 data terakhir untuk mesin ini
            cursor = collection.find({'Machine ID': mid}).sort([('timestamp', -1)]).limit(1)
            docs = await cursor.to_list(length=1)
            
            if not docs:
                results.append({'machine_id': mid, 'status': 'no-data'})
                continue

            reading = docs[0]
            # Normalisasi data (panggil helper dari service)
            payload = normalize_for_classifier(reading, fallback_type=str(mid)[:1])
            
            if not payload:
                results.append({'machine_id': mid, 'status': 'missing-fields'})
                continue

            # Prediksi Failure
            try:
                pred = classifier.predict(**payload)
                results.append({
                    'machine_id': mid,
                    'status': 'active',
                    'prediction': pred['prediction_label'], # 'No Failure' or 'Failure'
                    'confidence': pred['probabilities'],
                    'last_updated': reading.get('timestamp')
                })
            except Exception as e:
                results.append({'machine_id': mid, 'status': 'prediction-error', 'error': str(e)})

        return {'count': len(results), 'machines': results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating machine status: {e}")

class GeminiRequest(BaseModel):
    query: str
    context: Dict  # Menerima JSON object penuh, bukan string
    label: str = "semua fitur" # Default label

@app.post("/ask-gemini")
async def ask_gemini_explanation(payload: GeminiRequest, token: dict = Depends(verify_token)):
    try:
        # Panggil service baru
        explanation = explain_with_gemini(
            user_query=payload.query, 
            context_data=payload.context,
            label_key=payload.label.lower() # Pastikan lowercase agar match logic
        )
        
        return {
            "status": "success",
            "original_query": payload.query,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))