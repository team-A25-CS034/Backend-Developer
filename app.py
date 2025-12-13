import os
import aiofiles
from fastapi import FastAPI, HTTPException, Depends, status, Request, File, UploadFile, Query
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import jwt
from passlib.context import CryptContext
from dotenv import load_dotenv
from typing import Optional, List, Dict, Literal
import pandas as pd
import io
import asyncio
import json

# --- SERVICES IMPORT ---
from services.postgres_service import PostgresService
from services.forecasting_service import generate_forecast, prepare_forecast_data
from services.machine_failure_service import initialize_classifier, get_classifier, normalize_for_classifier
from services.pos_service import load_pos_resources, predict_pos
from services.classifier_service import load_classifier_resources, predict_prompt_type
from services.injection_service import load_injection_resources, predict_injection_status
from services.gemini_service import explain_with_gemini, configure_gemini

load_dotenv()

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'logs', 'user_llm_conversation.txt')
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# --- DATABASE CONFIG (POSTGRESQL) ---
DATABASE_URL = os.getenv('DATABASE_URL')

# JWT Config
ACCESS_TOKEN_KEY = os.getenv('ACCESS_TOKEN_KEY', 'secret_key')
ACCESS_TOKEN_AGE = int(os.getenv('ACCESS_TOKEN_AGE', 1800))
ALGORITHM = "HS256"
API_USERNAME = os.getenv('API_USERNAME')
API_PASSWORD = os.getenv('API_PASSWORD')

# Columns required for ML/Processing
REQUIRED_COLUMNS = {
    "UID", 
    "Product ID", 
    "Type", 
    "Air temperature [K]", 
    "Process temperature [K]", 
    "Rotational speed [rpm]", 
    "Torque [Nm]", 
    "Tool wear [min]", 
    "Machine failure",
    "TWF", "HDF", "PWF", "OSF", "RNF" 
}

COLUMN_MAPPING = {
    "productID": "machine_id",
    "Product ID": "machine_id",
    "UDI": "udi",
    "Type": "machine_type",
    "air temperature [K]": "air_temperature",
    "Air temperature [K]": "air_temperature",
    "process temperature [K]": "process_temperature",
    "Process temperature [K]": "process_temperature",
    "rotational speed [rpm]": "rotational_speed",
    "Rotational speed [rpm]": "rotational_speed",
    "torque [Nm]": "torque",
    "Torque [Nm]": "torque",
    "tool wear [min]": "tool_wear",
    "Tool wear [min]": "tool_wear",
    "machine failure": "machine_failure",
    "Machine failure": "machine_failure"
}

# Global DB Service Instance
db_service: Optional[PostgresService] = None

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)

class TimelineResponse(BaseModel):
    last_readings: List[Dict]
    forecast_minutes: int
    forecast_data: List[Dict]
    created_at: str

def serialize_readings(readings: List[Dict]) -> List[Dict]:
    """Helper untuk format datetime agar aman untuk JSON"""
    safe_readings = []
    for doc in readings:
        safe = {}
        for key, value in doc.items():
            if isinstance(value, datetime):
                safe[key] = value.isoformat()
            else:
                safe[key] = value
        safe_readings.append(safe)
    return safe_readings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage Database connection and ML models lifecycle"""
    global db_service
    
    configure_gemini()
    
    # --- INIT POSTGRESQL ---
    if DATABASE_URL:
        db_service = PostgresService(DATABASE_URL)
        await db_service.init_models()
        print(f"✅ Connected to PostgreSQL")
    else:
        print("⚠️ DATABASE_URL not set - running without database")
    
    # --- LOAD ML MODELS ---
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'classifier.h5')
    initialize_classifier(MODEL_PATH)
    
    BASE_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
    load_pos_resources(BASE_MODELS_DIR)
    load_classifier_resources(os.path.join(BASE_MODELS_DIR, 'prompt_classifier'))
    load_injection_resources(os.path.join(BASE_MODELS_DIR, 'prompt_injection'))

    yield
    
    # --- CLOSE CONNECTION ---
    if db_service:
        await db_service.close()
        print("🔌 Database connection closed")


app = FastAPI(title='Smart Machine Backend', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
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

class TicketCreate(BaseModel):
    machine_name: str
    priority: Literal["Low", "Medium", "High", "Critical"]
    issue_summary: str
    suggested_fix: str
    estimated_time_to_address: str

# Auth Helpers
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

# --- ENDPOINTS ---

@app.get('/')
def root():
    classifier = get_classifier()
    return {
        'status': 'ok',
        'database': 'postgresql',
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

@app.post('/predict/pos')
def get_pos_tags(request: TextRequest, token: dict = Depends(verify_token)):
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
    try:
        result = predict_prompt_type(payload.text)
        
        try:
            label_result = result.get('label', 'unknown') 
            log_entry = f"{payload.text}|{label_result}\n"
            async with aiofiles.open(LOG_FILE_PATH, mode='a', encoding='utf-8') as f:
                await f.write(log_entry)
        except Exception as log_error:
            print(f"Log Error: {log_error}")

        return {
            "original_text": payload.text,
            "classification": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/injection")
async def predict_injection(request: Request):
    try:
        data = await request.json()
        text = data.get("prompt", "")
        
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
        raise HTTPException(status_code=503, detail=str(ve))
    except Exception as e:
        print(f"Injection Error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post('/predict')
async def predict_machine_failure(sample: InputSample, token: dict = Depends(verify_token)):
    classifier = get_classifier()
    if not classifier or not classifier.is_loaded():
        raise HTTPException(status_code=500, detail='Model not loaded')
    try:
        result = classifier.predict(
            air_temperature=sample.Air_temperature,
            process_temperature=sample.Process_temperature,
            rotational_speed=sample.Rotational_speed,
            torque=sample.Torque,
            tool_wear=sample.Tool_wear,
            machine_type=sample.Type
        )

        is_failure = result['prediction_label'] != 'No Failure'
        
        # Simpan jika failure dan DB terkoneksi
        if is_failure and db_service:
            priority = "High"
            issue_summary = f"Auto-Detected: {result['prediction_label']} pada Mesin Tipe {sample.Type}"
            suggested_fix = "Lakukan inspeksi mendalam pada komponen rotasi."
            
            if sample.Tool_wear > 200:
                suggested_fix = "Ganti komponen Tool/Mata Pisau segera (Wear Level Tinggi)."
            
            ticket_data = {
                "machine_name": f"Machine_{sample.Type}_Auto",
                "priority": priority,
                "issue_summary": issue_summary,
                "suggested_fix": suggested_fix,
                "estimated_time_to_address": "1-2 Jam (Estimasi AI)",
                "source": "System (Auto-Generated)",
                "sensor_snapshot": sample.dict()
            }
            
            # Create Ticket in Postgres
            await db_service.create_ticket(ticket_data)

            # Broadcast Alert
            alert_payload = {
                "type": "CRITICAL_ALERT",
                "title": f"Terdeteksi Kerusakan Mesin!",
                "message": f"Tiket perbaikan otomatis telah dibuat.",
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
            await notification_manager.broadcast(alert_payload)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/forecast', response_model=ForecastResponse)
async def forecast_machine(request: ForecastRequest, token: dict = Depends(verify_token)):
    if not db_service:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    try:
        # Ambil data dari Postgres (30 hari terakhir)
        historical_readings = await db_service.get_readings_range(request.machine_id, days=30)
        
        if not historical_readings:
             # Fallback: ambil data terakhir tanpa filter hari
             historical_readings = await db_service.get_last_readings(limit=1440)
        
        if not historical_readings or len(historical_readings) < 7:
            raise HTTPException(status_code=400, detail="Insufficient data for forecasting (need min 7 points)")

        df_historical = prepare_forecast_data(historical_readings)
        
        last_reading = historical_readings[-1]
        machine_type = last_reading.get('machine_type') or last_reading.get('Type') or 'M'
        
        forecast_data = generate_forecast(
            historical_data=df_historical,
            forecast_minutes=request.forecast_minutes,
            machine_id=request.machine_id,
            machine_type=machine_type
        )
        
        await db_service.save_forecast(request.machine_id, request.forecast_minutes, forecast_data)
        
        return ForecastResponse(
            machine_id=request.machine_id,
            forecast_minutes=request.forecast_minutes,
            forecast_data=forecast_data,
            created_at=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        print(f"Forecast Error: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")

@app.get('/readings')
async def get_readings(machine_id: str, limit: int = 100, token: dict = Depends(verify_token)):
    if not db_service:
        raise HTTPException(status_code=500, detail="Database not connected")
    try:
        # Ambil list reading, karena get_readings_range ambil list, kita filter manual limitnya
        readings = await db_service.get_readings_range(machine_id, days=365) 
        
        # Sort descending by timestamp (asumsi service return chronological, kita balik)
        readings.sort(key=lambda x: x['timestamp'], reverse=True)
        limited_readings = readings[:limit]
        
        # Serialize datetime
        safe_readings = serialize_readings(limited_readings)
        
        return {'machine_id': machine_id, 'count': len(safe_readings), 'readings': safe_readings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/machines')
async def get_machines_list(token: dict = Depends(verify_token)):
    if not db_service:
        raise HTTPException(status_code=500, detail="Database not connected")
    try:
        # SQL Distinct Query
        ids = await db_service.get_distinct_machine_ids()
        
        # Clean nulls
        clean_ids = [mid for mid in ids if mid]
        
        return {'count': len(clean_ids), 'machines': [{'machine_id': mid} for mid in sorted(clean_ids)]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/timeline', response_model=TimelineResponse)
async def get_timeline(
    limit: int = 50,
    forecast_minutes: int = 100,
    token: dict = Depends(verify_token)
):
    if not db_service:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    try:
        readings = await db_service.get_last_readings(limit=limit)
        if not readings:
            raise HTTPException(status_code=404, detail="No historical data available")

        safe_readings = serialize_readings(readings)

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
    classifier = get_classifier()
    if not classifier:
        raise HTTPException(status_code=500, detail='Machine Classifier model not loaded')

    if not db_service:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        machine_ids = await db_service.get_distinct_machine_ids()
        
        results = []
        for mid in machine_ids:
            if not mid: continue
            
            # Ambil 1 data terakhir
            reading = await db_service.get_latest_reading(mid)
            
            if not reading:
                results.append({'machine_id': mid, 'status': 'no-data'})
                continue

            # Mapping Key SQL (snake_case) ke Feature Model
            payload = {
                'air_temperature': reading.get('air_temperature'),
                'process_temperature': reading.get('process_temperature'),
                'rotational_speed': reading.get('rotational_speed'),
                'torque': reading.get('torque'),
                'tool_wear': reading.get('tool_wear'),
                'machine_type': reading.get('machine_type')
            }
            
            if None in payload.values():
                results.append({'machine_id': mid, 'status': 'missing-fields'})
                continue

            try:
                pred = classifier.predict(**payload)
                results.append({
                    'machine_id': mid,
                    'status': 'active',
                    'prediction': pred['prediction_label'], 
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
    context: Dict 
    label: str = "semua fitur"

@app.post("/ask-gemini")
async def ask_gemini_explanation(payload: GeminiRequest, token: dict = Depends(verify_token)):
    try:
        explanation = explain_with_gemini(
            user_query=payload.query, 
            context_data=payload.context,
            label_key=payload.label.lower()
        )
        
        return {
            "status": "success",
            "original_query": payload.query,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/upload-data")
async def upload_sensor_data(
    file: UploadFile = File(...), 
    token: dict = Depends(verify_token)
):
    """
    Upload data baru (Excel/CSV) ke PostgreSQL.
    Otomatis mapping kolom CSV ke kolom Database (snake_case).
    Fix: Timestamp timezone issue.
    """
    if not db_service:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Format file harus .csv atau .xlsx")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca file: {str(e)}")

    # 1. Rename Kolom (CSV Header -> Snake Case DB Column)
    df.rename(columns=COLUMN_MAPPING, inplace=True)

    # 2. Pastikan kolom yang dibutuhkan ada
    required_internal_cols = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
    
    missing_cols = set(required_internal_cols) - set(df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=400, 
            detail=f"Mapping Gagal. Kolom database berikut tidak ditemukan: {missing_cols}. Cek header CSV anda."
        )

    # 3. Cleaning Numeric Data
    for col in required_internal_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=required_internal_cols, inplace=True)
    
    if df.empty:
        raise HTTPException(status_code=400, detail="File tidak berisi data valid setelah pembersihan.")

    # --- PERBAIKAN TIMESTAMPS (CRUCIAL FIX) ---
    if 'timestamp' not in df.columns:
        # Gunakan datetime.now() polos (Naive) agar cocok dengan PostgreSQL
        df['timestamp'] = datetime.now() 
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Jika pandas mendeteksi timezone, kita hapus (tz_localize(None))
        try:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        except Exception:
            pass # Sudah naive, biarkan

    try:
        records = df.to_dict(orient='records')
        count = await db_service.bulk_insert_readings(records)
        
        success_payload = {
            "type": "NEW_DATA",
            "title": "Data Sensor Baru",
            "message": f"File '{file.filename}' berhasil diupload ke PostgreSQL.",
            "timestamp": datetime.now().isoformat()
        }
        await notification_manager.broadcast(success_payload)
        
        return {
            "status": "success",
            "filename": file.filename,
            "inserted_count": count,
            "message": f"Berhasil menambahkan {count} data baru ke sistem."
        }
    except Exception as e:
        # Print error detail ke console agar lebih jelas
        print(f"❌ Upload Error Detail: {e}") 
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan ke database: {str(e)}")

class NotificationManager:
    def __init__(self):
        self.active_connections: List[asyncio.Queue] = []

    async def connect(self):
        queue = asyncio.Queue()
        self.active_connections.append(queue)
        print(f"📡 Client Connected. Total: {len(self.active_connections)}")
        return queue

    def disconnect(self, queue: asyncio.Queue):
        if queue in self.active_connections:
            self.active_connections.remove(queue)
            print(f"🔌 Client Disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        if 'timestamp' not in message:
            message['timestamp'] = datetime.now().isoformat()
        for queue in self.active_connections:
            await queue.put(message)

notification_manager = NotificationManager()

def verify_token_stream(
    token: Optional[str] = Query(None), 
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security_optional)
):
    """
    Helper Hybrid: Bisa baca token dari URL (?token=...) ATAU Header (Authorization: Bearer ...)
    """
    token_string = None
    
    # 1. Cek apakah ada di URL Query (?token=abc)
    if token:
        token_string = token
    # 2. Jika tidak ada di URL, cek di Header Authorization
    elif auth:
        token_string = auth.credentials
    
    # 3. Validasi
    if not token_string:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is required (Query Param 'token' or Bearer Header)",
        )

    try:
        payload = jwt.decode(token_string, ACCESS_TOKEN_KEY, algorithms=[ALGORITHM])
        if payload.get("sub") is None:
            raise ValueError("No subject in token")
        return payload
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

@app.get("/notifications/stream")
async def stream_notifications(token_payload: dict = Depends(verify_token_stream)):
    async def event_generator():
        queue = await notification_manager.connect()
        try:
            while True:
                data = await queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        except asyncio.CancelledError:
            notification_manager.disconnect(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/tickets", status_code=201)
async def create_manual_ticket(ticket: TicketCreate, token: dict = Depends(verify_token)):
    if not db_service:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    try:
        ticket_data = ticket.dict()
        ticket_data['source'] = "Manual (Engineer)"
        ticket_data['created_by'] = token.get('sub', 'Unknown User') 
        
        # Postgres Service call
        ticket_id = await db_service.create_ticket(ticket_data)
        
        return {
            "status": "success",
            "message": "Tiket berhasil dibuat",
            "ticket_id": ticket_id,
            "data": ticket_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membuat tiket: {e}")

@app.get("/tickets-machine")
async def get_all_tickets(status: str = None, token: dict = Depends(verify_token)):
    if not db_service:
        raise HTTPException(status_code=500, detail="Database not connected")
        
    try:
        tickets = await db_service.get_tickets(limit=100, status=status)
        return {
            "count": len(tickets),
            "tickets": tickets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))