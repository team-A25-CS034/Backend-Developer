import os
import aiofiles
from fastapi import FastAPI, HTTPException, Depends, status, Request, File, UploadFile
from fastapi.responses import StreamingResponse
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
import pandas as pd
import io
import asyncio
import json
from typing import List
from typing import Literal

from services.mongodb_service import MongoDBClient
from services.forecasting_service import generate_forecast, prepare_forecast_data
from services.machine_failure_service import initialize_classifier, get_classifier, normalize_for_classifier
from services.pos_service import load_pos_resources, predict_pos
from services.classifier_service import load_classifier_resources, predict_prompt_type
from services.injection_service import load_injection_resources, predict_injection_status
from services.gemini_service import explain_with_gemini, configure_gemini

load_dotenv()

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'logs', 'user_llm_conversation.txt')
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_URI_TICKETS = os.getenv('MONGODB_URI_TICKETS')
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'machine_monitoring_db')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'machine_monitoring')

# JWT Config
ACCESS_TOKEN_KEY = os.getenv('ACCESS_TOKEN_KEY', 'secret_key')
ACCESS_TOKEN_AGE = int(os.getenv('ACCESS_TOKEN_AGE', 1800))
ALGORITHM = "HS256"
API_USERNAME = os.getenv('API_USERNAME')
API_PASSWORD = os.getenv('API_PASSWORD')

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
    "productID": "Product ID",
    "UDI": "UDI",
    "Type": "Type",
    "air temperature [K]": "Air temperature [K]",
    "process temperature [K]": "Process temperature [K]",
    "rotational speed [rpm]": "Rotational speed [rpm]",
    "torque [Nm]": "Torque [Nm]",
    "tool wear [min]": "Tool wear [min]",
    "machine failure": "Machine failure"
}

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage MongoDB connection and ML models lifecycle"""
    global mongodb_client
    
    configure_gemini()
    
    if MONGODB_URI:
        mongodb_client = MongoDBClient(
            mongo_uri=MONGODB_URI, 
            database_name=MONGODB_DATABASE, 
            collection_name=MONGODB_COLLECTION,
            ticket_uri=MONGODB_URI_TICKETS
        )
        print(f"Connected to Main MongoDB: {MONGODB_DATABASE}")
        print(f"Connected to Ticket MongoDB: maintenance_db")
    else:
        print("MONGODB_URI not set - running without database")
    
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'classifier.h5')
    initialize_classifier(MODEL_PATH)
    
    BASE_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
    
    load_pos_resources(BASE_MODELS_DIR)
    
    load_classifier_resources(os.path.join(BASE_MODELS_DIR, 'prompt_classifier'))
    
    load_injection_resources(os.path.join(BASE_MODELS_DIR, 'prompt_injection'))

    yield
    
    if mongodb_client:
        await mongodb_client.close()
        print("🔌 MongoDB connection closed")


app = FastAPI(title='Smart Machine Backend', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    """Detect prompt injection attacks."""
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
async def predict_machine_failure(sample: InputSample, token: dict = Depends(verify_token)): # Ubah jadi ASYNC
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
        
        if is_failure and mongodb_client:
            # Tentukan prioritas berdasarkan confidence atau tipe failure
            priority = "High"
            
            # Buat summary otomatis
            issue_summary = f"Auto-Detected: {result['prediction_label']} pada Mesin Tipe {sample.Type}"
            
            # Logic sederhana untuk saran perbaikan (Bisa diganti pakai Gemini nanti)
            suggested_fix = "Lakukan inspeksi mendalam pada komponen rotasi dan cek riwayat tool wear."
            if sample.Tool_wear > 200:
                suggested_fix = "Ganti komponen Tool/Mata Pisau segera (Wear Level Tinggi)."
            
            ticket_data = {
                "machine_name": f"Machine_{sample.Type}_Auto", # Atau ambil dari ID jika ada
                "priority": priority,
                "issue_summary": issue_summary,
                "suggested_fix": suggested_fix,
                "estimated_time_to_address": "1-2 Jam (Estimasi AI)",
                "source": "System (Auto-Generated)",
                "sensor_snapshot": sample.dict() # Simpan data sensor saat kejadian
            }
            
            # Simpan ke MongoDB tanpa menunggu response (Fire & Forget task)
            await mongodb_client.create_ticket(ticket_data)

            # Kirim Notifikasi WebSocket (Kode lama Anda)
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
    if not mongodb_client:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    try:
        historical_readings = await mongodb_client.get_readings_range(request.machine_id, days=30)
        
        if not historical_readings:
             historical_readings = await mongodb_client.get_last_readings(limit=1440)
        
        if not historical_readings or len(historical_readings) < 7:
            raise HTTPException(status_code=400, detail="Insufficient data for forecasting (need min 7 points)")

        df_historical = prepare_forecast_data(historical_readings)
        
        last_reading = historical_readings[-1]
        machine_type = last_reading.get('Type', 'M') 
        
        forecast_data = generate_forecast(
            historical_data=df_historical,
            forecast_minutes=request.forecast_minutes,
            machine_id=request.machine_id,
            machine_type=machine_type
        )
        
        await mongodb_client.save_forecast(request.machine_id, request.forecast_minutes, forecast_data)
        
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
    if not mongodb_client:
        raise HTTPException(status_code=500, detail="Database not connected")
    try:
        # PERBAIKAN: Gunakan key "Machine ID" sesuai database
        cursor = mongodb_client.sensor_readings.find(
            {'Machine ID': machine_id}, 
            sort=[('timestamp', -1)]
        ).limit(limit)
        
        readings = await cursor.to_list(length=limit)
        
        # Serialize
        for doc in readings:
            doc['_id'] = str(doc.get('_id'))
            if 'timestamp' in doc and hasattr(doc['timestamp'], 'isoformat'):
                doc['timestamp'] = doc['timestamp'].isoformat()
        
        return {'machine_id': machine_id, 'count': len(readings), 'readings': readings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get('/machines')
# async def get_machines_list(token: dict = Depends(verify_token)):
#     if not mongodb_client:
#         raise HTTPException(status_code=500, detail="Database not connected")
#     try:
#         ids = await mongodb_client.sensor_readings.distinct('machine_id')
#         return {'count': len(ids), 'machines': [{'machine_id': mid} for mid in sorted(ids)]}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.get('/machines')
async def get_machines_list(token: dict = Depends(verify_token)):
    if not mongodb_client:
        raise HTTPException(status_code=500, detail="Database not connected")
    try:
        # PERBAIKAN: Gunakan key "Machine ID" (bukan machine_id)
        ids = await mongodb_client.sensor_readings.distinct('Machine ID')
        
        # Bersihkan hasil (hapus yang kosong/null)
        clean_ids = [mid for mid in ids if mid]
        
        # Return format standar frontend
        return {'count': len(clean_ids), 'machines': [{'machine_id': mid} for mid in sorted(clean_ids)]}
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
        readings = await mongodb_client.get_last_readings(limit=limit)
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

    try:
        collection = mongodb_client.sensor_readings
        # PERBAIKAN: Gunakan key "Machine ID"
        machine_ids = await collection.distinct('Machine ID')
        
        results = []
        for mid in machine_ids:
            if not mid: continue
            
            # Ambil data terakhir per mesin
            cursor = collection.find({'Machine ID': mid}).sort([('timestamp', -1)]).limit(1)
            docs = await cursor.to_list(length=1)
            
            if not docs:
                results.append({'machine_id': mid, 'status': 'no-data'})
                continue

            reading = docs[0]
            
            # Normalisasi manual agar aman (Map dari DB Key -> Model Feature Key)
            payload = {
                'air_temperature': reading.get('Air temperature [K]'),
                'process_temperature': reading.get('Process temperature [K]'),
                'rotational_speed': reading.get('Rotational speed [rpm]'),
                'torque': reading.get('Torque [Nm]'),
                'tool_wear': reading.get('Tool wear [min]'),
                'machine_type': reading.get('Type')
            }
            
            # Validasi kelengkapan data
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
    Upload data baru (Excel/CSV) ke database.
    Format kolom harus sesuai dengan standar Predictive Maintenance Dataset.
    """
    if not mongodb_client:
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

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    critical_missing = [col for col in missing_cols if col in [
        "Air temperature [K]", "Process temperature [K]", 
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ]]
    
    if critical_missing:
        raise HTTPException(
            status_code=400, 
            detail=f"Format file salah. Kolom berikut hilang: {critical_missing}"
        )

    # 2. Rename Kolom
    df.rename(columns=COLUMN_MAPPING, inplace=True)

    # 3. Cleaning Data (Gunakan nama kolom BARU / Internal)
    # Di kode asli, variable 'numeric_cols' menggunakan nama panjang/lama. 
    # Karena kita sudah rename, kita harus update list ini ke nama baru agar cocok dengan df.
    numeric_cols = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=numeric_cols, inplace=True)
    
    if df.empty:
        raise HTTPException(status_code=400, detail="File tidak berisi data valid setelah pembersihan.")

    if 'timestamp' not in df.columns:
        df['timestamp'] = datetime.now(timezone.utc)
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    try:
        records = df.to_dict(orient='records')
        count = await mongodb_client.bulk_insert_readings(records)
        
        success_payload = {
            "type": "NEW_DATA",
            "title": "Data Sensor Baru",
            "message": f"File '{file.filename}' berhasil diupload.",
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
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan ke database: {str(e)}")

class NotificationManager:
    """Class untuk mengelola koneksi real-time ke client"""
    def __init__(self):
        self.active_connections: List[asyncio.Queue] = []

    async def connect(self):
        """User baru connect -> buatkan antrian pesan baru"""
        queue = asyncio.Queue()
        self.active_connections.append(queue)
        print(f"📡 Client Connected. Total: {len(self.active_connections)}")
        return queue

    def disconnect(self, queue: asyncio.Queue):
        """User disconnect -> hapus antrian"""
        if queue in self.active_connections:
            self.active_connections.remove(queue)
            print(f"🔌 Client Disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Kirim pesan ke SEMUA user yang sedang connect"""
        if 'timestamp' not in message:
            message['timestamp'] = datetime.now().isoformat()
            
        for queue in self.active_connections:
            await queue.put(message)

notification_manager = NotificationManager()

def verify_token_query(token: str):
    try:
        payload = jwt.decode(token, ACCESS_TOKEN_KEY, algorithms=[ALGORITHM])
        if payload.get("sub") is None:
            raise ValueError("No subject in token")
        return payload
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token for stream",
        )

@app.get("/notifications/stream")
async def stream_notifications(token: str = Depends(verify_token_query)):
    """
    Endpoint ini akan menahan koneksi (Keep-Alive).
    Setiap kali ada trigger 'broadcast', data akan dikirim ke sini.
    """
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
    """
    Endpoint untuk insinyur membuat tiket secara manual melalui Form.
    """
    if not mongodb_client:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    try:
        ticket_data = ticket.dict()
        ticket_data['source'] = "Manual (Engineer)"
        ticket_data['created_by'] = token.get('sub', 'Unknown User') 
        
        ticket_id = await mongodb_client.create_ticket(ticket_data)
        
        if '_id' in ticket_data:
            ticket_data['_id'] = str(ticket_data['_id'])
        
        if 'created_at' in ticket_data and hasattr(ticket_data['created_at'], 'isoformat'):
             ticket_data['created_at'] = ticket_data['created_at'].isoformat()

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
    """
    Melihat daftar tiket (Dashboard Tiket).
    Bisa filter by status (Open/Closed).
    """
    if not mongodb_client:
        raise HTTPException(status_code=500, detail="Database not connected")
        
    try:
        tickets = await mongodb_client.get_tickets(limit=100, status=status)
        return {
            "count": len(tickets),
            "tickets": tickets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))