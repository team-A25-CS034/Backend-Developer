# Machine Monitoring Backend API

Backend untuk sistem monitoring mesin prediktif menggunakan FastAPI, XGBoost classifier, dan time-series forecasting.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Buat file `.env` dengan konfigurasi:
```
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/database
MONGODB_DATABASE=machine_monitoring_db
MONGODB_COLLECTION=machine_monitoring
API_USERNAME=admin
API_PASSWORD=your_password
ACCESS_TOKEN_KEY=your_secret_key
```

### 3. Run Server
```bash
uvicorn app:app --reload --port 8000
```

---

## API Routes

### Authentication

#### POST `/login`
Login dan dapatkan JWT token.

**Request:**
```json
{
  "username": "admin",
  "password": "password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

---

### Machine Data

#### GET `/readings`
Ambil daftar mesin atau sensor readings mesin spesifik.

**Query Parameters:**
- `machine_id` (optional): ID mesin untuk filter

**Response:**
```json
{
  "machines": ["M1", "M2", "M3"],
  "count": 3
}
```

atau jika dengan `machine_id`:
```json
{
  "readings": [
    {
      "Machine ID": "M1",
      "Air temperature [K]": 298.1,
      "Process temperature [K]": 308.6,
      "Rotational speed [rpm]": 1551,
      "Torque [Nm]": 42.8,
      "Tool wear [min]": 0,
      "Type": "M"
    }
  ],
  "count": 100
}
```

#### GET `/machine-status`
Ambil status dan klasifikasi terbaru dari semua mesin.

**Response:**
```json
{
  "count": 3,
  "machines": [
    {
      "machine_id": "M1",
      "prediction_numeric": 0,
      "prediction_label": "No Failure",
      "probabilities": [0.95, 0.02, 0.01, 0.01, 0.01, 0.0],
      "timestamp": "2025-12-06T10:30:00Z"
    },
    {
      "machine_id": "M2",
      "prediction_numeric": 4,
      "prediction_label": "Tool Wear Failure",
      "probabilities": [0.1, 0.05, 0.03, 0.02, 0.8, 0.0],
      "timestamp": "2025-12-06T10:29:45Z"
    }
  ]
}
```

---

### Predictions

#### POST `/predict`
Prediksi status kegagalan mesin dari sensor readings tunggal.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request:**
```json
{
  "Air_temperature": 298.1,
  "Process_temperature": 308.6,
  "Rotational_speed": 1551,
  "Torque": 42.8,
  "Tool_wear": 0,
  "Type": "M"
}
```

**Response:**
```json
{
  "prediction_numeric": 0,
  "prediction_label": "No Failure",
  "probabilities": [0.95, 0.02, 0.01, 0.01, 0.01, 0.0]
}
```

**Failure Types:**
- `0` - No Failure
- `1` - Heat Dissipation Failure
- `2` - Power Failure
- `3` - Overstrain Failure
- `4` - Tool Wear Failure
- `5` - Random Failures

---

### Forecasting

#### POST `/forecast`
Generate forecast untuk 5 sensor dalam N menit ke depan.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request:**
```json
{
  "forecast_minutes": 300
}
```

**Response:**
```json
{
  "forecast_minutes": 300,
  "forecast_data": [
    {
      "air_temperature": 299.2,
      "process_temperature": 309.1,
      "rotational_speed": 1550,
      "torque": 42.5,
      "tool_wear": 1
    },
    {
      "air_temperature": 299.5,
      "process_temperature": 309.4,
      "rotational_speed": 1552,
      "torque": 43.0,
      "tool_wear": 2
    }
  ],
  "created_at": "2025-12-06T10:30:00Z"
}
```

#### GET `/timeline`
Ambil 50 sensor readings terakhir + forecast untuk N menit ke depan.

**Query Parameters:**
- `limit` (int, default: 50): Jumlah readings terakhir
- `forecast_minutes` (int, default: 100): Menit forecast ke depan

**Response:**
```json
{
  "last_readings": [
    {
      "machine_id": "M1",
      "timestamp": "2025-12-06T10:15:00Z",
      "air_temperature": 298.0,
      "process_temperature": 308.5,
      "rotational_speed": 1550,
      "torque": 42.0,
      "tool_wear": 10
    },
    {
      "machine_id": "M1",
      "timestamp": "2025-12-06T10:16:00Z",
      "air_temperature": 298.2,
      "process_temperature": 308.7,
      "rotational_speed": 1551,
      "torque": 42.3,
      "tool_wear": 11
    }
  ],
  "forecast_minutes": 100,
  "forecast_data": [
    {
      "air_temperature": 298.5,
      "process_temperature": 309.0,
      "rotational_speed": 1552,
      "torque": 42.5,
      "tool_wear": 12
    }
  ],
  "created_at": "2025-12-06T10:30:00Z"
}
```

---

### Health Check

#### GET `/`
Status API dan model.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## Notes

- Semua route kecuali `GET /` dan `POST /login` memerlukan JWT token di header `Authorization: Bearer <token>`
- Token expires setelah 24 jam