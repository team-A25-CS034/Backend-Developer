## Quick Start

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the root directory based on `.env.example`

### 3. Run Server

```bash
uvicorn app:app --reload --port 8000
```

---

## API Routes

### Base URL
```
http://localhost:8000
```

### 1. Health Check

**GET** `/`

Check if the API and model are running properly.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 2. Login

**POST** `/login`

Authenticate and get a JWT access token.

**Request Body:**
```json
{
  "username": "your_username",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Status Codes:**
- `200 OK` - Login successful
- `401 Unauthorized` - Invalid credentials

### 3. Predict Machine Failure

**POST** `/predict`

Predict machine failure status based on sensor data.

**Headers:**
```
Authorization: Bearer <your_access_token>
```

**Request Body:**
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

**Parameters:**
- `Air_temperature` (float): Ambient air temperature in Kelvin
- `Process_temperature` (float): Process temperature in Kelvin
- `Rotational_speed` (float): Rotational speed in RPM
- `Torque` (float): Torque in Nm
- `Tool_wear` (float): Tool wear in minutes
- `Type` (string): Machine type - "L" (Low), "M" (Medium), or "H" (High)

**Response:**
```json
{
  "prediction": 0,
  "label": "No Failure",
  "probabilities": {
    "No Failure": 0.95,
    "Failure": 0.05
  }
}
```

**Status Codes:**
- `200 OK` - Prediction successful
- `400 Bad Request` - Invalid input data
- `401 Unauthorized` - Missing or invalid token
- `500 Internal Server Error` - Model error

### 4. Generate Forecast

**POST** `/forecast`

Generate forecast for machine sensor data for N days ahead.

**Headers:**
```
Authorization: Bearer <your_access_token>
```

**Request Body:**
```json
{
  "machine_id": "machine_001",
  "forecast_days": 7
}
```

**Parameters:**
- `machine_id` (string): Unique identifier for the machine
- `forecast_days` (integer): Number of days to forecast (1-30)

**Response:**
```json
{
  "machine_id": "machine_001",
  "forecast_days": 7,
  "forecast_data": [
    {
      "date": "2025-10-21",
      "Air_temperature": 298.5,
      "Process_temperature": 309.2,
      "Rotational_speed": 1550.0,
      "Torque": 43.1,
      "Tool_wear": 5.0
    }
  ],
  "created_at": "2025-10-20T12:00:00Z"
}
```

**Status Codes:**
- `200 OK` - Forecast generated successfully
- `400 Bad Request` - Invalid parameters or insufficient data
- `401 Unauthorized` - Missing or invalid token
- `404 Not Found` - No historical data for machine_id
- `500 Internal Server Error` - Database not connected or forecast error

---

## Authentication

All endpoints except `/` and `/login` require authentication using JWT Bearer tokens.

### How to authenticate:

1. **Login** to get an access token:
   ```bash
   curl -X POST http://localhost:8000/login \
     -H "Content-Type: application/json" \
     -d '{"username": "your_username", "password": "your_password"}'
   ```

2. **Use the token** in subsequent requests:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <your_access_token>" \
     -d '{...}'
   ```

### Token Expiration

Access tokens expire after 30 minutes (1800 seconds) by default. You'll need to login again to get a new token.