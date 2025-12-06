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

Generate forecast for machine sensor data. This service produces minute-level forecasts.
Each forecast step is one minute. The API accepts `forecast_minutes`
to request N minutes ahead (default fixed to 300 minutes).

**Headers:**
```
Authorization: Bearer <your_access_token>
```

**Request Body (minute example):**
```json
{
  "machine_id": "machine_001",
  "forecast_minutes": 300
}
```

Notes on parameters:
- `machine_id` (string): Unique identifier for the machine.
- `forecast_minutes` (integer): Number of minutes to forecast. Default is 300. Choose a reasonable upper bound (for example 1-1440).

Behavior and units:
- Each forecast step == 1 minute. Timestamps in the response are incremented by minutes from the last historical timestamp.

**Response (example - minute mode):**
```json
{
  "machine_id": "machine_001",
  "forecast_minutes": 300,
  "forecast_data": [
    {
      "timestamp": "2025-09-02T00:00:00Z",
      "day_ahead": 1,
      "Air_temperature": 298.309,
      "Process_temperature": 308.926,
      "Rotational_speed": 1500.0,
      "Torque": 40.00,
      "Tool_wear": 28.80
    }
  ],
  "created_at": "2025-09-02T00:00:00Z"
}
```

**Status Codes:**
- `200 OK` - Forecast generated successfully
- `400 Bad Request` - Invalid parameters (e.g., missing `machine_id`, both `forecast_days` and `forecast_minutes` missing, or values out of allowed range) or insufficient data
- `401 Unauthorized` - Missing or invalid token
- `404 Not Found` - No historical data for `machine_id`
- `500 Internal Server Error` - Database not connected or forecast error

### 5. Displaying Historical Data (MongoDB)

You can fetch historical sensor readings stored in MongoDB via the backend. There are two useful endpoints:

- **Authenticated**: `GET /readings?limit=<n>` — requires a valid JWT in the `Authorization: Bearer <token>` header and returns recent readings for the machine.

Quick curl example (admin endpoint):
```bash
curl -s "http://localhost:8000/readings?limit=500" \
-H "Authorization: Bearer <TOKEN>"
```

**Response (example - limit=1):**
```json
{
  "machine_id": "machine_01",
  "count": 1,
  "readings": [
    {
      "timestamp": "2025-09-01T23:59:00",
      "machine_type": "M",
      "process_temperature": 309.1,
      "machine_id": "machine_01",
      "torque": 40.4,
      "air_temperature": 298.5,
      "tool_wear": 28.8,
      "rotational_speed": 1500,
      "_id": "6909bd8c79033ed1e570f233"
    }
  ]
}
```

### 6. List Machines

**GET** `/machines`

Return a list of distinct `machine_id` values present in the readings collection. This endpoint is authenticated and intended for frontend clients to discover available machines.

**Headers:**
```
Authorization: Bearer <your_access_token>
```

**Response (example):**
```json
{
  "count": 2,
  "machines": [
    { "machine_id": "machine_01" },
    { "machine_id": "machine_02" }
  ]
}
```

Quick curl example (authenticated):
```bash
curl -H "Authorization: Bearer <your_access_token>" "http://127.0.0.1:8000/machines"
```

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