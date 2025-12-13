from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, BigInteger
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    air_temperature = Column(Float)
    process_temperature = Column(Float)
    rotational_speed = Column(Float)
    torque = Column(Float)
    tool_wear = Column(Float)
    machine_type = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "machine_id": self.machine_id,
            "timestamp": self.timestamp,
            "air_temperature": self.air_temperature,
            "process_temperature": self.process_temperature,
            "rotational_speed": self.rotational_speed,
            "torque": self.torque,
            "tool_wear": self.tool_wear,
            "machine_type": self.machine_type,
            "Air temperature [K]": self.air_temperature,
            "Process temperature [K]": self.process_temperature,
            "Rotational speed [rpm]": self.rotational_speed,
            "Torque [Nm]": self.torque,
            "Tool wear [min]": self.tool_wear,
            "Type": self.machine_type,
            "Machine ID": self.machine_id 
        }

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    prediction_label = Column(String)
    prediction_numeric = Column(Integer)
    probabilities = Column(JSON)
    input_data = Column(JSON)

class Forecast(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    forecast_minutes = Column(Integer)
    forecast_data = Column(JSON)

class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    machine_name = Column(String)
    priority = Column(String)
    issue_summary = Column(String)
    suggested_fix = Column(String)
    estimated_time = Column(String)
    status = Column(String, default="Open")
    created_by = Column(String, nullable=True)
    source = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    sensor_snapshot = Column(JSON, nullable=True)