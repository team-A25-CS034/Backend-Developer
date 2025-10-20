"""
MongoDB Client for Machine Monitoring
Handles sensor readings, predictions, and forecasts
"""

from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import os


class MongoDBClient:
    def __init__(self, mongo_uri: str, database_name: str = "machine_monitoring"):
        """Initialize MongoDB client"""
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[database_name]
        self.sensor_readings = self.db['sensor_readings']
        self.predictions = self.db['predictions']
        self.forecasts = self.db['forecasts']
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()
    
    async def get_latest_reading(self, machine_id: str) -> Optional[Dict]:
        """Get the latest sensor reading for a machine"""
        reading = await self.sensor_readings.find_one(
            {'machine_id': machine_id},
            sort=[('timestamp', -1)]
        )
        return reading
    
    async def get_readings_range(
        self, 
        machine_id: str, 
        days: int = 30
    ) -> List[Dict]:
        """Get sensor readings for the last N days"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        cursor = self.sensor_readings.find(
            {
                'machine_id': machine_id,
                'timestamp': {'$gte': cutoff_date}
            },
            sort=[('timestamp', 1)]
        )
        
        readings = await cursor.to_list(length=None)
        return readings
    
    async def save_prediction(
        self,
        machine_id: str,
        prediction_label: str,
        prediction_numeric: int,
        probabilities: Optional[List[float]],
        input_data: Dict
    ):
        """Save a prediction result"""
        document = {
            'timestamp': datetime.now(timezone.utc),
            'machine_id': machine_id,
            'prediction_label': prediction_label,
            'prediction_numeric': prediction_numeric,
            'probabilities': probabilities,
            'input_data': input_data
        }
        
        result = await self.predictions.insert_one(document)
        return str(result.inserted_id)
    
    async def save_forecast(
        self,
        machine_id: str,
        forecast_days: int,
        forecast_data: List[Dict]
    ):
        """Save forecast results"""
        document = {
            'created_at': datetime.now(timezone.utc),
            'machine_id': machine_id,
            'forecast_days': forecast_days,
            'forecast_data': forecast_data
        }
        
        result = await self.forecasts.insert_one(document)
        return str(result.inserted_id)
    
    async def get_latest_forecast(self, machine_id: str) -> Optional[Dict]:
        """Get the most recent forecast for a machine"""
        forecast = await self.forecasts.find_one(
            {'machine_id': machine_id},
            sort=[('created_at', -1)]
        )
        return forecast
    
    async def get_machine_statistics(self, machine_id: str, days: int = 30) -> Dict:
        """Get aggregated statistics for a machine"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        pipeline = [
            {
                '$match': {
                    'machine_id': machine_id,
                    'timestamp': {'$gte': cutoff_date}
                }
            },
            {
                '$group': {
                    '_id': None,
                    'avg_air_temp': {'$avg': '$air_temperature'},
                    'avg_process_temp': {'$avg': '$process_temperature'},
                    'avg_rotational_speed': {'$avg': '$rotational_speed'},
                    'avg_torque': {'$avg': '$torque'},
                    'max_tool_wear': {'$max': '$tool_wear'},
                    'min_tool_wear': {'$min': '$tool_wear'},
                    'count': {'$sum': 1}
                }
            }
        ]
        
        cursor = self.sensor_readings.aggregate(pipeline)
        result = await cursor.to_list(length=1)
        
        if result:
            stats = result[0]
            stats.pop('_id', None)
            return stats
        
        return {}
