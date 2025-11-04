"""
MongoDB Client for Machine Monitoring
Handles sensor readings, predictions, and forecasts
"""

from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import os


class MongoDBClient:
    def __init__(self, mongo_uri: str, database_name: str = None, collection_name: str = None):
        """Initialize MongoDB client

        By default this will look for environment variables provided by the
        application (MONGODB_DATABASE and MONGODB_COLLECTION). If those are
        not provided we default to the new database and collection names that
        the user indicated: `macnine_monitoring_db` and `machine_monitoring`.
        """
        # Lazy import of os to avoid global dependency assumptions
        import os

        if not database_name:
            # Default to the project's machine_monitoring_db as configured by user
            database_name = os.getenv('MONGODB_DATABASE', 'machine_monitoring_db')
        if not collection_name:
            collection_name = os.getenv('MONGODB_COLLECTION', 'machine_monitoring')

        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[database_name]

        # Primary collection for sensor readings (some deployments use a
        # single 'machine_monitoring' collection). We keep the legacy
        # collection names around for compatibility but prefer the
        # configured collection_name when present.
        self.sensor_readings = self.db[collection_name]
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

    async def get_last_readings(self, machine_id: str, limit: int = 1440) -> List[Dict]:
        """Return the last `limit` readings for a machine regardless of timestamp.

        This is a fallback for datasets that are historic (older than the
        requested days window) so the forecast endpoint can still operate on
        the most recent available data.
        """
        cursor = self.sensor_readings.find(
            {'machine_id': machine_id},
            sort=[('timestamp', -1)]
        ).limit(limit)

        docs = await cursor.to_list(length=limit)
        # Return in chronological order (oldest first)
        return list(reversed(docs))
    
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
        forecast_minutes: int,
        forecast_data: List[Dict]
    ):
        """Save forecast results (now storing minute-based horizon)

        Backwards compatible field name: previously this stored forecast_days
        — now we store forecast_minutes so callers and consumers can clearly
        distinguish minute-based forecasts.
        """
        document = {
            'created_at': datetime.now(timezone.utc),
            'machine_id': machine_id,
            'forecast_minutes': forecast_minutes,
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
