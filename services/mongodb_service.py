from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import os

class MongoDBClient:
    def __init__(self, mongo_uri: str, database_name: str = None, collection_name: str = None, ticket_uri: str = None):
        import os
        if not database_name:
            database_name = os.getenv('MONGODB_DATABASE', 'machine_monitoring_db')
        if not collection_name:
            collection_name = os.getenv('MONGODB_COLLECTION', 'machine_monitoring')

        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[database_name]
        self.sensor_readings = self.db[collection_name]
        self.predictions = self.db['predictions']
        self.forecasts = self.db['forecasts']
        
        if ticket_uri:
            print(f"🔌 Connecting to Dedicated Ticket Database...")
            self.ticket_client = AsyncIOMotorClient(ticket_uri)
            self.ticket_db = self.ticket_client['maintenance_db'] 
            self.tickets = self.ticket_db['maintenance']
        else:
            self.tickets = self.db['maintenance']
    
    def close(self):
        self.client.close()
        if hasattr(self, 'ticket_client'):
            self.ticket_client.close()
    
    async def get_latest_reading(self, machine_id: str) -> Optional[Dict]:
        """Get the latest sensor reading for a machine"""
        reading = await self.sensor_readings.find_one(
            {'Machine ID': machine_id},  # <-- FIX KEY
            sort=[('timestamp', -1)]
        )
        return reading
    
    async def get_readings_range(self, machine_id: str = None, days: int = 30) -> List[Dict]:
        """Ambil data historis"""
        filter_query = {}
        if machine_id:
            filter_query['Machine ID'] = machine_id # <-- FIX KEY
            
        # Optional: Filter timestamp jika ada
        # cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        # filter_query['timestamp'] = {'$gte': cutoff_date}
        
        cursor = self.sensor_readings.find(filter_query)
        readings = await cursor.to_list(length=None)
        return readings

    async def get_last_readings(self, limit: int = 1440) -> List[Dict]:
        """
        Ambil sejumlah data terakhir tanpa mempedulikan timestamp (Fallback untuk data statis/lama).
        Digunakan oleh endpoint /timeline.
        """
        cursor = self.sensor_readings.find({}).sort('_id', -1).limit(limit)
        docs = await cursor.to_list(length=limit)
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
    
    async def save_forecast(self, machine_id: str, forecast_minutes: int, forecast_data: List[Dict]):
        """Simpan hasil forecast ke DB"""
        document = {
            'machine_id': machine_id,
            'created_at': datetime.now(timezone.utc),
            'forecast_minutes': forecast_minutes,
            'forecast_data': forecast_data
        }
        await self.forecasts.insert_one(document)
    
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
                    'Machine ID': machine_id, # <-- FIX KEY
                    # 'timestamp': {'$gte': cutoff_date} # Uncomment jika timestamp tersedia
                }
            },
            {
                # AGGREGATION MENGGUNAKAN NAMA KOLOM ASLI DB
                '$group': {
                    '_id': None,
                    'avg_air_temp': {'$avg': '$Air temperature [K]'},
                    'avg_process_temp': {'$avg': '$Process temperature [K]'},
                    'avg_rotational_speed': {'$avg': '$Rotational speed [rpm]'},
                    'avg_torque': {'$avg': '$Torque [Nm]'},
                    'max_tool_wear': {'$max': '$Tool wear [min]'},
                    'min_tool_wear': {'$min': '$Tool wear [min]'},
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
    
    async def bulk_insert_readings(self, readings: List[Dict]) -> int:
        """
        Menyimpan banyak data sensor sekaligus ke database.
        Mengembalikan jumlah data yang berhasil disimpan.
        """
        if not readings:
            return 0
            
        current_time = datetime.now(timezone.utc)
        for r in readings:
            if 'uploaded_at' not in r:
                r['uploaded_at'] = current_time
        
        result = await self.sensor_readings.insert_many(readings)
        return len(result.inserted_ids)
    
    async def create_ticket(self, ticket_data: Dict) -> str:
        ticket_data['created_at'] = datetime.now(timezone.utc)
        ticket_data['status'] = 'Open' 
        
        result = await self.tickets.insert_one(ticket_data)
        return str(result.inserted_id)

    async def get_tickets(self, limit: int = 50, status: str = None) -> List[Dict]:
        query = {}
        if status:
            query['status'] = status
            
        cursor = self.tickets.find(query).sort('created_at', -1).limit(limit)
        tickets = await cursor.to_list(length=limit)
        
        for t in tickets:
            t['_id'] = str(t['_id'])
            if 'created_at' in t:
                t['created_at'] = t['created_at'].isoformat()
        return tickets