from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, desc, func
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Optional
import os
from datetime import datetime, timedelta, timezone
from services.models_db import Base, SensorReading, Prediction, Forecast, Ticket

class PostgresService:
    def __init__(self, db_url: str):
        # Async Engine untuk PostgreSQL
        self.engine = create_async_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def init_models(self):
        """Membuat tabel jika belum ada (Auto Migration sederhana)"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self):
        await self.engine.dispose()

    async def get_latest_reading(self, machine_id: str) -> Optional[Dict]:
        async with self.SessionLocal() as session:
            result = await session.execute(
                select(SensorReading)
                .where(SensorReading.machine_id == machine_id)
                .order_by(desc(SensorReading.timestamp))
                .limit(1)
            )
            item = result.scalars().first()
            return item.to_dict() if item else None

    async def get_readings_range(self, machine_id: str = None, days: int = 30) -> List[Dict]:
        async with self.SessionLocal() as session:
            query = select(SensorReading)
            if machine_id:
                query = query.where(SensorReading.machine_id == machine_id)
            
            # Bisa tambah filter tanggal disini jika mau
            # cutoff = datetime.utcnow() - timedelta(days=days)
            # query = query.where(SensorReading.timestamp >= cutoff)
            
            result = await session.execute(query)
            readings = result.scalars().all()
            return [r.to_dict() for r in readings]

    async def get_last_readings(self, limit: int = 1440) -> List[Dict]:
        async with self.SessionLocal() as session:
            # Ambil data terbaru, lalu reverse urutannya agar kronologis
            result = await session.execute(
                select(SensorReading)
                .order_by(desc(SensorReading.id)) # Gunakan ID atau Timestamp
                .limit(limit)
            )
            readings = result.scalars().all()
            # Kita reverse agar urutannya dari lama -> baru (timeline chart)
            return [r.to_dict() for r in reversed(readings)]

    async def save_prediction(self, machine_id: str, prediction_label: str, 
                            prediction_numeric: int, probabilities: list, input_data: dict):
        async with self.SessionLocal() as session:
            new_pred = Prediction(
                machine_id=machine_id,
                prediction_label=prediction_label,
                prediction_numeric=prediction_numeric,
                probabilities=probabilities,
                input_data=input_data,
                timestamp=datetime.utcnow()
            )
            session.add(new_pred)
            await session.commit()
            return str(new_pred.id)

    async def save_forecast(self, machine_id: str, forecast_minutes: int, forecast_data: List[Dict]):
        async with self.SessionLocal() as session:
            new_forecast = Forecast(
                machine_id=machine_id,
                forecast_minutes=forecast_minutes,
                forecast_data=forecast_data,
                created_at=datetime.utcnow()
            )
            session.add(new_forecast)
            await session.commit()

    async def get_latest_forecast(self, machine_id: str) -> Optional[Dict]:
        async with self.SessionLocal() as session:
            result = await session.execute(
                select(Forecast)
                .where(Forecast.machine_id == machine_id)
                .order_by(desc(Forecast.created_at))
                .limit(1)
            )
            item = result.scalars().first()
            if item:
                return {
                    "machine_id": item.machine_id,
                    "forecast_minutes": item.forecast_minutes,
                    "forecast_data": item.forecast_data,
                    "created_at": item.created_at
                }
            return None

    async def get_machine_statistics(self, machine_id: str, days: int = 30) -> Dict:
        """Aggregation menggunakan SQL Query"""
        async with self.SessionLocal() as session:
            stmt = select(
                func.avg(SensorReading.air_temperature).label("avg_air"),
                func.avg(SensorReading.process_temperature).label("avg_proc"),
                func.avg(SensorReading.rotational_speed).label("avg_rot"),
                func.avg(SensorReading.torque).label("avg_tor"),
                func.max(SensorReading.tool_wear).label("max_wear"),
                func.min(SensorReading.tool_wear).label("min_wear"),
                func.count(SensorReading.id).label("count")
            ).where(SensorReading.machine_id == machine_id)

            result = await session.execute(stmt)
            stats = result.first()
            
            if stats and stats.count > 0:
                return {
                    "avg_air_temp": stats.avg_air,
                    "avg_process_temp": stats.avg_proc,
                    "avg_rotational_speed": stats.avg_rot,
                    "avg_torque": stats.avg_tor,
                    "max_tool_wear": stats.max_wear,
                    "min_tool_wear": stats.min_wear,
                    "count": stats.count
                }
            return {}

    async def bulk_insert_readings(self, readings: List[Dict]) -> int:
        async with self.SessionLocal() as session:
            # Convert list dict ke List Object ORM
            orm_objects = []
            for r in readings:
                orm_objects.append(SensorReading(
                    machine_id=r.get("machine_id") or r.get("Product ID"),
                    timestamp=r.get("timestamp"),
                    air_temperature=r.get("air_temperature"),
                    process_temperature=r.get("process_temperature"),
                    rotational_speed=r.get("rotational_speed"),
                    torque=r.get("torque"),
                    tool_wear=r.get("tool_wear"),
                    machine_type=r.get("machine_type") or r.get("Type")
                ))
            
            session.add_all(orm_objects)
            await session.commit()
            return len(orm_objects)
    
    async def get_distinct_machine_ids(self):
        async with self.SessionLocal() as session:
            result = await session.execute(select(SensorReading.machine_id).distinct())
            return result.scalars().all()

    # --- TICKET METHODS ---
    async def create_ticket(self, ticket_data: Dict) -> str:
        async with self.SessionLocal() as session:
            new_ticket = Ticket(
                machine_name=ticket_data.get("machine_name"),
                priority=ticket_data.get("priority"),
                issue_summary=ticket_data.get("issue_summary"),
                suggested_fix=ticket_data.get("suggested_fix"),
                estimated_time=ticket_data.get("estimated_time_to_address"),
                status=ticket_data.get("status", "Open"),
                created_by=ticket_data.get("created_by"),
                source=ticket_data.get("source"),
                sensor_snapshot=ticket_data.get("sensor_snapshot")
            )
            session.add(new_ticket)
            await session.commit()
            return str(new_ticket.id)

    async def get_tickets(self, limit: int = 50, status: str = None) -> List[Dict]:
        async with self.SessionLocal() as session:
            query = select(Ticket).order_by(desc(Ticket.created_at)).limit(limit)
            if status:
                query = query.where(Ticket.status == status)
            
            result = await session.execute(query)
            tickets = result.scalars().all()
            
            # Serialize manual
            output = []
            for t in tickets:
                output.append({
                    "_id": str(t.id),
                    "machine_name": t.machine_name,
                    "priority": t.priority,
                    "issue_summary": t.issue_summary,
                    "suggested_fix": t.suggested_fix,
                    "status": t.status,
                    "created_at": t.created_at.isoformat() if t.created_at else None
                })
            return output