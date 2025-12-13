import pandas as pd
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from datetime import datetime, timedelta
from services.models_db import Base, SensorReading

load_dotenv()

# Ambil URL Database & Bersihkan driver async jika ada
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and '+asyncpg' in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace('+asyncpg', '')

# Mapping agar data CSV sesuai dengan nama kolom di Database (snake_case)
COLUMN_RENAME_MAP = {
    "Product ID": "machine_id",
    "productID": "machine_id",
    "UDI": "udi",
    "Type": "machine_type",
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure"
}

def seed_data():
    if not DATABASE_URL:
        print("❌ Error: DATABASE_URL belum di-set di file .env")
        return

    print(f"🔌 Menghubungkan ke PostgreSQL...")
    
    try:
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        print("🗑️  Menghapus tabel lama (Reset Schema)...")
        SensorReading.__table__.drop(engine, checkfirst=True)
        
        # 1. Buat Tabel Baru
        print("🛠️  Membuat ulang tabel database...")
        Base.metadata.create_all(engine)

        # 2. Baca CSV
        csv_file = 'dummy_sensor_data.csv'
        if not os.path.exists(csv_file):
            print(f"❌ File {csv_file} tidak ditemukan!")
            return

        print(f"📂 Membaca file {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # 3. Rename Kolom
        df.rename(columns=COLUMN_RENAME_MAP, inplace=True)
        
        # Validasi kolom penting
        if 'machine_id' not in df.columns:
            print(f"❌ Error: Kolom 'Product ID' tidak ditemukan di CSV.")
            return

        # 4. Proses Timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            print("⚠️ Kolom timestamp tidak ditemukan, membuat data dummy history...")
            base_time = datetime.now()
            # Buat list timestamp
            timestamps = [base_time - timedelta(minutes=i) for i in range(len(df))]
            df['timestamp'] = list(reversed(timestamps))

        # 5. Konversi DataFrame ke List of Objects
        print("🔄 Mengkonversi data ke format Database...")
        readings_to_insert = []
        
        current_time = datetime.now()

        for _, row in df.iterrows():
            def safe_float(val):
                try:
                    return float(val)
                except:
                    return 0.0

            reading = SensorReading(
                machine_id=str(row.get('machine_id')),
                timestamp=row.get('timestamp'),
                air_temperature=safe_float(row.get('air_temperature')),
                process_temperature=safe_float(row.get('process_temperature')),
                rotational_speed=safe_float(row.get('rotational_speed')),
                torque=safe_float(row.get('torque')),
                tool_wear=safe_float(row.get('tool_wear')),
                machine_type=str(row.get('machine_type')),
                uploaded_at=current_time
            )
            readings_to_insert.append(reading)

        # 6. Bulk Insert
        if readings_to_insert:
            print(f"🚀 Memulai insert {len(readings_to_insert)} baris data...")
            session.add_all(readings_to_insert)
            session.commit()
            print(f"✅ Berhasil memasukkan {len(readings_to_insert)} data ke PostgreSQL!")
        else:
            print("⚠️ File CSV kosong atau tidak ada data valid.")

    except Exception as e:
        if 'session' in locals():
            session.rollback()
        print(f"❌ Terjadi kesalahan: {e}")
    finally:
        if 'session' in locals():
            session.close()
        print("🔌 Koneksi ditutup.")

if __name__ == "__main__":
    seed_data()