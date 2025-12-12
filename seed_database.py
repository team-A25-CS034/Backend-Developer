import pandas as pd
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'machine_monitoring_db')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'machine_monitoring')

# Mapping agar data konsisten dengan kode backend (snake_case)
COLUMN_RENAME_MAP = {
    "Product ID": "machine_id",  # KUNCI UTAMA PERBAIKAN
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
    if not MONGODB_URI:
        print("❌ Error: MONGODB_URI belum di-set di file .env")
        return

    print(f"🔌 Menghubungkan ke MongoDB: {MONGODB_URI} ...")
    
    try:
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DATABASE]
        collection = db[MONGODB_COLLECTION]
        
        # Opsional: Reset data lama agar tidak duplikat/campur aduk
        # collection.delete_many({}) 
        # print("⚠️ Data lama dihapus (Reset).")

        csv_file = 'dummy_sensor_data.csv'
        if not os.path.exists(csv_file):
            print(f"❌ File {csv_file} tidak ditemukan!")
            return

        df = pd.read_csv(csv_file)
        
        # --- PERBAIKAN: Rename Kolom ---
        df.rename(columns=COLUMN_RENAME_MAP, inplace=True)
        
        # Validasi sederhana: Pastikan machine_id ada
        if 'machine_id' not in df.columns:
            print(f"❌ Error: Kolom 'Product ID' atau 'productID' tidak ditemukan di CSV.")
            print(f"Kolom yang ada: {list(df.columns)}")
            return

        # Preprocessing Timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            print("⚠️ Kolom timestamp tidak ditemukan, membuat timestamp otomatis...")
            # Buat timestamp mundur (history) agar terlihat di grafik
            df['timestamp'] = [datetime.now() for _ in range(len(df))]

        data = df.to_dict('records')
        
        if data:
            collection.insert_many(data)
            print(f"✅ Berhasil memasukkan {len(data)} baris data! (Key utama: 'machine_id')")
        else:
            print("⚠️ File CSV kosong.")

    except Exception as e:
        print(f"❌ Terjadi kesalahan: {e}")
    finally:
        if 'client' in locals():
            client.close()
            print("🔌 Koneksi ditutup.")

if __name__ == "__main__":
    seed_data()