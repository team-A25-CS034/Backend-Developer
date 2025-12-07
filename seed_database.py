import pandas as pd
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'machine_monitoring_db')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'machine_monitoring')

def seed_data():
    if not MONGODB_URI:
        print("❌ Error: MONGODB_URI belum di-set di file .env")
        return

    print(f"🔌 Menghubungkan ke MongoDB: {MONGODB_URI} ...")
    
    try:
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DATABASE]
        collection = db[MONGODB_COLLECTION]
        
        # Cek apakah data sudah ada
        count = collection.count_documents({})
        if count > 0:
            print(f"⚠️ Database sudah berisi {count} data. Skipping seed.")
            # Hapus baris return di bawah jika ingin menimpa/menambah data meski sudah ada
            return 

        # Baca file CSV
        csv_file = 'dummy_sensor_data.csv'
        if not os.path.exists(csv_file):
            print(f"❌ File {csv_file} tidak ditemukan!")
            return

        df = pd.read_csv(csv_file)
        
        # Preprocessing timestamp (PENTING: MongoDB butuh format datetime objek, bukan string)
        # Sesuaikan nama kolom 'Timestamp' atau 'timestamp' dengan isi CSV Anda
        if 'Timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Timestamp'])
            # Hapus kolom lama jika namanya beda agar rapi (opsional)
            if 'Timestamp' != 'timestamp': 
                del df['Timestamp']
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            # Jika tidak ada kolom timestamp, buat timestamp dummy
            print("⚠️ Kolom timestamp tidak ditemukan, membuat timestamp otomatis...")
            df['timestamp'] = [datetime.now() for _ in range(len(df))]

        # Konversi ke list of dict
        data = df.to_dict('records')
        
        # Insert ke MongoDB
        if data:
            collection.insert_many(data)
            print(f"✅ Berhasil memasukkan {len(data)} baris data ke koleksi '{MONGODB_COLLECTION}'!")
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