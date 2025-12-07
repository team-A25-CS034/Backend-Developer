import os
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

_gemini_model = None

def configure_gemini():
    global _gemini_model
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        print(f"❌ Gemini Error: {e}")

def analyze_forecast_data(forecast_data_list):
    """
    Menganalisa raw data list menjadi ringkasan tren.
    """
    if not forecast_data_list:
        return "Tidak ada data forecast tersedia."

    df = pd.DataFrame(forecast_data_list)
    
    summary = {}
    for col in df.columns:
        start_val = df[col].iloc[0]
        end_val = df[col].iloc[-1]
        delta = end_val - start_val
        trend_str = f"{round(start_val, 1)} -> {round(end_val, 1)} ({'+' if delta>0 else ''}{round(delta, 1)})"
        summary[col] = trend_str
    
    return summary

def explain_with_gemini(user_query: str, context_data: dict, label_key: str = "semua fitur"):
    if _gemini_model is None:
        configure_gemini()
    
    if _gemini_model is None:
        return "Layanan AI tidak tersedia."

    try:
        machine_id = context_data.get('machine_id', 'Unknown Machine')
        forecast_list = context_data.get('forecast_data', [])
        
        data_summary = analyze_forecast_data(forecast_list)
        
        prompt = f"""
        Bertindaklah sebagai Senior Engineer yang sedang melaporkan status kritis mesin.
        
        DATA TELEMETRI (Prediksi 60 menit):
        - ID Mesin: {machine_id}
        - Perubahan Data: {data_summary}
        
        PERTANYAAN USER: "{user_query}"
        FOKUS ANALISIS: "{label_key}"

        INSTRUKSI JAWABAN (STRICT):
        1. DILARANG menggunakan kata sapaan (Halo, Tentu, Baik, dll).
        2. WAJIB memulai kalimat pertama dengan variasi: "Berdasarkan data prediksi...", "Data sensor menunjukkan...", atau "Analisis telemetri mengindikasikan...".
        3. Jawaban harus dalam format LIST (Poin-poin) yang singkat dan padat.
        4. SETIAP poin analisis harus menyertakan ANGKA SPESIFIK dari data.
        5. Pisahkan antara "Analisis Masalah" dan "Rekomendasi Teknis".
        
        Contoh Format Output yang diinginkan:
        Berdasarkan data prediksi, mesin mengalami anomali beban kritis:
        * **Analisis:**
            * Tool Wear melonjak drastis dari X ke Y dalam 60 menit.
            * Suhu turun sebesar Z, mengindikasikan hilangnya gesekan operasional.
        * **Rekomendasi:**
            * Lakukan Emergency Stop segera.
            * Ganti komponen mata pisau.
        """

        response = _gemini_model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print(f"Generate Error: {e}")
        return "Gagal melakukan analisis data."