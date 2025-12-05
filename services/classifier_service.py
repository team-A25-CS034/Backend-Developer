import tensorflow as tf
import pickle
import joblib
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import fungsi cleaning dari file clean_raw_text.py
# Pastikan ada __init__.py kosong di folder services/ dan services/utils/
try:
    from .utils.clean_raw_text import clean_text
except ImportError:
    # Fallback jika struktur folder berbeda, sesuaikan import ini
    from clean_raw_text import clean_text

# Global Variables
_model = None
_tokenizer = None
_label_encoder = None
_max_len = 50 # Sesuaikan dengan MAX_LEN saat training (cek model.ipynb jika ragu)

def load_classifier_resources(model_dir: str):
    """
    Memuat Model, Tokenizer, dan LabelEncoder ke memori.
    """
    global _model, _tokenizer, _label_encoder
    
    print("⏳ Loading Prompt Classifier resources...")
    try:
        # 1. Load Model Keras
        model_path = os.path.join(model_dir, 'model.keras')
        _model = tf.keras.models.load_model(model_path)

        # 2. Load Tokenizer (Pickle)
        tokenizer_path = os.path.join(model_dir, 'tokenizer.pickle')
        with open(tokenizer_path, 'rb') as handle:
            _tokenizer = pickle.load(handle)

        # 3. Load Label Encoder (Joblib)
        le_path = os.path.join(model_dir, 'label_encoder.joblib')
        _label_encoder = joblib.load(le_path)

        print("✅ Prompt Classifier loaded successfully!")
    
    except Exception as e:
        print(f"❌ Error loading Classifier: {e}")
        # Kita tidak raise error agar aplikasi utama tetap jalan walau model ini gagal
        pass

def predict_prompt_type(text: str):
    """
    Menerima teks raw, membersihkan, dan mengembalikan kategori prompt.
    """
    if _model is None:
        return {"label": "Error: Model not loaded", "confidence": 0.0}

    # 1. Cleaning
    try:
        cleaned_text = clean_text(text)
    except Exception:
        cleaned_text = text # Fallback jika cleaning gagal
    
    # 2. Tokenizing
    sequences = _tokenizer.texts_to_sequences([cleaned_text])
    
    # 3. Padding
    padded = pad_sequences(sequences, maxlen=_max_len, padding='post', truncating='post')
    
    # 4. Predict
    prediction = _model.predict(padded, verbose=0)
    
    # 5. Decode Label
    predicted_index = np.argmax(prediction, axis=1)
    
    try:
        # OPSI 1: Jika menggunakan Scikit-Learn LabelEncoder (Standard)
        predicted_label = _label_encoder.inverse_transform(predicted_index)[0]
    except Exception:
        # OPSI 2: Jika menggunakan Category Encoders (Butuh DataFrame 'Label')
        # Error 'KeyError: Label' biasanya diselesaikan di sini
        try:
            df_pred = pd.DataFrame(predicted_index, columns=['Label'])
            decoded_df = _label_encoder.inverse_transform(df_pred)
            predicted_label = decoded_df['Label'].iloc[0]
        except Exception as e:
            # Fallback terakhir jika gagal total
            print(f"❌ Label decoding error: {e}")
            predicted_label = str(predicted_index[0])

    confidence = float(np.max(prediction))

    return {
        "label": predicted_label,
        "confidence": confidence
    }