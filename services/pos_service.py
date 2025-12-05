import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Global variables untuk resource model
_model = None
_word_to_int = None
_int_to_label = None
_max_len = 50  # Default value, nanti di-override saat load

def load_pos_resources(model_dir: str):
    """
    Memuat model, mapping json, dan konfigurasi lainnya ke memori.
    Dipanggil sekali saat startup aplikasi.
    """
    global _model, _word_to_int, _int_to_label, _max_len

    print("⏳ Loading POS Model resources...")
    
    try:
        # 1. Load Model Keras
        model_path = os.path.join(model_dir, 'POS.keras')
        _model = tf.keras.models.load_model(model_path)
        
        # 2. Load Word Mapping (Word -> Int)
        with open(os.path.join(model_dir, 'word_to_int.json'), 'r') as f:
            _word_to_int = json.load(f)
            
        # 3. Load Label Mapping (Int -> Label)
        # Note: Key di JSON biasanya string, kita perlu convert key jadi int
        with open(os.path.join(model_dir, 'int_to_label.json'), 'r') as f:
            raw_labels = json.load(f)
            _int_to_label = {int(k): v for k, v in raw_labels.items()}

        # 4. Load Max Len (jika ada file txt, jika tidak pakai default)
        max_len_path = os.path.join(model_dir, 'max_len.txt')
        if os.path.exists(max_len_path):
            with open(max_len_path, 'r') as f:
                _max_len = int(f.read().strip())
        
        print("✅ POS Model loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading POS resources: {e}")
        raise e

def predict_pos(text: str):
    """
    Melakukan inferensi POS Tagging pada teks input.
    """
    if _model is None:
        raise RuntimeError("POS Model belum dimuat. Pastikan load_pos_resources dipanggil.")

    # 1. Preprocessing (Logic sama seperti di inference.ipynb)
    # Lowercase dan split
    original_words = text.strip().lower().split()
    
    # Map words to integers (Handle unknown words dengan 'PAD' atau key khusus jika ada)
    # Asumsi: Menggunakan 'PAD' atau 0 untuk kata tidak dikenal, sesuaikan dengan training Anda
    pad_token = _word_to_int.get('PAD', 0) 
    
    int_sequence = [
        _word_to_int.get(word, pad_token) for word in original_words
    ]
    
    # Padding
    padded_sequence = pad_sequences(
        [int_sequence], 
        maxlen=_max_len, 
        padding='post', 
        value=pad_token
    )
    
    # 2. Prediction
    predicted_probs = _model.predict(padded_sequence, verbose=0)
    predicted_labels_int = np.argmax(predicted_probs, axis=-1)[0]
    
    # 3. Decoding (Int -> Label)
    # Hanya ambil label sepanjang kalimat asli (buang padding)
    original_len = len(original_words)
    result_labels = []
    
    for i in range(min(original_len, _max_len)):
        idx = predicted_labels_int[i]
        label = _int_to_label.get(idx, "O") # Default ke 'O' jika error
        result_labels.append(label)
        
    # Format output: Gabungkan kata dengan labelnya
    response = []
    for word, label in zip(original_words, result_labels):
        response.append({"word": word, "entity": label})
        
    return response