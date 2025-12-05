import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

_model = None
_word_to_int = None
_int_to_label = None
_max_len = 50 

def load_pos_resources(model_dir: str):
    global _model, _word_to_int, _int_to_label, _max_len

    try:
        model_path = os.path.join(model_dir, 'POS.keras')
        _model = tf.keras.models.load_model(model_path)
        
        with open(os.path.join(model_dir, 'word_to_int.json'), 'r') as f:
            _word_to_int = json.load(f)
            
        with open(os.path.join(model_dir, 'int_to_label.json'), 'r') as f:
            raw_labels = json.load(f)
            _int_to_label = {int(k): v for k, v in raw_labels.items()}

        max_len_path = os.path.join(model_dir, 'max_len.txt')
        if os.path.exists(max_len_path):
            with open(max_len_path, 'r') as f:
                _max_len = int(f.read().strip())
        
    except Exception as e:
        print(f"Error loading POS resources: {e}")
        raise e

def predict_pos(text: str):
    if _model is None:
        raise RuntimeError("POS Model belum dimuat. Pastikan load_pos_resources dipanggil.")

    original_words = text.strip().lower().split()
    
    pad_token = _word_to_int.get('PAD', 0) 
    
    int_sequence = [
        _word_to_int.get(word, pad_token) for word in original_words
    ]
    
    padded_sequence = pad_sequences(
        [int_sequence], 
        maxlen=_max_len, 
        padding='post', 
        value=pad_token
    )
    
    predicted_probs = _model.predict(padded_sequence, verbose=0)
    predicted_labels_int = np.argmax(predicted_probs, axis=-1)[0]
    
    original_len = len(original_words)
    result_labels = []
    
    for i in range(min(original_len, _max_len)):
        idx = predicted_labels_int[i]
        label = _int_to_label.get(idx, "O")
        result_labels.append(label)
        
    response = []
    for word, label in zip(original_words, result_labels):
        response.append({"word": word, "entity": label})
        
    return response