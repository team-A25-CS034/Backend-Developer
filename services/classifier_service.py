import tensorflow as tf
import pickle
import joblib
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

try:
    from .utils.clean_raw_text import clean_text
except ImportError:
    from clean_raw_text import clean_text

_model = None
_tokenizer = None
_label_encoder = None
_max_len = 50 

def load_classifier_resources(model_dir: str):
    global _model, _tokenizer, _label_encoder
    
    try:
        model_path = os.path.join(model_dir, 'model.keras')
        _model = tf.keras.models.load_model(model_path)

        tokenizer_path = os.path.join(model_dir, 'tokenizer.pickle')
        with open(tokenizer_path, 'rb') as handle:
            _tokenizer = pickle.load(handle)

        le_path = os.path.join(model_dir, 'label_encoder.joblib')
        _label_encoder = joblib.load(le_path)

    except Exception as e:
        print(f"Error loading Classifier: {e}")
        pass

def predict_prompt_type(text: str):
    """
    Menerima teks raw, membersihkan, dan mengembalikan kategori prompt.
    """
    if _model is None:
        return {"label": "Error: Model not loaded", "confidence": 0.0}

    try:
        cleaned_text = clean_text(text)
    except Exception:
        cleaned_text = text 
    
    sequences = _tokenizer.texts_to_sequences([cleaned_text])
    
    padded = pad_sequences(sequences, maxlen=_max_len, padding='post', truncating='post')
    
    prediction = _model.predict(padded, verbose=0)
    
    predicted_index = np.argmax(prediction, axis=1)
    
    try:
        predicted_label = _label_encoder.inverse_transform(predicted_index)[0]
    except Exception:
        try:
            df_pred = pd.DataFrame(predicted_index, columns=['Label'])
            decoded_df = _label_encoder.inverse_transform(df_pred)
            predicted_label = decoded_df['Label'].iloc[0]
        except Exception as e:
            print(f"Label decoding error: {e}")
            predicted_label = str(predicted_index[0])

    confidence = float(np.max(prediction))

    return {
        "label": predicted_label,
        "confidence": confidence
    }