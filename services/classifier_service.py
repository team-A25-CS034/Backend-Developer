import tensorflow as tf
import pickle
import joblib
import os
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.stem import WordNetLemmatizer

_stemmer = None
try:
    nltk.data.find('corpora/wordnet.zip')
    _stemmer = WordNetLemmatizer()
except LookupError:
    try:
        nltk.download('wordnet', quiet=True)
        _stemmer = WordNetLemmatizer()
    except Exception as e:
        print(f"NLTK Error: {e}. Lemmatization will be skipped.")

_model = None
_tokenizer = None
_int_to_label_map = {}
_max_len = 200

def load_classifier_resources(model_dir: str):
    global _model, _tokenizer, _int_to_label_map
    
    try:
        model_path = os.path.join(model_dir, 'model.keras')
        _model = tf.keras.models.load_model(model_path)

        tokenizer_path = os.path.join(model_dir, 'tokenizer.pickle')
        with open(tokenizer_path, 'rb') as handle:
            _tokenizer = pickle.load(handle)

        le_path = os.path.join(model_dir, 'label_encoder.joblib')
        encoder = joblib.load(le_path)
        
        _int_to_label_map = {}
        
        if hasattr(encoder, 'mapping'): 
            for col_map in encoder.mapping:
                mapping_series = col_map['mapping'] # Ini Series: Label -> Int
                for label, idx in mapping_series.items():
                    try:
                        idx_int = int(idx)
                        _int_to_label_map[idx_int] = str(label)
                    except:
                        continue
                        
        elif hasattr(encoder, 'classes_'):
            for idx, label in enumerate(encoder.classes_):
                _int_to_label_map[int(idx)] = str(label)
        
        print(f"Prompt Classifier loaded! (Mapped {len(_int_to_label_map)} labels)")

    except Exception as e:
        print(f"Error loading Classifier: {e}")

def _preprocess_text(text: str) -> str:
    try:
        text = str(text).lower()
        text = re.sub(r'[^a-z ]', '', text)
        text = re.sub(r'(^|\s)[a-z]\b', '', text)
        text = re.sub(r'\s+', ' ', text) 
        text = text.strip()
        
        if _stemmer:
            words = text.split()
            words = [_stemmer.lemmatize(word) for word in words]
            text = ' '.join(words)
            
        return text
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return text.lower().strip()

def predict_prompt_type(text: str):
    if _model is None:
        return {"label": "Error: Model not loaded", "confidence": 0.0}

    cleaned_text = _preprocess_text(text)
    
    sequences = _tokenizer.texts_to_sequences([cleaned_text])
    
    padded = pad_sequences(sequences, maxlen=_max_len, padding='post', truncating='post')
    
    prediction = _model.predict(padded, verbose=0)
    
    predicted_index = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))
    
    predicted_label = _int_to_label_map.get(predicted_index)
    
    if predicted_label is None:
        predicted_label = _int_to_label_map.get(predicted_index + 1)
        
    if predicted_label is None:
        predicted_label = "unknown"

    return {
        "label": predicted_label,
        "confidence": confidence
    }