import re
import string

def clean_text(text: str) -> str:
    """
    Membersihkan text input untuk inferensi model.
    """
    # Contoh logic (sesuaikan dengan training Anda):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Hapus simbol
    text = text.strip()
    return text