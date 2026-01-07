import re
import string
from transformers import AutoTokenizer

# Placeholder untuk daftar stopword Bahasa Jawa
# Idealnya, ini harus menjadi daftar yang lebih komprehensif dan dikelola secara eksternal
DEFAULT_JAWA_STOPWORDS = set([
    "lan", "ing", "karo", "saka", "kanggo", "menyang", "supaya", "amarga", "nanging",
    "utawa", "yen", "nalika", "dene", "marang", "dening", "iku", "iki", "ana", "ora",
    "uga", "wis", "durung", "tansah", "kabeh", "sawetara", "akeh", "sithik", "banget",
    "luwih", "paling", "mung", "wae", "bae", "kok", "to", "lho", "tha", "e", "ne",
    "ku", "mu", "ipun", "ingkang", "kula", "panjenengan", "kowe", "aku", "dheweke"
])

# Inisialisasi tokenizer dari Hugging Face
# Menggunakan model yang sama dengan yang akan di fine-tune untuk konsistensi
TOKENIZER_CHECKPOINT = "indobenchmark/indobert-base-p1"
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CHECKPOINT)
except Exception as e:
    print(f"Error loading tokenizer: {e}. Using basic split tokenizer as fallback.")
    tokenizer = None

def remove_punctuation(text):
    """Menghapus tanda baca dari teks."""
    if not isinstance(text, str):
        return ""
    return text.translate(str.maketrans('', '', string.punctuation))

def normalize_text(text):
    """
    Normalisasi teks:
    1. Mengubah ke huruf kecil.
    2. Menghapus tanda baca.
    3. Menghapus spasi berlebih.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_punctuation(text)
    text = re.sub(r'\s+', ' ', text).strip() # Menghapus spasi berlebih
    return text

def tokenize_text(text):
    """
    Melakukan tokenisasi teks.
    Menggunakan tokenizer dari Hugging Face jika tersedia, jika tidak menggunakan split sederhana.
    """
    if not isinstance(text, str):
        return []
    if tokenizer:
        return tokenizer.tokenize(text)
    return text.split()

def remove_stopwords(tokens, stopwords_list=None):
    """
    Menghapus stopwords dari daftar token.
    """
    if stopwords_list is None:
        stopwords_list = DEFAULT_JAWA_STOPWORDS
    if not isinstance(tokens, list):
        return []
    return [token for token in tokens if token.lower() not in stopwords_list]

def preprocess_text_advanced(text, custom_stopwords=None):
    """
    Pipeline lengkap untuk preprocessing teks lanjutan:
    1. Normalisasi (lowercase, remove punctuation, remove extra spaces).
    2. Tokenisasi.
    3. Penghapusan stopwords.
    """
    if not isinstance(text, str):
        return [], "" # Mengembalikan list token kosong dan string teks kosong

    normalized_text = normalize_text(text)
    tokens = tokenize_text(normalized_text)

    if custom_stopwords is not None:
        filtered_tokens = remove_stopwords(tokens, custom_stopwords)
    else:
        filtered_tokens = remove_stopwords(tokens)

    return filtered_tokens, " ".join(filtered_tokens)

if __name__ == '__main__':
    sample_text_1 = "Iki tuladha ukara basa Jawa, kanthi tandha wacan lan angka 123!"
    sample_text_2 = "Kula tresna sanget kaliyan Yogyakarta."
    sample_text_3 = "  REGANE  PANCEN  LARANG  TENAN  KOK."
    sample_text_4 = None # Contoh input tidak valid

    print(f"Teks Asli 1: {sample_text_1}")
    tokens_1, processed_text_1 = preprocess_text_advanced(sample_text_1)
    print(f"Tokens 1: {tokens_1}")
    print(f"Teks Diproses 1: {processed_text_1}\n")

    print(f"Teks Asli 2: {sample_text_2}")
    tokens_2, processed_text_2 = preprocess_text_advanced(sample_text_2)
    print(f"Tokens 2: {tokens_2}")
    print(f"Teks Diproses 2: {processed_text_2}\n")

    print(f"Teks Asli 3: {sample_text_3}")
    custom_stops = DEFAULT_JAWA_STOPWORDS.union({"regane", "pancen", "larang", "tenan"})
    tokens_3, processed_text_3 = preprocess_text_advanced(sample_text_3, custom_stopwords=custom_stops)
    print(f"Tokens 3 (custom stopwords): {tokens_3}")
    print(f"Teks Diproses 3 (custom stopwords): {processed_text_3}\n")

    print(f"Teks Asli 4: {sample_text_4}")
    tokens_4, processed_text_4 = preprocess_text_advanced(sample_text_4)
    print(f"Tokens 4: {tokens_4}")
    print(f"Teks Diproses 4: {processed_text_4}\n")

    # Contoh hanya normalisasi
    normalized_sample = normalize_text("Contoh Teks Dengan HURUF BESAR dan tanda baca!!!")
    print(f"Teks Normalisasi: '{normalized_sample}'")

    # Contoh hanya tokenisasi
    tokenized_sample = tokenize_text("iki teks sing arep di tokenisasi")
    print(f"Teks Tokenisasi: {tokenized_sample}")

    # Contoh hanya remove stopwords
    stopwords_removed_sample = remove_stopwords(["aku", "mangan", "sega", "lan", "krupuk"])
    print(f"Stopwords Dihapus: {stopwords_removed_sample}")
