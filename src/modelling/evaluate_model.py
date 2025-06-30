import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Asumsi struktur dataset mirip dengan yang digunakan untuk training
class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None): # Labels bersifat opsional, mungkin tidak ada untuk data inferensi murni
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        if self.labels is not None:
            return len(self.labels)
        # Jika tidak ada label, panjang berdasarkan input_ids (asumsi semua entri encoding sama panjangnya)
        return len(self.encodings['input_ids'])


def load_model_and_tokenizer(model_path):
    """Memuat model dan tokenizer dari path yang diberikan."""
    if not os.path.exists(model_path):
        logger.error(f"Direktori model tidak ditemukan di: {model_path}")
        return None, None
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info(f"Model dan tokenizer berhasil dimuat dari {model_path}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error saat memuat model atau tokenizer: {e}")
        return None, None

def prepare_evaluation_data(texts, tokenizer, labels=None, max_length=128):
    """Mempersiapkan data untuk evaluasi (tokenisasi dan pembuatan dataset)."""
    if not tokenizer:
        logger.error("Tokenizer tidak tersedia.")
        return None
    if not texts:
        logger.error("Tidak ada teks untuk dievaluasi.")
        return None

    try:
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt" if labels is None else None)
        eval_dataset = EvaluationDataset(encodings, labels)
        return eval_dataset
    except Exception as e:
        logger.error(f"Error saat mempersiapkan data evaluasi: {e}")
        return None

def predict(model, eval_dataset, device=None):
    """Melakukan prediksi menggunakan model pada dataset evaluasi."""
    if not model or not eval_dataset:
        logger.error("Model atau dataset evaluasi tidak valid.")
        return None, None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval() # Set model ke mode evaluasi

    predictions = []
    raw_outputs = [] # Untuk menyimpan logits jika diperlukan

    # Buat DataLoader untuk batching (opsional tapi baik untuk data besar)
    # Untuk kesederhanaan, jika eval_dataset kecil, kita bisa iterasi langsung
    # Namun, Trainer biasanya menggunakan DataLoader, jadi kita tiru itu.

    # Jika kita tidak menggunakan Trainer untuk prediksi, kita harus manual batch dan inferensi
    # Untuk contoh ini, kita asumsikan eval_dataset adalah hasil dari tokenizer yang sudah return_tensors="pt"
    # atau kita akan buat DataLoader sederhana.

    # Untuk implementasi yang lebih mudah dan konsisten dengan Trainer,
    # kita bisa menggunakan Trainer.predict() jika memungkinkan.
    # Namun, di sini kita akan coba implementasi manual untuk pemahaman.

    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=8) # Batch size bisa disesuaikan

    with torch.no_grad(): # Tidak perlu menghitung gradien saat evaluasi
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            raw_outputs.extend(logits.cpu().numpy())
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())

    return np.array(predictions), np.array(raw_outputs)


def compute_metrics(labels, predictions):
    """Menghitung metrik evaluasi."""
    if labels is None or predictions is None:
        logger.error("Label atau prediksi tidak valid untuk perhitungan metrik.")
        return None
    if len(labels) != len(predictions):
        logger.error(f"Jumlah label ({len(labels)}) dan prediksi ({len(predictions)}) tidak cocok.")
        return None

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    acc = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)

    metrics = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm.tolist() # Konversi ke list agar bisa di-serialize JSON
    }
    logger.info(f"Metrik Evaluasi: {metrics}")
    return metrics

def save_evaluation_results(results, output_path):
    """Menyimpan hasil evaluasi ke file JSON."""
    if results is None:
        logger.warn("Tidak ada hasil evaluasi untuk disimpan.")
        return
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"Hasil evaluasi disimpan di: {output_path}")
    except Exception as e:
        logger.error(f"Error saat menyimpan hasil evaluasi: {e}")

def evaluate_model(model_path, eval_data_path_or_texts, text_column='text', label_column='label',
                   output_file="evaluation_results.json", is_file=True):
    """
    Pipeline lengkap untuk evaluasi model.
    eval_data_path_or_texts: Bisa berupa path ke file CSV atau list of strings.
    is_file: True jika eval_data_path_or_texts adalah path file, False jika list of strings.
    """
    model, tokenizer = load_model_and_tokenizer(model_path)
    if not model or not tokenizer:
        return None

    texts_to_eval = []
    labels_to_eval = None

    if is_file:
        if not os.path.exists(eval_data_path_or_texts):
            logger.error(f"File data evaluasi tidak ditemukan: {eval_data_path_or_texts}")
            return None
        try:
            eval_df = pd.read_csv(eval_data_path_or_texts)
            if text_column not in eval_df.columns:
                logger.error(f"Kolom teks '{text_column}' tidak ditemukan di file evaluasi.")
                return None
            texts_to_eval = eval_df[text_column].astype(str).tolist()
            if label_column in eval_df.columns:
                labels_to_eval = eval_df[label_column].tolist()
            else:
                logger.warn(f"Kolom label '{label_column}' tidak ditemukan. Metrik tidak akan dihitung.")
        except Exception as e:
            logger.error(f"Error membaca file data evaluasi: {e}")
            return None
    else: # eval_data_path_or_texts adalah list of strings
        if not isinstance(eval_data_path_or_texts, list):
            logger.error("Data evaluasi (texts) harus berupa list jika is_file=False.")
            return None
        texts_to_eval = eval_data_path_or_texts
        # Untuk evaluasi teks mentah tanpa label, labels_to_eval akan tetap None

    if not texts_to_eval:
        logger.error("Tidak ada teks yang valid untuk dievaluasi.")
        return None

    eval_dataset = prepare_evaluation_data(texts_to_eval, tokenizer, labels=labels_to_eval)
    if not eval_dataset:
        return None

    predictions, raw_outputs = predict(model, eval_dataset)
    if predictions is None:
        return None

    results = {"predictions": predictions.tolist()} # Simpan prediksi mentah
    if raw_outputs is not None:
        results["raw_outputs_logits"] = raw_outputs.tolist()


    if labels_to_eval is not None: # Hanya hitung metrik jika ada label sebenarnya
        # Pastikan jumlah label sesuai dengan jumlah prediksi setelah dataset preparation
        # (misal, jika ada teks yang di-drop karena kosong setelah tokenisasi)
        # Untuk implementasi ini, kita asumsikan prepare_evaluation_data tidak drop data jika ada label
        # dan panjangnya akan sama dengan texts_to_eval.
        if len(labels_to_eval) != len(predictions):
            logger.error(f"Panjang label ({len(labels_to_eval)}) dan prediksi ({len(predictions)}) tidak cocok setelah persiapan data. Tidak dapat menghitung metrik.")
        else:
            metrics = compute_metrics(labels_to_eval, predictions)
            if metrics:
                results.update(metrics) # Gabungkan metrik ke hasil
    else:
        logger.info("Tidak ada label yang disediakan, hanya prediksi yang akan disimpan.")

    # Tentukan path output final
    # Jika model_path adalah models/bert_jawa_hate_speech, simpan di sana.
    # Jika tidak, simpan di direktori kerja.
    if model_path and os.path.isdir(model_path):
        final_output_path = os.path.join(model_path, output_file)
    else:
        final_output_path = output_file

    save_evaluation_results(results, final_output_path)
    return results


if __name__ == '__main__':
    # Contoh penggunaan:
    # 1. Pastikan model sudah dilatih dan tersimpan di MODEL_DIR
    #    Misalnya, setelah menjalankan train_model.py, akan ada di 'models/bert_jawa_hate_speech'
    # 2. Siapkan file data evaluasi (misalnya, 'eval_data.csv') dengan kolom 'text' dan 'label'

    MODEL_DIR = "models/bert_jawa_hate_speech" # Path ke model yang sudah dilatih

    # Buat file evaluasi dummy
    dummy_eval_data = {
        'text': [
            "iki conto teks sing apik tenan",
            "teks iki ngemot ujaran kebencian ringan",
            "rasah kakean omong!",
            "produk iki pancen nggilani tenan",
            "netral wae lah teks iki"
        ],
        'label': [0, 1, 2, 3, 0] # 0:Bukan, 1:Ringan, 2:Sedang, 3:Berat
    }
    dummy_eval_df = pd.DataFrame(dummy_eval_data)
    dummy_eval_file = "dummy_eval_data.csv"
    dummy_eval_df.to_csv(dummy_eval_file, index=False)

    if os.path.exists(MODEL_DIR):
        logger.info(f"Mengevaluasi model dari: {MODEL_DIR} dengan data: {dummy_eval_file}")
        evaluation_results = evaluate_model(MODEL_DIR, dummy_eval_file, text_column='text', label_column='label')
        if evaluation_results:
            logger.info("Evaluasi selesai.")
            # Hasil akan tersimpan di MODEL_DIR/evaluation_results.json
        else:
            logger.error("Evaluasi gagal.")
    else:
        logger.warn(f"Direktori model {MODEL_DIR} tidak ditemukan. Tidak dapat menjalankan contoh evaluasi.")
        logger.warn("Jalankan train_model.py terlebih dahulu untuk menghasilkan model.")

    # Contoh evaluasi teks mentah tanpa label
    raw_texts_to_evaluate = [
        "matur nuwun sanget nggih",
        "dasar wong ora duwe aturan!"
    ]
    if os.path.exists(MODEL_DIR):
        logger.info(f"Mengevaluasi teks mentah dengan model dari: {MODEL_DIR}")
        raw_text_results = evaluate_model(MODEL_DIR, raw_texts_to_evaluate, is_file=False, output_file="raw_text_predictions.json")
        if raw_text_results:
            logger.info("Evaluasi teks mentah selesai.")
            # Hasil prediksi akan tersimpan di MODEL_DIR/raw_text_predictions.json
        else:
            logger.error("Evaluasi teks mentah gagal.")

    # Hapus file dummy
    if os.path.exists(dummy_eval_file):
        os.remove(dummy_eval_file)
        logger.info(f"File evaluasi dummy {dummy_eval_file} dihapus.")
