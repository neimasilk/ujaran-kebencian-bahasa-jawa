import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TOKENIZER_CHECKPOINT = "indobenchmark/indobert-base-p1"
MODEL_OUTPUT_DIR = "models/bert_jawa_hate_speech"
NUM_LABELS = 4 # Contoh: Bukan Ujaran Kebencian, Ringan, Sedang, Berat

# Muat tokenizer global
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CHECKPOINT)
except Exception as e:
    logger.error(f"Gagal memuat tokenizer: {e}. Beberapa fungsi mungkin tidak berjalan semestinya.")
    tokenizer = None

class JavaneseHateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_processed_data(processed_data_path):
    """
    Memuat data yang sudah diproses (misalnya, dari output text_preprocessing).
    Asumsi data adalah CSV dengan kolom 'processed_text' dan 'label'.
    Label diasumsikan sudah di-encode menjadi integer.
    """
    if not os.path.exists(processed_data_path):
        logger.error(f"File data yang diproses tidak ditemukan di: {processed_data_path}")
        return None
    try:
        df = pd.read_csv(processed_data_path)
        if 'processed_text' not in df.columns or 'label' not in df.columns:
            logger.error("Kolom 'processed_text' atau 'label' tidak ditemukan dalam data.")
            return None
        # Hapus baris dengan teks kosong setelah preprocessing jika ada
        df = df.dropna(subset=['processed_text'])
        df = df[df['processed_text'].str.strip() != '']
        return df
    except Exception as e:
        logger.error(f"Error saat memuat data yang diproses: {e}")
        return None

def prepare_datasets(df, text_column='processed_text', label_column='label', test_size=0.2):
    """
    Membagi data menjadi train dan validation, lalu melakukan tokenisasi.
    """
    if df is None or df.empty:
        logger.error("DataFrame input kosong atau None.")
        return None, None

    if not tokenizer:
        logger.error("Tokenizer tidak tersedia. Tidak dapat melanjutkan persiapan dataset.")
        return None, None

    try:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df[text_column].tolist(), df[label_column].tolist(), test_size=test_size, random_state=42
        )

        # Pastikan semua label ada dalam rentang yang valid (0 hingga NUM_LABELS-1)
        logger.debug(f"prepare_datasets: NUM_LABELS = {NUM_LABELS}")
        logger.debug(f"prepare_datasets: Original train_labels sample: {train_labels[:5]}")
        logger.debug(f"prepare_datasets: Original val_labels sample: {val_labels[:5]}")

        combined_labels = train_labels + val_labels
        invalid_labels = [lbl for lbl in combined_labels if not (0 <= lbl < NUM_LABELS)]
        if invalid_labels:
            logger.error(f"Label mengandung nilai di luar rentang yang diharapkan [0, {NUM_LABELS-1}]. Label tidak valid: {list(set(invalid_labels))[:5]}") # Tampilkan beberapa contoh label tidak valid
            return None, None

        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

        train_dataset = JavaneseHateSpeechDataset(train_encodings, train_labels)
        val_dataset = JavaneseHateSpeechDataset(val_encodings, val_labels)

        return train_dataset, val_dataset
    except Exception as e:
        logger.error(f"Error saat mempersiapkan dataset: {e}")
        return None, None

def train_model(train_dataset, val_dataset, model_output_dir=MODEL_OUTPUT_DIR, num_labels=NUM_LABELS,
                num_train_epochs=1, per_device_train_batch_size=8, per_device_eval_batch_size=8,
                learning_rate=5e-5, weight_decay=0.01, logging_steps=10, save_total_limit=2):
    """
    Melatih (fine-tune) model BERT untuk klasifikasi teks.
    """
    if train_dataset is None or val_dataset is None:
        logger.error("Train atau validation dataset tidak valid.")
        return None

    if not tokenizer: # Perlu tokenizer untuk model juga
        logger.error("Tokenizer tidak tersedia. Tidak dapat melanjutkan pelatihan.")
        return None

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            TOKENIZER_CHECKPOINT, # Bisa juga model checkpoint lain jika berbeda dengan tokenizer
            num_labels=num_labels
        )

        training_args = TrainingArguments(
            output_dir=model_output_dir,
            num_train_epochs=num_train_epochs, # Jumlah epoch kecil untuk contoh/tes
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            # evaluation_strategy="epoch", # Sementara dihapus untuk mengatasi TypeError
            # save_strategy="epoch",       # Sementara dihapus
            load_best_model_at_end=False, # Dinonaktifkan sementara karena butuh eval_strategy
            logging_steps=logging_steps,
            save_total_limit=save_total_limit,
            # report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        logger.info("Memulai pelatihan model...")
        trainer.train()
        logger.info("Pelatihan selesai.")

        # Simpan model dan tokenizer terakhir
        os.makedirs(model_output_dir, exist_ok=True)
        trainer.save_model(model_output_dir)
        tokenizer.save_pretrained(model_output_dir) # Simpan tokenizer juga
        logger.info(f"Model dan tokenizer disimpan di {model_output_dir}")

        return model_output_dir # Mengembalikan path ke model yang disimpan
    except Exception as e:
        logger.error(f"Error saat melatih model: {e}")
        # Cetak traceback untuk detail lebih lanjut jika diperlukan
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == '__main__':
    # Contoh penggunaan (membutuhkan file data dummy)
    # 1. Buat file CSV dummy 'dummy_processed_data.csv'
    # Kolom: processed_text,label
    # Contoh isi:
    # "iki teks proses siji",0
    # "teks loro kanggo conto",1
    # "telu papat limo enem pitu",2
    # "wolu songo sepuluh sewelas",3
    # "iki maneh teks proses siji",0
    # "teks loro liyane kanggo conto",1
    # "telu papat limo enem pitu maneh",2
    # "wolu songo sepuluh sewelas liyane",3
    # "siji loro telu papat",0
    # "limo enem pitu wolu",1
    # "songo sepuluh sewelas rolas",2
    # "telulas patbelas limolas nembelas",3

    dummy_data = {
        'processed_text': [
            "iki teks proses siji", "teks loro kanggo conto", "telu papat limo enem pitu", "wolu songo sepuluh sewelas",
            "iki maneh teks proses siji", "teks loro liyane kanggo conto", "telu papat limo enem pitu maneh", "wolu songo sepuluh sewelas liyane",
            "siji loro telu papat", "limo enem pitu wolu", "songo sepuluh sewelas rolas", "telulas patbelas limolas nembelas",
            "iki teks positif banget", "iki teks negatif tenan", "iki teks netral wae", "iki teks super positif",
            "iki teks super negatif", "iki teks cukup netral", "ora ono opo opo", "seneng banget aku",
            "sedih rasane atiku", "biasa wae ora ono sing spesial", "mantap jiwa", "elek banget iki barang"
        ],
        'label': [
            0, 1, 2, 3,
            0, 1, 2, 3,
            0, 1, 2, 3,
            0, 3, 1, 0, # positif, ujaran kebencian berat (contoh salah), ujaran kebencian ringan, positif
            3, 1, 0, 0, # ujaran kebencian berat (contoh salah), ringan, netral, netral
            1, 2, 0, 3  # ringan, sedang, netral, berat (contoh salah)
        ]
    }
    # Pastikan ada cukup sampel per kelas untuk stratifikasi jika digunakan (default train_test_split tidak stratify)
    # dan untuk evaluasi yang bermakna.
    # Untuk contoh ini, kita hanya punya 24 sampel. Batch size 8, jadi 3 step per epoch.

    dummy_df = pd.DataFrame(dummy_data)
    dummy_file_path = "dummy_processed_data.csv"
    dummy_df.to_csv(dummy_file_path, index=False)

    logger.info(f"Memuat data dummy dari {dummy_file_path}")
    df_loaded = load_processed_data(dummy_file_path)

    if df_loaded is not None:
        logger.info("Mempersiapkan dataset...")
        train_ds, val_ds = prepare_datasets(df_loaded)

        if train_ds and val_ds:
            logger.info(f"Jumlah data latih: {len(train_ds)}")
            logger.info(f"Jumlah data validasi: {len(val_ds)}")

            # Hapus direktori model output sebelumnya jika ada, untuk contoh ini
            # import shutil
            # if os.path.exists(MODEL_OUTPUT_DIR):
            #     shutil.rmtree(MODEL_OUTPUT_DIR)

            logger.info("Memulai contoh pelatihan dengan data dummy...")
            # Kurangi epoch dan batch size untuk pengujian cepat
            trained_model_path = train_model(
                train_ds, val_ds,
                num_train_epochs=1,
                per_device_train_batch_size=4, # Lebih kecil agar bisa jalan di CPU dengan memori terbatas
                per_device_eval_batch_size=4,
                logging_steps=5
            )
            if trained_model_path:
                logger.info(f"Contoh pelatihan selesai. Model disimpan di: {trained_model_path}")
            else:
                logger.error("Contoh pelatihan gagal.")
        else:
            logger.error("Gagal mempersiapkan dataset dummy.")
    else:
        logger.error("Gagal memuat data dummy.")

    # Hapus file dummy setelah selesai
    if os.path.exists(dummy_file_path):
        os.remove(dummy_file_path)
        logger.info(f"File dummy {dummy_file_path} dihapus.")

    # Untuk menjalankan ini, pastikan PyTorch terinstal.
    # Jika tidak ada GPU, Hugging Face Trainer akan otomatis menggunakan CPU.
    # Proses fine-tuning BERT, bahkan untuk 1 epoch dengan data kecil, bisa memakan waktu beberapa menit di CPU.
    # Outputnya akan ada di folder models/bert_jawa_hate_speech/
