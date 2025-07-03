import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    logger.info("GPU tidak tersedia. Training akan menggunakan CPU (lebih lambat).")
    logger.info("Untuk training yang lebih cepat, install PyTorch dengan CUDA support.")
    logger.info("Panduan: https://pytorch.org/get-started/locally/")

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

def load_labeled_data(labeled_data_path="src/data_collection/hasil-labeling.csv"):
    """
    Memuat data yang sudah dilabeli dari hasil-labeling.csv.
    Melakukan mapping label dari string ke numerik dan filtering berdasarkan confidence score.
    """
    if not os.path.exists(labeled_data_path):
        logger.error(f"File data berlabel tidak ditemukan di: {labeled_data_path}")
        return None
    try:
        df = pd.read_csv(labeled_data_path)
        
        # Cek kolom yang diperlukan
        required_columns = ['text', 'final_label', 'confidence_score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Kolom yang diperlukan tidak ditemukan: {missing_columns}")
            return None
        
        # Filter data berdasarkan confidence score (>= 0.7)
        logger.info(f"Data sebelum filtering: {len(df)} samples")
        df = df[df['confidence_score'] >= 0.7]
        logger.info(f"Data setelah filtering (confidence >= 0.7): {len(df)} samples")
        
        # Hapus baris dengan error atau teks kosong
        df = df[df['error'].isna() | (df['error'] == '')]
        df = df.dropna(subset=['text', 'final_label'])
        df = df[df['text'].str.strip() != '']
        
        # Mapping label dari string ke numerik
        label_mapping = {
            "Bukan Ujaran Kebencian": 0,
            "Ujaran Kebencian - Ringan": 1, 
            "Ujaran Kebencian - Sedang": 2,
            "Ujaran Kebencian - Berat": 3
        }
        
        # Apply mapping
        df['label_numeric'] = df['final_label'].map(label_mapping)
        
        # Debug: Check what's in final_label and label_numeric
        logger.debug(f"Sample final_label values: {df['final_label'].head().values.tolist()}")
        logger.debug(f"Sample label_numeric values before dropna: {df['label_numeric'].head().values.tolist()}")
        
        # Hapus baris dengan label yang tidak dikenali
        df = df.dropna(subset=['label_numeric'])
        df['label_numeric'] = df['label_numeric'].astype(int)
        
        # Debug: Check after conversion
        logger.debug(f"Sample label_numeric values after conversion: {df['label_numeric'].head().values.tolist()}")
        
        # Drop original label column to avoid conflicts and rename columns for consistency
        df = df.drop(columns=['label'], errors='ignore')  # Drop original label column if it exists
        df = df.rename(columns={'text': 'processed_text', 'label_numeric': 'label'})
        
        # Debug: Check final labels
        logger.debug(f"Sample final labels: {df['label'].head().values.tolist()}")
        
        # Log distribusi label
        try:
            label_distribution = df['label'].value_counts().sort_index()
            logger.info(f"Distribusi label: {label_distribution.to_dict()}")
        except Exception as e:
            logger.warning(f"Tidak dapat menghitung distribusi label: {e}")
            logger.info(f"Jumlah data final: {len(df)}")
        
        return df[['processed_text', 'label']]
        
    except Exception as e:
        logger.error(f"Error saat memuat data berlabel: {e}")
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
            df[text_column].values.tolist(), df[label_column].values.tolist(), test_size=test_size, random_state=42
        )

        # Pastikan semua label ada dalam rentang yang valid (0 hingga NUM_LABELS-1)
        logger.debug(f"prepare_datasets: NUM_LABELS = {NUM_LABELS}")
        logger.debug(f"prepare_datasets: Original train_labels sample: {train_labels[:5]}")
        logger.debug(f"prepare_datasets: Original val_labels sample: {val_labels[:5]}")

        # Validasi label - pastikan semua label adalah integer dalam rentang yang benar
        combined_labels = train_labels + val_labels
        invalid_labels = []
        for lbl in combined_labels:
            try:
                # Handle case where label might be a list or other type
                if isinstance(lbl, (list, tuple)):
                    invalid_labels.append(str(lbl))
                    continue
                    
                if not isinstance(lbl, (int, float)) or not (0 <= int(lbl) < NUM_LABELS):
                    invalid_labels.append(str(lbl))
            except (TypeError, ValueError):
                invalid_labels.append(str(lbl))
        
        if invalid_labels:
            # Konversi invalid_labels ke string untuk menghindari error unhashable
            unique_invalid = list(set(invalid_labels))[:5]
            logger.error(f"Label mengandung nilai di luar rentang yang diharapkan [0, {NUM_LABELS-1}]. Label tidak valid: {unique_invalid}")
            return None, None
        
        # Konversi semua label ke integer
        train_labels = [int(lbl) for lbl in train_labels]
        val_labels = [int(lbl) for lbl in val_labels]

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
        logger.info(f"Loading model: {TOKENIZER_CHECKPOINT}")
        model = AutoModelForSequenceClassification.from_pretrained(
            TOKENIZER_CHECKPOINT,
            num_labels=num_labels
        )
        
        # Move model to appropriate device
        model.to(device)
        logger.info(f"Model moved to device: {device}")
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model size: ~{total_params * 4 / 1024**2:.1f} MB")

        # Optimasi batch size berdasarkan device
        if torch.cuda.is_available():
            # GPU: batch size lebih besar untuk efisiensi
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb >= 24:  # High-end GPU (A100, RTX 4090, etc.)
                per_device_train_batch_size = min(per_device_train_batch_size * 4, 64)
                per_device_eval_batch_size = min(per_device_eval_batch_size * 4, 128)
            elif gpu_memory_gb >= 12:  # Mid-range GPU (RTX 3080, RTX 4070, etc.)
                per_device_train_batch_size = min(per_device_train_batch_size * 2, 32)
                per_device_eval_batch_size = min(per_device_eval_batch_size * 2, 64)
            logger.info(f"GPU detected. Optimized batch sizes - Train: {per_device_train_batch_size}, Eval: {per_device_eval_batch_size}")
        else:
            # CPU: batch size lebih kecil untuk menghindari memory issues
            per_device_train_batch_size = min(per_device_train_batch_size, 8)
            per_device_eval_batch_size = min(per_device_eval_batch_size, 16)
            logger.info(f"CPU detected. Conservative batch sizes - Train: {per_device_train_batch_size}, Eval: {per_device_eval_batch_size}")

        training_args = TrainingArguments(
            output_dir=model_output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=logging_steps,
            save_total_limit=save_total_limit,
            dataloader_pin_memory=torch.cuda.is_available(),  # Pin memory untuk GPU
            dataloader_num_workers=4 if torch.cuda.is_available() else 0,  # Parallel data loading untuk GPU
            fp16=torch.cuda.is_available(),  # Mixed precision untuk GPU
            report_to="none"
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
    # Menggunakan dataset hasil labeling yang sudah ada
    logger.info("Memuat data berlabel dari hasil-labeling.csv")
    df_loaded = load_labeled_data()

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

            logger.info("Memulai pelatihan model IndoBERT untuk deteksi ujaran kebencian...")
            # Konfigurasi training sesuai technical briefing
            trained_model_path = train_model(
                train_ds, val_ds,
                num_train_epochs=3,  # Sesuai rekomendasi technical briefing
                per_device_train_batch_size=16,  # Sesuai rekomendasi
                per_device_eval_batch_size=32,   # Sesuai rekomendasi
                learning_rate=2e-5,              # Sesuai rekomendasi
                weight_decay=0.01,
                logging_steps=50
            )
            if trained_model_path:
                logger.info(f"Pelatihan model selesai. Model disimpan di: {trained_model_path}")
                logger.info("Model siap untuk evaluasi dan deployment.")
            else:
                logger.error("Pelatihan model gagal.")
        else:
            logger.error("Gagal mempersiapkan dataset untuk training.")
    else:
        logger.error("Gagal memuat data berlabel dari hasil-labeling.csv.")

    logger.info("Training pipeline selesai.")
    logger.info("\n=== PANDUAN TRAINING ===")
    logger.info("1. Pastikan PyTorch dan transformers terinstal")
    if torch.cuda.is_available():
        logger.info("2. ✅ GPU tersedia - Training akan lebih cepat")
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("2. ⚠️  GPU tidak tersedia - Training akan menggunakan CPU (lebih lambat)")
        logger.info("   Install PyTorch dengan CUDA: https://pytorch.org/get-started/locally/")
    logger.info("3. Model akan disimpan di: models/bert_jawa_hate_speech/")
    logger.info("4. Gunakan evaluate_model.py untuk evaluasi performa")
    logger.info("5. Monitor training dengan: nvidia-smi -l 1 (untuk GPU)")
    logger.info("========================\n")
