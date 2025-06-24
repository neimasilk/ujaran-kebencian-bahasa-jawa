"""
Utilitas untuk pelatihan dan evaluasi model deteksi ujaran kebencian.
"""

# Import library yang umum digunakan untuk ML
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
# import pandas as pd
# import numpy as np

# Jika menggunakan Hugging Face Transformers & PyTorch/TensorFlow:
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
# from torch.utils.data import DataLoader, TensorDataset

def split_data(df, label_column='sentiment', test_size=0.2, random_state=42):
    """
    Membagi DataFrame menjadi set pelatihan dan validasi/test.
    (Ini adalah placeholder, implementasi sebenarnya akan bergantung pada library yang digunakan)

    Args:
        df (pandas.DataFrame): DataFrame input yang sudah diproses dan berlabel.
        label_column (str): Nama kolom yang berisi label.
        test_size (float): Proporsi dataset untuk dijadikan set tes/validasi.
        random_state (int): Seed untuk random state demi reproduktifitas.

    Returns:
        tuple: Berisi (X_train, X_val, y_train, y_val) atau format lain tergantung library.
               Atau (train_df, val_df) jika ingin membagi DataFrame secara langsung.
    """
    if df is None or df.empty:
        print("DataFrame kosong, tidak bisa melakukan pembagian data.")
        return None, None, None, None

    print(f"Membagi data menjadi train/validation set dengan test_size={test_size}...")
    # Contoh menggunakan scikit-learn (jika sudah diinstal)
    # from sklearn.model_selection import train_test_split
    # texts = df[df.columns.difference([label_column])] # Semua kolom kecuali label
    # labels = df[label_column]
    # X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=test_size, random_state=random_state, stratify=labels if label_column in df else None)
    # print("Pembagian data selesai.")
    # return X_train, X_val, y_train, y_val

    # Placeholder: Mengembalikan DataFrame yang dibagi secara acak (tidak ideal untuk ML sebenarnya)
    # Pastikan pandas sudah diimport
    try:
        val_df = df.sample(frac=test_size, random_state=random_state)
        train_df = df.drop(val_df.index)
        print("Pembagian data (placeholder) selesai.")
        # Jika ingin mengembalikan fitur dan label terpisah:
        # X_train = train_df.drop(label_column, axis=1)
        # y_train = train_df[label_column]
        # X_val = val_df.drop(label_column, axis=1)
        # y_val = val_df[label_column]
        # return X_train, X_val, y_train, y_val
        return train_df, val_df
    except Exception as e:
        print(f"Error saat membagi data (placeholder): {e}")
        return None, None


def train_model(train_texts, train_labels, model_name="indobenchmark/indobert-base-p1"):
    """
    Melatih model klasifikasi teks (placeholder untuk fine-tuning IndoBERT).

    Args:
        train_texts (list atau pd.Series): Teks untuk pelatihan.
        train_labels (list atau pd.Series): Label untuk pelatihan.
        model_name (str): Nama model Hugging Face yang akan di-fine-tune.

    Returns:
        object: Model yang sudah dilatih (misalnya, model Hugging Face).
    """
    print(f"Memulai pelatihan model (placeholder untuk {model_name})...")
    print(f"Jumlah data latih: {len(train_texts)}")

    # Implementasi sebenarnya akan melibatkan:
    # 1. Inisialisasi tokenizer dan model dari Hugging Face.
    #    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_CLASSES) # NUM_CLASSES = 4
    # 2. Tokenisasi teks pelatihan.
    #    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
    # 3. Membuat PyTorch Dataset & DataLoader.
    #    train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
    #                                  torch.tensor(train_encodings['attention_mask']),
    #                                  torch.tensor(list(train_labels))) # Pastikan label sudah di-encode ke integer
    #    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 4. Setup optimizer dan learning rate scheduler.
    #    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    #    num_training_steps = NUM_EPOCHS * len(train_loader)
    #    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    # 5. Loop pelatihan (training loop).
    #    model.train()
    #    for epoch in range(NUM_EPOCHS):
    #        for batch in train_loader:
    #            input_ids, attention_mask, labels = batch
    #            optimizer.zero_grad()
    #            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    #            loss = outputs.loss
    #            loss.backward()
    #            optimizer.step()
    #            lr_scheduler.step()
    #        print(f"Epoch {epoch+1} selesai, loss: {loss.item()}")
    print("Pelatihan model (placeholder) selesai.")
    # Placeholder: Mengembalikan string sebagai model dummy
    return "trained_dummy_model"


def evaluate_model(model, val_texts, val_labels):
    """
    Mengevaluasi model pada data validasi/test (placeholder).

    Args:
        model (object): Model yang sudah dilatih.
        val_texts (list atau pd.Series): Teks untuk validasi/evaluasi.
        val_labels (list atau pd.Series): Label sebenarnya untuk validasi/evaluasi.

    Returns:
        dict: Berisi metrik evaluasi (misalnya, akurasi, presisi, recall, f1).
    """
    print(f"Memulai evaluasi model (placeholder) pada {len(val_texts)} data validasi...")

    # Implementasi sebenarnya akan melibatkan:
    # 1. Tokenisasi teks validasi.
    # 2. Melakukan prediksi dengan model (model.eval()).
    # 3. Menghitung metrik (accuracy, precision, recall, f1-score, confusion matrix).
    #    Contoh:
    #    - Dapatkan prediksi dari model
    #    - predictions = model.predict(val_encodings) # atau cara lain sesuai framework
    #    - predicted_labels = np.argmax(predictions.logits, axis=1)
    #    - accuracy = accuracy_score(val_labels, predicted_labels)
    #    - precision, recall, f1, _ = precision_recall_fscore_support(val_labels, predicted_labels, average='weighted')
    #    - conf_matrix = confusion_matrix(val_labels, predicted_labels)
    #    - print(f"Akurasi: {accuracy}")
    #    - print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}")
    #    - print(f"Confusion Matrix:\n{conf_matrix}")

    # Placeholder: Mengembalikan metrik dummy
    dummy_metrics = {
        'accuracy': 0.75, # Contoh akurasi
        'precision': 0.70,
        'recall': 0.75,
        'f1_score': 0.72
    }
    print(f"Evaluasi model (placeholder) selesai. Metrik dummy: {dummy_metrics}")
    return dummy_metrics


def save_model(model, path="models/trained_bert_javanese_hatespeech.pt"):
    """
    Menyimpan model yang sudah dilatih (placeholder).

    Args:
        model (object): Model yang akan disimpan.
        path (str): Path untuk menyimpan model.
    """
    print(f"Menyimpan model (placeholder) ke {path}...")
    # Implementasi sebenarnya:
    # Jika model Hugging Face: model.save_pretrained(path)
    # Jika model PyTorch biasa: torch.save(model.state_dict(), path)
    # Pastikan direktori 'models/' sudah ada atau dibuat
    # import os
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Model (placeholder) berhasil disimpan di {path}.")


if __name__ == '__main__':
    # Contoh penggunaan (untuk pengujian modul secara mandiri)
    # Ini memerlukan data dummy atau data yang sudah diproses dari data_utils.py
    print("--- Menguji train_utils (Placeholder) ---")

    # Buat data dummy untuk pengujian
    import pandas as pd
    dummy_data = {
        'review': [
            "teks contoh satu untuk pelatihan",
            "contoh teks kedua yang sedikit lebih panjang",
            "ini adalah teks ketiga",
            "teks validasi pertama",
            "contoh validasi kedua"
        ],
        'sentiment': [0, 1, 0, 1, 0] # Asumsi label sudah di-encode: 0=Bukan, 1=Ringan, 2=Sedang, 3=Berat
    }
    dummy_df = pd.DataFrame(dummy_data)

    print(f"\n--- Menguji split_data (Placeholder) ---")
    # train_df, val_df = split_data(dummy_df, label_column='sentiment', test_size=0.4)
    # if train_df is not None and val_df is not None:
    #     print(f"Data latih: {len(train_df)} baris, Data validasi: {len(val_df)} baris")
    #     X_train_texts = train_df['review']
    #     y_train_labels = train_df['sentiment']
    #     X_val_texts = val_df['review']
    #     y_val_labels = val_df['sentiment']

        # print(f"\n--- Menguji train_model (Placeholder) ---")
        # trained_model = train_model(X_train_texts, y_train_labels)

        # if trained_model:
        #     print(f"\n--- Menguji evaluate_model (Placeholder) ---")
        #     evaluation_results = evaluate_model(trained_model, X_val_texts, y_val_labels)
        #     print(f"Hasil evaluasi (placeholder): {evaluation_results}")

        #     print(f"\n--- Menguji save_model (Placeholder) ---")
        #     # Perlu membuat direktori models/ jika belum ada untuk path default
        #     import os
        #     os.makedirs("models", exist_ok=True)
        #     save_model(trained_model, path="models/dummy_model_test.pt")
    # else:
    #     print("Pembagian data gagal.")

    print("\nPengujian train_utils (placeholder) selesai. Implementasi fungsi-fungsi utama memerlukan library ML seperti Transformers dan PyTorch/TensorFlow.")
    print("Untuk menjalankan ini, pastikan pandas sudah terinstal (`pip install pandas`).")
