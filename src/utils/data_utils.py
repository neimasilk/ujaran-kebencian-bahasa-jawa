"""
Utilitas untuk memuat dan melakukan pra-pemrosesan data.
"""
import pandas as pd

def load_data_from_google_sheets(sheet_id, sheet_name, credentials_path="path/to/your/credentials.json"):
    """
    Memuat data dari Google Sheets.
    (Ini adalah placeholder, implementasi sebenarnya memerlukan Google Sheets API client)

    Args:
        sheet_id (str): ID dari Google Sheet.
        sheet_name (str): Nama sheet di dalam Google Sheet.
        credentials_path (str): Path ke file kredensial Google Cloud.

    Returns:
        pandas.DataFrame: DataFrame yang berisi data dari sheet, atau None jika gagal.
    """
    print(f"Placeholder: Memuat data dari Google Sheet ID: {sheet_id}, Nama Sheet: {sheet_name}")
    print(f"Placeholder: Menggunakan kredensial dari: {credentials_path}")
    # Implementasi sebenarnya akan menggunakan google-api-python-client
    # Contoh:
    # from google.oauth2.service_account import Credentials
    # from googleapiclient.discovery import build
    #
    # scope = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    # creds = Credentials.from_service_account_file(credentials_path, scopes=scope)
    # service = build('sheets', 'v4', credentials=creds)
    # sheet = service.spreadsheets()
    # result = sheet.values().get(spreadsheetId=sheet_id, range=sheet_name).execute()
    # values = result.get('values', [])
    # if not values:
    #     print('No data found.')
    #     return None
    # else:
    #     # Asumsikan baris pertama adalah header
    #     df = pd.DataFrame(values[1:], columns=values[0])
    #     return df
    # Untuk sekarang, kita return DataFrame kosong sebagai placeholder
    return pd.DataFrame()


def load_data_from_csv(file_path):
    """
    Memuat data dari file CSV.

    Args:
        file_path (str): Path ke file CSV.

    Returns:
        pandas.DataFrame: DataFrame yang berisi data dari CSV, atau None jika gagal.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data berhasil dimuat dari {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {file_path}")
        return None
    except Exception as e:
        print(f"Error saat memuat data dari CSV: {e}")
        return None


def preprocess_data(df, text_column='review'):
    """
    Melakukan pra-pemrosesan dasar pada DataFrame.
    (Ini adalah placeholder, implementasi sebenarnya akan lebih kompleks)

    Args:
        df (pandas.DataFrame): DataFrame input.
        text_column (str): Nama kolom yang berisi teks untuk diproses.

    Returns:
        pandas.DataFrame: DataFrame setelah pra-pemrosesan.
    """
    if df is None or df.empty:
        print("DataFrame kosong, tidak ada pra-pemrosesan yang dilakukan.")
        return df

    print("Memulai pra-pemrosesan data...")
    # Contoh pra-pemrosesan dasar:
    # 1. Menghapus baris dengan nilai NaN di kolom teks
    df.dropna(subset=[text_column], inplace=True)

    # 2. Mengubah teks menjadi huruf kecil
    df[text_column] = df[text_column].astype(str).str.lower()

    # 3. Menghapus spasi berlebih (leading/trailing)
    df[text_column] = df[text_column].str.strip()

    # (Tambahkan langkah pra-pemrosesan lain di sini sesuai kebutuhan,
    #  misalnya: menghapus tanda baca, tokenisasi, stemming, stopword removal,
    #  normalisasi slang/singkatan Bahasa Jawa, dll.)

    # Contoh filtering duplikat berdasarkan kolom teks
    initial_rows = len(df)
    df.drop_duplicates(subset=[text_column], keep='first', inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"Menghapus {rows_dropped} baris duplikat berdasarkan kolom '{text_column}'.")

    print("Pra-pemrosesan data selesai.")
    return df

if __name__ == '__main__':
    # Contoh penggunaan (untuk pengujian modul secara mandiri)

    # Contoh memuat dari CSV (menggunakan dataset yang ada sebagai contoh)
    # Sesuaikan path jika file CSV Anda berada di lokasi berbeda
    # Atau jika Anda sudah memindahkannya ke data/raw/
    # Path diubah agar relatif terhadap root proyek, bukan lokasi skrip
    csv_file = 'src/data_collection/raw-dataset.csv'
    # Jika raw-dataset.csv sudah dipindah ke data/raw/raw-dataset.csv
    # csv_file = 'data/raw/raw-dataset.csv'

    print(f"\n--- Menguji load_data_from_csv ---")
    data_df = load_data_from_csv(csv_file)

    if data_df is not None and not data_df.empty:
        print(f"\nJumlah baris sebelum pra-pemrosesan: {len(data_df)}")
        print(f"Beberapa baris awal sebelum pra-pemrosesan:\n{data_df.head()}")

        # Pra-pemrosesan data
        print(f"\n--- Menguji preprocess_data ---")
        processed_df = preprocess_data(data_df.copy(), text_column='review') # Gunakan .copy() agar tidak mengubah df asli

        if processed_df is not None and not processed_df.empty:
            print(f"\nJumlah baris setelah pra-pemrosesan: {len(processed_df)}")
            print(f"Beberapa baris awal setelah pra-pemrosesan:\n{processed_df.head()}")
        else:
            print("Tidak ada data untuk ditampilkan setelah pra-pemrosesan.")
    else:
        print("Gagal memuat data dari CSV atau DataFrame kosong.")

    # Contoh placeholder untuk Google Sheets (tidak akan berjalan tanpa implementasi & kredensial)
    # print(f"\n--- Menguji load_data_from_google_sheets (Placeholder) ---")
    # sheet_df = load_data_from_google_sheets("YOUR_SHEET_ID", "Sheet1")
    # if sheet_df is not None and not sheet_df.empty:
    #     print(f"Berhasil memuat (placeholder) {len(sheet_df)} baris dari Google Sheets.")
    # else:
    #     print("Gagal memuat (placeholder) data dari Google Sheets atau sheet kosong.")
