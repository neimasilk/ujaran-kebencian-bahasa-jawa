# API Dokumentasi - Data Loading

## Overview
Modul data loading menyediakan fungsi-fungsi untuk memuat dan memproses dataset ujaran kebencian bahasa Jawa. Modul ini mendukung loading dari file CSV dan Google Sheets (placeholder).

## Modules

### `src.data_collection.load_csv_dataset`

#### `inspect_dataset(csv_file_path)`
Melakukan inspeksi dasar terhadap dataset CSV.

**Parameters:**
- `csv_file_path` (str): Path ke file CSV yang akan diinspeksi

**Returns:**
- `dict`: Dictionary berisi informasi dataset atau None jika terjadi error

**Raises:**
- `FileNotFoundError`: Jika file tidak ditemukan
- `pd.errors.EmptyDataError`: Jika file CSV kosong
- `pd.errors.ParserError`: Jika format CSV tidak valid

**Example:**
```python
from src.data_collection.load_csv_dataset import inspect_dataset

result = inspect_dataset('data/raw/dataset.csv')
if result:
    print(f"Dataset memiliki {result['num_rows']} baris")
```

### `src.utils.data_utils`

#### `load_data_from_csv(csv_file_path)`
Memuat data dari file CSV dengan error handling.

**Parameters:**
- `csv_file_path` (str): Path ke file CSV

**Returns:**
- `pandas.DataFrame`: DataFrame berisi data atau None jika gagal

**Example:**
```python
from src.utils.data_utils import load_data_from_csv

df = load_data_from_csv('data/raw/dataset.csv')
if df is not None:
    print(f"Berhasil memuat {len(df)} baris data")
```

#### `load_data_from_google_sheets(sheet_id, sheet_name)`
Memuat data dari Google Sheets (placeholder implementation).

**Parameters:**
- `sheet_id` (str): ID Google Sheets
- `sheet_name` (str): Nama sheet

**Returns:**
- `pandas.DataFrame`: DataFrame kosong (placeholder)

**Note:** Implementasi aktual memerlukan Google Sheets API client.

#### `preprocess_data(df, text_column='review')`
Melakukan pra-pemrosesan dasar pada DataFrame.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame input
- `text_column` (str): Nama kolom teks yang akan diproses (default: 'review')

**Returns:**
- `pandas.DataFrame`: DataFrame setelah pra-pemrosesan

**Processing Steps:**
1. Menghapus baris dengan nilai NaN di kolom teks
2. Mengubah teks menjadi lowercase
3. Menghapus whitespace di awal dan akhir
4. Menghapus duplikasi berdasarkan kolom teks

**Example:**
```python
from src.utils.data_utils import preprocess_data

processed_df = preprocess_data(df, text_column='review')
print(f"Data setelah preprocessing: {len(processed_df)} baris")
```

## Error Handling

Semua fungsi dilengkapi dengan error handling yang komprehensif:

- **File tidak ditemukan**: Menampilkan pesan error dan mengembalikan None
- **Format CSV tidak valid**: Menangani ParserError dan EmptyDataError
- **DataFrame kosong**: Menangani kasus DataFrame None atau kosong

## Testing

Modul ini dilengkapi dengan unit tests komprehensif yang mencakup:

- Test untuk semua fungsi utama
- Test untuk berbagai skenario error
- Test untuk edge cases
- Mock testing untuk print statements
- Validation testing untuk struktur data

**Menjalankan Tests:**
```bash
python -m pytest tests/test_data_loading.py -v
```

**Coverage Testing:**
```bash
python -m pytest tests/test_data_loading.py --cov=src --cov-report=term-missing
```

## Dependencies

- `pandas`: Untuk manipulasi data
- `pytest`: Untuk unit testing
- `unittest.mock`: Untuk mocking dalam tests

## File Structure

```
src/
├── data_collection/
│   ├── __init__.py
│   ├── load_csv_dataset.py
│   └── raw-dataset.csv
└── utils/
    ├── __init__.py
    └── data_utils.py

tests/
└── test_data_loading.py
```

## Future Enhancements

1. **Google Sheets Integration**: Implementasi penuh untuk Google Sheets API
2. **Advanced Preprocessing**: Tokenisasi, stemming, dan normalisasi teks Jawa
3. **Data Validation**: Schema validation untuk memastikan konsistensi data
4. **Caching**: Implementasi caching untuk performa yang lebih baik
5. **Logging**: Sistem logging yang lebih komprehensif