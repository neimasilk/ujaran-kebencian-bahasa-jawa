#!/usr/bin/env python3
"""
Script untuk melakukan pelabelan data ujaran kebencian Bahasa Jawa
menggunakan DeepSeek V3 API untuk akurasi yang lebih tinggi.
"""

import pandas as pd
import os
import time
from typing import Tuple, Optional
from pathlib import Path
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI SDK tidak ditemukan. Install dengan: pip install openai")
    OpenAI = None

class DeepSeekLabeler:
    def __init__(self, api_key: str):
        if OpenAI is None:
            raise ImportError("OpenAI SDK diperlukan. Install dengan: pip install openai")
        
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        # System prompt untuk pelabelan ujaran kebencian Bahasa Jawa
        self.system_prompt = """
Anda adalah ahli linguistik Bahasa Jawa yang bertugas melabeli ujaran kebencian. 
Anda memahami nuansa budaya, tingkatan bahasa (ngoko, krama), dan konteks sosial Jawa.

Tugas: Klasifikasikan teks Bahasa Jawa ke dalam 4 kategori:

1. **bukan_ujaran_kebencian**: Teks netral, positif, atau kritik membangun tanpa unsur hinaan/provokasi

2. **ujaran_kebencian_ringan**: 
   - Sindiran halus, ejekan terselubung
   - Metafora budaya Jawa yang menyiratkan ketidaksukaan
   - Penggunaan pasemon atau peribahasa untuk menyindir

3. **ujaran_kebencian_sedang**:
   - Hinaan langsung, cercaan, bahasa kasar
   - Penggunaan ngoko yang tidak pantas dalam konteks
   - Lebih eksplisit dari kategori ringan

4. **ujaran_kebencian_berat**:
   - Ancaman kekerasan fisik
   - Hasutan untuk melakukan kekerasan
   - Dehumanisasi, diskriminasi sistematis
   - Penghinaan ekstrem terkait SARA

Perhatikan:
- Tingkatan bahasa Jawa (ngoko vs krama)
- Konteks budaya dan metafora lokal
- Maksud (intent) di balik ujaran
- Unsur SARA (Suku, Agama, Ras, Antargolongan)

WAJIB: Berikan respons dalam format JSON yang valid:
{
  "label": "bukan_ujaran_kebencian|ujaran_kebencian_ringan|ujaran_kebencian_sedang|ujaran_kebencian_berat",
  "confidence": 85,
  "reasoning": "penjelasan singkat alasan pelabelan dalam 1-2 kalimat"
}

Catatan: confidence adalah angka 1-100 (persentase keyakinan).
"""
    
    def create_user_prompt(self, text: str) -> str:
        """Membuat prompt untuk teks yang akan dilabeli."""
        return f"""
Labelkan teks Bahasa Jawa berikut:

Teks: "{text}"

Berikan respons dalam format JSON yang PERSIS seperti ini:
{{
  "label": "bukan_ujaran_kebencian",
  "confidence": 85,
  "reasoning": "Teks ini netral dan tidak mengandung unsur ujaran kebencian"
}}

Pastikan JSON valid dan gunakan salah satu label: bukan_ujaran_kebencian, ujaran_kebencian_ringan, ujaran_kebencian_sedang, ujaran_kebencian_berat.
"""
    
    def call_deepseek_api(self, text: str, max_retries: int = 3) -> Optional[dict]:
        """Memanggil DeepSeek API untuk pelabelan menggunakan OpenAI SDK."""
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.create_user_prompt(text)}
                    ],
                    temperature=0.1,  # Low temperature untuk konsistensi
                    max_tokens=200,
                    stream=False
                )
                
                content = response.choices[0].message.content
                
                # Parse JSON response
                try:
                    import json
                    parsed_result = json.loads(content)
                    return parsed_result
                except json.JSONDecodeError:
                    # Jika tidak bisa parse JSON, coba ekstrak manual
                    return self._extract_from_text(content)
                    
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "429" in error_msg:
                    wait_time = 2 ** attempt
                    print(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"API Error (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
        
        return None
    
    def _extract_from_text(self, content: str) -> dict:
        """Ekstrak informasi dari respons teks jika JSON parsing gagal."""
        import re
        
        # Fallback parsing untuk respons yang tidak dalam format JSON sempurna
        content_lower = content.lower()
        
        # Default values
        label = "bukan_ujaran_kebencian"
        confidence = 50  # default confidence
        reasoning = "Fallback parsing dari response text"
        
        # Extract label dengan prioritas
        if 'ujaran_kebencian_berat' in content_lower:
            label = "ujaran_kebencian_berat"
            confidence = 75
        elif 'ujaran_kebencian_sedang' in content_lower:
            label = "ujaran_kebencian_sedang"
            confidence = 70
        elif 'ujaran_kebencian_ringan' in content_lower:
            label = "ujaran_kebencian_ringan"
            confidence = 65
        elif 'bukan_ujaran_kebencian' in content_lower:
            label = "bukan_ujaran_kebencian"
            confidence = 60
        
        # Try to extract confidence number
        confidence_match = re.search(r'confidence["\s]*:?["\s]*(\d+)', content_lower)
        if confidence_match:
            try:
                extracted_conf = int(confidence_match.group(1))
                if 1 <= extracted_conf <= 100:
                    confidence = extracted_conf
            except ValueError:
                pass
        
        # Try to extract reasoning
        reasoning_patterns = [
            r'reasoning["\s]*:?["\s]*([^"\n}]+)',
            r'alasan["\s]*:?["\s]*([^"\n}]+)',
            r'penjelasan["\s]*:?["\s]*([^"\n}]+)'
        ]
        
        for pattern in reasoning_patterns:
            reasoning_match = re.search(pattern, content, re.IGNORECASE)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                break
        
        return {
            "label": label,
            "confidence": confidence,
            "reasoning": reasoning
        }
    
    def label_text(self, text: str) -> Tuple[str, int, str]:
        """Melabeli satu teks dan mengembalikan hasil."""
        if pd.isna(text) or text.strip() == '':
            return 'bukan_ujaran_kebencian', 1, 'Teks kosong'
        
        result = self.call_deepseek_api(text)
        
        if result:
            label = result.get('label', 'bukan_ujaran_kebencian')
            confidence = result.get('confidence', 3)
            reasoning = result.get('reasoning', 'DeepSeek API response')
            return label, confidence, reasoning
        else:
            return 'bukan_ujaran_kebencian', 1, 'API call failed'

def process_dataset_with_deepseek(api_key: str, input_file: str = None, sample_size: int = 500):
    """
    Memproses dataset menggunakan DeepSeek API untuk pelabelan.
    """
    if not api_key:
        print("Error: API key DeepSeek diperlukan!")
        return None
    
    # Tentukan file input
    if input_file is None:
        input_file = "data/processed/labeling_template.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: File input tidak ditemukan di {input_file}")
        return None
    
    # Baca dataset
    df = pd.read_csv(input_file)
    
    # Batasi jumlah data jika perlu (untuk menghemat API calls)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Menggunakan sampel {sample_size} data dari {len(pd.read_csv(input_file))} total data")
    
    # Inisialisasi labeler
    labeler = DeepSeekLabeler(api_key)
    
    print(f"Memulai pelabelan dengan DeepSeek V3 untuk {len(df)} data...")
    print("Estimasi waktu: ~{:.1f} menit".format(len(df) * 2 / 60))  # Asumsi 2 detik per request
    
    # Proses setiap baris
    results = []
    failed_count = 0
    
    for idx, row in df.iterrows():
        text = row['text']
        
        print(f"\rMemproses: {idx + 1}/{len(df)} | Gagal: {failed_count}", end="", flush=True)
        
        label, confidence, reasoning = labeler.label_text(text)
        
        # Update dataframe
        df.at[idx, 'new_label'] = label
        df.at[idx, 'confidence'] = confidence
        df.at[idx, 'notes'] = f"DeepSeek: {reasoning}"
        
        if 'failed' in reasoning.lower():
            failed_count += 1
        
        # Rate limiting - jeda antar request
        time.sleep(0.5)  # 500ms delay
        
        # Save progress setiap 50 data
        if (idx + 1) % 50 == 0:
            temp_path = "data/processed/deepseek_progress.csv"
            df.to_csv(temp_path, index=False)
    
    print("\n")
    
    # Simpan hasil final
    output_path = "data/processed/deepseek_labeled_dataset.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n=== PELABELAN DEEPSEEK SELESAI ===")
    print(f"File output: {output_path}")
    print(f"Total data diproses: {len(df)}")
    print(f"Gagal: {failed_count}")
    print(f"Sukses: {len(df) - failed_count}")
    
    # Statistik hasil
    print(f"\nDistribusi label DeepSeek:")
    print(df['new_label'].value_counts())
    
    print(f"\nDistribusi confidence:")
    print(df['confidence'].value_counts().sort_index())
    
    return df

def create_validation_subset(df, validation_size: int = 100):
    """
    Membuat subset untuk validasi manual dari hasil DeepSeek.
    """
    if df is None or len(df) == 0:
        return
    
    # Ambil sampel stratified berdasarkan label
    validation_samples = []
    
    for label in df['new_label'].unique():
        label_df = df[df['new_label'] == label]
        sample_size = min(validation_size // 4, len(label_df))  # Bagi rata untuk 4 kategori
        if sample_size > 0:
            samples = label_df.sample(n=sample_size, random_state=42)
            validation_samples.append(samples)
    
    if validation_samples:
        validation_df = pd.concat(validation_samples)
        
        # Simpan untuk validasi
        validation_path = "data/processed/deepseek_validation_subset.csv"
        validation_df.to_csv(validation_path, index=False)
        
        print(f"\n=== SUBSET VALIDASI DIBUAT ===")
        print(f"File: {validation_path}")
        print(f"Jumlah data validasi: {len(validation_df)}")
        print("Distribusi:")
        print(validation_df['new_label'].value_counts())

if __name__ == "__main__":
    print("=== DEEPSEEK LABELING SYSTEM ===")
    print("Script untuk pelabelan ujaran kebencian Bahasa Jawa menggunakan DeepSeek V3")
    print()
    
    # Load API key dari environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('DEEPSEEK_API_KEY')
        
        if not api_key:
            print("❌ API key tidak ditemukan di file .env")
            print("Pastikan file .env sudah dibuat dengan DEEPSEEK_API_KEY")
            sys.exit(1)
        
        print(f"✅ API key ditemukan: {api_key[:8]}...")
        print("✅ Menggunakan model: deepseek-chat (DeepSeek-V3)")
        print("✅ Base URL: https://api.deepseek.com")
        print()
        
        # Test koneksi API dulu
        print("Testing koneksi API...")
        labeler = DeepSeekLabeler(api_key)
        test_result = labeler.call_deepseek_api("Test koneksi")
        
        if test_result:
            print("✅ Koneksi API berhasil!")
            print()
            
            # Proses dataset dengan sampel kecil dulu
            print("Memulai pelabelan dengan sampel 10 data...")
            df = process_dataset_with_deepseek(api_key, sample_size=10)
            
            if df is not None:
                # Tampilkan hasil pelabelan untuk review
                print("\n=== HASIL PELABELAN (10 SAMPEL) ===")
                print("Format: [Text] -> [Label] (Confidence: X%) [Reasoning]")
                print("=" * 80)
                
                for idx, row in df.iterrows():
                    text = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
                    label = row['deepseek_label']
                    confidence = row['deepseek_confidence']
                    reasoning = row['deepseek_reasoning'][:150] + "..." if len(row['deepseek_reasoning']) > 150 else row['deepseek_reasoning']
                    
                    print(f"\n{idx+1}. [{text}]")
                    print(f"   -> {label} (Confidence: {confidence}%)")
                    print(f"   Reasoning: {reasoning}")
                    print("-" * 80)
                
                # Buat subset validasi
                create_validation_subset(df)
                
                print("\n=== PELABELAN SELESAI ===")
                print("✅ Hasil tersimpan di data/processed/deepseek_labeled_dataset.csv")
                print("✅ Subset validasi di data/processed/deepseek_validation_subset.csv")
                print()
                print("=== ANALISIS KUALITAS ===")
                label_counts = df['deepseek_label'].value_counts()
                avg_confidence = df['deepseek_confidence'].mean()
                print(f"Distribusi Label: {dict(label_counts)}")
                print(f"Rata-rata Confidence: {avg_confidence:.1f}%")
                print()
                print("=== LANGKAH SELANJUTNYA ===")
                print("1. Review hasil pelabelan di atas")
                print("2. Jika kualitas memuaskan, proses dataset lengkap")
                print("3. Jika perlu perbaikan, sesuaikan system prompt")
                print("4. Update training pipeline dengan dataset berlabel")
            else:
                print("❌ Pelabelan gagal. Periksa koneksi internet dan API key.")
        else:
            print("❌ Test koneksi API gagal. Periksa API key dan koneksi internet.")
            
    except ImportError as e:
        if "openai" in str(e).lower():
            print("❌ OpenAI SDK tidak ditemukan")
            print("Install dengan: pip install openai")
        else:
            print("❌ Module python-dotenv tidak ditemukan")
            print("Install dengan: pip install python-dotenv openai")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nUntuk menjalankan secara manual:")
        print("from deepseek_labeling import process_dataset_with_deepseek")
        print("df = process_dataset_with_deepseek('your_api_key_here')")