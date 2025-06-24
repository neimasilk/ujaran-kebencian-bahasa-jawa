#!/usr/bin/env python3
"""
Script untuk melakukan pelabelan otomatis awal berdasarkan pola-pola
yang dapat diidentifikasi dalam teks Bahasa Jawa.
"""

import pandas as pd
import re
import os
from typing import Tuple

class JavaneseHateSpeechLabeler:
    def __init__(self):
        # Kata-kata kunci untuk kategori berat
        self.heavy_keywords = [
            'mateni', 'bunuh', 'pukul', 'gebuk', 'tembak', 'bacok',
            'ancam', 'ngancam', 'neraka', 'mati', 'mampus',
            'virus', 'penyakit', 'sampah', 'bangsat', 'anjing'
        ]
        
        # Kata-kata kunci untuk kategori sedang
        self.medium_keywords = [
            'bodho', 'goblok', 'elek', 'jelek', 'ora becik',
            'ora apik', 'kurang ajar', 'bajingan', 'brengsek',
            'ora duwe', 'ora ngerti', 'ora iso'
        ]
        
        # Kata-kata kunci untuk kategori ringan (sindiran)
        self.light_keywords = [
            'mbok', 'lho', 'kok', 'mosok', 'piye to',
            'aneh', 'lucu', 'ora umum', 'ora wajar'
        ]
        
        # Kata-kata netral/positif
        self.neutral_keywords = [
            'apik', 'becik', 'bagus', 'seneng', 'suka',
            'trima kasih', 'matur nuwun', 'sugeng',
            'selamat', 'sukses', 'berhasil'
        ]
        
        # Pola SARA dan diskriminasi
        self.sara_patterns = [
            r'wong\s+(tionghoa|cina|arab|india|batak|jawa|sunda)',
            r'agama\s+(islam|kristen|hindu|buddha)',
            r'(gay|lesbian|homo|lgbt)',
            r'(wanita|wong\s+wadon|perempuan)',
            r'(imigran|migran|pendatang)'
        ]
    
    def detect_patterns(self, text: str) -> dict:
        """
        Mendeteksi pola-pola dalam teks.
        """
        text_lower = text.lower()
        
        patterns = {
            'has_heavy': any(keyword in text_lower for keyword in self.heavy_keywords),
            'has_medium': any(keyword in text_lower for keyword in self.medium_keywords),
            'has_light': any(keyword in text_lower for keyword in self.light_keywords),
            'has_neutral': any(keyword in text_lower for keyword in self.neutral_keywords),
            'has_sara': any(re.search(pattern, text_lower) for pattern in self.sara_patterns),
            'has_threat': any(word in text_lower for word in ['ancam', 'bunuh', 'mateni', 'tembak']),
            'has_profanity': any(word in text_lower for word in ['anjing', 'bangsat', 'bajingan', 'brengsek'])
        }
        
        return patterns
    
    def auto_label(self, text: str) -> Tuple[str, int, str]:
        """
        Melakukan pelabelan otomatis berdasarkan pola yang terdeteksi.
        
        Returns:
            Tuple[label, confidence, reasoning]
        """
        if pd.isna(text) or text.strip() == '':
            return 'bukan_ujaran_kebencian', 1, 'Teks kosong'
        
        patterns = self.detect_patterns(text)
        reasoning_parts = []
        
        # Logika pelabelan berdasarkan prioritas
        if patterns['has_threat'] or (patterns['has_heavy'] and patterns['has_sara']):
            label = 'ujaran_kebencian_berat'
            confidence = 4
            reasoning_parts.append('Mengandung ancaman/kekerasan')
            if patterns['has_sara']:
                reasoning_parts.append('Terkait SARA')
        
        elif patterns['has_heavy'] or patterns['has_profanity']:
            label = 'ujaran_kebencian_sedang'
            confidence = 3
            reasoning_parts.append('Mengandung kata kasar/hinaan langsung')
        
        elif patterns['has_medium'] and patterns['has_sara']:
            label = 'ujaran_kebencian_sedang'
            confidence = 3
            reasoning_parts.append('Hinaan dengan unsur SARA')
        
        elif patterns['has_medium']:
            label = 'ujaran_kebencian_ringan'
            confidence = 2
            reasoning_parts.append('Mengandung ejekan/sindiran')
        
        elif patterns['has_light'] and patterns['has_sara']:
            label = 'ujaran_kebencian_ringan'
            confidence = 2
            reasoning_parts.append('Sindiran halus dengan unsur SARA')
        
        elif patterns['has_neutral']:
            label = 'bukan_ujaran_kebencian'
            confidence = 3
            reasoning_parts.append('Mengandung kata positif/netral')
        
        else:
            # Default untuk teks yang tidak jelas
            label = 'bukan_ujaran_kebencian'
            confidence = 1
            reasoning_parts.append('Tidak terdeteksi pola ujaran kebencian')
        
        reasoning = '; '.join(reasoning_parts) if reasoning_parts else 'Analisis otomatis'
        
        return label, confidence, reasoning

def process_labeling_template():
    """
    Memproses template pelabelan dengan auto-labeling.
    """
    template_path = "data/processed/labeling_template.csv"
    
    if not os.path.exists(template_path):
        print(f"Error: Template tidak ditemukan di {template_path}")
        return
    
    # Baca template
    df = pd.read_csv(template_path)
    
    # Inisialisasi labeler
    labeler = JavaneseHateSpeechLabeler()
    
    print("Memulai auto-labeling...")
    
    # Proses setiap baris
    for idx, row in df.iterrows():
        text = row['text']
        label, confidence, reasoning = labeler.auto_label(text)
        
        # Update dataframe
        df.at[idx, 'new_label'] = label
        df.at[idx, 'confidence'] = confidence
        df.at[idx, 'notes'] = f"Auto: {reasoning}"
        
        if idx % 100 == 0:
            print(f"Diproses: {idx + 1}/{len(df)} baris")
    
    # Simpan hasil
    output_path = "data/processed/auto_labeled_template.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n=== AUTO-LABELING SELESAI ===")
    print(f"File output: {output_path}")
    
    # Statistik hasil
    print(f"\nDistribusi label otomatis:")
    print(df['new_label'].value_counts())
    
    print(f"\nDistribusi confidence:")
    print(df['confidence'].value_counts().sort_index())
    
    return df

def create_manual_review_subset(df, low_confidence_threshold=2):
    """
    Membuat subset untuk review manual berdasarkan confidence rendah.
    """
    if df is None:
        return
    
    # Filter data dengan confidence rendah
    low_conf_df = df[df['confidence'] <= low_confidence_threshold].copy()
    
    # Tambahkan beberapa sampel random untuk validasi
    high_conf_df = df[df['confidence'] > low_confidence_threshold].sample(
        n=min(50, len(df[df['confidence'] > low_confidence_threshold])), 
        random_state=42
    ).copy()
    
    # Gabungkan
    review_df = pd.concat([low_conf_df, high_conf_df]).sample(frac=1, random_state=42)
    
    # Simpan untuk review manual
    review_path = "data/processed/manual_review_needed.csv"
    review_df.to_csv(review_path, index=False)
    
    print(f"\n=== SUBSET REVIEW MANUAL DIBUAT ===")
    print(f"File: {review_path}")
    print(f"Jumlah data untuk review: {len(review_df)}")
    print(f"- Confidence rendah (≤{low_confidence_threshold}): {len(low_conf_df)}")
    print(f"- Sampel validasi: {len(high_conf_df)}")

if __name__ == "__main__":
    print("Memulai proses auto-labeling...\n")
    
    # Proses auto-labeling
    df = process_labeling_template()
    
    if df is not None:
        # Buat subset untuk review manual
        create_manual_review_subset(df)
        
        print("\n=== LANGKAH SELANJUTNYA ===")
        print("1. Review file data/processed/auto_labeled_template.csv")
        print("2. Fokus review manual pada data/processed/manual_review_needed.csv")
        print("3. Perbaiki label yang confidence-nya rendah")
        print("4. Validasi beberapa sampel dengan confidence tinggi")
    else:
        print("Gagal memproses auto-labeling.")