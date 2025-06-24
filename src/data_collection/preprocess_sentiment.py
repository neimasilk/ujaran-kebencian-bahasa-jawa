#!/usr/bin/env python3
"""
Preprocessing Script untuk Strategi Optimasi Biaya DeepSeek Labeling

Script ini memisahkan dataset berdasarkan sentimen:
- Data positif: Auto-assign sebagai 'bukan_ujaran_kebencian'
- Data negatif: Siap untuk diproses dengan DeepSeek V3

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import os
from pathlib import Path
from loguru import logger

def preprocess_sentiment_data(input_file: str, output_dir: str = "data/processed") -> dict:
    """
    Memisahkan dataset berdasarkan sentimen dan auto-assign label untuk data positif
    
    Args:
        input_file (str): Path ke file dataset mentah
        output_dir (str): Direktori output untuk file hasil
        
    Returns:
        dict: Statistik preprocessing
    """
    
    # Pastikan direktori output ada
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset dari {input_file}")
    df = pd.read_csv(input_file)
    
    # Validasi kolom yang diperlukan
    required_columns = ['review', 'sentiment']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Kolom yang diperlukan tidak ditemukan: {missing_columns}")
    
    # Pisahkan berdasarkan sentimen
    data_positif = df[df['sentiment'] == 'positive'].copy()
    data_negatif = df[df['sentiment'] == 'negative'].copy()
    
    logger.info(f"Data positif: {len(data_positif)} samples")
    logger.info(f"Data negatif: {len(data_negatif)} samples")
    
    # Auto-assign untuk data positif
    data_positif['hate_speech_label'] = 'bukan_ujaran_kebencian'
    data_positif['confidence'] = 100
    data_positif['reasoning'] = 'Auto-assigned: Sentimen positif tidak mengandung ujaran kebencian'
    data_positif['processing_method'] = 'auto_sentiment_based'
    
    # Siapkan data negatif untuk DeepSeek processing
    data_negatif['hate_speech_label'] = None
    data_negatif['confidence'] = None
    data_negatif['reasoning'] = None
    data_negatif['processing_method'] = 'deepseek_pending'
    
    # Simpan file terpisah
    positive_file = os.path.join(output_dir, "positive_auto_labeled.csv")
    negative_file = os.path.join(output_dir, "negative_for_deepseek.csv")
    combined_file = os.path.join(output_dir, "preprocessed_dataset.csv")
    
    data_positif.to_csv(positive_file, index=False)
    data_negatif.to_csv(negative_file, index=False)
    
    # Gabungkan untuk dataset lengkap
    combined_data = pd.concat([data_positif, data_negatif], ignore_index=True)
    combined_data.to_csv(combined_file, index=False)
    
    # Statistik
    stats = {
        'total_samples': len(df),
        'positive_samples': len(data_positif),
        'negative_samples': len(data_negatif),
        'positive_percentage': (len(data_positif) / len(df)) * 100,
        'negative_percentage': (len(data_negatif) / len(df)) * 100,
        'files_created': {
            'positive_labeled': positive_file,
            'negative_pending': negative_file,
            'combined': combined_file
        }
    }
    
    logger.info("=== PREPROCESSING SELESAI ===")
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Positif (auto-labeled): {stats['positive_samples']} ({stats['positive_percentage']:.1f}%)")
    logger.info(f"Negatif (perlu DeepSeek): {stats['negative_samples']} ({stats['negative_percentage']:.1f}%)")
    logger.info(f"Files saved to: {output_dir}")
    
    return stats

def estimate_cost_savings(stats: dict) -> dict:
    """
    Estimasi penghematan biaya dengan strategi baru
    
    Args:
        stats (dict): Statistik dari preprocessing
        
    Returns:
        dict: Estimasi biaya
    """
    
    # Asumsi biaya (berdasarkan analisis sebelumnya)
    cost_per_sample = 0.0000893  # ~$0.0893 per 1000 samples
    
    old_cost = stats['total_samples'] * cost_per_sample
    new_cost = stats['negative_samples'] * cost_per_sample
    savings = old_cost - new_cost
    savings_percentage = (savings / old_cost) * 100
    
    cost_analysis = {
        'old_strategy_cost': old_cost,
        'new_strategy_cost': new_cost,
        'savings_amount': savings,
        'savings_percentage': savings_percentage,
        'positive_samples_cost': 0.0,  # Auto-assigned, no API cost
        'negative_samples_cost': new_cost
    }
    
    logger.info("=== ANALISIS BIAYA ===")
    logger.info(f"Strategi lama: ${old_cost:.4f}")
    logger.info(f"Strategi baru: ${new_cost:.4f}")
    logger.info(f"Penghematan: ${savings:.4f} ({savings_percentage:.1f}%)")
    
    return cost_analysis

def main():
    """
    Main function untuk menjalankan preprocessing
    """
    
    # Konfigurasi
    input_file = "src/data_collection/raw-dataset.csv"
    output_dir = "data/processed"
    
    try:
        # Preprocessing
        stats = preprocess_sentiment_data(input_file, output_dir)
        
        # Analisis biaya
        cost_analysis = estimate_cost_savings(stats)
        
        # Simpan statistik
        stats_file = os.path.join(output_dir, "preprocessing_stats.csv")
        stats_df = pd.DataFrame([
            {'metric': 'total_samples', 'value': stats['total_samples']},
            {'metric': 'positive_samples', 'value': stats['positive_samples']},
            {'metric': 'negative_samples', 'value': stats['negative_samples']},
            {'metric': 'positive_percentage', 'value': stats['positive_percentage']},
            {'metric': 'negative_percentage', 'value': stats['negative_percentage']},
            {'metric': 'old_strategy_cost', 'value': cost_analysis['old_strategy_cost']},
            {'metric': 'new_strategy_cost', 'value': cost_analysis['new_strategy_cost']},
            {'metric': 'savings_amount', 'value': cost_analysis['savings_amount']},
            {'metric': 'savings_percentage', 'value': cost_analysis['savings_percentage']}
        ])
        stats_df.to_csv(stats_file, index=False)
        
        logger.success(f"Preprocessing berhasil! Statistik disimpan di {stats_file}")
        
        print("\n=== NEXT STEPS ===")
        print(f"1. Review file: {stats['files_created']['negative_pending']}")
        print("2. Jalankan DeepSeek labeling untuk data negatif:")
        print("   python src/data_collection/deepseek_labeling_optimized.py")
        print(f"3. Hasil akhir akan tersedia di: {stats['files_created']['combined']}")
        
    except Exception as e:
        logger.error(f"Error dalam preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()