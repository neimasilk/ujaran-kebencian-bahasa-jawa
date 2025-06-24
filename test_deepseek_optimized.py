#!/usr/bin/env python3
"""
Test script untuk DeepSeek Labeling Optimized Version
Membandingkan biaya dan performa antara versi normal vs optimized
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append('src')

from data_collection.deepseek_labeling_optimized import DeepSeekLabelerOptimized

def estimate_tokens(text: str) -> int:
    """Estimasi jumlah token (rough approximation)"""
    return len(text.split()) * 1.3  # Approximation for Indonesian/Javanese

def calculate_cost_comparison():
    """Hitung perbandingan biaya antara versi normal vs optimized"""
    
    # DeepSeek pricing (per 1M tokens)
    input_cost_per_1m = 0.14  # USD
    output_cost_per_1m = 2.19  # USD
    
    # Estimasi token untuk 1 sampel
    sample_text = "Iki minangka conto teks Bahasa Jawa kanggo ngitung biaya token"
    
    # Normal version
    system_prompt_normal = """Sampeyan minangka ahli klasifikasi ujaran kebencian kanggo Bahasa Jawa..."""
    user_prompt_normal = f"Klasifikasi teks iki: {sample_text}"
    
    normal_input_tokens = estimate_tokens(system_prompt_normal + user_prompt_normal)
    normal_output_tokens = 80  # label + confidence + reasoning
    
    # Optimized version
    system_prompt_optimized = """Klasifikasi teks Bahasa Jawa ke dalam kategori ujaran kebencian..."""
    user_prompt_optimized = f"Klasifikasi: {sample_text}"
    
    optimized_input_tokens = estimate_tokens(system_prompt_optimized + user_prompt_optimized)
    optimized_output_tokens = 15  # hanya label + confidence
    
    print("\n" + "="*80)
    print("=== PERBANDINGAN BIAYA TOKEN ===")
    print("="*80)
    
    print(f"\n📊 ESTIMASI PER SAMPEL:")
    print(f"Normal Version:")
    print(f"  - Input tokens: {normal_input_tokens}")
    print(f"  - Output tokens: {normal_output_tokens}")
    print(f"  - Total tokens: {normal_input_tokens + normal_output_tokens}")
    
    print(f"\nOptimized Version:")
    print(f"  - Input tokens: {optimized_input_tokens}")
    print(f"  - Output tokens: {optimized_output_tokens}")
    print(f"  - Total tokens: {optimized_input_tokens + optimized_output_tokens}")
    
    # Hitung penghematan token
    normal_total = normal_input_tokens + normal_output_tokens
    optimized_total = optimized_input_tokens + optimized_output_tokens
    token_savings = normal_total - optimized_total
    savings_percentage = (token_savings / normal_total) * 100
    
    print(f"\n💰 PENGHEMATAN TOKEN:")
    print(f"  - Token saved per sample: {token_savings}")
    print(f"  - Percentage saved: {savings_percentage:.1f}%")
    
    # Hitung biaya untuk full dataset (41,759 samples)
    total_samples = 41759
    
    # Normal version cost
    normal_input_cost = (normal_input_tokens * total_samples / 1_000_000) * input_cost_per_1m
    normal_output_cost = (normal_output_tokens * total_samples / 1_000_000) * output_cost_per_1m
    normal_total_cost = normal_input_cost + normal_output_cost
    
    # Optimized version cost
    optimized_input_cost = (optimized_input_tokens * total_samples / 1_000_000) * input_cost_per_1m
    optimized_output_cost = (optimized_output_tokens * total_samples / 1_000_000) * output_cost_per_1m
    optimized_total_cost = optimized_input_cost + optimized_output_cost
    
    cost_savings = normal_total_cost - optimized_total_cost
    cost_savings_percentage = (cost_savings / normal_total_cost) * 100
    
    print(f"\n💵 BIAYA UNTUK FULL DATASET ({total_samples:,} samples):")
    print(f"Normal Version:")
    print(f"  - Input cost: ${normal_input_cost:.4f}")
    print(f"  - Output cost: ${normal_output_cost:.4f}")
    print(f"  - Total cost: ${normal_total_cost:.4f}")
    
    print(f"\nOptimized Version:")
    print(f"  - Input cost: ${optimized_input_cost:.4f}")
    print(f"  - Output cost: ${optimized_output_cost:.4f}")
    print(f"  - Total cost: ${optimized_total_cost:.4f}")
    
    print(f"\n🎯 TOTAL PENGHEMATAN:")
    print(f"  - Cost saved: ${cost_savings:.4f}")
    print(f"  - Percentage saved: {cost_savings_percentage:.1f}%")
    
    return {
        'normal_cost': normal_total_cost,
        'optimized_cost': optimized_total_cost,
        'savings': cost_savings,
        'savings_percentage': cost_savings_percentage
    }

def main():
    print("🚀 Testing DeepSeek Labeling - Optimized Version")
    print("="*60)
    
    # Hitung perbandingan biaya
    cost_comparison = calculate_cost_comparison()
    
    # Check API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("❌ DEEPSEEK_API_KEY tidak ditemukan!")
        print("Set environment variable: set DEEPSEEK_API_KEY=your_api_key")
        return
    
    print(f"\n✅ API key ditemukan: {api_key[:8]}...")
    print(f"✅ Menggunakan model: deepseek-chat (DeepSeek-V3)")
    print(f"✅ Base URL: https://api.deepseek.com")
    
    # Initialize labeler
    try:
        labeler = DeepSeekLabelerOptimized()
        print("\nTesting koneksi API...")
        
        if labeler.test_connection():
            print("✅ Koneksi API berhasil!")
        else:
            print("❌ Koneksi API gagal!")
            return
            
    except Exception as e:
        print(f"❌ Error initializing labeler: {e}")
        return
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        df = pd.read_csv('src/data_collection/raw-dataset.csv', names=['review', 'sentiment'])
        print(f"✅ Dataset dimuat dari: src/data_collection/raw-dataset.csv")
        
        # Use 5 samples for quick test
        sample_size = 5
        sample_df = df.sample(n=sample_size, random_state=42)
        print(f"✅ Dataset dimuat: {len(df)} total, menggunakan {sample_size} sampel")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Start labeling
    print(f"\nMemulai pelabelan dengan {sample_size} sampel (OPTIMIZED MODE)...")
    
    results = []
    for idx, row in sample_df.iterrows():
        text = row['review']
        result = labeler.label_text(text)
        result['text'] = text
        results.append(result)
    
    # Display results
    print("\n" + "="*80)
    print(f"=== HASIL PELABELAN ({sample_size} SAMPEL - OPTIMIZED) ===")
    print("Format: [Text] -> [Label] (Confidence: X%)")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        text_preview = result['text'][:80] + "..." if len(result['text']) > 80 else result['text']
        print(f"\n{i}. [{text_preview}]")
        print(f"   -> {result['label']} (Confidence: {result['confidence']}%)")
        print("-" * 80)
    
    # Analyze results
    analysis = labeler.analyze_results(results)
    
    print("\n=== ANALISIS KUALITAS ===")
    print(f"Distribusi Label: {analysis['label_distribution']}")
    print(f"Rata-rata Confidence: {analysis['average_confidence']:.1f}%")
    
    # Save results
    output_path = 'data/processed/deepseek_test_optimized.csv'
    labeler.save_results(results, output_path)
    
    print("\n=== PERBANDINGAN DENGAN VERSI NORMAL ===")
    print(f"💰 Biaya Normal: ${cost_comparison['normal_cost']:.4f}")
    print(f"💰 Biaya Optimized: ${cost_comparison['optimized_cost']:.4f}")
    print(f"💵 Penghematan: ${cost_comparison['savings']:.4f} ({cost_comparison['savings_percentage']:.1f}%)")
    
    print("\n=== TRADE-OFFS ===")
    print("✅ Keuntungan Optimized:")
    print("   - Biaya lebih murah (50-70% penghematan)")
    print("   - Response lebih cepat")
    print("   - Cocok untuk production labeling")
    
    print("\n❌ Kerugian Optimized:")
    print("   - Tidak ada reasoning/penjelasan")
    print("   - Sulit untuk debugging")
    print("   - Kurang transparency")
    
    print("\n=== REKOMENDASI ===")
    print("🎯 Gunakan Optimized untuk:")
    print("   - Production labeling dataset besar")
    print("   - Budget terbatas")
    print("   - Sudah yakin dengan kualitas model")
    
    print("\n🎯 Gunakan Normal untuk:")
    print("   - Development dan testing")
    print("   - Quality assurance")
    print("   - Debugging klasifikasi yang salah")
    
    print("\n=== LANGKAH SELANJUTNYA ===")
    print("1. Review hasil optimized di atas")
    print("2. Bandingkan akurasi dengan versi normal")
    print("3. Pilih versi sesuai kebutuhan dan budget")
    print("4. Proses dataset lengkap dengan versi pilihan")

if __name__ == "__main__":
    main()