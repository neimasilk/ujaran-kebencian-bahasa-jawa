#!/usr/bin/env python3
"""Demo script untuk menunjukkan strategi cost-efficient DeepSeek API labeling.

Script ini mendemonstrasikan:
1. Pembagian data positif/negatif
2. Cost savings analysis
3. Performance comparison
4. Quality metrics

Usage:
    python demo_cost_efficient_labeling.py
"""

import pandas as pd
import time
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from utils.deepseek_labeling import DeepSeekLabelingStrategy
from utils.deepseek_client import create_deepseek_client
from config.settings import Settings

def create_sample_dataset():
    """Buat sample dataset untuk demo."""
    sample_data = [
        # Data Positif (bukan ujaran kebencian)
        {"text": "Sugeng enjing, piye kabare?", "label": "positive"},
        {"text": "Aku seneng banget karo kowe", "label": "positive"},
        {"text": "Mangga mlebu, monggo pinarak", "label": "positive"},
        {"text": "Selamat ulang tahun, mugi panjang umur", "label": "positive"},
        {"text": "Terima kasih wis gelem mbantu", "label": "positive"},
        {"text": "Apik tenan pemandangane kene", "label": "positive"},
        
        # Data Negatif (berpotensi ujaran kebencian)
        {"text": "Wong edan iki, ora ngerti aturan", "label": "negative"},
        {"text": "Dasar bodoh, ora iso mikir", "label": "negative"},
        {"text": "Mati wae kowe, ora guna", "label": "negative"},
        {"text": "Setan kabeh wong kaya ngene", "label": "negative"},
    ]
    
    return pd.DataFrame(sample_data)

def demonstrate_cost_analysis(df):
    """Demonstrasi analisis cost savings."""
    print("\n" + "="*60)
    print("           ANALISIS COST SAVINGS")
    print("="*60)
    
    total_samples = len(df)
    positive_count = len(df[df['label'] == 'positive'])
    negative_count = len(df[df['label'] == 'negative'])
    
    # Estimasi biaya (contoh: $0.002 per API call)
    cost_per_api_call = 0.002
    
    # Scenario 1: Tanpa strategi (semua pakai API)
    cost_without_strategy = total_samples * cost_per_api_call
    
    # Scenario 2: Dengan strategi (hanya negatif pakai API)
    cost_with_strategy = negative_count * cost_per_api_call
    
    # Savings
    savings = cost_without_strategy - cost_with_strategy
    savings_percentage = (savings / cost_without_strategy) * 100
    
    print(f"📊 DATASET OVERVIEW:")
    print(f"   Total sampel: {total_samples}")
    print(f"   Data positif: {positive_count} ({positive_count/total_samples*100:.1f}%)")
    print(f"   Data negatif: {negative_count} ({negative_count/total_samples*100:.1f}%)")
    
    print(f"\n💰 ANALISIS BIAYA:")
    print(f"   Biaya per API call: ${cost_per_api_call}")
    print(f"   ")
    print(f"   TANPA STRATEGI:")
    print(f"   - Semua {total_samples} sampel → API")
    print(f"   - Total biaya: ${cost_without_strategy:.3f}")
    print(f"   ")
    print(f"   DENGAN STRATEGI:")
    print(f"   - {positive_count} sampel → Rule-based ($0)")
    print(f"   - {negative_count} sampel → API (${cost_with_strategy:.3f})")
    print(f"   - Total biaya: ${cost_with_strategy:.3f}")
    print(f"   ")
    print(f"   💡 PENGHEMATAN: ${savings:.3f} ({savings_percentage:.1f}%)")
    
    return {
        'total_samples': total_samples,
        'positive_samples': positive_count,
        'negative_samples': negative_count,
        'cost_without_strategy': cost_without_strategy,
        'cost_with_strategy': cost_with_strategy,
        'savings': savings,
        'savings_percentage': savings_percentage
    }

def demonstrate_labeling_process(df):
    """Demonstrasi proses labeling dengan strategi."""
    print("\n" + "="*60)
    print("           PROSES LABELING")
    print("="*60)
    
    strategy = DeepSeekLabelingStrategy()
    
    # Split data
    positive_data, negative_data = strategy.filter_data_by_initial_label(
        df, text_column='text', label_column='label'
    )
    
    print(f"\n🔄 PEMBAGIAN DATA:")
    print(f"   Data positif: {len(positive_data)} sampel")
    print(f"   Data negatif: {len(negative_data)} sampel")
    
    # Process positive data (rule-based)
    print(f"\n✅ PROCESSING DATA POSITIF (Rule-based):")
    positive_results = []
    for idx, row in positive_data.iterrows():
        result = {
            'text': row['text'],
            'original_label': row['label'],
            'final_label': 'Bukan Ujaran Kebencian',
            'confidence': 1.0,
            'method': 'rule_based',
            'processing_time': 0.001  # Sangat cepat
        }
        positive_results.append(result)
        print(f"   ✓ '{row['text'][:30]}...' → Bukan Ujaran Kebencian")
    
    # Process negative data (mock API)
    print(f"\n🤖 PROCESSING DATA NEGATIF (DeepSeek API - Mock):")
    negative_results = []
    
    # Mock API responses untuk demo
    mock_responses = [
        {'label': 'Ujaran Kebencian - Ringan', 'confidence': 0.85},
        {'label': 'Ujaran Kebencian - Sedang', 'confidence': 0.92},
        {'label': 'Ujaran Kebencian - Berat', 'confidence': 0.95},
        {'label': 'Ujaran Kebencian - Sedang', 'confidence': 0.88},
    ]
    
    for idx, (_, row) in enumerate(negative_data.iterrows()):
        # Simulate API call delay
        time.sleep(0.1)
        
        mock_response = mock_responses[idx % len(mock_responses)]
        result = {
            'text': row['text'],
            'original_label': row['label'],
            'final_label': mock_response['label'],
            'confidence': mock_response['confidence'],
            'method': 'deepseek_api',
            'processing_time': 0.8 + (idx * 0.1)  # Simulate varying response times
        }
        negative_results.append(result)
        print(f"   🤖 '{row['text'][:30]}...' → {mock_response['label']} (conf: {mock_response['confidence']:.2f})")
    
    return positive_results + negative_results

def demonstrate_quality_metrics(results):
    """Demonstrasi quality metrics."""
    print("\n" + "="*60)
    print("           QUALITY METRICS")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    
    # Label distribution
    label_dist = df_results['final_label'].value_counts()
    print(f"\n🏷️  DISTRIBUSI LABEL FINAL:")
    for label, count in label_dist.items():
        percentage = (count / len(df_results)) * 100
        print(f"   {label}: {count} ({percentage:.1f}%)")
    
    # Method distribution
    method_dist = df_results['method'].value_counts()
    print(f"\n⚙️  DISTRIBUSI METODE:")
    for method, count in method_dist.items():
        percentage = (count / len(df_results)) * 100
        print(f"   {method}: {count} ({percentage:.1f}%)")
    
    # Confidence analysis
    avg_confidence = df_results['confidence'].mean()
    min_confidence = df_results['confidence'].min()
    max_confidence = df_results['confidence'].max()
    low_confidence = len(df_results[df_results['confidence'] < 0.7])
    
    print(f"\n🎯 ANALISIS CONFIDENCE:")
    print(f"   Rata-rata: {avg_confidence:.3f}")
    print(f"   Minimum: {min_confidence:.3f}")
    print(f"   Maximum: {max_confidence:.3f}")
    print(f"   Low confidence (<0.7): {low_confidence} sampel")
    
    # Performance analysis
    total_time = df_results['processing_time'].sum()
    avg_time = df_results['processing_time'].mean()
    api_time = df_results[df_results['method'] == 'deepseek_api']['processing_time'].sum()
    
    print(f"\n⚡ ANALISIS PERFORMA:")
    print(f"   Total waktu: {total_time:.2f} detik")
    print(f"   Rata-rata per sampel: {avg_time:.3f} detik")
    print(f"   Waktu API: {api_time:.2f} detik")
    print(f"   Throughput: {len(df_results)/total_time:.1f} sampel/detik")
    
    return {
        'label_distribution': label_dist.to_dict(),
        'method_distribution': method_dist.to_dict(),
        'avg_confidence': avg_confidence,
        'low_confidence_count': low_confidence,
        'total_processing_time': total_time,
        'throughput': len(df_results)/total_time
    }

def demonstrate_comparison():
    """Demonstrasi perbandingan dengan approach tradisional."""
    print("\n" + "="*60)
    print("           PERBANDINGAN APPROACH")
    print("="*60)
    
    print(f"\n📋 TRADITIONAL APPROACH:")
    print(f"   ❌ Semua data → Manual labeling")
    print(f"   ❌ Waktu: ~5 menit per sampel")
    print(f"   ❌ Biaya: Tinggi (human annotator)")
    print(f"   ❌ Konsistensi: Bervariasi antar annotator")
    print(f"   ✅ Akurasi: Tinggi untuk konteks budaya")
    
    print(f"\n🤖 FULL API APPROACH:")
    print(f"   ❌ Semua data → API calls")
    print(f"   ❌ Biaya: Tinggi untuk data yang jelas")
    print(f"   ✅ Waktu: Cepat (~1-2 detik per sampel)")
    print(f"   ✅ Konsistensi: Tinggi")
    print(f"   ⚠️  Akurasi: Baik tapi mungkin miss konteks lokal")
    
    print(f"\n🎯 HYBRID APPROACH (Cost-Efficient):")
    print(f"   ✅ Data jelas → Rule-based (gratis)")
    print(f"   ✅ Data ambigu → API calls")
    print(f"   ✅ Biaya: Optimal (50-70% lebih murah)")
    print(f"   ✅ Waktu: Cepat")
    print(f"   ✅ Konsistensi: Tinggi")
    print(f"   ✅ Akurasi: Baik dengan human validation")

def save_demo_results(cost_analysis, quality_metrics):
    """Simpan hasil demo ke file."""
    results = {
        'demo_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'cost_analysis': cost_analysis,
        'quality_metrics': quality_metrics,
        'strategy_summary': {
            'approach': 'hybrid_cost_efficient',
            'key_benefits': [
                'Significant cost savings (50-70%)',
                'Maintained quality through selective API usage',
                'Fast processing for obvious cases',
                'Scalable approach'
            ],
            'recommended_use_cases': [
                'Large datasets with mixed content',
                'Budget-constrained projects',
                'Production environments',
                'Continuous labeling workflows'
            ]
        }
    }
    
    # Save to file
    output_file = 'demo_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Hasil demo disimpan ke: {output_file}")

def main():
    """Main demo function."""
    print("🚀 DEMO: Cost-Efficient DeepSeek API Labeling Strategy")
    print("="*60)
    print("Demo ini menunjukkan bagaimana strategi pembagian positif/negatif")
    print("dapat menghemat biaya API sambil mempertahankan kualitas labeling.")
    
    # 1. Create sample dataset
    print("\n📝 Membuat sample dataset...")
    df = create_sample_dataset()
    print(f"Dataset dibuat: {len(df)} sampel")
    
    # 2. Demonstrate cost analysis
    cost_analysis = demonstrate_cost_analysis(df)
    
    # 3. Demonstrate labeling process
    results = demonstrate_labeling_process(df)
    
    # 4. Demonstrate quality metrics
    quality_metrics = demonstrate_quality_metrics(results)
    
    # 5. Show comparison
    demonstrate_comparison()
    
    # 6. Save results
    save_demo_results(cost_analysis, quality_metrics)
    
    print("\n" + "="*60)
    print("                    KESIMPULAN")
    print("="*60)
    print(f"✅ Strategi cost-efficient berhasil menghemat {cost_analysis['savings_percentage']:.1f}% biaya")
    print(f"✅ Kualitas labeling tetap terjaga dengan confidence rata-rata {quality_metrics['avg_confidence']:.3f}")
    print(f"✅ Throughput tinggi: {quality_metrics['throughput']:.1f} sampel/detik")
    print(f"✅ Pendekatan hybrid optimal untuk production use")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Setup DeepSeek API key di .env file")
    print("2. Test dengan dataset real menggunakan --mock flag")
    print("3. Run production labeling dengan confidence monitoring")
    print("4. Implement continuous quality assurance")
    
    print("\n📚 DOKUMENTASI LENGKAP:")
    print("- docs/deepseek-api-strategy.md")
    print("- memory-bank/petunjuk-pekerjaan-manual.md")
    print("- src/data_collection/deepseek_labeling_pipeline.py")

if __name__ == "__main__":
    main()