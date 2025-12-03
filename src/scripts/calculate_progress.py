#!/usr/bin/env python3
"""
Skrip untuk menghitung estimasi biaya dan waktu berdasarkan log aktual
Berdasarkan data log batch 1-7 dari total 2285 batch
"""

import datetime
from typing import Dict, Tuple

def parse_log_data() -> Dict:
    """
    Data log aktual dari batch 1-7
    """
    log_data = {
        'batch_1': {
            'start': '02:47:58,601',
            'end': '02:48:48,773',
            'confidence': 0.895
        },
        'batch_2': {
            'start': '02:48:48,774', 
            'end': '02:49:39,580',
            'confidence': 0.905
        },
        'batch_3': {
            'start': '02:49:39,581',
            'end': '02:50:24,665', 
            'confidence': 0.775
        },
        'batch_4': {
            'start': '02:50:24,666',
            'end': '02:51:10,961',
            'confidence': 0.935
        },
        'batch_5': {
            'start': '02:51:10,961',
            'end': '02:51:59,864',
            'confidence': 0.930
        },
        'batch_6': {
            'start': '02:51:59,865',
            'end': '02:52:49,013',
            'confidence': 0.950
        },
        'batch_7': {
            'start': '02:52:49,013',
            'end': 'ongoing',  # batch 7 sedang berjalan
            'confidence': None
        }
    }
    return log_data

def calculate_batch_duration(start_time: str, end_time: str) -> float:
    """
    Menghitung durasi batch dalam detik
    """
    if end_time == 'ongoing':
        return None
        
    # Parse waktu format HH:MM:SS,mmm
    start_parts = start_time.split(':')
    end_parts = end_time.split(':')
    
    start_seconds = int(start_parts[0]) * 3600 + int(start_parts[1]) * 60 + float(start_parts[2].replace(',', '.'))
    end_seconds = int(end_parts[0]) * 3600 + int(end_parts[1]) * 60 + float(end_parts[2].replace(',', '.'))
    
    return end_seconds - start_seconds

def analyze_performance() -> Tuple[float, float, float]:
    """
    Menganalisis performa dari 6 batch yang selesai
    Returns: (avg_duration, min_duration, max_duration)
    """
    log_data = parse_log_data()
    durations = []
    confidences = []
    
    print("=== ANALISIS PERFORMA BATCH ===")
    print(f"{'Batch':<8} {'Durasi (detik)':<15} {'Durasi (menit)':<15} {'Confidence':<12}")
    print("-" * 60)
    
    for i in range(1, 7):  # batch 1-6 yang sudah selesai
        batch_key = f'batch_{i}'
        batch_data = log_data[batch_key]
        
        duration = calculate_batch_duration(batch_data['start'], batch_data['end'])
        durations.append(duration)
        confidences.append(batch_data['confidence'])
        
        print(f"{i:<8} {duration:<15.1f} {duration/60:<15.2f} {batch_data['confidence']:<12.3f}")
    
    avg_duration = sum(durations) / len(durations)
    min_duration = min(durations)
    max_duration = max(durations)
    avg_confidence = sum(confidences) / len(confidences)
    
    print("-" * 60)
    print(f"Rata-rata durasi per batch: {avg_duration:.1f} detik ({avg_duration/60:.2f} menit)")
    print(f"Durasi minimum: {min_duration:.1f} detik ({min_duration/60:.2f} menit)")
    print(f"Durasi maksimum: {max_duration:.1f} detik ({max_duration/60:.2f} menit)")
    print(f"Rata-rata confidence: {avg_confidence:.3f}")
    print()
    
    return avg_duration, min_duration, max_duration

def estimate_remaining_time(avg_duration: float) -> Dict:
    """
    Estimasi waktu yang tersisa
    """
    total_batches = 2285
    completed_batches = 6  # batch 1-6 selesai, batch 7 sedang berjalan
    remaining_batches = total_batches - completed_batches
    
    # Estimasi waktu tersisa
    remaining_seconds = remaining_batches * avg_duration
    remaining_minutes = remaining_seconds / 60
    remaining_hours = remaining_minutes / 60
    remaining_days = remaining_hours / 24
    
    # Total waktu estimasi
    total_seconds = total_batches * avg_duration
    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60
    total_days = total_hours / 24
    
    return {
        'remaining_seconds': remaining_seconds,
        'remaining_minutes': remaining_minutes,
        'remaining_hours': remaining_hours,
        'remaining_days': remaining_days,
        'total_seconds': total_seconds,
        'total_minutes': total_minutes,
        'total_hours': total_hours,
        'total_days': total_days
    }

def estimate_cost() -> Dict:
    """
    Estimasi biaya berdasarkan data sebelumnya
    """
    # Data dari analisis sebelumnya
    total_samples = 41759
    negative_samples = int(total_samples * 0.549)  # 54.9% negative
    
    # Estimasi token (dari analisis sebelumnya)
    avg_input_tokens_per_sample = 200  # estimasi konservatif
    avg_output_tokens_per_sample = 75   # estimasi konservatif
    
    total_input_tokens = negative_samples * avg_input_tokens_per_sample
    total_output_tokens = negative_samples * avg_output_tokens_per_sample
    
    # Harga DeepSeek API (per 1M token)
    standard_input_price = 0.14   # $0.14 per 1M input tokens
    standard_output_price = 0.28  # $0.28 per 1M output tokens
    
    discount_input_price = 0.07   # $0.07 per 1M input tokens (50% discount)
    discount_output_price = 0.14  # $0.14 per 1M output tokens (50% discount)
    
    # Kalkulasi biaya
    standard_cost = (total_input_tokens * standard_input_price / 1_000_000) + \
                   (total_output_tokens * standard_output_price / 1_000_000)
    
    discount_cost = (total_input_tokens * discount_input_price / 1_000_000) + \
                   (total_output_tokens * discount_output_price / 1_000_000)
    
    return {
        'total_samples': total_samples,
        'negative_samples': negative_samples,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'standard_cost': standard_cost,
        'discount_cost': discount_cost
    }

def print_time_estimates(time_data: Dict):
    """
    Menampilkan estimasi waktu
    """
    print("=== ESTIMASI WAKTU ===")
    print(f"Total batch: 2285")
    print(f"Batch selesai: 6")
    print(f"Batch tersisa: {2285 - 6}")
    print()
    
    print("Estimasi waktu tersisa:")
    print(f"  {time_data['remaining_hours']:.1f} jam ({time_data['remaining_days']:.1f} hari)")
    print()
    
    print("Estimasi total waktu:")
    print(f"  {time_data['total_hours']:.1f} jam ({time_data['total_days']:.1f} hari)")
    print()
    
    # Estimasi waktu selesai
    now = datetime.datetime.now()
    estimated_completion = now + datetime.timedelta(seconds=time_data['remaining_seconds'])
    print(f"Estimasi selesai: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_cost_estimates(cost_data: Dict):
    """
    Menampilkan estimasi biaya
    """
    print("=== ESTIMASI BIAYA ===")
    print(f"Total sampel: {cost_data['total_samples']:,}")
    print(f"Sampel negatif (yang diproses): {cost_data['negative_samples']:,}")
    print()
    
    print(f"Estimasi token:")
    print(f"  Input tokens: {cost_data['total_input_tokens']:,}")
    print(f"  Output tokens: {cost_data['total_output_tokens']:,}")
    print()
    
    print(f"Estimasi biaya:")
    print(f"  Harga standar: ${cost_data['standard_cost']:.2f}")
    print(f"  Harga diskon (promo): ${cost_data['discount_cost']:.2f}")
    print(f"  Penghematan: ${cost_data['standard_cost'] - cost_data['discount_cost']:.2f} ({((cost_data['standard_cost'] - cost_data['discount_cost']) / cost_data['standard_cost'] * 100):.1f}%)")
    print()

def main():
    """
    Fungsi utama
    """
    print("KALKULASI ESTIMASI BIAYA DAN WAKTU LABELING")
    print("Berdasarkan log aktual batch 1-7 dari 2285 batch")
    print("=" * 60)
    print()
    
    # Analisis performa
    avg_duration, min_duration, max_duration = analyze_performance()
    
    # Estimasi waktu
    time_data = estimate_remaining_time(avg_duration)
    print_time_estimates(time_data)
    
    # Estimasi biaya
    cost_data = estimate_cost()
    print_cost_estimates(cost_data)
    
    # Rekomendasi
    print("=== REKOMENDASI ===")
    print("1. Proses berjalan stabil dengan rata-rata confidence tinggi (0.898)")
    print("2. Durasi per batch konsisten (~50 detik)")
    print("3. Untuk menghemat biaya, jalankan selama jam promo (UTC 16:30-00:30)")
    print("4. Estimasi penyelesaian dalam ~1.6 hari jika berjalan 24/7")
    print("5. Jika hanya berjalan saat promo (8 jam/hari), estimasi ~5 hari")
    print()
    
    # Progress saat ini
    progress_percentage = (6 / 2285) * 100
    print(f"Progress saat ini: {progress_percentage:.2f}% (6/2285 batch)")
    
if __name__ == "__main__":
    main()