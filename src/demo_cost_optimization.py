#!/usr/bin/env python3
"""Demo Cost Optimization untuk DeepSeek API.

Script ini mendemonstrasikan:
1. Deteksi periode diskon/standar
2. Estimasi biaya real-time
3. Strategi optimasi yang berbeda
4. Monitoring dan peringatan
"""

import os
import sys
import argparse
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.cost_optimizer import CostOptimizer
from utils.logger import setup_logger

def demo_current_status():
    """Demo status saat ini."""
    print("\n" + "="*60)
    print("🕒 STATUS COST OPTIMIZER SAAT INI")
    print("="*60)
    
    optimizer = CostOptimizer()
    status = optimizer.get_status_report()
    
    # Waktu saat ini
    print(f"📅 Waktu UTC: {status['current_time']['utc']}")
    print(f"📅 Waktu WIB: {status['current_time']['local']}")
    
    # Periode harga
    print(f"\n💰 Periode Harga: {status['pricing']['current_period']}")
    print(f"💰 Diskon Aktif: {'✅ YA' if status['pricing']['is_discount'] else '❌ TIDAK'}")
    
    # Harga saat ini
    print(f"\n💵 Harga Input (Cache Hit): ${status['pricing']['input_cache_hit']}/1M tokens")
    print(f"💵 Harga Input (Cache Miss): ${status['pricing']['input_cache_miss']}/1M tokens")
    print(f"💵 Harga Output: ${status['pricing']['output']}/1M tokens")
    
    # Periode diskon berikutnya
    print(f"\n⏰ Periode Diskon Berikutnya:")
    print(f"   Mulai: {status['next_discount_period']['start']}")
    print(f"   Selesai: {status['next_discount_period']['end']}")
    print(f"   Durasi: {status['next_discount_period']['duration_hours']:.1f} jam")
    
    # Potensi penghematan
    print(f"\n💡 Potensi Penghematan:")
    print(f"   Input (Cache Hit): {status['savings_potential']['input_cache_hit']}")
    print(f"   Input (Cache Miss): {status['savings_potential']['input_cache_miss']}")
    print(f"   Output: {status['savings_potential']['output']}")

def demo_cost_calculation():
    """Demo perhitungan biaya."""
    print("\n" + "="*60)
    print("💰 DEMO PERHITUNGAN BIAYA")
    print("="*60)
    
    optimizer = CostOptimizer()
    
    # Simulasi dataset 41,759 samples
    input_tokens = 8_350_000  # 8.35M tokens
    output_tokens = 3_130_000  # 3.13M tokens
    cache_hit_ratio = 0.1  # 10% cache hit
    
    print(f"📊 Dataset: 41,759 samples")
    print(f"📊 Input Tokens: {input_tokens:,}")
    print(f"📊 Output Tokens: {output_tokens:,}")
    print(f"📊 Cache Hit Ratio: {cache_hit_ratio*100:.0f}%")
    
    # Hitung biaya saat ini
    current_cost = optimizer.calculate_cost(input_tokens, output_tokens, cache_hit_ratio)
    
    print(f"\n💵 BIAYA SAAT INI ({current_cost['pricing_period']}):")
    print(f"   Input Cost: ${current_cost['input_cost']:.2f}")
    print(f"   Output Cost: ${current_cost['output_cost']:.2f}")
    print(f"   Total Cost: ${current_cost['total_cost']:.2f}")
    
    # Simulasi biaya di periode berbeda
    print(f"\n📈 PERBANDINGAN BIAYA:")
    
    # Hitung biaya standar
    temp_discount = optimizer.discount_pricing
    optimizer.discount_pricing = optimizer.standard_pricing
    standard_cost = optimizer.calculate_cost(input_tokens, output_tokens, cache_hit_ratio)
    optimizer.discount_pricing = temp_discount
    
    # Hitung biaya diskon
    temp_standard = optimizer.standard_pricing
    optimizer.standard_pricing = optimizer.discount_pricing
    discount_cost = optimizer.calculate_cost(input_tokens, output_tokens, cache_hit_ratio)
    optimizer.standard_pricing = temp_standard
    
    print(f"   Standard Price: ${standard_cost['total_cost']:.2f}")
    print(f"   Discount Price: ${discount_cost['total_cost']:.2f}")
    
    savings = standard_cost['total_cost'] - discount_cost['total_cost']
    savings_percent = (savings / standard_cost['total_cost']) * 100
    
    print(f"   💡 Penghematan: ${savings:.2f} ({savings_percent:.0f}%)")

def demo_strategies():
    """Demo strategi yang berbeda."""
    print("\n" + "="*60)
    print("🎯 DEMO STRATEGI OPTIMASI")
    print("="*60)
    
    optimizer = CostOptimizer()
    strategies = ["discount_only", "always", "warn_expensive"]
    
    for strategy in strategies:
        should_process, reason = optimizer.should_process_now(strategy)
        
        print(f"\n🔧 Strategi: {strategy.upper()}")
        print(f"   Hasil: {'✅ PROSES' if should_process else '⏳ TUNGGU'}")
        print(f"   Alasan: {reason}")
        
        if strategy == "discount_only":
            print(f"   💡 Cocok untuk: Dataset besar, budget terbatas")
            print(f"   ⚠️ Catatan: Hanya 8 jam per hari")
        elif strategy == "always":
            print(f"   💡 Cocok untuk: Deadline ketat, dataset kecil")
            print(f"   ⚠️ Catatan: Biaya lebih tinggi saat periode standar")
        elif strategy == "warn_expensive":
            print(f"   💡 Cocok untuk: Sebagian besar use case")
            print(f"   ⚠️ Catatan: Memerlukan monitoring manual")

def demo_time_simulation():
    """Demo simulasi waktu untuk melihat perubahan periode."""
    print("\n" + "="*60)
    print("⏰ DEMO SIMULASI WAKTU")
    print("="*60)
    
    optimizer = CostOptimizer()
    
    # Simulasi 24 jam ke depan
    current_time = optimizer.get_current_utc_time()
    
    print(f"📅 Simulasi 24 jam dari: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"\n🕒 Jadwal Periode Harga:")
    
    for hour in range(24):
        sim_time = current_time + timedelta(hours=hour)
        is_discount = optimizer.is_discount_period(sim_time)
        
        # Konversi ke WIB
        wib_time = sim_time.replace(tzinfo=optimizer.utc).astimezone(optimizer.timezone)
        
        status = "🟢 DISKON" if is_discount else "🔴 STANDAR"
        print(f"   {wib_time.strftime('%H:%M')} WIB - {status}")

def demo_cost_monitoring():
    """Demo monitoring biaya real-time."""
    print("\n" + "="*60)
    print("📊 DEMO MONITORING BIAYA REAL-TIME")
    print("="*60)
    
    optimizer = CostOptimizer()
    
    # Simulasi processing 5 batch
    total_cost = 0.0
    total_savings = 0.0
    
    print(f"🚀 Simulasi processing 5 batch...")
    
    for batch in range(1, 6):
        # Simulasi token per batch
        batch_input = 200_000  # 200K tokens
        batch_output = 75_000   # 75K tokens
        
        # Hitung biaya
        cost_info = optimizer.calculate_cost(batch_input, batch_output, 0.1)
        total_cost += cost_info['total_cost']
        
        # Hitung penghematan jika diskon
        if cost_info['is_discount']:
            # Simulasi biaya standar
            temp_pricing = optimizer.discount_pricing
            optimizer.discount_pricing = optimizer.standard_pricing
            standard_info = optimizer.calculate_cost(batch_input, batch_output, 0.1)
            optimizer.discount_pricing = temp_pricing
            
            batch_savings = standard_info['total_cost'] - cost_info['total_cost']
            total_savings += batch_savings
            
            print(f"   Batch {batch}: ${cost_info['total_cost']:.4f} ({cost_info['pricing_period']}) - Hemat: ${batch_savings:.4f}")
        else:
            print(f"   Batch {batch}: ${cost_info['total_cost']:.4f} ({cost_info['pricing_period']})")
        
        # Simulasi delay
        time.sleep(0.5)
    
    print(f"\n📈 RINGKASAN:")
    print(f"   Total Cost: ${total_cost:.4f}")
    if total_savings > 0:
        print(f"   Total Savings: ${total_savings:.4f}")
        print(f"   Efficiency: {(total_savings/(total_cost+total_savings))*100:.1f}% saved")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Demo Cost Optimization untuk DeepSeek API")
    parser.add_argument("--demo", choices=["status", "cost", "strategies", "time", "monitoring", "all"],
                       default="all", help="Pilih demo yang ingin dijalankan")
    
    args = parser.parse_args()
    
    print("🚀 DEMO COST OPTIMIZATION - DEEPSEEK API")
    print("="*60)
    print("📍 Timezone: Asia/Jakarta (GMT+7)")
    print("📍 API: DeepSeek Chat Model")
    print("📍 Dataset: 41,759 samples (Javanese Hate Speech)")
    
    if args.demo == "status" or args.demo == "all":
        demo_current_status()
    
    if args.demo == "cost" or args.demo == "all":
        demo_cost_calculation()
    
    if args.demo == "strategies" or args.demo == "all":
        demo_strategies()
    
    if args.demo == "time" or args.demo == "all":
        demo_time_simulation()
    
    if args.demo == "monitoring" or args.demo == "all":
        demo_cost_monitoring()
    
    print("\n" + "="*60)
    print("✅ DEMO SELESAI!")
    print("📚 Lihat dokumentasi lengkap di: docs/cost-optimization-strategy.md")
    print("🔧 Gunakan persistent_labeling_pipeline.py dengan parameter cost_strategy")
    print("="*60)

if __name__ == "__main__":
    main()