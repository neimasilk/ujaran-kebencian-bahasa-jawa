"""Cost Optimizer untuk DeepSeek API berdasarkan jam operasional.

Modul ini mengoptimalkan biaya dengan:
1. Mendeteksi jam murah/mahal berdasarkan UTC dan GMT+7
2. Memberikan peringatan saat jam mahal
3. Otomatis pause/resume berdasarkan strategi biaya
4. Estimasi biaya real-time
"""

import datetime
import pytz
import time
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PricingInfo:
    """Informasi harga DeepSeek API."""
    input_cache_hit: float
    input_cache_miss: float
    output: float
    is_discount: bool
    period_name: str

class CostOptimizer:
    """Optimizer biaya untuk DeepSeek API berdasarkan jam operasional."""
    
    def __init__(self, timezone: str = "Asia/Jakarta"):
        """Initialize cost optimizer.
        
        Args:
            timezone: Timezone untuk konversi waktu (default: Asia/Jakarta GMT+7)
        """
        self.timezone = pytz.timezone(timezone)
        self.utc = pytz.UTC
        
        # Pricing DeepSeek API (per 1M tokens)
        self.standard_pricing = PricingInfo(
            input_cache_hit=0.07,
            input_cache_miss=0.27,
            output=1.10,
            is_discount=False,
            period_name="Standard (UTC 00:30-16:30)"
        )
        
        self.discount_pricing = PricingInfo(
            input_cache_hit=0.035,  # 50% OFF
            input_cache_miss=0.135,  # 50% OFF
            output=0.550,  # 50% OFF
            is_discount=True,
            period_name="Discount (UTC 16:30-00:30)"
        )
        
        self.logger = logging.getLogger(__name__)
    
    def get_current_utc_time(self) -> datetime.datetime:
        """Dapatkan waktu UTC saat ini."""
        return datetime.datetime.now(self.utc)
    
    def get_current_local_time(self) -> datetime.datetime:
        """Dapatkan waktu lokal (GMT+7) saat ini."""
        return datetime.datetime.now(self.timezone)
    
    def is_discount_period(self, utc_time: Optional[datetime.datetime] = None) -> bool:
        """Cek apakah saat ini dalam periode diskon.
        
        Periode diskon: UTC 16:30-00:30 (GMT+7: 23:30-07:30)
        
        Args:
            utc_time: Waktu UTC untuk dicek (default: waktu sekarang)
            
        Returns:
            True jika dalam periode diskon
        """
        if utc_time is None:
            utc_time = self.get_current_utc_time()
        
        # Konversi ke waktu UTC
        utc_hour = utc_time.hour
        utc_minute = utc_time.minute
        utc_time_decimal = utc_hour + utc_minute / 60.0
        
        # Periode diskon: 16:30-00:30 UTC (melintasi midnight)
        discount_start = 16.5  # 16:30
        discount_end = 0.5     # 00:30
        
        # Cek apakah dalam periode diskon
        if discount_start <= utc_time_decimal <= 23.99:  # 16:30-23:59
            return True
        elif 0.0 <= utc_time_decimal <= discount_end:     # 00:00-00:30
            return True
        else:
            return False
    
    def get_current_pricing(self) -> PricingInfo:
        """Dapatkan informasi harga saat ini."""
        if self.is_discount_period():
            return self.discount_pricing
        else:
            return self.standard_pricing
    
    def get_next_discount_period(self) -> Tuple[datetime.datetime, datetime.datetime]:
        """Dapatkan waktu mulai dan selesai periode diskon berikutnya.
        
        Returns:
            Tuple (start_time, end_time) dalam timezone lokal
        """
        now_utc = self.get_current_utc_time()
        
        # Hitung periode diskon berikutnya
        if self.is_discount_period(now_utc):
            # Sedang dalam periode diskon, cari akhir periode
            if now_utc.hour >= 16:  # Masih hari yang sama
                end_utc = now_utc.replace(hour=0, minute=30, second=0, microsecond=0) + datetime.timedelta(days=1)
            else:  # Sudah lewat midnight
                end_utc = now_utc.replace(hour=0, minute=30, second=0, microsecond=0)
            
            start_utc = now_utc  # Sudah dimulai
        else:
            # Tidak dalam periode diskon, cari periode berikutnya
            if now_utc.hour < 16 or (now_utc.hour == 16 and now_utc.minute < 30):
                # Hari yang sama
                start_utc = now_utc.replace(hour=16, minute=30, second=0, microsecond=0)
                end_utc = now_utc.replace(hour=0, minute=30, second=0, microsecond=0) + datetime.timedelta(days=1)
            else:
                # Besok
                start_utc = now_utc.replace(hour=16, minute=30, second=0, microsecond=0) + datetime.timedelta(days=1)
                end_utc = now_utc.replace(hour=0, minute=30, second=0, microsecond=0) + datetime.timedelta(days=2)
        
        # Konversi ke timezone lokal
        start_local = start_utc.astimezone(self.timezone)
        end_local = end_utc.astimezone(self.timezone)
        
        return start_local, end_local
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, 
                     cache_hit_ratio: float = 0.0) -> Dict:
        """Hitung biaya untuk jumlah token tertentu.
        
        Args:
            input_tokens: Jumlah input tokens
            output_tokens: Jumlah output tokens
            cache_hit_ratio: Rasio cache hit (0.0-1.0)
            
        Returns:
            Dictionary dengan informasi biaya
        """
        pricing = self.get_current_pricing()
        
        # Hitung biaya input
        cache_hit_tokens = int(input_tokens * cache_hit_ratio)
        cache_miss_tokens = input_tokens - cache_hit_tokens
        
        input_cost = (
            (cache_hit_tokens / 1_000_000) * pricing.input_cache_hit +
            (cache_miss_tokens / 1_000_000) * pricing.input_cache_miss
        )
        
        # Hitung biaya output
        output_cost = (output_tokens / 1_000_000) * pricing.output
        
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_hit_tokens": cache_hit_tokens,
            "cache_miss_tokens": cache_miss_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "pricing_period": pricing.period_name,
            "is_discount": pricing.is_discount,
            "savings_vs_standard": 0 if not pricing.is_discount else total_cost,
            "timestamp_utc": self.get_current_utc_time().isoformat(),
            "timestamp_local": self.get_current_local_time().isoformat()
        }
    
    def should_process_now(self, strategy: str = "discount_only") -> Tuple[bool, str]:
        """Tentukan apakah harus memproses sekarang berdasarkan strategi.
        
        Args:
            strategy: Strategi biaya ('discount_only', 'always', 'warn_expensive')
            
        Returns:
            Tuple (should_process, reason)
        """
        is_discount = self.is_discount_period()
        pricing = self.get_current_pricing()
        
        if strategy == "discount_only":
            if is_discount:
                return True, f"✅ Periode diskon aktif: {pricing.period_name}"
            else:
                next_start, next_end = self.get_next_discount_period()
                return False, f"⏳ Menunggu periode diskon berikutnya: {next_start.strftime('%H:%M')} - {next_end.strftime('%H:%M')} WIB"
        
        elif strategy == "always":
            if is_discount:
                return True, f"✅ Memproses dengan harga diskon: {pricing.period_name}"
            else:
                return True, f"⚠️ Memproses dengan harga standar: {pricing.period_name}"
        
        elif strategy == "warn_expensive":
            if is_discount:
                return True, f"✅ Periode diskon aktif: {pricing.period_name}"
            else:
                next_start, next_end = self.get_next_discount_period()
                return True, f"⚠️ PERINGATAN: Harga mahal! {pricing.period_name}. Diskon berikutnya: {next_start.strftime('%H:%M')} WIB"
        
        else:
            raise ValueError(f"Strategi tidak dikenal: {strategy}")
    
    def wait_for_discount_period(self, check_interval: int = 300) -> None:
        """Tunggu hingga periode diskon dimulai.
        
        Args:
            check_interval: Interval pengecekan dalam detik (default: 5 menit)
        """
        if self.is_discount_period():
            self.logger.info("✅ Sudah dalam periode diskon")
            return
        
        next_start, next_end = self.get_next_discount_period()
        wait_seconds = (next_start - self.get_current_local_time()).total_seconds()
        
        self.logger.info(f"⏳ Menunggu periode diskon: {next_start.strftime('%Y-%m-%d %H:%M:%S')} WIB")
        self.logger.info(f"⏳ Estimasi waktu tunggu: {wait_seconds/3600:.1f} jam")
        
        while not self.is_discount_period():
            remaining = (next_start - self.get_current_local_time()).total_seconds()
            if remaining <= 0:
                break
            
            self.logger.info(f"⏳ Sisa waktu tunggu: {remaining/3600:.1f} jam")
            time.sleep(min(check_interval, remaining))
        
        self.logger.info("✅ Periode diskon dimulai!")
    
    def get_status_report(self) -> Dict:
        """Dapatkan laporan status lengkap."""
        current_pricing = self.get_current_pricing()
        next_start, next_end = self.get_next_discount_period()
        
        return {
            "current_time": {
                "utc": self.get_current_utc_time().isoformat(),
                "local": self.get_current_local_time().isoformat(),
                "timezone": str(self.timezone)
            },
            "pricing": {
                "current_period": current_pricing.period_name,
                "is_discount": current_pricing.is_discount,
                "input_cache_hit": current_pricing.input_cache_hit,
                "input_cache_miss": current_pricing.input_cache_miss,
                "output": current_pricing.output
            },
            "next_discount_period": {
                "start": next_start.isoformat(),
                "end": next_end.isoformat(),
                "duration_hours": (next_end - next_start).total_seconds() / 3600
            },
            "savings_potential": {
                "input_cache_hit": f"{((self.standard_pricing.input_cache_hit - self.discount_pricing.input_cache_hit) / self.standard_pricing.input_cache_hit * 100):.0f}%",
                "input_cache_miss": f"{((self.standard_pricing.input_cache_miss - self.discount_pricing.input_cache_miss) / self.standard_pricing.input_cache_miss * 100):.0f}%",
                "output": f"{((self.standard_pricing.output - self.discount_pricing.output) / self.standard_pricing.output * 100):.0f}%"
            }
        }

# Contoh penggunaan
if __name__ == "__main__":
    optimizer = CostOptimizer()
    
    # Cek status saat ini
    print("=== STATUS COST OPTIMIZER ===")
    status = optimizer.get_status_report()
    print(f"Waktu sekarang (WIB): {status['current_time']['local']}")
    print(f"Periode harga: {status['pricing']['current_period']}")
    print(f"Diskon aktif: {status['pricing']['is_discount']}")
    
    # Cek apakah harus memproses
    should_process, reason = optimizer.should_process_now("discount_only")
    print(f"\nHarus memproses: {should_process}")
    print(f"Alasan: {reason}")
    
    # Estimasi biaya untuk 1000 samples
    cost_info = optimizer.calculate_cost(
        input_tokens=8_350_000,  # 8.35M tokens
        output_tokens=3_130_000,  # 3.13M tokens
        cache_hit_ratio=0.1
    )
    print(f"\n=== ESTIMASI BIAYA (41,759 samples) ===")
    print(f"Total biaya: ${cost_info['total_cost']:.2f}")
    print(f"Periode: {cost_info['pricing_period']}")
    if cost_info['is_discount']:
        standard_cost = optimizer.calculate_cost(
            input_tokens=8_350_000,
            output_tokens=3_130_000,
            cache_hit_ratio=0.1
        )
        # Hitung dengan harga standar untuk perbandingan
        temp_pricing = optimizer.discount_pricing
        optimizer.discount_pricing = optimizer.standard_pricing
        standard_info = optimizer.calculate_cost(
            input_tokens=8_350_000,
            output_tokens=3_130_000,
            cache_hit_ratio=0.1
        )
        optimizer.discount_pricing = temp_pricing
        
        savings = standard_info['total_cost'] - cost_info['total_cost']
        print(f"Penghematan vs harga standar: ${savings:.2f} ({savings/standard_info['total_cost']*100:.0f}%)")