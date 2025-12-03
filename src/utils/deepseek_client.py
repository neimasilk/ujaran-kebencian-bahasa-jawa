"""DeepSeek API client untuk labeling data ujaran kebencian Bahasa Jawa.

Modul ini menyediakan interface untuk berinteraksi dengan DeepSeek API
untuk melakukan labeling otomatis pada data yang berlabel 'negative'.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import Settings

# Setup logger
logger = logging.getLogger(__name__)

@dataclass
class LabelingResult:
    """Hasil labeling dari DeepSeek API."""
    text: str
    label_id: int
    confidence: float
    response_time: float
    error: Optional[str] = None

class DeepSeekAPIClient:
    """Client untuk berinteraksi dengan DeepSeek API."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Inisialisasi DeepSeek API client.
        
        Args:
            settings: Instance Settings, jika None akan membuat instance baru
        """
        self.settings = settings or Settings()
        self.base_url = self.settings.deepseek_base_url
        self.api_key = self.settings.deepseek_api_key
        self.model = self.settings.deepseek_model
        self.max_tokens = self.settings.deepseek_max_tokens
        self.temperature = self.settings.deepseek_temperature
        self.rate_limit = self.settings.deepseek_rate_limit
        
        # Initialize OpenAI client with DeepSeek configuration
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.request_window_start = time.time()
        
        # Validate API key
        if not self.api_key:
            logger.warning("DeepSeek API key not provided. Set DEEPSEEK_API_KEY environment variable.")
    
    def _check_rate_limit(self) -> None:
        """Implementasi rate limiting sederhana."""
        current_time = time.time()
        
        # Reset counter setiap menit
        if current_time - self.request_window_start >= 60:
            self.request_count = 0
            self.request_window_start = current_time
        
        # Check if we've hit the rate limit
        if self.request_count >= self.rate_limit:
            sleep_time = 60 - (current_time - self.request_window_start)
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
                self.request_count = 0
                self.request_window_start = time.time()
    
    def _create_prompt(self, text: str) -> str:
        """Membuat prompt untuk DeepSeek API.
        
        Args:
            text: Teks Bahasa Jawa yang akan diklasifikasi
            
        Returns:
            Prompt yang diformat untuk DeepSeek
        """
        prompt = f"""
Anda adalah ahli dalam mendeteksi ujaran kebencian dalam Bahasa Jawa dengan pemahaman mendalam tentang konteks budaya dan linguistik Jawa.

Klasifikasikan teks Bahasa Jawa berikut ke dalam salah satu kategori:

0: Bukan Ujaran Kebencian - Teks netral, positif, atau kritik membangun tanpa unsur merendahkan
1: Ujaran Kebencian Ringan - Sindiran halus, ejekan terselubung, atau pasemon yang menyiratkan ketidaksukaan
2: Ujaran Kebencian Sedang - Hinaan langsung, cercaan, atau penggunaan bahasa kasar yang merendahkan
3: Ujaran Kebencian Berat - Ancaman kekerasan, hasutan, dehumanisasi, atau diskriminasi sistematis

Pertimbangkan:
- Tingkatan bahasa (ngoko vs krama) dan konteks penggunaannya
- Metafora dan ungkapan khas Jawa (pasemon)
- Konteks budaya dan sosial
- Target ujaran (SARA, gender, orientasi seksual, kondisi fisik)

Teks: "{text}"

Jawab HANYA dengan format: angka|kepercayaan
Contoh: 2|0.85

Jangan berikan penjelasan tambahan.
"""
        return prompt
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _make_api_request(self, prompt: str) -> str:
        """Melakukan request ke DeepSeek API dengan retry logic.
        
        Args:
            prompt: Prompt untuk dikirim ke API
            
        Returns:
            Response dari API
            
        Raises:
            Exception: Jika request gagal setelah retry
        """
        if not self.api_key:
            raise ValueError("DeepSeek API key not configured")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for Javanese hate speech detection."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"DeepSeek API request failed: {e}")
            raise
    
    def _parse_response(self, response: str) -> Tuple[int, float]:
        """Parse response dari DeepSeek API.
        
        Args:
            response: Response string dari DeepSeek
            
        Returns:
            Tuple berisi (label_id, confidence_score)
        """
        try:
            # Expected format: "2|0.85"
            parts = response.strip().split('|')
            if len(parts) == 2:
                label_id = int(parts[0])
                confidence = float(parts[1])
                
                # Validasi label_id
                if label_id not in [0, 1, 2, 3]:
                    logger.warning(f"Invalid label_id: {label_id}, defaulting to 1")
                    label_id = 1
                    
                # Validasi confidence
                if not (0.0 <= confidence <= 1.0):
                    logger.warning(f"Invalid confidence: {confidence}, defaulting to 0.5")
                    confidence = 0.5
                    
                return label_id, confidence
            else:
                logger.error(f"Invalid response format: {response}")
                return 1, 0.5  # Default ke "Ringan"
                
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing DeepSeek response '{response}': {e}")
            return 1, 0.5  # Default ke "Ringan"
    
    def label_single_text(self, text: str) -> LabelingResult:
        """Melabeli satu teks menggunakan DeepSeek API.
        
        Args:
            text: Teks Bahasa Jawa yang akan diklasifikasi
            
        Returns:
            LabelingResult dengan hasil klasifikasi
        """
        start_time = time.time()
        
        try:
            # Rate limiting
            self._check_rate_limit()
            
            # Create prompt
            prompt = self._create_prompt(text)
            
            # Make API request
            response = self._make_api_request(prompt)
            
            # Parse response
            label_id, confidence = self._parse_response(response)
            
            # Update rate limiting counter
            self.request_count += 1
            
            response_time = time.time() - start_time
            
            return LabelingResult(
                text=text,
                label_id=label_id,
                confidence=confidence,
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Error labeling text: {e}")
            
            return LabelingResult(
                text=text,
                label_id=1,  # Default ke "Ringan"
                confidence=0.3,  # Low confidence untuk error
                response_time=response_time,
                error=str(e)
            )
    
    def label_batch(self, texts: List[str], 
                         batch_delay: float = 1.0) -> List[LabelingResult]:
        """Melabeli batch teks menggunakan DeepSeek API.
        
        Args:
            texts: List teks yang akan diklasifikasi
            batch_delay: Delay antar request dalam detik
            
        Returns:
            List LabelingResult
        """
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            
            result = self.label_single_text(text)
            results.append(result)
            
            # Add delay between requests (additional rate limiting)
            if i < len(texts) - 1:  # Don't delay after last request
                time.sleep(batch_delay)
        
        return results
    
    def get_usage_stats(self) -> Dict:
        """Mendapatkan statistik penggunaan API.
        
        Returns:
            Dictionary berisi statistik penggunaan
        """
        current_time = time.time()
        window_remaining = 60 - (current_time - self.request_window_start)
        
        return {
            "requests_in_current_window": self.request_count,
            "rate_limit": self.rate_limit,
            "window_time_remaining": max(0, window_remaining),
            "requests_remaining": max(0, self.rate_limit - self.request_count)
        }


class MockDeepSeekClient(DeepSeekAPIClient):
    """Mock client untuk testing tanpa menggunakan API sebenarnya."""
    
    def __init__(self, settings: Optional[Settings] = None):
        super().__init__(settings)
        self.mock_responses = {
            # Contoh responses untuk testing
            "positive_text": (0, 0.9),
            "ringan_text": (1, 0.7),
            "sedang_text": (2, 0.8),
            "berat_text": (3, 0.95)
        }
    
    def _make_api_request(self, prompt: str) -> str:
        """Mock API request yang mengembalikan response simulasi."""
        # Extract text from prompt untuk mock response
        text_start = prompt.find('Teks: "') + 7
        text_end = prompt.find('"', text_start)
        text = prompt[text_start:text_end].lower() if text_start > 6 and text_end > text_start else "default"
        
        # Simulasi network delay berdasarkan text length (deterministik)
        delay = 0.1 + (len(text) % 5) * 0.01  # 0.1-0.14 seconds
        time.sleep(delay)
        
        # Mock responses berdasarkan content (deterministik) - SAMA dengan MockParallelDeepSeekClient
        if any(word in text for word in ['positif', 'bagus', 'seneng', 'terima kasih', 'selamat']):
            return "0|0.9"
        elif any(word in text for word in ['sialan', 'brengsek', 'bodoh', 'anjing', 'bangsat']):
            return "2|0.8"
        elif any(word in text for word in ['mateni', 'bunuh', 'berantas', 'pateni', 'usir']):
            return "3|0.95"
        elif any(word in text for word in ['angel', 'ora', 'kudu', 'memang', 'tipik']):
            return "1|0.75"
        else:
            # Default berdasarkan text hash untuk konsistensi
            text_hash = abs(hash(text)) % 4
            if text_hash == 0:
                return "0|0.6"
            elif text_hash == 1:
                return "1|0.7"
            elif text_hash == 2:
                return "2|0.65"
            else:
                return "1|0.7"


# Factory function untuk membuat client
def create_deepseek_client(mock: bool = False, 
                          settings: Optional[Settings] = None) -> DeepSeekAPIClient:
    """Factory function untuk membuat DeepSeek client.
    
    Args:
        mock: Jika True, akan mengembalikan MockDeepSeekClient
        settings: Instance Settings
        
    Returns:
        DeepSeekAPIClient instance
    """
    if mock:
        return MockDeepSeekClient(settings)
    else:
        return DeepSeekAPIClient(settings)


if __name__ == "__main__":
    # Example usage
    def main():
        # Create mock client for testing
        client = create_deepseek_client(mock=True)
        
        # Test texts
        test_texts = [
            "Aku ora sengit marang Muslim",  # Should be 0 (Bukan ujaran kebencian)
            "Wong cendhek piye wae ora pinter",  # Should be 1 (Ringan)
            "Dalit minangka sampah masyarakat",  # Should be 2 (Sedang)
            "Wong Yahudi kudu mati"  # Should be 3 (Berat)
        ]
        
        # Label batch
        results = client.label_batch(test_texts)
        
        # Print results
        print("\n=== HASIL LABELING DEEPSEEK ===")
        for result in results:
            print(f"Text: {result.text[:50]}...")
            print(f"Label: {result.label_id}, Confidence: {result.confidence:.2f}")
            print(f"Response time: {result.response_time:.2f}s")
            if result.error:
                print(f"Error: {result.error}")
            print("-" * 50)
        
        # Print usage stats
        stats = client.get_usage_stats()
        print("\n=== USAGE STATS ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    # Run example
    main()