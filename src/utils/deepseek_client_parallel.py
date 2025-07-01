#!/usr/bin/env python3
"""
Parallel DeepSeek Client untuk labeling data secara paralel.

Modul ini mengimplementasikan versi paralel dari DeepSeek client yang dapat:
- Memproses multiple requests secara concurrent menggunakan asyncio
- Menggunakan ThreadPoolExecutor untuk I/O bound operations
- Mengelola rate limiting dengan lebih efisien
- Memberikan progress tracking yang real-time

Usage:
    from utils.deepseek_client_parallel import ParallelDeepSeekClient
    
    client = ParallelDeepSeekClient(max_workers=5)
    results = await client.label_batch_parallel(texts)
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import Settings
from utils.deepseek_client import LabelingResult, DeepSeekAPIClient

# Setup logger
logger = logging.getLogger(__name__)

class RateLimiter:
    """Thread-safe rate limiter untuk parallel requests."""
    
    def __init__(self, max_requests_per_minute: int = 100):
        self.max_requests = max_requests_per_minute
        self.requests = Queue()
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """Acquire permission untuk membuat request.
        
        Returns:
            True jika boleh membuat request, False jika harus wait
        """
        current_time = time.time()
        
        with self.lock:
            # Remove old requests (older than 1 minute)
            while not self.requests.empty():
                if current_time - self.requests.queue[0] > 60:
                    self.requests.get()
                else:
                    break
            
            # Check if we can make a new request
            if self.requests.qsize() < self.max_requests:
                self.requests.put(current_time)
                return True
            else:
                return False
    
    def wait_time(self) -> float:
        """Calculate berapa lama harus wait sebelum bisa request lagi."""
        if self.requests.empty():
            return 0.0
        
        oldest_request = self.requests.queue[0]
        wait_time = 60 - (time.time() - oldest_request)
        return max(0.0, wait_time)

class ParallelDeepSeekClient(DeepSeekAPIClient):
    """Parallel version dari DeepSeek client dengan async support."""
    
    def __init__(self, settings: Optional[Settings] = None, max_workers: int = 5):
        """Initialize parallel DeepSeek client.
        
        Args:
            settings: Settings instance
            max_workers: Maximum number of concurrent workers
        """
        super().__init__(settings)
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(self.rate_limit)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"Initialized ParallelDeepSeekClient with {max_workers} workers")
    
    def _make_api_request_with_rate_limit(self, prompt: str) -> str:
        """Make API request dengan rate limiting yang thread-safe.
        
        Args:
            prompt: Prompt untuk dikirim ke API
            
        Returns:
            Response dari API
        """
        # Wait for rate limit if needed
        while not self.rate_limiter.acquire():
            wait_time = self.rate_limiter.wait_time()
            if wait_time > 0:
                logger.debug(f"Rate limit hit, waiting {wait_time:.2f} seconds")
                time.sleep(min(wait_time, 1.0))  # Sleep max 1 second at a time
        
        # Make the actual API request
        return self._make_api_request(prompt)
    
    def label_single_text_sync(self, text: str) -> LabelingResult:
        """Synchronous version untuk digunakan dalam thread pool.
        
        Args:
            text: Teks yang akan diklasifikasi
            
        Returns:
            LabelingResult
        """
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self._create_prompt(text)
            
            # Make API request with rate limiting
            response = self._make_api_request_with_rate_limit(prompt)
            
            # Parse response
            label_id, confidence = self._parse_response(response)
            
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
    
    async def label_batch_parallel(self, texts: List[str], 
                                 progress_callback=None) -> List[LabelingResult]:
        """Label batch teks secara paralel menggunakan asyncio.
        
        Args:
            texts: List teks yang akan diklasifikasi
            progress_callback: Optional callback untuk progress updates
            
        Returns:
            List LabelingResult dalam urutan yang sama dengan input
        """
        if not texts:
            return []
        
        logger.info(f"Starting parallel labeling of {len(texts)} texts with {self.max_workers} workers")
        
        # Create futures untuk semua texts
        loop = asyncio.get_event_loop()
        futures = []
        
        for i, text in enumerate(texts):
            future = loop.run_in_executor(
                self.executor, 
                self.label_single_text_sync, 
                text
            )
            futures.append((i, future))
        
        # Process results as they complete
        results = [None] * len(texts)  # Pre-allocate dengan urutan yang benar
        completed = 0
        
        for i, future in futures:
            try:
                result = await future
                results[i] = result
                completed += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(completed, len(texts), result)
                
                # Log progress
                if completed % 10 == 0 or completed == len(texts):
                    logger.info(f"Completed {completed}/{len(texts)} texts")
                    
            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                # Create error result
                results[i] = LabelingResult(
                    text=texts[i],
                    label_id=1,
                    confidence=0.3,
                    response_time=0.0,
                    error=str(e)
                )
                completed += 1
        
        logger.info(f"Parallel labeling completed. Processed {len(texts)} texts")
        return results
    
    def label_batch_parallel_sync(self, texts: List[str], 
                                progress_callback=None) -> List[LabelingResult]:
        """Synchronous wrapper untuk parallel labeling.
        
        Args:
            texts: List teks yang akan diklasifikasi
            progress_callback: Optional callback untuk progress updates
            
        Returns:
            List LabelingResult
        """
        # Run async function in new event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.label_batch_parallel(texts, progress_callback))
        finally:
            loop.close()
    
    def get_parallel_stats(self) -> Dict:
        """Get statistics tentang parallel processing.
        
        Returns:
            Dictionary berisi statistik
        """
        base_stats = self.get_usage_stats()
        
        parallel_stats = {
            "max_workers": self.max_workers,
            "rate_limiter_queue_size": self.rate_limiter.requests.qsize(),
            "estimated_wait_time": self.rate_limiter.wait_time()
        }
        
        return {**base_stats, **parallel_stats}
    
    def __del__(self):
        """Cleanup executor saat object dihapus."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

class MockParallelDeepSeekClient(ParallelDeepSeekClient):
    """Mock version untuk testing tanpa API calls."""
    
    def _make_api_request_with_rate_limit(self, prompt: str) -> str:
        """Mock API request dengan simulasi delay."""
        # Extract text from prompt untuk mock response
        text_start = prompt.find('Teks: "') + 7
        text_end = prompt.find('"', text_start)
        text = prompt[text_start:text_end].lower() if text_start > 6 and text_end > text_start else "default"
        
        # Simulasi network delay berdasarkan text length (deterministik)
        delay = 0.1 + (len(text) % 5) * 0.01  # 0.1-0.14 seconds
        time.sleep(delay)
        
        # Mock responses berdasarkan content (deterministik)
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

def create_parallel_deepseek_client(mock: bool = False, 
                                  settings: Optional[Settings] = None,
                                  max_workers: int = 5) -> ParallelDeepSeekClient:
    """Factory function untuk membuat parallel DeepSeek client.
    
    Args:
        mock: Jika True, return mock client
        settings: Settings instance
        max_workers: Maximum concurrent workers
        
    Returns:
        ParallelDeepSeekClient instance
    """
    if mock:
        return MockParallelDeepSeekClient(settings, max_workers)
    else:
        return ParallelDeepSeekClient(settings, max_workers)