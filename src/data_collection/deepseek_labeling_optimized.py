import os
import json
import re
from typing import Dict, Any, Optional
from openai import OpenAI
import pandas as pd
from datetime import datetime

class DeepSeekLabelerOptimized:
    """
    Versi optimized DeepSeek Labeler untuk menghemat biaya token.
    Menghilangkan reasoning dan menggunakan prompt yang lebih singkat.
    """
    
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("API key tidak ditemukan. Set DEEPSEEK_API_KEY environment variable.")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model
        
        # System prompt yang lebih singkat dan fokus
        self.system_prompt = """Klasifikasi teks Bahasa Jawa ke dalam kategori ujaran kebencian.

Kategori:
- bukan_ujaran_kebencian: Teks netral/positif
- ujaran_kebencian_ringan: Sindiran halus, stereotip ringan
- ujaran_kebencian_sedang: Generalisasi negatif, diskriminasi
- ujaran_kebencian_berat: Ancaman, dehumanisasi, hasutan kekerasan

Response format JSON:
{"label": "kategori", "confidence": 85}

Confidence: 1-100 (seberapa yakin dengan klasifikasi)"""
    
    def create_user_prompt(self, text: str) -> str:
        """Membuat prompt user yang singkat"""
        return f"Klasifikasi: {text}"
    
    def test_connection(self) -> bool:
        """Test koneksi API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Respond with: OK"},
                    {"role": "user", "content": "Test"}
                ],
                max_tokens=10,
                temperature=0
            )
            return "OK" in response.choices[0].message.content
        except Exception as e:
            print(f"❌ Koneksi gagal: {e}")
            return False
    
    def label_text(self, text: str) -> Dict[str, Any]:
        """Label single text dengan format optimized"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.create_user_prompt(text)}
                ],
                max_tokens=50,  # Sangat kecil karena hanya butuh label + confidence
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(content)
                
                # Validasi format
                if 'label' not in result or 'confidence' not in result:
                    raise ValueError("Missing required fields")
                
                # Validasi label
                valid_labels = [
                    'bukan_ujaran_kebencian', 
                    'ujaran_kebencian_ringan',
                    'ujaran_kebencian_sedang', 
                    'ujaran_kebencian_berat'
                ]
                
                if result['label'] not in valid_labels:
                    result['label'] = 'bukan_ujaran_kebencian'  # default
                
                # Validasi confidence
                confidence = int(result['confidence'])
                if confidence < 1 or confidence > 100:
                    confidence = 50  # default
                result['confidence'] = confidence
                
                return {
                    'label': result['label'],
                    'confidence': result['confidence'],
                    'reasoning': 'Optimized mode - no reasoning to save costs'
                }
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # Fallback parsing jika JSON gagal
                return self._extract_from_text_optimized(content)
                
        except Exception as e:
            print(f"❌ Error labeling: {e}")
            return {
                'label': 'bukan_ujaran_kebencian',
                'confidence': 30,
                'reasoning': f'Error: {str(e)}'
            }
    
    def _extract_from_text_optimized(self, text: str) -> Dict[str, Any]:
        """Fallback parsing untuk response non-JSON yang optimized"""
        # Extract confidence
        confidence_match = re.search(r'confidence["\s]*:?["\s]*(\d+)', text, re.IGNORECASE)
        confidence = int(confidence_match.group(1)) if confidence_match else 50
        
        # Extract label
        label = 'bukan_ujaran_kebencian'  # default
        if 'ujaran_kebencian_berat' in text.lower():
            label = 'ujaran_kebencian_berat'
            confidence = max(confidence, 70)
        elif 'ujaran_kebencian_sedang' in text.lower():
            label = 'ujaran_kebencian_sedang'
            confidence = max(confidence, 60)
        elif 'ujaran_kebencian_ringan' in text.lower():
            label = 'ujaran_kebencian_ringan'
            confidence = max(confidence, 55)
        
        return {
            'label': label,
            'confidence': min(confidence, 100),
            'reasoning': 'Parsed from text response (optimized)'
        }
    
    def label_batch(self, texts: list, show_progress: bool = True) -> list:
        """Label multiple texts"""
        results = []
        total = len(texts)
        
        for i, text in enumerate(texts, 1):
            if show_progress:
                print(f"Processing {i}/{total}: {text[:50]}...")
            
            result = self.label_text(text)
            result['text'] = text
            results.append(result)
        
        return results
    
    def save_results(self, results: list, output_path: str) -> None:
        """Save results to CSV"""
        df = pd.DataFrame(results)
        
        # Reorder columns
        columns = ['text', 'label', 'confidence', 'reasoning']
        df = df[columns]
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ Results saved to: {output_path}")
    
    def analyze_results(self, results: list) -> Dict[str, Any]:
        """Analyze labeling results"""
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        analysis = {
            'total_samples': len(results),
            'label_distribution': df['label'].value_counts().to_dict(),
            'average_confidence': df['confidence'].mean(),
            'confidence_by_label': df.groupby('label')['confidence'].mean().to_dict(),
            'low_confidence_count': len(df[df['confidence'] < 50]),
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis

def process_negative_dataset(api_key: str, input_file: str = "data/processed/negative_for_deepseek.csv", 
                               output_file: str = "data/processed/deepseek_negative_labeled.csv",
                               sample_size: int = None) -> pd.DataFrame:
    """
    Memproses dataset negatif dengan DeepSeek V3 optimized
    
    Args:
        api_key (str): DeepSeek API key
        input_file (str): File data negatif hasil preprocessing
        output_file (str): File output hasil labeling
        sample_size (int): Jumlah sample untuk testing (None = semua data)
        
    Returns:
        pd.DataFrame: Dataset yang sudah dilabeli
    """
    
    # Load data negatif
    print(f"Loading negative data dari {input_file}")
    df = pd.read_csv(input_file)
    
    if sample_size:
        df = df.head(sample_size)
        print(f"Processing sample: {len(df)} dari {sample_size} requested")
    else:
        print(f"Processing full dataset: {len(df)} samples")
    
    # Initialize labeler
    labeler = DeepSeekLabelerOptimized(api_key)
    
    results = []
    total_tokens = 0
    successful = 0
    failed = 0
    
    print("Memulai proses labeling...")
    
    for idx, row in df.iterrows():
        try:
            result = labeler.label_text(row['review'])
            
            results.append({
                'review': row['review'],
                'sentiment': row['sentiment'],
                'hate_speech_label': result['label'],
                'confidence': result['confidence'],
                'processing_method': 'deepseek_optimized'
            })
            
            successful += 1
            
            # Estimate tokens
            estimated_tokens = len(row['review'].split()) * 1.3
            total_tokens += estimated_tokens
            
            if idx % 10 == 0:
                print(f"Processed {idx+1}/{len(df)} samples")
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            failed += 1
            
            # Add failed sample with default values
            results.append({
                'review': row['review'],
                'sentiment': row['sentiment'],
                'hate_speech_label': 'error',
                'confidence': 0,
                'processing_method': 'deepseek_failed'
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_file, index=False)
    
    # Summary
    print("=== LABELING SELESAI ===")
    print(f"Berhasil: {successful}/{len(df)}")
    print(f"Gagal: {failed}/{len(df)}")
    print(f"Total tokens: {total_tokens:.0f}")
    print(f"Estimasi biaya: ${(total_tokens * 0.00014 / 1000):.6f}")
    print(f"Hasil disimpan di: {output_file}")
    
    return results_df

def combine_with_positive_data(negative_labeled_file: str = "data/processed/deepseek_negative_labeled.csv",
                              positive_file: str = "data/processed/positive_auto_labeled.csv",
                              final_output: str = "data/processed/final_labeled_dataset.csv") -> pd.DataFrame:
    """
    Menggabungkan data negatif yang sudah dilabeli dengan data positif yang auto-assigned
    
    Args:
        negative_labeled_file (str): File data negatif hasil DeepSeek
        positive_file (str): File data positif auto-assigned
        final_output (str): File output gabungan final
        
    Returns:
        pd.DataFrame: Dataset final lengkap
    """
    
    # Load both datasets
    print("Menggabungkan data positif dan negatif...")
    
    df_negative = pd.read_csv(negative_labeled_file)
    df_positive = pd.read_csv(positive_file)
    
    # Ensure consistent columns
    required_columns = ['review', 'sentiment', 'hate_speech_label', 'confidence', 'processing_method']
    
    for col in required_columns:
        if col not in df_positive.columns:
            if col == 'processing_method':
                df_positive[col] = 'auto_sentiment_based'
            else:
                df_positive[col] = None
    
    # Combine datasets
    final_df = pd.concat([df_positive, df_negative], ignore_index=True)
    
    # Save final dataset
    final_df.to_csv(final_output, index=False)
    
    # Summary statistics
    stats = {
        'total_samples': len(final_df),
        'positive_samples': len(df_positive),
        'negative_samples': len(df_negative),
        'hate_speech_count': len(final_df[final_df['hate_speech_label'] == 'ujaran_kebencian']),
        'non_hate_speech_count': len(final_df[final_df['hate_speech_label'] == 'bukan_ujaran_kebencian']),
        'avg_confidence': final_df['confidence'].mean()
    }
    
    print("=== DATASET FINAL ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Positif (auto): {stats['positive_samples']}")
    print(f"Negatif (DeepSeek): {stats['negative_samples']}")
    print(f"Ujaran kebencian: {stats['hate_speech_count']}")
    print(f"Bukan ujaran kebencian: {stats['non_hate_speech_count']}")
    print(f"Rata-rata confidence: {stats['avg_confidence']:.1f}%")
    print(f"Dataset final disimpan di: {final_output}")
    
    return final_df

def main():
    """
    Main function untuk menjalankan optimized labeling workflow
    """
    
    import os
    from pathlib import Path
    
    # Ensure directories exist
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Get API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    
    if not api_key:
        api_key = input("Masukkan DeepSeek API key: ").strip()
        
    if not api_key or api_key == "your_deepseek_api_key_here":
        print("API key tidak valid. Silakan set DEEPSEEK_API_KEY di environment atau masukkan manual.")
        return
    
    try:
        # Check if preprocessing sudah dilakukan
        negative_file = "data/processed/negative_for_deepseek.csv"
        positive_file = "data/processed/positive_auto_labeled.csv"
        
        if not os.path.exists(negative_file) or not os.path.exists(positive_file):
            print("File preprocessing tidak ditemukan. Jalankan preprocessing dulu:")
            print("python src/data_collection/preprocess_sentiment.py")
            return
        
        # Ask for sample size
        sample_input = input("Masukkan jumlah sample untuk testing (Enter untuk semua data): ").strip()
        sample_size = int(sample_input) if sample_input else None
        
        # Process negative data with DeepSeek
        print("=== MEMULAI DEEPSEEK LABELING (OPTIMIZED) ===")
        negative_labeled = process_negative_dataset(api_key, sample_size=sample_size)
        
        # Combine with positive data
        final_dataset = combine_with_positive_data()
        
        print("✅ Optimized labeling selesai!")
        
        print("\n=== NEXT STEPS ===")
        print("1. Review hasil di: data/processed/final_labeled_dataset.csv")
        print("2. Validasi kualitas labeling")
        print("3. Update training pipeline dengan dataset baru")
        
    except Exception as e:
        print(f"Error dalam optimized labeling: {e}")
        raise

if __name__ == "__main__":
    main()