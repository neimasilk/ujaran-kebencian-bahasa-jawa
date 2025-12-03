"""Utilitas untuk data labeling menggunakan DeepSeek API dengan strategi cost-efficient.

Modul ini mengimplementasikan strategi pembagian label positif/negatif untuk menghemat
biaya penggunaan DeepSeek API, di mana:
- Data berlabel 'positive' dianggap bukan ujaran kebencian (tidak perlu labeling ulang)
- Data berlabel 'negative' perlu dilabeli lebih detail dengan 4 kategori ujaran kebencian
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Setup logger
logger = logging.getLogger(__name__)

class DeepSeekLabelingStrategy:
    """Kelas untuk mengelola strategi labeling dengan DeepSeek API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Inisialisasi strategi labeling.
        
        Args:
            api_key: API key untuk DeepSeek (opsional, bisa diambil dari environment)
        """
        self.api_key = api_key
        self.positive_label = "positive"
        self.negative_label = "negative"
        
        # Mapping kategori detail untuk data negative
        self.detailed_categories = {
            0: "Bukan Ujaran Kebencian",
            1: "Ujaran Kebencian - Ringan", 
            2: "Ujaran Kebencian - Sedang",
            3: "Ujaran Kebencian - Berat"
        }
    
    def filter_data_by_initial_label(self, df: pd.DataFrame, 
                                   text_column: str = 'text', 
                                   label_column: str = 'label') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Memisahkan data berdasarkan label positif/negatif awal.
        
        Args:
            df: DataFrame dengan data dan label awal
            text_column: Nama kolom yang berisi teks
            label_column: Nama kolom yang berisi label (positive/negative)
            
        Returns:
            Tuple berisi (positive_data, negative_data)
        """
        if df.empty:
            logger.warning("DataFrame kosong")
            return pd.DataFrame(), pd.DataFrame()
            
        # Pisahkan data berdasarkan label
        positive_data = df[df[label_column] == self.positive_label].copy()
        negative_data = df[df[label_column] == self.negative_label].copy()
        
        logger.info(f"Data positive: {len(positive_data)} sampel")
        logger.info(f"Data negative: {len(negative_data)} sampel (perlu labeling detail)")
        
        return positive_data, negative_data
    
    def prepare_positive_data(self, positive_df: pd.DataFrame, 
                            text_column: str = 'text') -> pd.DataFrame:
        """Menyiapkan data positive dengan label final 'Bukan Ujaran Kebencian'.
        
        Args:
            positive_df: DataFrame dengan data berlabel positive
            text_column: Nama kolom teks
            
        Returns:
            DataFrame dengan label final
        """
        if positive_df.empty:
            return pd.DataFrame()
            
        result_df = positive_df.copy()
        result_df['final_label'] = self.detailed_categories[0]  # "Bukan Ujaran Kebencian"
        result_df['confidence_score'] = 1.0  # High confidence untuk data positive
        result_df['labeling_method'] = 'rule_based_positive'
        
        logger.info(f"Prepared {len(result_df)} positive samples as 'Bukan Ujaran Kebencian'")
        return result_df
    
    def prepare_negative_data_for_deepseek(self, negative_df: pd.DataFrame,
                                         text_column: str = 'text',
                                         batch_size: int = 100) -> List[Dict]:
        """Menyiapkan data negative untuk labeling dengan DeepSeek API.
        
        Args:
            negative_df: DataFrame dengan data berlabel negative
            text_column: Nama kolom teks
            batch_size: Ukuran batch untuk processing
            
        Returns:
            List of dictionaries siap untuk DeepSeek API
        """
        if negative_df.empty:
            return []
            
        # Siapkan prompt template untuk DeepSeek
        prompt_template = """
Anda adalah ahli dalam mendeteksi ujaran kebencian dalam Bahasa Jawa. 
Klasifikasikan teks berikut ke dalam salah satu kategori:

0: Bukan Ujaran Kebencian
1: Ujaran Kebencian - Ringan (sindiran halus, ejekan terselubung)
2: Ujaran Kebencian - Sedang (hinaan langsung, cercaan)
3: Ujaran Kebencian - Berat (ancaman kekerasan, hasutan, dehumanisasi)

Teks: "{text}"

Jawab hanya dengan angka (0, 1, 2, atau 3) dan tingkat kepercayaan (0.0-1.0).
Format: angka|kepercayaan
Contoh: 2|0.85
"""
        
        batches = []
        for i in range(0, len(negative_df), batch_size):
            batch_df = negative_df.iloc[i:i+batch_size]
            batch_data = {
                'batch_id': i // batch_size,
                'texts': batch_df[text_column].tolist(),
                'original_indices': batch_df.index.tolist(),
                'prompts': [prompt_template.format(text=text) for text in batch_df[text_column]]
            }
            batches.append(batch_data)
            
        logger.info(f"Prepared {len(batches)} batches for DeepSeek labeling")
        return batches
    
    def parse_deepseek_response(self, response: str) -> Tuple[int, float]:
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
                if label_id not in self.detailed_categories:
                    logger.warning(f"Invalid label_id: {label_id}, defaulting to 0")
                    label_id = 0
                    
                # Validasi confidence
                if not (0.0 <= confidence <= 1.0):
                    logger.warning(f"Invalid confidence: {confidence}, defaulting to 0.5")
                    confidence = 0.5
                    
                return label_id, confidence
            else:
                logger.error(f"Invalid response format: {response}")
                return 0, 0.5  # Default fallback
                
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing DeepSeek response '{response}': {e}")
            return 0, 0.5  # Default fallback
    
    def combine_labeled_data(self, positive_df: pd.DataFrame, 
                           negative_df: pd.DataFrame,
                           deepseek_results: List[Dict]) -> pd.DataFrame:
        """Menggabungkan hasil labeling dari data positive dan negative.
        
        Args:
            positive_df: DataFrame data positive yang sudah dilabeli
            negative_df: DataFrame data negative original
            deepseek_results: Hasil labeling dari DeepSeek untuk data negative
            
        Returns:
            DataFrame gabungan dengan label final
        """
        # Siapkan data positive
        final_positive = self.prepare_positive_data(positive_df)
        
        # Siapkan data negative dengan hasil DeepSeek
        final_negative = negative_df.copy()
        
        # Placeholder untuk hasil DeepSeek (implementasi actual akan menggunakan API)
        # Untuk sekarang, kita buat contoh hasil
        if deepseek_results:
            # Process DeepSeek results
            for result in deepseek_results:
                batch_indices = result['original_indices']
                batch_labels = result.get('labels', [1] * len(batch_indices))  # Default to ringan
                batch_confidences = result.get('confidences', [0.7] * len(batch_indices))
                
                for idx, label_id, confidence in zip(batch_indices, batch_labels, batch_confidences):
                    if idx in final_negative.index:
                        final_negative.loc[idx, 'final_label'] = self.detailed_categories[label_id]
                        final_negative.loc[idx, 'confidence_score'] = confidence
                        final_negative.loc[idx, 'labeling_method'] = 'deepseek_api'
        
        # Gabungkan data
        if not final_positive.empty and not final_negative.empty:
            combined_df = pd.concat([final_positive, final_negative], ignore_index=True)
        elif not final_positive.empty:
            combined_df = final_positive
        elif not final_negative.empty:
            combined_df = final_negative
        else:
            combined_df = pd.DataFrame()
            
        logger.info(f"Combined dataset: {len(combined_df)} total samples")
        return combined_df
    
    def generate_labeling_report(self, df: pd.DataFrame) -> Dict:
        """Generate laporan hasil labeling.
        
        Args:
            df: DataFrame hasil labeling
            
        Returns:
            Dictionary berisi statistik labeling
        """
        if df.empty:
            return {}
            
        report = {
            'total_samples': len(df),
            'label_distribution': df['final_label'].value_counts().to_dict(),
            'labeling_method_distribution': df['labeling_method'].value_counts().to_dict(),
            'average_confidence': df['confidence_score'].mean(),
            'low_confidence_samples': len(df[df['confidence_score'] < 0.6])
        }
        
        # Estimasi cost saving
        positive_samples = len(df[df['labeling_method'] == 'rule_based_positive'])
        negative_samples = len(df[df['labeling_method'] == 'deepseek_api'])
        
        # Asumsi: tanpa strategi ini, semua data perlu DeepSeek API
        total_samples = positive_samples + negative_samples
        cost_saving_percentage = (positive_samples / total_samples * 100) if total_samples > 0 else 0
        
        report['cost_analysis'] = {
            'samples_processed_by_rule': positive_samples,
            'samples_processed_by_api': negative_samples,
            'estimated_cost_saving_percentage': round(cost_saving_percentage, 2)
        }
        
        return report


def load_and_process_dataset(file_path: str, 
                           text_column: str = 'text',
                           label_column: str = 'label') -> Tuple[pd.DataFrame, Dict]:
    """Function utama untuk memproses dataset dengan strategi cost-efficient.
    
    Args:
        file_path: Path ke file CSV dataset
        text_column: Nama kolom teks
        label_column: Nama kolom label awal
        
    Returns:
        Tuple berisi (processed_dataframe, labeling_report)
    """
    # Load dataset
    try:
        df = pd.read_csv(file_path, names=[text_column, label_column])
        logger.info(f"Loaded dataset: {len(df)} samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return pd.DataFrame(), {}
    
    # Initialize strategy
    strategy = DeepSeekLabelingStrategy()
    
    # Filter data
    positive_data, negative_data = strategy.filter_data_by_initial_label(
        df, text_column, label_column
    )
    
    # Prepare negative data for DeepSeek (simulation)
    negative_batches = strategy.prepare_negative_data_for_deepseek(negative_data, text_column)
    
    # Simulate DeepSeek results (dalam implementasi nyata, ini akan memanggil API)
    simulated_results = []
    for batch in negative_batches:
        # Simulasi hasil - dalam implementasi nyata akan memanggil DeepSeek API
        simulated_result = {
            'batch_id': batch['batch_id'],
            'original_indices': batch['original_indices'],
            'labels': [1] * len(batch['texts']),  # Default ke "Ringan"
            'confidences': [0.75] * len(batch['texts'])
        }
        simulated_results.append(simulated_result)
    
    # Combine results
    final_df = strategy.combine_labeled_data(positive_data, negative_data, simulated_results)
    
    # Generate report
    report = strategy.generate_labeling_report(final_df)
    
    return final_df, report


if __name__ == "__main__":
    # Example usage
    dataset_path = "../data_collection/raw-dataset.csv"
    
    # Process dataset
    processed_df, report = load_and_process_dataset(dataset_path)
    
    # Print report
    print("\n=== LAPORAN LABELING ===")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    # Save processed data
    if not processed_df.empty:
        output_path = "../data_collection/labeled-dataset.csv"
        processed_df.to_csv(output_path, index=False)
        print(f"\nProcessed dataset saved to: {output_path}")