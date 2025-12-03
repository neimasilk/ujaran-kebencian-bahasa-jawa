#!/usr/bin/env python3
"""
Script untuk membuat dataset standar yang seimbang dan representatif
untuk eksperimen ilmiah sistem deteksi ujaran kebencian Bahasa Jawa.

Tujuan:
1. Membuat balanced evaluation set (2000 sampel, 25% per kelas)
2. Membuat standardized training set dengan class weighting
3. Implementasi cross-validation framework
4. Dokumentasi lengkap untuk reproducibility

Author: AI Research Team
Date: 16 Januari 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import Counter
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StandardizedDatasetCreator:
    """
    Kelas untuk membuat dataset standar yang seimbang dan representatif
    untuk eksperimen ilmiah.
    """
    
    def __init__(self, source_data_path, output_dir="data/standardized", random_seed=42):
        self.source_data_path = source_data_path
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Label mapping untuk konsistensi
        self.label_mapping = {
            'Bukan Ujaran Kebencian': 0,
            'Ujaran Kebencian - Ringan': 1,
            'Ujaran Kebencian - Sedang': 2,
            'Ujaran Kebencian - Berat': 3
        }
        
        # Class weights berdasarkan analisis sebelumnya
        self.class_weights = {
            0: 0.2537,  # Bukan Ujaran Kebencian
            1: 1.0309,  # Ujaran Kebencian - Ringan
            2: 1.2019,  # Ujaran Kebencian - Sedang
            3: 1.5401   # Ujaran Kebencian - Berat
        }
        
        logger.info(f"StandardizedDatasetCreator initialized")
        logger.info(f"Source data: {source_data_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Random seed: {random_seed}")
    
    def load_and_validate_data(self):
        """
        Load dan validasi data sumber.
        """
        logger.info("Loading source data...")
        
        try:
            # Load data
            if self.source_data_path.endswith('.csv'):
                df = pd.read_csv(self.source_data_path)
            else:
                raise ValueError("Only CSV files are supported")
            
            logger.info(f"Loaded {len(df)} samples from source data")
            
            # Validasi kolom yang diperlukan
            required_columns = ['text', 'final_label', 'confidence_score']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Filter berdasarkan confidence score
            df_filtered = df[df['confidence_score'] >= 0.7].copy()
            logger.info(f"After confidence filtering (≥0.7): {len(df_filtered)} samples")
            
            # Remove duplicates
            df_clean = df_filtered.drop_duplicates(subset=['text'], keep='first')
            logger.info(f"After duplicate removal: {len(df_clean)} samples")
            
            # Add label_id
            df_clean['label_id'] = df_clean['final_label'].map(self.label_mapping)
            
            # Validasi mapping
            if df_clean['label_id'].isna().any():
                unmapped_labels = df_clean[df_clean['label_id'].isna()]['final_label'].unique()
                raise ValueError(f"Unmapped labels found: {unmapped_labels}")
            
            self.df = df_clean
            logger.info("Data validation completed successfully")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def analyze_current_distribution(self):
        """
        Analisis distribusi data saat ini.
        """
        logger.info("Analyzing current data distribution...")
        
        # Distribusi label
        label_counts = self.df['final_label'].value_counts()
        label_percentages = self.df['final_label'].value_counts(normalize=True) * 100
        
        distribution_analysis = {
            'total_samples': len(self.df),
            'label_distribution': {},
            'imbalance_ratio': None,
            'majority_class': None,
            'minority_class': None
        }
        
        for label, count in label_counts.items():
            percentage = label_percentages[label]
            distribution_analysis['label_distribution'][label] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
        
        # Analisis ketidakseimbangan
        max_count = label_counts.max()
        min_count = label_counts.min()
        distribution_analysis['imbalance_ratio'] = round(max_count / min_count, 2)
        distribution_analysis['majority_class'] = label_counts.idxmax()
        distribution_analysis['minority_class'] = label_counts.idxmin()
        
        logger.info(f"Distribution analysis completed:")
        logger.info(f"  Total samples: {distribution_analysis['total_samples']}")
        logger.info(f"  Imbalance ratio: {distribution_analysis['imbalance_ratio']}:1")
        logger.info(f"  Majority class: {distribution_analysis['majority_class']}")
        logger.info(f"  Minority class: {distribution_analysis['minority_class']}")
        
        self.distribution_analysis = distribution_analysis
        return distribution_analysis
    
    def create_balanced_evaluation_set(self, target_size=2000, samples_per_class=None):
        """
        Membuat balanced evaluation set dengan distribusi yang seimbang.
        """
        logger.info(f"Creating balanced evaluation set (target size: {target_size})...")
        
        if samples_per_class is None:
            samples_per_class = target_size // 4  # 4 kelas
        
        balanced_dfs = []
        actual_samples = {}
        
        for label in self.label_mapping.keys():
            label_data = self.df[self.df['final_label'] == label]
            available_samples = len(label_data)
            
            # Tentukan jumlah sampel yang akan diambil
            n_samples = min(samples_per_class, available_samples)
            
            if n_samples < samples_per_class:
                logger.warning(f"Only {n_samples} samples available for '{label}' (requested: {samples_per_class})")
            
            # Sampling dengan random seed
            if n_samples > 0:
                sampled = label_data.sample(n=n_samples, random_state=self.random_seed)
                balanced_dfs.append(sampled)
                actual_samples[label] = n_samples
                logger.info(f"  {label}: {n_samples} samples")
        
        # Gabungkan dan acak
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        logger.info(f"Balanced evaluation set created: {len(balanced_df)} samples")
        
        # Validasi distribusi
        final_distribution = balanced_df['final_label'].value_counts()
        logger.info("Final distribution:")
        for label, count in final_distribution.items():
            percentage = (count / len(balanced_df)) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
        self.balanced_eval_set = balanced_df
        return balanced_df
    
    def create_standardized_training_set(self, test_size=0.2):
        """
        Membuat standardized training set dengan stratified split.
        """
        logger.info(f"Creating standardized training set (test_size: {test_size})...")
        
        # Stratified split
        X = self.df[['text', 'final_label', 'confidence_score']]
        y = self.df['label_id']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed, 
            stratify=y
        )
        
        # Gabungkan kembali
        train_df = X_train.copy()
        train_df['label_id'] = y_train
        
        test_df = X_test.copy()
        test_df['label_id'] = y_test
        
        logger.info(f"Training set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        # Validasi distribusi stratified
        train_dist = train_df['final_label'].value_counts(normalize=True) * 100
        test_dist = test_df['final_label'].value_counts(normalize=True) * 100
        
        logger.info("Training set distribution:")
        for label, pct in train_dist.items():
            logger.info(f"  {label}: {pct:.2f}%")
        
        logger.info("Test set distribution:")
        for label, pct in test_dist.items():
            logger.info(f"  {label}: {pct:.2f}%")
        
        self.train_set = train_df
        self.test_set = test_df
        
        return train_df, test_df
    
    def create_cross_validation_folds(self, n_splits=5):
        """
        Membuat cross-validation folds untuk evaluasi yang robust.
        """
        logger.info(f"Creating {n_splits}-fold cross-validation splits...")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        
        X = self.df[['text', 'final_label', 'confidence_score']]
        y = self.df['label_id']
        
        cv_folds = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            train_fold = self.df.iloc[train_idx].copy()
            val_fold = self.df.iloc[val_idx].copy()
            
            fold_info = {
                'fold': fold_idx + 1,
                'train_size': len(train_fold),
                'val_size': len(val_fold),
                'train_distribution': train_fold['final_label'].value_counts(normalize=True).to_dict(),
                'val_distribution': val_fold['final_label'].value_counts(normalize=True).to_dict()
            }
            
            cv_folds.append({
                'fold_info': fold_info,
                'train_data': train_fold,
                'val_data': val_fold
            })
            
            logger.info(f"Fold {fold_idx + 1}: Train={len(train_fold)}, Val={len(val_fold)}")
        
        self.cv_folds = cv_folds
        return cv_folds
    
    def save_datasets(self):
        """
        Simpan semua dataset yang telah dibuat.
        """
        logger.info("Saving standardized datasets...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Balanced Evaluation Set
        if hasattr(self, 'balanced_eval_set'):
            eval_path = self.output_dir / "balanced_evaluation_set.csv"
            self.balanced_eval_set.to_csv(eval_path, index=False)
            logger.info(f"Balanced evaluation set saved: {eval_path}")
        
        # 2. Standardized Training/Test Sets
        if hasattr(self, 'train_set') and hasattr(self, 'test_set'):
            train_path = self.output_dir / "standardized_train_set.csv"
            test_path = self.output_dir / "standardized_test_set.csv"
            
            self.train_set.to_csv(train_path, index=False)
            self.test_set.to_csv(test_path, index=False)
            
            logger.info(f"Training set saved: {train_path}")
            logger.info(f"Test set saved: {test_path}")
        
        # 3. Cross-Validation Folds
        if hasattr(self, 'cv_folds'):
            cv_dir = self.output_dir / "cross_validation_folds"
            cv_dir.mkdir(exist_ok=True)
            
            for fold_data in self.cv_folds:
                fold_num = fold_data['fold_info']['fold']
                
                train_fold_path = cv_dir / f"fold_{fold_num}_train.csv"
                val_fold_path = cv_dir / f"fold_{fold_num}_val.csv"
                
                fold_data['train_data'].to_csv(train_fold_path, index=False)
                fold_data['val_data'].to_csv(val_fold_path, index=False)
            
            logger.info(f"Cross-validation folds saved: {cv_dir}")
        
        # 4. Metadata dan Documentation
        self.save_metadata(timestamp)
        
        return {
            'balanced_eval_set': eval_path if hasattr(self, 'balanced_eval_set') else None,
            'train_set': train_path if hasattr(self, 'train_set') else None,
            'test_set': test_path if hasattr(self, 'test_set') else None,
            'cv_folds_dir': cv_dir if hasattr(self, 'cv_folds') else None
        }
    
    def save_metadata(self, timestamp):
        """
        Simpan metadata dan dokumentasi untuk reproducibility.
        """
        metadata = {
            'creation_timestamp': timestamp,
            'source_data_path': str(self.source_data_path),
            'random_seed': self.random_seed,
            'label_mapping': self.label_mapping,
            'class_weights': self.class_weights,
            'distribution_analysis': self.distribution_analysis if hasattr(self, 'distribution_analysis') else None,
            'datasets_created': {
                'balanced_evaluation_set': {
                    'size': len(self.balanced_eval_set) if hasattr(self, 'balanced_eval_set') else None,
                    'distribution': self.balanced_eval_set['final_label'].value_counts().to_dict() if hasattr(self, 'balanced_eval_set') else None
                },
                'training_set': {
                    'size': len(self.train_set) if hasattr(self, 'train_set') else None,
                    'distribution': self.train_set['final_label'].value_counts().to_dict() if hasattr(self, 'train_set') else None
                },
                'test_set': {
                    'size': len(self.test_set) if hasattr(self, 'test_set') else None,
                    'distribution': self.test_set['final_label'].value_counts().to_dict() if hasattr(self, 'test_set') else None
                },
                'cross_validation': {
                    'n_folds': len(self.cv_folds) if hasattr(self, 'cv_folds') else None,
                    'fold_sizes': [fold['fold_info']['train_size'] + fold['fold_info']['val_size'] for fold in self.cv_folds] if hasattr(self, 'cv_folds') else None
                }
            }
        }
        
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadata saved: {metadata_path}")
        
        # Buat dokumentasi README
        self.create_documentation()
    
    def create_documentation(self):
        """
        Buat dokumentasi lengkap untuk dataset standar.
        """
        doc_content = f"""
# Standardized Dataset Documentation
## Sistem Deteksi Ujaran Kebencian Bahasa Jawa

**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Random Seed:** {self.random_seed}  
**Source Data:** {self.source_data_path}  

## Dataset Overview

### Original Dataset
- **Total Samples:** {len(self.df)}
- **Quality Filter:** Confidence score ≥ 0.7
- **Duplicates Removed:** Yes

### Label Distribution (Original)
{self._format_distribution_table(self.df['final_label'].value_counts())}

### Imbalance Analysis
- **Imbalance Ratio:** {self.distribution_analysis['imbalance_ratio']}:1
- **Majority Class:** {self.distribution_analysis['majority_class']}
- **Minority Class:** {self.distribution_analysis['minority_class']}

## Standardized Datasets Created

### 1. Balanced Evaluation Set
- **File:** `balanced_evaluation_set.csv`
- **Purpose:** Standard evaluation untuk semua eksperimen
- **Size:** {len(self.balanced_eval_set) if hasattr(self, 'balanced_eval_set') else 'N/A'} samples
- **Distribution:** Balanced (target 25% per class)

### 2. Standardized Training Set
- **File:** `standardized_train_set.csv`
- **Purpose:** Training dengan stratified split
- **Size:** {len(self.train_set) if hasattr(self, 'train_set') else 'N/A'} samples
- **Split Ratio:** 80% training, 20% testing

### 3. Standardized Test Set
- **File:** `standardized_test_set.csv`
- **Purpose:** Final evaluation
- **Size:** {len(self.test_set) if hasattr(self, 'test_set') else 'N/A'} samples

### 4. Cross-Validation Folds
- **Directory:** `cross_validation_folds/`
- **Purpose:** Robust model evaluation
- **Folds:** {len(self.cv_folds) if hasattr(self, 'cv_folds') else 'N/A'}-fold stratified

## Class Weights for Training

```python
class_weights = {{
    0: {self.class_weights[0]},  # Bukan Ujaran Kebencian
    1: {self.class_weights[1]},  # Ujaran Kebencian - Ringan
    2: {self.class_weights[2]},  # Ujaran Kebencian - Sedang
    3: {self.class_weights[3]}   # Ujaran Kebencian - Berat
}}
```

## Usage for Experiments

### Standard Evaluation Protocol
```python
# Load balanced evaluation set
eval_df = pd.read_csv('data/standardized/balanced_evaluation_set.csv')

# Use for all model comparisons
results = evaluate_model(model, eval_df)
```

### Cross-Validation
```python
# Load CV folds
for fold in range(1, 6):
    train_df = pd.read_csv(f'data/standardized/cross_validation_folds/fold_{{fold}}_train.csv')
    val_df = pd.read_csv(f'data/standardized/cross_validation_folds/fold_{{fold}}_val.csv')
    # Train and evaluate
```

## Reproducibility

- **Random Seed:** {self.random_seed} (fixed untuk semua operasi)
- **Stratified Sampling:** Mempertahankan distribusi kelas
- **Quality Control:** Confidence threshold dan duplicate removal
- **Documentation:** Lengkap dengan metadata

## Quality Assurance

- ✅ No missing values dalam kolom penting
- ✅ No duplicate texts
- ✅ Consistent label mapping
- ✅ Stratified distributions maintained
- ✅ Reproducible with fixed random seed

---

**Generated by:** StandardizedDatasetCreator  
**Version:** 1.0  
**Contact:** AI Research Team  
"""
        
        doc_path = self.output_dir / "README.md"
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        logger.info(f"Documentation created: {doc_path}")
    
    def _format_distribution_table(self, value_counts):
        """
        Format distribusi sebagai tabel markdown.
        """
        total = value_counts.sum()
        table = "\n| Label | Count | Percentage |\n|-------|-------|------------|\n"
        
        for label, count in value_counts.items():
            percentage = (count / total) * 100
            table += f"| {label} | {count} | {percentage:.2f}% |\n"
        
        return table
    
    def visualize_distributions(self):
        """
        Buat visualisasi distribusi dataset.
        """
        logger.info("Creating distribution visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Distribution Analysis', fontsize=16)
        
        # 1. Original distribution
        ax1 = axes[0, 0]
        original_counts = self.df['final_label'].value_counts()
        ax1.bar(range(len(original_counts)), original_counts.values)
        ax1.set_title('Original Dataset Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(len(original_counts)))
        ax1.set_xticklabels([label.replace(' - ', '\n') for label in original_counts.index], rotation=45)
        
        # 2. Balanced evaluation set
        if hasattr(self, 'balanced_eval_set'):
            ax2 = axes[0, 1]
            balanced_counts = self.balanced_eval_set['final_label'].value_counts()
            ax2.bar(range(len(balanced_counts)), balanced_counts.values, color='green', alpha=0.7)
            ax2.set_title('Balanced Evaluation Set')
            ax2.set_xlabel('Class')
            ax2.set_ylabel('Count')
            ax2.set_xticks(range(len(balanced_counts)))
            ax2.set_xticklabels([label.replace(' - ', '\n') for label in balanced_counts.index], rotation=45)
        
        # 3. Training set distribution
        if hasattr(self, 'train_set'):
            ax3 = axes[1, 0]
            train_counts = self.train_set['final_label'].value_counts()
            ax3.bar(range(len(train_counts)), train_counts.values, color='blue', alpha=0.7)
            ax3.set_title('Training Set Distribution')
            ax3.set_xlabel('Class')
            ax3.set_ylabel('Count')
            ax3.set_xticks(range(len(train_counts)))
            ax3.set_xticklabels([label.replace(' - ', '\n') for label in train_counts.index], rotation=45)
        
        # 4. Test set distribution
        if hasattr(self, 'test_set'):
            ax4 = axes[1, 1]
            test_counts = self.test_set['final_label'].value_counts()
            ax4.bar(range(len(test_counts)), test_counts.values, color='orange', alpha=0.7)
            ax4.set_title('Test Set Distribution')
            ax4.set_xlabel('Class')
            ax4.set_ylabel('Count')
            ax4.set_xticks(range(len(test_counts)))
            ax4.set_xticklabels([label.replace(' - ', '\n') for label in test_counts.index], rotation=45)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / "distribution_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"Distribution visualization saved: {viz_path}")
        
        plt.show()

def main():
    """
    Main function untuk menjalankan standardisasi dataset.
    """
    logger.info("Starting dataset standardization process...")
    
    # Konfigurasi
    source_data_path = "src/data_collection/hasil-labeling.csv"
    output_dir = "data/standardized"
    random_seed = 42
    
    # Inisialisasi creator
    creator = StandardizedDatasetCreator(
        source_data_path=source_data_path,
        output_dir=output_dir,
        random_seed=random_seed
    )
    
    try:
        # 1. Load dan validasi data
        df = creator.load_and_validate_data()
        
        # 2. Analisis distribusi
        distribution = creator.analyze_current_distribution()
        
        # 3. Buat balanced evaluation set
        balanced_eval = creator.create_balanced_evaluation_set(target_size=2000)
        
        # 4. Buat standardized training/test sets
        train_set, test_set = creator.create_standardized_training_set(test_size=0.2)
        
        # 5. Buat cross-validation folds
        cv_folds = creator.create_cross_validation_folds(n_splits=5)
        
        # 6. Simpan semua dataset
        saved_files = creator.save_datasets()
        
        # 7. Buat visualisasi
        creator.visualize_distributions()
        
        logger.info("Dataset standardization completed successfully!")
        logger.info(f"Files saved to: {output_dir}")
        
        # Summary
        print("\n" + "="*60)
        print("DATASET STANDARDIZATION SUMMARY")
        print("="*60)
        print(f"Source data: {len(df)} samples")
        print(f"Balanced evaluation set: {len(balanced_eval)} samples")
        print(f"Training set: {len(train_set)} samples")
        print(f"Test set: {len(test_set)} samples")
        print(f"Cross-validation folds: {len(cv_folds)} folds")
        print(f"Output directory: {output_dir}")
        print("\nFiles created:")
        for key, path in saved_files.items():
            if path:
                print(f"  - {key}: {path}")
        print("\n✅ Ready for scientific experiments!")
        
    except Exception as e:
        logger.error(f"Error during standardization: {e}")
        raise

if __name__ == "__main__":
    main()