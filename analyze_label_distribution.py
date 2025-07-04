import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the shuffled dataset
df = pd.read_csv('data/processed/final_dataset_shuffled.csv')

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Analyze label distribution
print("\n=== DISTRIBUSI LABEL ===")
print("\nDistribusi berdasarkan kolom 'label':")
label_counts = df['label'].value_counts()
print(label_counts)
print(f"\nPersentase:")
for label, count in label_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{label}: {count} ({percentage:.2f}%)")

# Check if final_label column exists and analyze it
if 'final_label' in df.columns:
    print("\nDistribusi berdasarkan kolom 'final_label':")
    final_label_counts = df['final_label'].value_counts()
    print(final_label_counts)
    print(f"\nPersentase:")
    for label, count in final_label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{label}: {count} ({percentage:.2f}%)")

# Create visualization
plt.figure(figsize=(12, 8))

if 'final_label' in df.columns:
    plt.subplot(2, 1, 1)
    final_label_counts.plot(kind='bar')
    plt.title('Distribusi Final Label')
    plt.xticks(rotation=45)
    plt.ylabel('Jumlah')
    
    plt.subplot(2, 1, 2)
    label_counts.plot(kind='bar')
    plt.title('Distribusi Label')
    plt.xticks(rotation=45)
    plt.ylabel('Jumlah')
else:
    label_counts.plot(kind='bar')
    plt.title('Distribusi Label')
    plt.xticks(rotation=45)
    plt.ylabel('Jumlah')

plt.tight_layout()
plt.savefig('data/processed/label_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== ANALISIS KETIDAKSEIMBANGAN ===")
if 'final_label' in df.columns:
    # Use final_label for analysis
    main_counts = final_label_counts
    print("Menggunakan kolom 'final_label' untuk analisis:")
else:
    # Use label for analysis
    main_counts = label_counts
    print("Menggunakan kolom 'label' untuk analisis:")

# Calculate imbalance ratio
max_count = main_counts.max()
min_count = main_counts.min()
imbalance_ratio = max_count / min_count

print(f"\nKelas terbanyak: {main_counts.idxmax()} ({max_count} sampel)")
print(f"Kelas tersedikit: {main_counts.idxmin()} ({min_count} sampel)")
print(f"Rasio ketidakseimbangan: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print("\n⚠️  DATASET SANGAT TIDAK SEIMBANG!")
    print("   Rasio > 3:1 dapat menyebabkan masalah dalam training.")
elif imbalance_ratio > 2:
    print("\n⚠️  Dataset cukup tidak seimbang.")
    print("   Pertimbangkan teknik balancing atau class weighting.")
else:
    print("\n✅ Dataset relatif seimbang.")

print("\n=== REKOMENDASI ===")
if imbalance_ratio > 2:
    print("1. Gunakan class_weight='balanced' dalam model")
    print("2. Pertimbangkan teknik oversampling (SMOTE) atau undersampling")
    print("3. Gunakan stratified sampling untuk train/validation split")
    print("4. Monitor precision, recall, dan F1-score per kelas")
else:
    print("Dataset sudah cukup seimbang untuk training langsung.")