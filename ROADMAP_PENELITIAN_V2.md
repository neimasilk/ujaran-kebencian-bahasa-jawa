# ROADMAP PENELITIAN V2 - Updated dengan Status Terbaru

**Update Terakhir:** 6 Januari 2026 (Setelah Experimen 5-7)
**Status Best Result:** 81.38% F1-Macro (Label Smoothing)

---

## PART 1: REVIEW HASIL YANG SUDAH DICAPAI

### Hasil Eksperimen Lengkap

| Experiment | Teknik | Dataset | F1-Macro | Status | Catatan |
|------------|--------|---------|----------|--------|---------|
| **Exp 5** | IndoBERT Base | Improved 10K | 79.19% | ✅ | Baseline baru |
| **Exp 6A** | Focal Loss | Improved 10K | 80.73% | ✅ | +1.54% |
| **Exp 6B** | Weighted Ensemble | Improved 10K | ~80% | ✅ | Tidak signifikan |
| **Exp 6C** | Label Smoothing + Hyperparam | Improved 10K | **81.38%** | ✅ | **BEST** |
| **Exp 7** | Simple Ensemble (Voting) | Improved 10K | 79.50% | ✅ | Tidak bekerja |
| **Exp 8** | Extended DAPT | - | ⏳ | 🔄 IN PROGRESS | Custom BERT v3 |

### Per-Class Breakdown (Best Model - Label Smoothing)

| Class | F1-Score | Issue |
|-------|----------|-------|
| Neutral | 78.21% | Perlu improvement |
| Light | 78.67% | **WEAKEST** - paling sulit |
| Moderate | 84.17% | Sudah cukup baik |
| Severe | 84.49% | Sudah sangat baik |

**Key Insight:** Class **Light** adalah bottleneck utama. Ini masuk akal karena:
- Borderline dengan Neutral (bisa ambigu)
- Bahasa yang lebih halus/sarkastik
- Kurang contoh ekstrem

---

## PART 2: ANALISIS GAP

### Gap ke Target

| Target | F1-Macro | Gap | Aksi Diperlukan |
|--------|----------|-----|----------------|
| Current | 81.38% | - | - |
| **Target Workshop** | 82% | **0.62%** | Quick Win |
| **Konservatif** | 85% | 3.62% | Moderate effort |
| **Moderate** | 87% | 5.62% | Signifikan effort |
| **Optimistik** | 90% | 8.62% | Radical approach |

### Bottleneck Analysis

```
Error Breakdown (Estimasi dari 81.38% F1):
├── Model Architecture (40%)
│   ├── Base model sudah optimal (IndoBERT)
│   └── Butuh: Javanese-specific pre-training
├── Dataset (35%)
│   ├── Size sudah cukup (10K)
│   ├── Balance sudah optimal (1.38:1)
│   └── BUTuh: Quality dan diversity
├── Method (20%)
│   ├── Label smoothing sudah best
│   ├── Ensemble tidak bekerja (model too similar)
│   └── BUTuh: Method yang fundamentally different
└── Label Quality (5%)
    └── AI-generated labels mungkin ada noise
```

---

## PART 3: REKOMENDASI TEKNIS (UPDATE)

### A. QUICK WINS (Gap 0.62% → 82%)

#### 1. Test-Time Augmentation (TTA) ⭐ **BARU - HIGH IMPACT**
**Konsep:** Augment saat inference, vote hasilnya

```python
def predict_with_tta(model, text, augmentations=5):
    votes = []
    for _ in range(augmentations):
        # Random augmentation: synonym replacement, word drop, etc.
        aug_text = augment(text)
        pred = model.predict(aug_text)
        votes.append(pred)
    return mode(votes)  # Majority voting
```

**Expected:** +0.5-1% → target 82%
**Why:** Memanfaatkan model uncertainty, gratis tanpa retraining
**Priority:** HIGH - Implementasi cepat

#### 2. Threshold Optimization per-Class ⭐ **BARU**
**Konsep:** Optimasi decision threshold per class

```python
# Current: argmax (global threshold 0.5)
# Better: Find optimal threshold per class
from sklearn.metrics import roc_curve

def optimize_thresholds(val_probs, val_labels):
    thresholds = {}
    for class_id in range(4):
        fpr, tpr, thr = roc_curve(val_labels == class_id, val_probs[:, class_id])
        # Find best F1 threshold
        thresholds[class_id] = thr[np.argmax(2*tpr/(tpr+fpr+1e-8) - 1)]
    return thresholds
```

**Expected:** +0.3-0.8% → target 82%
**Why:** Decision boundary tidak selalu di 0.5
**Priority:** HIGH - Implementasi sangat cepat

---

### B. MODERATE EFFORT (Gap 3.62% → 85%)

#### 3. 🚀 **RADIKAL: LLM-as-Judge untuk Re-Labeling**
**Konsep:** Gunakan LLM (GPT-4/Claude) untuk re-label uncertain samples

```python
# Step 1: Identify uncertain samples
uncertain = get_uncertain_samples(model, test_data, threshold=0.6)

# Step 2: LLM-as-Judge re-labeling
def llm_judge(text):
    prompt = f"""
    Classify this Javanese text into ONE of:
    0=Neutral, 1=Light Hate, 2=Moderate Hate, 3=Severe Hate

    Text: "{text}"

    Reason step by step, then output FINAL: X
    """
    return parse_llm_response(prompt)

# Step 3: Add confident predictions to training set
new_labels = {text: llm_judge(text) for text in uncertain}
```

**Expected:** +1-2% (better labels)
**Cost:** ~$50-100 untuk 5K samples
**Priority:** HIGH - Quality improvement

#### 4. 🚀 **RADIKAL: Cross-Lingual Transfer dari Indonesian Hate Speech**
**Konsep:** Pre-train dulu di Indonesian hate speech dataset, lalu transfer ke Javanese

```python
# Indonesian hate speech dataset (20K+ samples available)
# 1. Train on Indonesian hate speech
id_model = train_model(indonesian_hate_data)

# 2. Adapter-based transfer to Javanese
javanese_adapter = train_adapter(id_model, javanese_data, freeze_base=True)

# 3. Fine-tune adapter on Javanese hate speech
final_model = finetune(javanese_adapter, javanese_hate_data)
```

**Sources Indonesian Hate Speech:**
- [Indonesian Hate Speech & Abusive Language](https://github.com/ialahsa/indonesian-hate-speech)
- [NLP Indonesia](https://github.com/kata-ai/hatespeech-id) datasets

**Expected:** +1-3% (domain knowledge transfer)
**Priority:** HIGH - Leveraging existing resources

#### 5. Contrastive Learning Pre-Training ⭐ **BARU**
**Konsep:** Pre-train dengan contrastive objective untuk separate classes

```python
class ContrastivePreTrainer:
    def __init__(self):
        # Positive pairs: same class, similar
        # Negative pairs: different class or dissimilar

    def create_pairs(self, dataset):
        pairs = []
        for anchor in dataset:
            # Find positive: same class
            positives = [s for s in dataset if s.label == anchor.label]
            # Find negative: different class
            negatives = [s for s in dataset if s.label != anchor.label]

            pairs.append((anchor, random.choice(positives)))
            pairs.append((anchor, random.choice(negatives)))
        return pairs

    def contrastive_loss(self, anchor, positive, negative, temperature=0.07):
        # SimCSE-style loss
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=-1)

        loss = -torch.log(torch.exp(pos_sim/temperature) /
                             (torch.exp(pos_sim/temperature) + torch.exp(neg_sim/temperature)))
        return loss
```

**Expected:** +0.5-1% (better representation)
**Priority:** MEDIUM - Novel approach

#### 6. Hierarchical Classification ⭐ **BARU**
**Konsep:** Two-stage classification

```python
class HierarchicalClassifier:
    def __init__(self):
        # Stage 1: Binary (Hate vs Non-Hate)
        self.binary_model = IndoBERT(num_labels=2)

        # Stage 2a: Non-Hate → Neutral
        # Stage 2b: Hate → Severity (Light/Moderate/Severe)
        self.severity_model = IndoBERT(num_labels=3)

    def predict(self, text):
        # Stage 1
        is_hate = self.binary_model.predict(text) == 1

        if not is_hate:
            return 0  # Neutral
        else:
            # Stage 2
            severity = self.severity_model.predict(text)
            return {1: 1, 2: 2, 3: 3}[severity]  # Map to 1,2,3
```

**Expected:** +0.5-1% (simpler problems)
**Priority:** MEDIUM - Architectural change

---

### C. RADICAL APPROACH (Gap 5.62%+ → 87-90%)

#### 7. 🚀 **SUPER RADIKAL: GAN-Based Data Generation**
**Konsep:** Generate synthetic hate speech samples dengan GAN

```python
class HateSpeechGAN:
    def __init__(self):
        # Generator: Generate fake hate speech
        self.generator = TransformerGenerator(
            vocab_size=tokenizer.vocab_size,
            latent_dim=128,
            seq_length=128
        )

        # Discriminator: Classify real vs fake hate speech
        self.discriminator = TransformerDiscriminator(num_classes=2)

        # Auxiliary classifier: Classify severity
        self.aux_classifier = TransformerClassifier(num_classes=4)

    def train(self, real_data):
        # Train GAN to generate realistic hate speech
        # Use auxiliary classifier to ensure label consistency
```

**Expected:** +2-4% (more diverse data)
**Risk:** Training complexity
**Priority:** MEDIUM - High effort, high reward

#### 8. 🚀 **SUPER RADIKAL: Instruction Tuning dengan Chain-of-Thought**
**Konsep:** Instruction-tune model dengan reasoning explisit

```python
instruction_dataset = [
    {
        "input": "Dasar lu tolol!",
        "instruction": "Classify hate speech severity",
        "output": "Moderate Hate (2). Reason: 'tolol' is insult but not severe hate speech."
    },
    # ... more examples with reasoning
]

model = instruct_tune(base_model, instruction_dataset)
```

**Expected:** +1-3% (better reasoning)
**Priority:** MEDIUM - Modern approach

#### 9. 🚀 **SUPER RADIKAL: Multi-Modal Approach (Text + Context)**
**Konsep:** Include additional context (user metadata, conversation history)

```python
class MultiModalHateClassifier:
    def __init__(self):
        self.text_encoder = IndoBERT()
        self.context_encoder = Transformer()
        self.fusion = CrossAttentionFusion()

    def predict(self, text, context):
        text_features = self.text_encoder(text)
        context_features = self.context_encoder(context)
        fused = self.fusion(text_features, context_features)
        return self.classifier(fused)
```

**Expected:** +0.5-2% (more information)
**Priority:** LOW - Needs additional data

#### 10. 🚀 **SUPER RADIKAL: Retrieval-Augmented Generation (RAG)**
**Konsep:** Retrieve similar examples untuk context saat prediction

```python
class RAGHateClassifier:
    def __init__(self, classifier, retriever):
        self.classifier = classifier
        self.retriever = retriever  # FAISS index of training examples

    def predict(self, text):
        # Retrieve top-K similar examples
        examples = self.retriever.retrieve(text, k=5)

        # Use examples as context
        prompt = f"Similar cases: {examples}\\nClassify: {text}"
        return self.classifier.predict_with_context(prompt)
```

**Expected:** +0.5-1% (better generalization)
**Priority:** LOW - Novel approach

---

## PART 4: ROADMAP UPDATE (PRIORITAS BARU)

### Phase 8: Quick Wins untuk Tembus 82%

| Task | Method | Expected | Timeline | Status |
|------|--------|----------|----------|--------|
| 8.1 | Test-Time Augmentation (TTA) | +0.5-1% | 1 jam | 🆕 NEW |
| 8.2 | Threshold Optimization | +0.3-0.8% | 30 menit | 🆕 NEW |
| 8.3 | LLM Re-labeling Uncertain Samples | +0.5-1% | 2-4 jam | 🆕 NEW |

**Cumulative Expected:** +1.3-2.8% → **82.7-84.2%**

### Phase 9: Moderate Effort untuk Tembus 85%

| Task | Method | Expected | Timeline | Status |
|------|--------|----------|----------|--------|
| 9.1 | Cross-Lingual Transfer (ID→JV) | +1-3% | 4-8 jam | 🆕 NEW |
| 9.2 | Extended DAPT (v3) | +1-2% | ⏳ Running | 🔄 IN PROGRESS |
| 9.3 | Contrastive Pre-Training | +0.5-1% | 4-6 jam | 🆕 NEW |
| 9.4 | Hierarchical Classification | +0.5-1% | 2-3 jam | 🆕 NEW |

**Cumulative Expected:** +3-7% → **85-88%**

### Phase 10: Radical Approaches untuk Tembus 87-90%

| Task | Method | Expected | Timeline | Status |
|------|--------|----------|----------|--------|
| 10.1 | GAN-Based Data Generation | +2-4% | 1-2 hari | 🆕 NEW |
| 10.2 | Instruction Tuning + CoT | +1-3% | 4-8 jam | 🆕 NEW |
| 10.3 | RAG-Based Classification | +0.5-1% | 4-6 jam | 🆕 NEW |

---

## PART 5: STRATEGI REKOMENDASI

### 🎯 UNTUK TEMBUS 82% TARGET WORKSHOP (1-2 HARI)

**Do This First:**
1. **Test-Time Augmentation** - 1 jam, paling cepat
2. **Threshold Optimization** - 30 menit, gratis
3. **LLM Re-labeling** - 2-4 jam, untuk quality

**Expected Result:** 82.7-84.2%

### 🎯 UNTUK TEMBUS 85% (1 MINGGU)

**Setelah 82% tercapai:**
1. **Cross-Lingual Transfer** - Leverage Indonesian resources
2. **Extended DAPT v3** - Sedang berjalan, monitor hasil
3. **Hierarchical Classification** - Architectural improvement

### 🎯 UNTUK TEMBUS 87-90% (JANGKA PANJANG)

**Research-oriented:**
1. **GAN-Based Generation** - Novel approach
2. **Instruction Tuning** - Modern SOTA methods
3. **RAG-Based** - Contextual prediction

---

## PART 6: IDE RADIKAL TAMBAHAN

### 💡 RADIKAL IDEA 1: Few-Shot Learning untuk Rare Classes

```python
# Problem: Class Light (78%) lemah
# Solution: Few-shot learning untuk rare classes

class FewShotHateClassifier:
    def __init__(self):
        self.base_model = IndoBERT()
        self.support_set = {
            "light": [
                ("bodoh kamu", 1),
                ("goblok banget", 1),
                # ... 5-10 examples
            ],
            "severe": [
                ("matamu kejam", 3),
                # ... examples
            ]
        }

    def predict_fewshot(self, text):
        # Compare with support set
        similarities = []
        for cls, examples in self.support_set.items():
            sim = compare_with_examples(text, examples)
            similarities.append((cls, sim))
        return max(similarities)[0]
```

**Expected:** +0.3-0.5% untuk weak classes

### 💡 RADIKAL IDEA 2: Curriculum Learning

```python
# Train dari easy → hard samples

class CurriculumTrainer:
    def __init__(self):
        self.difficulty_scores = {}

    def sort_by_difficulty(self, data):
        # Score by: prediction confidence, length, complexity
        for sample in data:
            score = self.difficulty_scores[sample.id]
        return sorted(data, key=lambda x: x.difficulty)

    def train(self):
        # Stage 1: Easy samples (confidence > 0.9)
        # Stage 2: Medium samples (0.7 < confidence < 0.9)
        # Stage 3: Hard samples (confidence < 0.7)
        pass
```

**Expected:** +0.5-1% (better learning dynamics)

### 💡 RADIKAL IDEA 3: Ensemble dengan Diverse Base Models

```python
# Problem: Ensemble tidak bekerja karena model too similar
# Solution: Use fundamentally different approaches

models = [
    # BERT-based (IndoBERT, mBERT, XLM-R)
    IndoBERT(),
    mBERT(),
    XLMRoBERTa(),

    # Non-BERT (diversify!)
    LSTMWithAttention(),      # Different architecture
    CNNTextClassifier(),      # CNN for text
    TransformerXL(),         # Long-range dependency

    # Pre-trained embeddings
    FastTextWithNgrams(),     # Bag-of-words + ngrams
    UniversalSentenceEncoder(), # USE
]

# Meta-learner: XGBoost dengan features dari semua models
```

**Expected:** +1-2% (true diversity)

---

## PART 7: IMPLEMENTATION PLAN HARI INI

### Immediate (Sisa 2 jam hari ini)

1. ✅ **Extended DAPT v3** - Sedang berjalan, biarkan selesai
2. ⏳ **Implement TTA** - Buat script, test besok
3. ⏳ **Implement Threshold Optimization** - Buat script, test besok

### Besok (di rumah/kantor)

1. ⏳ **Cek hasil DAPT v3** - Jika bagus, fine-tune untuk hate speech
2. ⏳ **Run TTA + Threshold Opt** - Harusnya tembus 82%
3. ⏳ **LLM Re-labeling** - Kalau masih belum tembus 82%

### Checkpoint System

Semua progress tersimpan di:
- `experiments/experiment_8_progress.json` - DAPT progress
- GitHub - Semua code sudah pushed
- Bisa resume kapan saja

---

## SUMMARY

### Current Status
- **Best F1-Macro:** 81.38%
- **Bottleneck:** Class Light (78.67%), model diversity
- **Target Workshop (82%):** Gap 0.62%

### Top 3 Recommendations (Highest Impact)

1. **Test-Time Augmentation** - Cepat, gratis, +0.5-1%
2. **Threshold Optimization** - Cepat, gratis, +0.3-0.8%
3. **Cross-Lingual Transfer** - Medium effort, +1-3%

### Radikal Ideas untuk Consider

1. **LLM-as-Judge Re-labeling** - Quality improvement
2. **GAN-Based Data Generation** - Novel approach
3. **Instruction Tuning** - Modern SOTA

---

**Update:** 6 Januari 2026, 14:00
**Status:** Siap untuk implementasi Phase 8 (Quick Wins)
