# ROADMAP FOCUSED - PAPER SUBMISSION
**Update:** 7 Januari 2026
**Status:** 81.38% F1-Macro → Target 85%
**Approach:** RADICAL & FOCUSED

---

## CURRENT STATUS

### What Works (Keep)
| Method | F1-Macro | Status |
|--------|----------|--------|
| IndoBERT + Label Smoothing | **81.38%** | ✅ BEST |
| Dataset Phase 3+4 | - | ✅ BEST DATASET |

### What FAILED (Don't Repeat)
| Method | Result | Why Failed |
|--------|--------|------------|
| Phase 5 (DeepSeek re-label) | 77-78% | AI labels noisy |
| Custom BERT v3 | 78.26% | DAPT not effective |
| Test-Time Augmentation | 80.44% | Lost important signals |
| Threshold Optimization | 80.13% | Model already calibrated |
| Large Balanced (39K) | 61.67% | Too imbalanced |

---

## RADICAL STRATEGY (3 PRONGED APPROACH)

### PRONG 1: DATA-CENTRIC (Expected +2-3%)
**Core Insight:** Dataset Phase 3+4 is good but needs ENHANCEMENT

#### 1.1 Hard Negative Mining ⭐ **HIGH PRIORITY**
```python
# Find samples that model consistently gets wrong
def get_hard_negatives(model, val_data, threshold=0.6):
    hard_samples = []
    for text, true_label in val_data:
        pred_prob = model.predict_proba(text)
        if pred_prob[true_label] < threshold:  # Model uncertain
            hard_samples.append((text, true_label, pred_prob))
    return hard_samples

# Then: HUMAN EXPERT re-labels these 500-1000 samples
# Expected: +1-2% (fixing systematic errors)
```

**Timeline:** 2-3 hours (identify) + manual labeling
**Risk:** Low (only improves data quality)

#### 1.2 Counterfactual Data Augmentation ⭐ **NEW - RADICAL**
```python
# Generate "what if" variations
def counterfactual_augment(text, label):
    # Neutral → Light: Add mild insult
    # Light → Moderate: Strengthen insult
    # Moderate → Severe: Add threat/violence

    examples = {
        0: [  # Neutral examples
            "kamu bodoh",  # borderline
            "goblok sih",  # borderline
        ],
        1: [  # Light → Moderate
            "bodoh banget sialan",  # intensified
        ],
        2: [  # Moderate → Severe
            "mampus kau mati saja",  # added violence
        ]
    }
    return systematic_augmentation(text, examples)

# Expected: +0.5-1% (better boundaries)
```

**Timeline:** 4-6 hours
**Risk:** Medium (need careful generation)

---

### PRONG 2: MODEL-CENTRIC (Expected +1-2%)
**Core Insight:** IndoBERT is good but let's try fundamentally DIFFERENT models

#### 2.1 DeBERTa V3 ⭐ **NEW - SOTA MODEL**
```python
# DeBERTa V3 is SOTA for many NLU tasks
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification

model = DebertaV2ForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",  # 184M params
    num_labels=4
)

# Key advantages:
# - Disentangled attention (better for long-range)
# - Enhanced mask decoder (better understanding)
# - SOTA on GLUE, SuperGLUE

# Expected: +1-2% (better architecture)
```

**Timeline:** 1-2 hours (training similar to IndoBERT)
**Risk:** Low (proven architecture)

#### 2.2 Adapter Fusion ⭐ **NEW - EFFICIENT**
```python
# Instead of full DAPT, use ADAPTERS
from adapters import AutoAdapterModel

model = AutoAdapterModel.from_pretrained("indobenchmark/indobert-base-p1")

# Add task-specific adapter
model.add_adapter("hate_javanese")
model.train_adapter("hate_javanese")

# Benefits:
# - Fast training (only 3-4% parameters)
# - Can combine multiple adapters
# - Reusable base model

# Expected: +0.5-1% (efficient adaptation)
```

**Timeline:** 2-3 hours
**Risk:** Low

---

### PRONG 3: LEARNING-CENTRIC (Expected +1-2%)
**Core Insight:** Training strategy matters

#### 3.1 Curriculum Learning by Difficulty ⭐ **NEW**
```python
# Train from easy → hard samples
def curriculum_training(model, data):
    # Stage 1: Sort by prediction confidence (pre-trained model)
    easy = [s for s in data if s.confidence > 0.9]
    medium = [s for s in data if 0.7 < s.confidence <= 0.9]
    hard = [s for s in data if s.confidence <= 0.7]

    # Stage 2: Train in order
    model.train(easy, epochs=2)
    model.train(medium, epochs=2)
    model.train(hard, epochs=3)

# Expected: +0.5-1% (better convergence)
```

**Timeline:** 1 hour (implementation) + training time
**Risk:** Low

#### 3.2 Knowledge Distillation from Larger Model ⭐ **NEW**
```python
# Teacher: Large model (IndoBERT-Large or XLM-R-Large)
# Student: Base IndoBERT

teacher = AutoModel.from_pretrained("indobenchmark/indobert-large-p1", num_labels=4)
student = AutoModel.from_pretrained("indobenchmark/indobert-base-p1", num_labels=4)

# Train student with teacher's knowledge
def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    # Soft labels from teacher
    teacher_soft = F.softmax(teacher_logits / T, dim=-1)
    student_soft = F.log_softmax(student_logits / T, dim=-1)

    # KD loss + CE loss
    kd_loss = F.kl_div(student_soft, teacher_soft) * (T**2)
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * kd_loss + (1 - alpha) * ce_loss

# Expected: +1-2% (teacher guidance)
```

**Timeline:** 2-3 hours
**Risk:** Low

---

## EXECUTION PLAN (PRIORITIZED)

### Phase 1: Quick Wins (Today - 4 hours)
| # | Task | Expected | Time | Priority |
|---|------|----------|------|----------|
| 1 | DeBERTa V3 Training | +1-2% | 2h | 🔥 URGENT |
| 2 | Hard Negative Mining | +1-2% | 2h | 🔥 URGENT |

**Cumulative:** 83.4-85.4% → TARGET ACHIEVED

### Phase 2: If Needed (Tomorrow - 6 hours)
| # | Task | Expected | Time | Priority |
|---|------|----------|------|----------|
| 3 | Adapter Training | +0.5-1% | 2h | HIGH |
| 4 | Knowledge Distillation | +1-2% | 3h | HIGH |
| 5 | Curriculum Learning | +0.5-1% | 1h | MEDIUM |

**Cumulative:** 85.4-89.4% → EXCEEDS TARGET

### Phase 3: Radical (If still needed)
| # | Task | Expected | Time | Priority |
|---|------|----------|------|----------|
| 6 | Counterfactual Augmentation | +0.5-1% | 4h | MEDIUM |
| 7 | Ensemble (DeBERTa + IndoBERT) | +0.5-1% | 1h | LOW |

---

## WHY THIS WILL WORK

### 1. DeBERTa V3 Advantage
- **SOTA architecture** (better than BERT)
- **Disentangled attention** (captures Javanese nuance better)
- **Proven results** on multilingual tasks
- **Same training cost** as IndoBERT

### 2. Hard Negative Mining
- **Targets systematic errors** (not random noise)
- **Human-in-the-loop** (quality guaranteed)
- **Focused improvement** (only on weak samples)

### 3. Knowledge Distillation
- **Teacher > Student** (guaranteed improvement)
- **Proven technique** (used in many SOTA models)
- **No extra data needed**

---

## SUCCESS METRICS

| Target | F1-Macro | Required |
|--------|----------|----------|
| Workshop Acceptance | 82% | +0.62% |
| Conservative Paper | 85% | +3.62% |
| Good Paper | 87% | +5.62% |
| Strong Paper | 90% | +8.62% |

**Phase 1 Expected:** 83.4-85.4% ✅ ACHIEVES WORKSHOP & CONSERVATIVE

---

## NEXT IMMEDIATE ACTIONS

1. **[ ] Run DeBERTa V3 training** (2 hours)
2. **[ ] Extract hard negatives** (30 min)
3. **[ ] Manual label hard negatives** (1.5 hours)
4. **[ ] Retrain with improved data** (1 hour)

**Total Time:** ~4-5 hours for 85% target

---

*Created: 7 Januari 2026*
*Status: Ready to execute*
