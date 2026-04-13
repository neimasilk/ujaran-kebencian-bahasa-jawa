# New References for Revised Paper
# Target: 30-40 total refs (currently ~31, need ~15 new)

## COMPARISON TABLE REFERENCES (for Table: Prior Work Comparison)

### [NEW] R1: Susanto et al. (2024) — IndoToxic2024
L. Susanto et al., "IndoToxic2024: A Demographically-Enriched Dataset of Hate Speech and Toxicity Types for Indonesian Language," arXiv:2406.19349, 2024.
- Indonesian, 7 binary tasks, 43,692 entries
- IndoBERTweet macro-F1 = 78%
- Most recent large Indonesian hate speech dataset

### [NEW] R2: Pamungkas & Chiril (2025) — Ngalawan (Javanese code-mixed)
E. W. Pamungkas and P. Chiril, "Ngalawan Ujaran Sengit: Hate speech detection in Indonesian code-mixed social media data," Language Resources and Evaluation, vol. 59, pp. 2387-2414, 2025.
- Indonesian-Javanese code-mixed, binary
- LR, RF, RNN, XLM-R, IndoBERT
- Most relevant Javanese work (2025)

### [EXISTING] Davidson et al. (2017)
Already cited. English, 3 classes, 24,802 tweets, LR+TF-IDF, F1=90% (weighted)
NOTE: Must add footnote that this is F1-weighted, not macro

### [EXISTING] Alfina et al. (2017)
Already cited. Indonesian, binary, 713 tweets, RFDT, F1=93.5%

### [EXISTING] Ibrohim & Budi (2019)
Already cited. Indonesian, multi-label, 13,169, RFDT+LP, ~77% accuracy

### [EXISTING] Putri et al. (2021)
Already cited. Javanese, binary, ~2,500, SVM, F1>60%

---

## GAP ANALYSIS REFERENCES (for Introduction)

### [NEW] R3: Gorman & Bedrick (2019) — Standard splits unreliable
K. Gorman and S. Bedrick, "We Need to Talk about Standard Splits," in Proc. 57th ACL, Florence, Italy, 2019, pp. 2786-2791. [Outstanding Paper Award]
- Rankings of 9 POS taggers on standard splits failed to reproduce with random splits
- KEY FOR GAP: Supports argument that single-split evaluation is misleading

### [NEW] R4: Søgaard et al. (2021) — Random splits overly optimistic
A. Søgaard, S. Ebert, J. Bastings, and K. Filippova, "We Need To Talk About Random Splits," in Proc. EACL 2021, pp. 1823-1832.
- Random splits lead to overly optimistic estimates
- Recommends multiple independent test sets
- KEY: Directly supports our dual-track evaluation protocol

### [NEW] R5: Jahan et al. (2024) — Comprehensive augmentation for hate speech
M. S. Jahan et al., "A Comprehensive Study on NLP Data Augmentation for Hate Speech Detection: Legacy Methods, BERT, and LLMs," Findings of ACL 2024, arXiv:2404.00303.
- GPT-3 augmentation improved F1 by +1.4% over traditional methods
- KEY: Shows LLM augmentation helps but doesn't address evaluation bias

---

## AUGMENTATION & SYNTHETIC DATA REFERENCES

### [NEW] R6: Hartvigsen et al. (2022) — ToxiGen
T. Hartvigsen et al., "ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection," in Proc. 60th ACL, Dublin, 2022, pp. 3309-3326.
- 274,000 machine-generated toxic/benign statements
- Shows synthetic data CAN improve classification
- KEY: But raises evaluation questions — does it generalize to real data?

### [NEW] R7: Vidgen et al. (2021) — Dynamic hate speech datasets
B. Vidgen, T. Thrush, Z. Waseem, and D. Kiela, "Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection," in Proc. 59th ACL, 2021, pp. 1667-1682.
- 40,000 entries generated dynamically
- Models on later rounds perform better and are harder to trick

### [NEW] R8: Juuti et al. (2020) — GPT-2 augmentation for toxic language
M. Juuti, T. Grondahl, A. Flanagan, and N. Asokan, "A little goes a long way: Improving toxic language classification despite data scarcity," Findings of EMNLP 2020, pp. 2991-3009.
- GPT-2 augmentation + BERT/shallow classifiers
- Key: Shallow classifiers on augmented data ≈ BERT performance

### [NEW] R9: Rizwan et al. (2022) — Conditional LM augmentation
H. Rizwan, M. H. Luqman, and A. M. Qamar, "Exploring Conditional Language Model Based Data Augmentation Approaches for Hate Speech Classification," in LNCS, vol. 12882, Springer, 2022.
- GPT-2 augmentation: up to 7.3% and 25.0% relative F1 improvement
- On two hate speech corpora

---

## CROSS-LINGUAL & MODEL REFERENCES

### [NEW] R10: Conneau et al. (2020) — XLM-RoBERTa
A. Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale," in Proc. 58th ACL, 2020, pp. 8440-8451.
- XLM-R: 100 languages, 2TB CommonCrawl
- +14.6% accuracy vs mBERT on XNLI
- KEY: Explains why XLM-R outperforms IndoBERT in our experiments

### [NEW] R11: Aji et al. (2022) — Indonesia 700+ Languages
A. F. Aji et al., "One Country, 700+ Languages: NLP Challenges for Underrepresented Languages and Dialects in Indonesia," in Proc. 60th ACL, Dublin, 2022, pp. 7226-7249.
- Survey of Indonesian language NLP challenges
- Highlights Javanese as low-resource despite 80M+ speakers

---

## ANNOTATION & METHODOLOGY REFERENCES

### [NEW] R12: Paun et al. (2018) — Annotation models
S. Paun, B. Carpenter, J. Chamberlain, D. Hovy, U. Kruschwitz, and M. Poesio, "Comparing Bayesian Models of Annotation," Transactions of the ACL, vol. 6, pp. 571-585, 2018.
- Bayesian models for handling annotator disagreement
- Supports our κ=0.72 discussion

### [NEW] R13: Aroyo & Welty (2015) — CrowdTruth
L. Aroyo and C. Welty, "Truth Is a Lie: Crowd Truth and the Seven Myths of Human Annotation," AI Magazine, vol. 36, no. 1, pp. 15-24, 2015.
- Argues disagreement IS informative, not noise
- Supports our finding that severity boundaries are subjective

---

## TOTAL NEW REFERENCES: 13
## TOTAL WITH EXISTING (~31): ~44 references

## RECOMMENDED REFERENCE NUMBERING (IEEE, order of appearance)
Keep existing [1]-[31] where possible, insert new ones at appropriate points.
Better approach: renumber completely in the new paper since structure is changing.
