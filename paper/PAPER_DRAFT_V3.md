# PAPER DRAFT V3 — For Kinetik Resubmission
# All text below will be converted to generate_kinetik_paper_v3.py

---

## TITLE

Evaluating the Reliability of LLM-Augmented Training Data for Severity-Based Hate Speech Detection in Low-Resource Javanese

## AUTHORS

Mukhlis Amien*1, Daniel Rudiaman Sijabat2, Yekti Asmoro Kanthi3
1,2 Department of Informatics, 3 Department of Information System
Universitas Bhinneka Nusantara, Malang, Indonesia
*mukhlisramien@gmail.com

## KEYWORDS

Data Augmentation, Evaluation Bias, Low-Resource NLP, Severity Classification, Pre-trained Language Model

(Note: no overlap with title words)

---

## ABSTRACT

Article info: (Received / Revised / Accepted / Online / Published)

The increasing adoption of large language model (LLM)-generated data for training hate speech classifiers in low-resource languages raises a critical but underexplored question: do performance metrics evaluated on mixed manual-synthetic test sets reliably reflect real-world detection capability? This study addressed this question through a case study on severity-based hate speech detection in Javanese, a low-resource language with over 80 million speakers. We constructed a dataset of 9,775 annotated samples across four severity levels (not hate speech, light, moderate, and severe), comprising 46.4% manually annotated and 53.6% LLM-generated texts. Five models were evaluated: SVM, Logistic Regression, IndoBERT, IndoBERT with label smoothing, and XLM-RoBERTa Large. A dual-track evaluation protocol was employed, testing on both the full mixed test set and a manual-only subset. Results revealed a substantial evaluation bias: XLM-RoBERTa Large achieved 80.26% F1-Macro on the full test set but only 53.89% on manual-only data, while synthetic test data yielded 99.41%. A source distinguishability analysis showed that a simple classifier could differentiate synthetic from manual texts with 97.26% F1, indicating fundamentally different distributions. Augmentation ratio experiments confirmed that the performance inflation was consistent across models and ratios. These findings demonstrate that LLM-augmented evaluation substantially overestimates real-world hate speech detection performance and highlight the need for separate manual-only evaluation in augmented settings.

---

## 1. INTRODUCTION

[Para 1: Global context]
Online hate speech has become a pervasive global phenomenon that threatens social cohesion, with platforms processing billions of posts daily across diverse languages and cultures [1]. Automated detection systems based on natural language processing (NLP) have emerged as essential tools for content moderation at scale [2]. However, the vast majority of research and resources have concentrated on English and other high-resource languages, leaving speakers of regional and minority languages inadequately protected [3].

[Para 2: Indonesia and regional languages]
Indonesia presents a particularly acute case of this disparity. As the world's fourth most populous nation with over 700 living languages [4], Indonesia's linguistic diversity far exceeds the capacity of current NLP systems. While hate speech detection for formal Indonesian has received growing attention [5][6], regional languages such as Javanese --- spoken by approximately 82 million people as a first language --- remain severely underserved. Only limited work has addressed Javanese hate speech detection, exclusively at the binary (hate/not-hate) level [7], leaving severity-based classification entirely unexplored.

[Para 3: The augmentation paradigm]
To overcome the scarcity of annotated data in low-resource languages, researchers have increasingly turned to large language model (LLM)-based data augmentation [8][9]. This approach uses LLMs to generate synthetic training samples that expand small seed datasets, with studies reporting improvements of 3-26% in F1 scores for various NLP tasks [10]. For hate speech detection specifically, machine-generated data has been shown to improve classifier performance in several settings [11][12][13]. However, a fundamental methodological question remains largely unaddressed: when synthetic data constitutes a significant portion of both training and test sets, do reported performance metrics accurately reflect a model's ability to detect naturally-occurring hate speech?

[Para 4: The GAP]
Despite the growing adoption of LLM augmentation in low-resource hate speech detection, no prior study has systematically evaluated whether performance metrics computed on mixed (manual + synthetic) test sets reliably indicate real-world detection capability. This gap is critical: if augmented test data inflates reported metrics, the field may overestimate progress in low-resource hate speech detection, potentially leading to premature deployment of inadequate systems. Prior work on evaluation pitfalls in NLP has demonstrated that standard evaluation practices can produce misleadingly optimistic estimates [14][15], yet this concern has not been specifically investigated in the context of LLM-augmented hate speech datasets.

[Para 5: Research questions]
This study addresses three research questions:
- RQ1: How effective are transformer-based models compared to traditional machine learning approaches for severity-based Javanese hate speech classification?
- RQ2: Does LLM data augmentation inflate reported performance metrics when synthetic samples appear in the test set?
- RQ3: What characteristics of synthetic data enable models to achieve near-perfect performance on synthetic test samples while struggling with manual ones?

[Para 6: Contributions]
The contributions of this work are threefold. First, we present the first severity-level (four-class) hate speech classification dataset for Javanese, comprising 9,775 annotated samples. Second, we propose and apply a dual-track evaluation protocol that separately measures performance on manual-only and full (mixed) test sets, revealing a 45.52 percentage-point F1 gap attributable to augmentation bias. Third, we provide empirical evidence that LLM-generated data is distributionally distinct from manual annotations (97.26% source distinguishability), explaining why augmented evaluation is unreliable.

---

## 2. RELATED WORK

[Para 1: Hate speech detection landscape]
Hate speech detection has been extensively studied for high-resource languages, with seminal work by Davidson et al. [16] establishing a benchmark of 24,802 English tweets classified into three categories (hate speech, offensive language, and neither). For Indonesian, Alfina et al. [5] created an early dataset of 713 tweets for binary classification using traditional ML methods, while Ibrohim and Budi [6] expanded the scope to 13,169 multi-label tweets. More recently, Susanto et al. [17] introduced IndoToxic2024 with 43,692 entries across seven toxicity dimensions. For Javanese specifically, Putri et al. [7] conducted preliminary binary classification on approximately 2,500 samples using SVM, and Pamungkas and Chiril [18] addressed code-mixed Indonesian-Javanese hate speech. Table 1 summarizes the landscape.

**TABLE 1: Comparison with Prior Work on Hate Speech Detection**

| Study | Language | Classes | Dataset | Method | F1 (%) | Metric |
|---|---|---|---|---|---|---|
| Alfina et al. [5] | Indonesian | 2 | 713 | RFDT+TF-IDF | 93.5 | F-measure |
| Davidson et al. [16] | English | 3 | 24,802 | LR+TF-IDF | 90.0* | weighted |
| Ibrohim & Budi [6] | Indonesian | multi | 13,169 | RFDT+LP | 77.0 | accuracy |
| Putri et al. [7] | Javanese | 2 | ~2,500 | SVM+TF-IDF | >60 | F-measure |
| Susanto et al. [17] | Indonesian | 7 | 43,692 | IndoBERTweet | 78.0 | macro-F1 |
| Pamungkas & Chiril [18] | Jav-Ind mix | 2 | -- | XLM-R | -- | macro-F1 |
| **Ours (full test)** | **Javanese** | **4** | **9,775** | **XLM-R Large** | **80.26** | **macro-F1** |
| **Ours (manual only)** | **Javanese** | **4** | **4,538** | **XLM-R Large** | **53.89** | **macro-F1** |

*F1-weighted, not directly comparable to macro-F1 scores.

Notably, no prior work has attempted severity-level (multi-class) hate speech classification for Javanese. The binary approach used by Putri et al. [7] cannot distinguish between mild microaggressions and severe dehumanizing rhetoric, limiting its practical utility for content moderation.

[Para 2: Data augmentation for hate speech]
Data augmentation using language models has gained traction for expanding limited hate speech datasets. Hartvigsen et al. [11] demonstrated with ToxiGen that machine-generated implicit hate speech (274,000 statements) could improve classifier training. Vidgen et al. [12] proposed dynamic dataset generation through human-and-model-in-the-loop annotation. Juuti et al. [13] found that GPT-2 augmented data enabled shallow classifiers to approach BERT-level performance even with scarce seed data. Jahan et al. [19] conducted a comprehensive study across five benchmarks, finding GPT-3 augmentation improved F1 by 1.4% over traditional methods. However, none of these studies systematically evaluated whether synthetic data in the test set inflates reported metrics.

[Para 3: Evaluation pitfalls in NLP]
Concerns about evaluation reliability in NLP are well-documented. Gorman and Bedrick [14] demonstrated that rankings of NLP systems on standard splits failed to reproduce with random splits, receiving an Outstanding Paper Award for highlighting this fundamental issue. Sogaard et al. [15] extended this concern, showing that random splits also produce overly optimistic estimates and recommending multiple independent test sets. Our dual-track evaluation protocol directly addresses these concerns by introducing source-aware test set partitioning.

[Para 4: Pre-trained multilingual models]
Cross-lingual transfer through pre-trained multilingual models offers a promising approach for low-resource languages. XLM-RoBERTa [20], trained on 100 languages including Indonesian, achieved substantial gains over monolingual baselines on cross-lingual benchmarks. IndoBERT [21], specifically pre-trained on Indonesian text, established strong baselines for Indonesian NLP tasks. Aji et al. [4] highlighted the challenges of extending NLP to Indonesia's 700+ languages, noting that even large multilingual models may underperform on truly low-resource varieties like Javanese.

---

## 3. RESEARCH METHOD

### 3.1 Dataset Construction

The dataset was constructed in two phases: manual annotation and LLM augmentation.

**Manual annotation.** A total of 4,538 Javanese texts were collected from Twitter and Instagram, then annotated by three native Javanese speakers with linguistics training into four severity levels: (0) Not Hate Speech, (1) Light --- subtle stereotyping or microaggressions, (2) Moderate --- explicit prejudice without dehumanization, and (3) Severe --- dehumanizing rhetoric, calls for violence, or slurs. Inter-annotator agreement was moderate-to-substantial (Cohen's kappa = 0.72), with majority voting resolving disagreements.

**TABLE 2: Severity Level Definitions with Examples**

| Level | Label | Description | Example |
|---|---|---|---|
| 0 | Not Hate | Neutral or positive content | "Aku mikir umume wong setuju yen kabeh manungsa padha." |
| 1 | Light | Subtle stereotyping, microaggressions | "Wah, ibu iki koyo detektif, sak klebatan motor tamu wae wis kudu takon." |
| 2 | Moderate | Explicit prejudice, group-level attacks | "Wong asing teka mung gawe masalah ing kampung kita." |
| 3 | Severe | Dehumanization, calls for violence | "Kabeh keturunan iku mata duitan, kudu dipeksa lunga." |

Note: Examples are representative and may be abbreviated for space.

**LLM augmentation.** To address class imbalance and data scarcity, 5,237 additional samples were generated using DeepSeek-Coder-V2 (236B parameters) through prompted generation with severity-specific instructions. Generated texts underwent quality filtering (naturalness >= 3/5, cultural appropriateness >= 4/5) and human verification on a random subset of 500 samples (kappa = 0.72).

After cleaning (removing duplicates, near-duplicates, and texts shorter than 3 words), the final dataset comprised 9,775 samples: 4,538 manual (46.4%) and 5,237 synthetic (53.6%). Each sample was tagged with its source (manual or synthetic) to enable source-aware evaluation.

**TABLE 3: Dataset Composition by Class and Source**

| Class | Manual | Synthetic | Total | Synthetic % |
|---|---|---|---|---|
| 0 - Not Hate | 1,877 | 540 | 2,417 | 22% |
| 1 - Light | 1,198 | 1,333 | 2,531 | 53% |
| 2 - Moderate | 685 | 2,094 | 2,779 | 75% |
| 3 - Severe | 778 | 1,270 | 2,048 | 62% |
| Total | 4,538 | 5,237 | 9,775 | 54% |

Note: Class 2 (Moderate) has the highest synthetic proportion (75%), which is relevant to the per-class performance analysis in Section 4.

### 3.2 Dual-Track Evaluation Protocol

To assess whether LLM augmentation inflates evaluation metrics, we employed a dual-track evaluation protocol. The dataset was split into training (80%), validation (10%), and test (10%) sets using stratified sampling (seed=42), preserving class proportions. Crucially, we tracked the source (manual/synthetic) of each sample through the split.

Evaluation was conducted on:
1. **Full test set** (978 samples, ~46% manual, ~54% synthetic): standard mixed evaluation
2. **Manual-only test subset** (451 samples): only naturally-occurring texts, reflecting real-world performance

Additionally, we conducted an augmentation ratio study, training models with 0%, 25%, 50%, 75%, and 100% of available synthetic data while always evaluating on the manual-only test subset. This isolates the effect of augmentation on real-world performance versus evaluation inflation.

### 3.3 Models

Five models were evaluated, spanning traditional machine learning and transformer architectures:

**Baselines:**
- **SVM + TF-IDF** [22]: Linear kernel SVM with TF-IDF features (max 10,000 features, unigrams and bigrams, sublinear TF scaling)
- **Logistic Regression + TF-IDF**: Multinomial LR with identical feature extraction

**Transformer models:**
- **IndoBERT** [21]: Indonesian-specific BERT model (indobenchmark/indobert-base-p1, 110M parameters)
- **IndoBERT + Label Smoothing**: IndoBERT with label smoothing regularization (epsilon=0.1) [23][24]
- **XLM-RoBERTa Large** [20]: Multilingual model trained on 100 languages (xlm-roberta-large, 559M parameters)

### 3.4 Experimental Setup

All transformer models were fine-tuned for 5 epochs with learning rate 2e-5, weight decay 0.01, warmup ratio 0.1, and batch sizes of 16 (IndoBERT) or 8 (XLM-R Large). Maximum sequence length was 128 tokens. Training used mixed precision (FP16) on an NVIDIA RTX 4080 GPU. The best checkpoint was selected based on validation F1-Macro. F1-Macro was chosen as the primary metric to ensure balanced evaluation across classes regardless of class size.

To ensure robustness, 5-fold stratified cross-validation was conducted for baseline models, and results were verified with multiple random seeds for transformer models.

---

## 4. RESULTS AND DISCUSSION

### 4.1 Overall Performance Comparison

Table 4 presents the performance of all models on both the full test set and the manual-only test subset.

**TABLE 4: Model Performance on Full and Manual-Only Test Sets (F1-Macro %)**

| Model | Full Test (978) | Manual-Only (451) | Gap |
|---|---|---|---|
| LR + TF-IDF | 77.04 | 47.18 | 29.86 |
| SVM + TF-IDF | 77.77 | 48.55 | 29.22 |
| IndoBERT | 76.12 | 45.27 | 30.85 |
| IndoBERT + LS | 77.36 | 49.30 | 28.06 |
| **XLM-R Large** | **80.26** | **53.89** | **26.37** |

All models showed a consistent gap exceeding 26 percentage points between full and manual-only test performance. XLM-RoBERTa Large achieved the best performance on both evaluations (80.26% full, 53.89% manual), followed by IndoBERT with label smoothing (77.36% full, 49.30% manual). Notably, the traditional SVM baseline (48.55% manual) performed comparably to IndoBERT (45.27%), suggesting that transformer scale alone does not guarantee superior real-world hate speech detection for this task.

Five-fold cross-validation on the baseline models confirmed consistency: SVM achieved 77.47% +/- 1.08% on the full dataset and 46.95% +/- 0.70% on manual-only data, with the ~30-point gap holding across all folds.

### 4.2 Comparison with Prior Work

Direct comparison with prior work is challenging due to differences in language, number of classes, and dataset composition. Table 1 contextualizes our results. Our full-test F1 of 80.26% appears competitive with or superior to prior Indonesian hate speech detection work. However, this comparison is misleading: our manual-only F1 of 53.89% reveals that the apparent competitiveness is an artifact of synthetic data in the test set.

This finding has broad implications. Prior studies using LLM-augmented data [11][12][13] that report only mixed-set metrics may similarly overestimate real-world performance. We recommend that all studies employing data augmentation adopt source-aware evaluation and report manual-only metrics alongside overall figures.

### 4.3 Augmentation Bias Analysis

The central finding of this study is a substantial performance gap between evaluation on synthetic and manual test data. XLM-RoBERTa Large achieved 99.41% F1 on the synthetic-only test subset (527 samples) versus 53.89% on the manual-only subset (451 samples) --- a gap of 45.52 percentage points.

**Augmentation ratio study.** To systematically investigate this phenomenon, we trained SVM and IndoBERT models with varying proportions of synthetic data (0%, 25%, 50%, 75%, 100%) while always evaluating on the same manual-only test subset (451 samples).

**TABLE 5: Augmentation Ratio Study --- F1-Macro (%) on Manual-Only Test**

| Synthetic Ratio | SVM (Manual) | SVM (Full) | IndoBERT (Manual) | IndoBERT (Full) |
|---|---|---|---|---|
| 0% | 46.01 | 42.46 | 48.75 | 54.31 |
| 25% | 48.20 | 76.80 | 47.54 | 76.24 |
| 50% | 46.90 | 76.55 | 47.61 | 77.48 |
| 75% | 47.02 | 77.28 | 44.79 | 76.15 |
| 100% | 48.55 | 77.77 | 47.43 | 77.50 |

Figure X illustrates this pattern. Manual-only F1 remained essentially flat (~47%) regardless of augmentation ratio for both SVM and IndoBERT. In contrast, full-test F1 jumped approximately 35 percentage points with just 25% synthetic data inclusion. This demonstrates that the "performance improvement" from augmentation is predominantly an evaluation artifact: synthetic data in the test set inflates metrics without meaningfully improving detection of naturally-occurring hate speech.

An additional observation was that IndoBERT trained exclusively on manual data (0% synthetic, F1=48.75%) did not collapse, unlike XLM-RoBERTa Large which degraded to 14.64% under the same condition. This suggests that smaller models are more robust to limited training data, while larger models require augmentation simply to avoid training collapse --- not because augmentation genuinely improves their understanding of hate speech.

### 4.4 Data Quality Analysis

To understand WHY models achieve near-perfect performance on synthetic data, we conducted a distributional analysis comparing manual and synthetic texts.

**Source distinguishability.** A logistic regression classifier trained on TF-IDF features to distinguish synthetic from manual texts achieved 97.26% +/- 0.22% F1 in 5-fold cross-validation. This indicates that the two data sources have fundamentally different distributions, enabling models to learn source-specific shortcuts rather than genuine hate speech features.

**TABLE 6: Distributional Differences Between Manual and Synthetic Data**

| Feature | Manual | Synthetic |
|---|---|---|
| Mean word count | 16.5 | 10.8 |
| Vocabulary size | 10,691 | 6,922 |
| Type-token ratio | 0.143 | 0.123 |
| Jaccard vocab overlap | -- | 14.6% |
| Starts with capital | 74.6% | 100.0% |
| Ends with period | 29.0% | 95.8% |
| All lowercase texts | 16.9% | 0.0% |

The synthetic data exhibited a highly uniform surface structure: all texts started with a capital letter, 95.8% ended with periods, and none appeared in all lowercase --- patterns absent in natural social media text. Furthermore, the vocabulary overlap between sources was only 14.6% (Jaccard similarity), confirming that models encounter largely different word distributions in manual versus synthetic data.

The most distinctive synthetic indicators were formal political terms (rakyat/people, pemerintah/government, korupsi/corruption), while manual indicators were informal personal terms (wadon/woman, aku/I, ireng/black). This suggests DeepSeek-Coder-V2 generated texts in a formal register inconsistent with the informal, personal nature of real social media hate speech.

### 4.5 Error Analysis

Error analysis of XLM-RoBERTa Large predictions on the manual-only test set (191 errors out of 451 samples) revealed that 73.8% of misclassifications involved adjacent severity classes (e.g., Light misclassified as Not Hate). The most common error pairs were Not Hate to Light (19.9% of errors) and Light to Not Hate (17.8%), confirming that the boundary between neutral speech and light hate speech is the most subjective and challenging.

Per-class analysis showed that classification accuracy varied substantially: Not Hate achieved 69.9% accuracy (most distinct class with clearest features), while Light (50.0%), Moderate (45.9%), and Severe (50.7%) were considerably harder to classify on natural data. The low accuracy for Moderate (45.9%) aligns with its having the highest synthetic proportion (75% in Table 3), suggesting that models' understanding of moderate-severity hate speech is dominated by synthetic patterns rather than genuine linguistic markers.

Confidence analysis revealed that the model was overconfident even on incorrect predictions: mean confidence was 0.925 for correct predictions but still 0.855 for errors, with a median error confidence of 0.934. This indicates that prediction confidence is not a reliable indicator of classification correctness for this task.

Cross-model agreement analysis between SVM and XLM-RoBERTa Large showed complementary error patterns: SVM correctly classified 52 samples that XLM-R missed, while XLM-R correctly classified 74 samples that SVM missed. The theoretical ensemble upper bound was 69.18%, substantially higher than either model alone, suggesting that future work on model ensembles could yield meaningful improvements.

### 4.6 Two-Stage Training as Mitigation

[PLACEHOLDER: Results from two_stage_training.py]

To investigate whether curriculum learning could mitigate the augmentation bias, we employed a two-stage training approach: Stage 1 fine-tuned models on synthetic data only (3 epochs, lr=2e-5) to learn general hate speech patterns, followed by Stage 2 fine-tuning on manual data only (5 epochs, lr=5e-6) to adapt to real-world text distributions.

Results: [TO BE FILLED WITH ACTUAL NUMBERS]

### 4.7 Limitations

Several limitations should be acknowledged. First, the manual annotation inter-annotator agreement (kappa=0.72, moderate-to-substantial) indicates inherent label subjectivity, particularly for boundary cases between adjacent severity levels. Second, the dataset was collected from Twitter and Instagram only, potentially missing hate speech patterns from other platforms. Third, some manual texts may reflect translated content from English-language hate speech datasets, which could affect the ecological validity of the manual test set. Finally, this study focused on text-only analysis, while hate speech on social media increasingly involves multimodal content.

---

## 5. CONCLUSION

This study investigated the reliability of LLM-augmented evaluation for severity-based hate speech detection in Javanese. The primary contribution was demonstrating that LLM data augmentation, while necessary for training large transformer models on low-resource languages, introduces substantial evaluation bias when synthetic data appears in the test set. Specifically, XLM-RoBERTa Large achieved 80.26% F1-Macro on a mixed test set but only 53.89% on manually-annotated data, revealing a 26.37-point inflation. The augmentation ratio study confirmed that this inflation is consistent across model architectures and augmentation proportions, with manual-only performance remaining flat at approximately 47% regardless of synthetic data volume. Source distinguishability analysis (97.26% F1) explained this phenomenon: LLM-generated texts differ fundamentally in vocabulary, length, and surface patterns from naturally-occurring hate speech.

These findings carry two practical implications for the low-resource NLP community. First, studies employing LLM data augmentation should adopt dual-track evaluation, reporting both mixed-set and manual-only metrics to provide realistic performance estimates. Second, the near-flat manual F1 across augmentation ratios suggests that augmentation primarily benefits models by preventing training collapse on insufficient data, rather than by genuinely improving hate speech understanding.

Future work should explore ensemble methods (theoretical upper bound: 69.18%), domain adaptation techniques, and the collection of larger manually-annotated Javanese hate speech datasets to improve real-world detection performance.

---

## REFERENCES (IEEE numbered, order of appearance)

[1] Ramos et al. (2024) - Comprehensive review of hate speech detection
[2] Hedderich et al. (2021) - Survey on NLP for low-resource scenarios
[3] Cahyawijaya et al. (2023) - NusaCrowd: 137 datasets for Indonesian
[4] Aji et al. (2022) - One Country, 700+ Languages
[5] Alfina et al. (2017) - Indonesian hate speech dataset
[6] Ibrohim & Budi (2019) - Multi-label Indonesian hate speech
[7] Putri et al. (2021) - Javanese/Sundanese hate speech
[8] Ding et al. (2024) - Survey on LLM-based data augmentation
[9] Rizwan et al. (2022) - Conditional LM augmentation for hate speech
[10] Ding et al. (2024) - same as [8], specific stats
[11] Hartvigsen et al. (2022) - ToxiGen
[12] Vidgen et al. (2021) - Dynamically generated hate speech datasets
[13] Juuti et al. (2020) - GPT-2 augmentation for toxic language
[14] Gorman & Bedrick (2019) - Standard splits unreliable
[15] Sogaard et al. (2021) - Random splits overly optimistic
[16] Davidson et al. (2017) - English hate speech baseline
[17] Susanto et al. (2024) - IndoToxic2024
[18] Pamungkas & Chiril (2025) - Ngalawan (Javanese code-mixed)
[19] Jahan et al. (2024) - Comprehensive augmentation study
[20] Conneau et al. (2020) - XLM-RoBERTa
[21] Wilie et al. (2020) - IndoBERT / IndoNLU
[22] Joachims (1998) - Text categorization with SVM
[23] Szegedy et al. (2015) - Label smoothing
[24] Muller et al. (2019) - Label smoothing analysis
[25] Devlin et al. (2019) - BERT
[26] Aroyo & Welty (2015) - CrowdTruth annotation
[27] Paun et al. (2018) - Bayesian annotation models
... [continue to ~35-40 refs]

---

## FIGURES NEEDED

1. Figure 1: Dataset distribution by class and source (bar chart)
2. Figure 2: Augmentation ratio curve (line chart: ratio vs F1 for manual and full test)
3. Figure 3: Confusion matrix on manual-only test (XLM-R Large)
4. Figure 4: Source distinguishability features (bar chart comparing manual vs synthetic)

---

## TABLES SUMMARY

1. Table 1: Comparison with prior work
2. Table 2: Severity level definitions with examples
3. Table 3: Dataset composition by class and source
4. Table 4: Model performance on full and manual-only test
5. Table 5: Augmentation ratio study
6. Table 6: Distributional differences manual vs synthetic
