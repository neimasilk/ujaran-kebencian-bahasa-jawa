# Literature Review: Implicit Hate Speech Detection
## Javanese Implicit Hate Speech Research

**Date:** 2025-01-08
**Focus:** Implicit hate speech, low-resource languages, Javanese sociolinguistics

---

## 1. Foundational Papers on Implicit Hate Speech

### 1.1 Latent Hatred: A Benchmark for Understanding Implicit Hate Speech (ElSherief et al., 2021)

**Venue:** EMNLP 2021
**Citations:** 347+ (highly influential)
**Link:** [ACL Anthology](https://aclanthology.org/2021.emnlp-main.29/) | [arXiv](https://arxiv.org/abs/2109.05322)

**Key Contributions:**
- First benchmark dataset for implicit hate speech detection
- ~20,000 English tweets annotated
- Defines implicit hate as: "indirectly targeting protected groups through subtle language"

**Implicit Hate Categories (from this paper):**
| Category | Description |
|----------|-------------|
| Not Hate | Non-hateful content |
| Implicit Hate | Hate expressed through indirect/subtle means |
| Explicit Hate | Overt hate speech |

**Methodology:**
- Crowdsourced annotation with quality control
- Inter-annotator agreement: κ = 0.68-0.72
- Human evaluation required for context understanding

**Limitations:**
- English-only
- Does not address low-resource languages
- Limited cultural context consideration

---

### 1.2 Cracking the Code: Enhancing Detection through Coding Classification (Wei et al., 2025)

**Venue:** TrustNLP @ ACL 2025
**Link:** [ACL Anthology](https://aclanthology.org/2025.trustnlp-main.9.pdf) | [arXiv](https://arxiv.org/html/2506.04693v1)

**Key Contributions:**
- Introduces **six codetypes** for implicit hate speech encoding strategies
- LLM-based classification approach
- Multi-task learning framework

**The Six Codetypes:**
1. **Stereotyping** - Using negative generalizations about groups
2. **Othering** - Creating separation between "us" vs "them"
3. **Threatening** - Veiled threats of violence/harm
4. **Dehumanizing** - Denying humanity to target groups
5. **Mocking** - Ridicule or derision
6. **Blaming** - Attributing problems to target groups

**Methodological Insights:**
- Codetype annotation improves model performance
- LLMs can be prompted to identify codetypes
- Multi-label classification approach works best

---

### 1.3 Amplifying Attention for Versatile Implicit Hate Detection (EMNLP 2025)

**Venue:** EMNLP 2025
**Link:** [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1469.pdf)

**Key Contributions:**
- Focuses on hate targeting protected groups through subtle language
- Attention-based mechanisms for implicit detection
- Context-aware modeling

---

### 1.4 Unmasking Implicit and Subtle Hate Speech (2025)

**Venue:** HAL Science / NAACL 2024 referenced
**Link:** [HAL Archive](https://hal.science/tel-05247463v2/file/2025COAZ4008.pdf)

**Key Contributions:**
- Survey of implicit hate speech detection methods
- Analysis of subtle vs. overt hate speech
- Framework for categorization

---

## 2. Low-Resource and Regional Languages

### 2.1 SGHateCheck: Functional Tests for Southeast Asian Languages (Ng et al., 2024)

**Venue:** WOAH @ ACL 2024
**Link:** [ACL Anthology](https://aclanthology.org/2024.woah-1.24/) | [arXiv](https://arxiv.org/abs/2405.01842) | [GitHub](https://github.com/Social-AI-Studio/SGHateCheck)

**Key Contributions:**
- Functional testing framework for Singapore/SEA languages
- Addresses low-resource language challenge
- Uses LLMs for translation and paraphrasing
- Culturally-aware test generation

**Languages Covered:**
- English (Singapore context)
- Mandarin Chinese
- Malay
- Tamil

**Methodology:**
- Extends HateCheck framework
- Functional tests rather than just accuracy metrics
- LLM-assisted translation and paraphrase generation

**Relevance to Our Work:**
- Demonstrates approach for regional languages
- Shows LLMs can help low-resource scenarios
- Functional testing is more informative than raw accuracy

---

### 2.2 Survey on Low-Resource Hate Speech Detection (2024)

**Publication:** ResearchGate
**Link:** [ResearchGate](https://www.researchgate.net/publication/386335700_A_Survey_on_Automatic_Online_Hate_Speech_Detection_in_Low-Resource_Languages)

**Key Findings:**
- Majority of hate speech research focuses on English
- Low-resource languages lack: datasets, models, benchmarks
- Cross-lingual transfer is promising but imperfect
- Cultural adaptation is crucial

---

### 2.3 South Asian Languages Survey (ACM 2024)

**Publisher:** ACM Digital Library
**Link:** [ACM DL](https://dl.acm.org/doi/10.1145/3711710)

**Key Insights:**
- Similar challenges to Southeast Asian context
- Code-mixing is a major challenge
- Cultural context is critical for detection

---

## 3. Datasets and Benchmarks

### 3.1 ImplicitHate Corpus (IHC)

**Repository:** [SALT-NLP/implicit-hate](https://github.com/SALT-NLP/implicit-hate)
**Hugging Face:** [SALT-NLP/ImplicitHate](https://huggingface.co/datasets/SALT-NLP/ImplicitHate)
**OpenDataLab:** [Implicit_Hate](https://opendatalab.com/OpenDataLab/Implicit_Hate)

**Statistics:**
- Total tweets: 22,056
- Implicit hate tweets: 6,346
- Source: Extremist groups in the US
- Fine-grained labels available

**Annotation Scheme:**
- Binary: Hate vs Non-Hate
- Implicit vs Explicit
- Free-text explanation of meaning
- Target identity

---

### 3.2 Hate Speech Dataset Catalogue

**Website:** [hatespeechdata.com](https://hatespeechdata.com/)

**Useful for:**
- Finding related datasets
- Comparing annotation schemes
- Benchmark selection

---

## 4. Javanese Sociolinguistics

### 4.1 Speech Levels (Undha-Usuk)

**Sources:**
- [Language Politeness in Javanese](https://www.researchgate.net/publication/355247752_Language_Politeness_in_the_Javanese_Verb_Speech_Level)
- [Javanese Language Varieties](https://www.atlantis-press.com/article/25903341.pdf)
- [Linguistic Socialization](https://ojs.linguistik-indonesia.org/index.php/linguistik_indonesia/article/download/177/117/687)

**Three Primary Levels:**

| Level | Use Context | Politeness |
|-------|-------------|------------|
| **Ngoko** | Close friends, equals, younger | Low/Informal |
| **Madya** | Neutral social situations | Medium |
| **Krama** | Formal, superiors, elders | High/Formal |

**Sociolinguistic Significance:**
- Speech level encodes social hierarchy
- Using wrong level = social violation
- Code-switching between levels is common
- Violation of expected levels can be used as hate/insult

**Relevance to Hate Speech:**
- Using **Ngoko** to superior = intentional disrespect
- Using **Krama** mockingly = sarcasm/insult
- Speech level violations = implicit hostility

---

### 4.2 Javanese Dialects

**Major Dialect Groups:**
1. **Western Javanese (Banyumasan)**
   - Banyumas, Tegal, Pekalongan
   - Distinct vocabulary and pronunciation

2. **Central Javanese (Mataraman)**
   - Yogyakarta, Surakarta (standard Javanese)
   - Considered "prestige" dialect

3. **Eastern Javanese (Arekan)**
   - Surabaya, Malang
   - More direct, less hierarchical

**Dialect-based Discrimination:**
- Stereotypes about each dialect group
- "Arek-arek" seen as rough/loud
- "Wong Banyumas" seen as rural
- Dialect mocking is common implicit hate

---

### 4.3 Cultural Context Factors

**Relevant Elements for Hate Speech:**

1. **Hierarchical Society**
   - Age-based respect
   - Status-based language
   - Inversion of hierarchy = insult

2. **Religious Syncretism**
   - Islam + Hindu-Buddhist + Indigenous
   - Kejawen (traditional Javanese beliefs)
   - Targeting traditional practices = implicit hate

3. **Historical Trauma**
   - Colonial history references
   - 1965-66 tragedy implications
   - Regional conflicts

4. **Regional Identity**
   - "Wong Jawa" vs non-Javanese
   - Intra-Javanese regional tensions
   - Migration-based tensions

---

## 5. Research Gaps

### 5.1 Identified Gaps in Current Research

| Gap | Description | Opportunity |
|-----|-------------|-------------|
| **Low-resource implicit hate** | No work on implicit hate in low-resource languages | Our work: Javanese |
| **Cultural taxonomy** | No taxonomy for non-Western contexts | Our work: Javanese-specific codetypes |
| **Sociolinguistic features** | Speech levels not used for hate detection | Our work: Register manipulation codetype |
| **Southeast Asian focus** | Limited work beyond Singapore/major languages | Our work: Regional Indonesian language |

### 5.2 Our Novel Contributions

1. **First implicit hate dataset for Javanese**
2. **Sociolinguistically-informed taxonomy**
3. **Speech-level aware hate detection**
4. **Cultural context annotations**
5. **Dialect-based othering analysis**

---

## 6. Related Work Citation Template

For the paper, we will cite:

```bibtex
@inproceedings{elsherief2021latent,
  title={Latent Hatred: A Benchmark for Understanding Implicit Hate Speech},
  author={ElSherief, Mai and Ziems, Caleb and Muchlinski, David and Anupindi, Vaishnavi and Yang, Di and and Ng, Aylin},
  booktitle={Proceedings of EMNLP},
  year={2021},
  url={https://aclanthology.org/2021.emnlp-main.29/}
}

@inproceedings{wei2025cracking,
  title={Cracking the Code: Enhancing Implicit Hate Speech Detection through Coding Classification},
  author={Wei, Lu and [others]},
  booktitle={Proceedings of TrustNLP @ ACL},
  year={2025},
  url={https://aclanthology.org/2025.trustnlp-main.9/}
}

@inproceedings{ng2024sghatecheck,
  title={SGHateCheck: Functional Tests for Detecting Hate Speech in Singapore and Southeast Asian Languages},
  author={Ng, Robin Chen and [others]},
  booktitle={Proceedings of WOAH @ ACL},
  year={2024},
  url={https://aclanthology.org/2024.woah-1.24/}
}
```

---

## 7. Next Steps from Literature Review

1. ✅ **Understand established codetypes** - Wei et al. (2025)
2. ✅ **Learn from low-resource approaches** - SGHateCheck
3. ✅ **Study Javanese sociolinguistics** - Speech levels, dialects
4. 🔄 **Develop Javanese-specific codetypes** - Current phase
5. ⏳ **Design annotation guidelines** - Next phase
6. ⏳ **Collect and annotate pilot data** - Following phase

---

**Status:** Literature Review Complete
**Next:** Develop JIHST (Javanese Implicit Hate Speech Taxonomy)

---

## Sources

- [ElSherief et al. 2021 - Latent Hatred](https://aclanthology.org/2021.emnlp-main.29/)
- [Wei et al. 2025 - Cracking the Code](https://arxiv.org/html/2506.04693v1)
- [Amplifying Attention - EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1469.pdf)
- [SGHateCheck - WOAH 2024](https://aclanthology.org/2024.woah-1.24/)
- [SGHateCheck GitHub](https://github.com/Social-AI-Studio/SGHateCheck)
- [ImplicitHate Corpus](https://github.com/SALT-NLP/implicit-hate)
- [Hate Speech Dataset Catalogue](https://hatespeechdata.com/)
- [Javanese Speech Levels - ResearchGate](https://www.researchgate.net/publication/355247752_Language_Politeness_in_the_Javanese_Verb_Speech_Level)
- [Javanese Language Varieties - Atlantis Press](https://www.atlantis-press.com/article/25903341.pdf)
- [Linguistic Socialization in Javanese](https://ojs.linguistik-indonesia.org/index.php/linguistik_indonesia/article/download/177/117/687)
