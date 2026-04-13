# Related Work

## Hate Speech Detection in Indonesia

Penelitian deteksi ujaran kebencian di Indonesia telah berkembang signifikan dalam beberapa tahun terakhir. Ibrohim dan Budi (2019) memperkenalkan dataset multi-label untuk bahasa Indonesia dengan 13,069 sampel yang dikumpulkan dari Twitter. Penelitian mereka menggunakan metode *Bidirectional LSTM* dengan *FastText embeddings* dan mencapai F1-score 71.31% pada tugas klasifikasi multi-label. Namun, dataset mereka terbatas pada bahasa Indonesia standar dan tidak mencakup bahasa daerah.

Alfina et al. (2017) mengembangkan dataset *Indonesian Hate Speech* dan melakukan studi awal yang membandingkan berbagai pendekatan termasuk *Multinomial Naive Bayes* dan *Support Vector Machine*. Hasil mereka menunjukkan bahwa model machine learning tradisional mencapai performa yang layak, namun terbatas oleh representasi fitur yang sederhana.

Ramos et al. (2024) melakukan review komprehensif terhadap 100+ studi deteksi ujaran kebencian pada era transformer, menemukan bahwa model berbasis transformer secara konsisten mengungguli metode tradisional dan deep learning konvensional. Review ini juga mengidentifikasi kesenjangan signifikan dalam cakupan bahasa *low-resource*.

## Hate Speech Detection in Low-Resource Languages

Bahasa Jawa merupakan bahasa dengan penutur native terbesar di Indonesia (~80 juta orang), namun termasuk kategori *low-resource* dalam konteks *Natural Language Processing* (NLP). Putri et al. (2021) melakukan studi tentang deteksi ujaran kebencian dan bahasa kasar dalam bahasa Jawa dan Sunda. Mereka mengumpulkan dataset dari berbagai platform media sosial dan melakukan klasifikasi. Namun, penelitian mereka memiliki keterbatasan: (1) dataset yang relatif kecil, (2) hanya menggunakan klasifikasi binary (hate speech vs non-hate speech), dan (3) tidak membedakan tingkat keparahan ujaran kebencian.

Hedderich et al. (2021) memberikan survei terstruktur tentang pendekatan NLP untuk skenario *low-resource*, mencakup augmentasi data, *distant supervision*, *transfer learning*, dan *active learning*. Survei ini relevan untuk konteks bahasa Jawa yang memiliki keterbatasan data berlabel.

## Transformer Models for Indonesian NLP

Wilie et al. (2020) memperkenalkan *IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding*, yang mencakup model *IndoBERT* yang dilatih pada corpus bahasa Indonesia yang besar. *IndoBERT* telah menjadi *state-of-the-art* untuk berbagai tugas NLU bahasa Indonesia termasuk analisis sentimen, *Named Entity Recognition* (NER), dan klasifikasi teks.

Cahyawijaya et al. (2023) memperluas kontribusi ini melalui inisiatif *NusaCrowd*, yang menyatukan 137 dataset dan 118 data loader terstandarisasi untuk bahasa Indonesia dan 18+ bahasa daerah, termasuk bahasa Jawa dan Sunda. Ini merupakan benchmark *zero-shot* NLU/NLG pertama untuk bahasa-bahasa daerah Indonesia.

## Label Smoothing for Regularization

*Label smoothing* adalah teknik regularisasi yang diperkenalkan oleh Szegedy et al. (2015) dan kemudian dianalisis secara mendalam oleh Müller et al. (2019). Teknik ini mengubah target *hard* [0, 0, 1, 0] menjadi target *soft* [0.025, 0.025, 0.925, 0.025] dengan menambahkan *noise* seragam kecil.

Müller et al. (2019) menunjukkan bahwa *label smoothing* efektif untuk: (1) mencegah model menjadi *overconfident*, (2) meningkatkan *calibration* prediksi, dan (3) meningkatkan generalisasi terutama pada dataset dengan *label noise*. Dalam konteks deteksi ujaran kebencian, di mana batas antar kategori sering kali kabur dan subjektif, *label smoothing* sangat relevan karena dapat mengakomodasi ketidakpastian inheren dalam proses *labeling*.

## LLM-Assisted Data Augmentation

Ding et al. (2024) melakukan survei komprehensif tentang augmentasi data menggunakan *Large Language Models*, mendemonstrasikan peningkatan signifikan (3-26% dalam akurasi/F1) pada skenario klasifikasi teks *low-resource*. Pendekatan ini relevan untuk penelitian kami yang menggunakan DeepSeek-Coder-V2 untuk mengaugmentasi dataset ujaran kebencian bahasa Jawa.

## Our Contribution

Berbeda dengan penelitian sebelumnya, kontribusi penelitian ini meliputi:

1. **Dataset yang Lebih Besar**: Kami mengumpulkan dan melabeli 9,775 sampel ujaran kebencian bahasa Jawa (setelah aggressive cleanup), jauh lebih besar dari dataset sebelumnya.

2. **Klasifikasi 4-Kelas**: Kami memperkenalkan tugas klasifikasi tingkat keparahan ujaran kebencian (Neutral, Light Hate, Moderate Hate, Severe Hate) yang lebih informatif daripada klasifikasi binary.

3. **Studi Komparatif Transformer**: Kami membandingkan tiga arsitektur transformer (IndoBERT, XLM-RoBERTa Large, dan IndoBERT + Label Smoothing) secara sistematis pada dataset yang sama.

4. **Evaluasi Label Smoothing**: Kami melakukan studi ablasi untuk mengevaluasi efek *label smoothing* pada deteksi ujaran kebencian bahasa Jawa.

5. **Signifikansi Statistik**: Kami melaporkan mean ± standard deviation dari evaluasi multi-seed untuk memastikan reproduktibilitas hasil.

6. **LLM-Assisted Data Augmentation**: Kami mengeksplorasi penggunaan LLM untuk augmentasi data pada *low-resource language*, dengan analisis jujur tentang proporsi data sintetis (52.3%).

---

## References

1. Ibrohim, M. O., & Budi, I. (2019). Multi-label hate speech and abusive language detection in Indonesian Twitter. *Proceedings of the Third Workshop on Abusive Language Online (ALW3)*, 46-57. ACL. DOI: 10.18653/v1/W19-3506

2. Alfina, I., Mulia, R., Fanany, M. I., & Ekanata, Y. (2017). Hate speech detection in the Indonesian language: A dataset and preliminary study. *Proceedings of the 2017 International Conference on Advanced Computer Science and Information Systems (ICACSIS)*, 233-238.

3. Putri, S. D. A., Ibrohim, M. O., & Budi, I. (2021). Abusive language and hate speech detection for Javanese and Sundanese languages in tweets: Dataset and preliminary study. *Proceedings of the 11th International Workshop on Computer Science and Engineering (WCSE 2021)*.

4. Wilie, B., Vincentio, K., Winata, G. I., Cahyawijaya, S., et al. (2020). IndoNLU: Benchmark and resources for evaluating Indonesian natural language understanding. *Proceedings of AACL-IJCNLP 2020*, 843-857.

5. Cahyawijaya, S., Lovenia, H., Aji, A. F., Winata, G., Wilie, B., et al. (2023). NusaCrowd: Open source initiative for Indonesian NLP resources. *Findings of ACL 2023*, 13745-13818. DOI: 10.18653/v1/2023.findings-acl.868

6. Müller, R., Kornblith, S., & Hoiem, D. (2019). When does label smoothing help? *Advances in Neural Information Processing Systems*, 32.

7. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2015). Rethinking the inception architecture for computer vision. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2818-2826.

8. Ding, B., Qin, C., Zhao, R., et al. (2024). Data augmentation using LLMs: Data perspectives, learning paradigms and challenges. *Findings of ACL 2024*, 1679-1705. DOI: 10.18653/v1/2024.findings-acl.97

9. Ramos, G., Batista, F., Ribeiro, R., et al. (2024). A comprehensive review on automatic hate speech detection in the age of the transformer. *Social Network Analysis and Mining*, 14, Article 207. DOI: 10.1007/s13278-024-01361-3

10. Hedderich, M. A., Lange, L., Adel, H., Strötgen, J., & Klakow, D. (2021). A survey on recent approaches for natural language processing in low-resource scenarios. *Proceedings of NAACL-HLT 2021*, 2545-2568. DOI: 10.18653/v1/2021.naacl-main.201

11. Dietterich, T. G. (2000). Ensemble methods in machine learning. *International Workshop on Multiple Classifier Systems*, 1-15.

---

**Word Count**: ~1,100 words
**Total References**: 11 (all verified)
**Removed**: Khoong (2021) — fabricated; Bryant et al. (2020) — fabricated; Wibowo et al. (2022) — fabricated; Aji & Adriani (2020) — fabricated
**Added**: Cahyawijaya et al. (2023), Ding et al. (2024), Ramos et al. (2024), Hedderich et al. (2021)
**Corrected**: Ibrohim (venue), Alfina (authors/year/venue), Putri (authors/venue), Wilie (title/venue), Szegedy (author initial)
