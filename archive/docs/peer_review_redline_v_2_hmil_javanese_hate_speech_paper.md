# Peer Review + Redline (v2)

**Manuscript**: *Human-and-Model-in-the-Loop Ensemble Learning for Javanese Hate Speech Detection: A Sociolinguistically-Informed Approach*

---

## Executive Verdict
Strong technical contribution (ensemble + calibration + fairness) with a compelling sociolinguistic angle. To be competitive for a top-tier NLP venue, focus on (A) provenance clarity for the dataset (tie it cleanly to HMIL, rounds, and ethics), (B) numerical coherence (single “94.09 macro-F1” story across paper + doc), (C) reproducibility (replace placeholders with working artifacts), and (D) sharper ablations isolating what drives gains (features vs ensemble vs HMIL).

---

## Major Strengths
- **HMIL framing** and sociolinguistic features (speech levels, code-mixing, cultural markers) form a distinctive contribution.
- **Robust evaluation portfolio**: cross-domain, adversarial, calibration, and fairness are all present (rare to see all four).
- **Clear engineering**: ensemble pipeline and meta-learner are well motivated; performance is high and plausible.

---

## High-Priority Revision Objectives (Actionable)

### 1) Dataset Provenance & HMIL Alignment (must-fix)
**Issue:** The dataset narrative alternates between (i) dynamic HMIL-style creation (annotator-led, perturbations) and (ii) long-horizon collection of real-world content. Readers need a single, precise story.

**Fix—drop-in text for Methods §2.x “Data Collection & HMIL Rounds”**
> **Data Collection Protocol.** We followed a human-and-model-in-the-loop (HMIL) process across four iterative rounds. In each round r ∈ {1…4}, annotators (trained linguists with Javanese expertise) produced candidate texts designed to elicit model errors, guided by curated prompts reflecting authentic Javanese online discourse (code-mixed Javanese–Indonesian–English, common domains, and speech levels). Annotators also created *paired perturbations* (minimal edits preserving semantics) to stress-test decision boundaries. After each round we retrained the model on the newly collected set and used the improved model in the next round. To prevent leakage of real users’ content and protect privacy, entries were either fully synthetic or substantially transformed from observed patterns; no raw scraped text was stored. Full protocol details, annotator training, well-being safeguards, and examples appear in Appendix E and the data card.

**Also add immediately after:** an at-a-glance table of **Round sizes**, **% perturbations**, **hate/not-hate mix**, and **example themes**.

**If you truly used non-synthetic real posts**: invert the paragraph to clearly name the platforms, licensing/compliance, and de-identification steps; remove “synthetic/perturbation” claims and keep the HMIL loop only for *selection* rather than *generation*.

---

### 2) Numerical Coherence & Claim Tightening (must-fix)
**Issue:** Two performance scales appear (e.g., 0.94 vs 94.09%) and different sections emphasize slightly different top-line numbers.

**Fix—house style**
- Use **Macro-F1** as the primary headline metric everywhere.
- Report to **two decimals** in the main text (e.g., 94.09 → **94.09** if that’s the exact cross-validated figure; otherwise round consistently to **94.10** or **94.09** throughout).
- Always accompany the headline with **Δ vs best single model** and **CI/σ** (e.g., `94.09 ± 0.08 macro-F1; +7.21 over best single`).

**Drop-in text for Abstract**
> Our stacked ensemble attains **94.09 macro-F1** (±0.08 over 5 seeds), **+7.21** over the best single transformer, with robust calibration (ECE **2.5%**) and stable fairness across demographic slices.

**Drop-in caption style for the main table**: “Mean ± SD over 5 seeds. Macro-F1 primary; accuracy and F1-weighted secondary.”

---

### 3) Reproducibility: Replace Placeholders & Pin Environments (must-fix)
**Issue:** Appendix E still has placeholder links and partially specified environments.

**Fix—checklist**
- Replace `[GitHub URL to be provided]` with a **public repo** URL (or an **archived Zenodo DOI** if the repo must remain private). Provide a **single `requirements.txt`/`environment.yml`**, a **`scripts/`** directory, and a **`Makefile`** (or `justfile`) with targets: `download`, `preprocess`, `train`, `evaluate`, `paper-tables`.
- Publish **frozen artifacts**: base-model checkpoints (or HF model cards), ensemble weights, and calibration binning configs.
- Add **a 1-command reproducibility path**: `make reproduce` → produces the main results, tables, and reliability diagrams.

**Drop-in text for Appendix E**
> Code and artifacts are available at **[link]** (commit **<shortsha>**, release **v1.0.0**). We provide a one-command script (`make reproduce`) that regenerates all main tables/figures. Environments are pinned via `environment.yml` (CUDA 11.6) and we export a `cudnn_deterministic=True` training profile.

---

### 4) Fairness & Significance: Add Statistics (should-fix)
**Issue:** Group metrics are shown without uncertainty or tests.

**Fix—drop-in for §5.4/§5.6**
- Report **95% bootstrap CIs** for per-group F1.
- For pairwise gaps (e.g., Urban vs Rural), report **absolute diff** and **p-value** from **permutation test** (10k resamples).
- Clarify definitions for **Equalized Odds** and **Demographic Parity** (macro-averaged over classes; thresholding protocol). Provide the **decision threshold** used.

**Drop-in table column additions**: `F1 (95% CI)`, `Δ vs Global`, `p (perm)`.

---

### 5) Calibration Methods: Be Specific (should-fix)
**Issue:** Calibration is reported (ECE/MCE/Brier) but the *how* is under-specified.

**Fix—drop-in for §3.x “Calibration & Uncertainty”**
- Specify **ECE** with **15 uniform probability bins**, **class-wise then macro-averaged**; provide **logit-binning** as a robustness check.
- State **prediction source** (pre- or post-ensemble) and **temperature scaling fit** (on validation). Include **confidence histogram** in the reliability figure.
- Add **selective prediction** analysis: coverage vs expected error; recommend **handoff threshold** for HMIL moderation.

---

### 6) Ablations to Attribute the Gain (should-fix)
**Issue:** Readers will ask “what matters most?”

**Fix—add three ablations**
1. **No sociolinguistic features** → ensemble only.
2. **No stacking (weighted voting only)** → shows meta-learner value.
3. **No HMIL (static data)** → trains on a static split of round-1-like data.

Report each on the same test split; add a **waterfall plot** of macro-F1 gains.

---

### 7) Meta-Learner Specifics (should-fix)
Provide: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, early stopping rounds, seed, class weights, and feature list (probabilities, entropy, agreement). Add a **Shapley summary** showing which features dominate meta decisions.

---

### 8) Error Taxonomy & Qualitative Slices (nice-to-have)
Add a short **typology** with 6–8 representative errors (implicit hate via proverbs, honorific misuse, sarcasm, orthographic variants, dialectal shifts, heavy code-switch). For each: input, gold, prediction, confidence, sociolinguistic cues, and a 1–2 line analysis.

---

### 9) Compute & Footprint (nice-to-have)
Include total GPU-hours, peak memory, and energy estimate (via `codecarbon`), plus carbon intensity region. Helps with reproducibility and ethics.

---

## Targeted Redlines (Copy-ready)

### Abstract (replace existing abstract)
> We present a **human-and-model-in-the-loop (HMIL)** approach for Javanese hate speech detection that integrates sociolinguistic features with a stacked transformer ensemble. Across four HMIL rounds with expert annotators, we curate a culturally informed dataset and train a meta-learner over base models’ probability-, entropy-, and agreement-features. On a held-out test set, our method achieves **94.09 macro-F1** (±0.08 over 5 seeds), **+7.21** over the best single model, with **ECE 2.5%** and stable fairness across demographic slices. We release code, artifacts, and ethical guidance for responsible use.

### Methods §3.x “Calibration & Uncertainty” (insert)
> We calibrate post-ensemble probabilities via **temperature scaling** fit on validation. **ECE** is computed with **15 uniform probability bins**, macro-averaged over classes; we confirm robustness with **logit-binning**. **MCE** and **Brier** follow standard definitions. **Selective prediction** curves show precision vs coverage for thresholds τ ∈ {0.7, 0.8, 0.9}, motivating human review for τ < 0.8.

### Experiments §4.1 “Data Card” (tighten)
> Our HMIL dataset records per-round counts, perturbation ratios, and domain labels. Each entry includes code-switch flags, speech-level markers (ngoko/madya/krama), and dialect labels. To protect privacy, content is either synthetic or substantially transformed; no user-identifiable text is retained. We publish the data card, annotation guidelines, and well-being protocol.

### Results §5.x “Fairness” (augment)
> We report **macro-F1 (95% CI)** by demographic groups and test the gap to the global score via **10k-permutation tests**. **Equalized Odds** is computed as the mean absolute difference of TPR/FPR to the global model across classes; **Demographic Parity** as the absolute difference in positive prediction rate. Thresholds are fixed at τ = 0.5 unless otherwise stated.

### Appendix E “Reproducibility” (replace placeholders)
> Code + models: **https://github.com/<org>/<repo>** (release **v1.0.0**, DOI **10.5281/zenodo.<id>**). `make reproduce` regenerates Tables 1–8 and Figures 1–3. Environments pinned via `environment.yml` (PyTorch 1.12, Transformers 4.21, CUDA 11.6). Checkpoints and calibration configs are hosted on the release page.

---

## Camera-Ready Checklist
- [ ] Single headline metric policy (Macro-F1, two decimals) applied everywhere.
- [ ] Abstract states macro-F1, Δ vs single model, ECE.
- [ ] Methods specify calibration binning; Uncertainty section explains entropy, disagreement, MC-dropout.
- [ ] Reproducibility links live; artifact versions frozen.
- [ ] Ablations added (features / stacking / HMIL).
- [ ] Fairness includes CIs and permutation tests.
- [ ] Ethics clarifies synthetic vs transformed content and consent.
- [ ] Compute footprint reported.

---

## Minor Edits (Style & Clarity)
- Convert all metric tables to **mean ± SD** (5 seeds). Use consistent decimal places.
- Replace “state-of-the-art” with numbers (avoid promissory language). Spell out abbreviations at first use.
- Normalize terminology: “macro-F1” (not Macro F1 / Macro-F1), “ECE” (Expected Calibration Error), “CI”.
- Ensure figures have legible fonts and describe binning/thresholds in captions.
- Add an **appendix table** mapping sociolinguistic features to examples.

---

## Optional Additions
- **Deployment note**: a short guide for integrating selective prediction + human review thresholds in production.
- **Data statement**: include a paragraph on annotator well-being and compensation.

---

**End of Review (v2)**

