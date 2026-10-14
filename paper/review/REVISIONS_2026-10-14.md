# Revisions after supervisor review (2026-10-14)

Changes made in response to the comments on the review draft of 2026-10-09. No experiment was re-run and
no result changed; every number in the paper is still generated from the saved results.

1. **"Related Work" renamed to "Literature Review"** (Section II). It now has four subsections: noise-robust
   and physics-informed CNNs, evaluation leakage, calibration under dataset shift, and a new *Research Gap*
   subsection that states what the paper adds.

2. **Abbreviations.** The abstract writes out every abbreviation it uses, with the short form in brackets
   (CWRU, 1D-CNN, PAC-1DCNN, WDCNN, SNR). The body defines each abbreviation again at its first use
   (e.g. CNN, CWRU, PAC-1DCNN, WDCNN, SNR, AdaBN, ReLU, AUC, CNN+HP, CNN+LHP, FIR, ECE, PU, GPU, CPU,
   ONNX, FP32, INT8) and uses only the short form after that. The abstract was also brought within the
   150-250 words that IEEE asks for (now 249 words).

3. **References from the last five years.** 12 older references were removed, and 9 references from
   2021-2024 were added and checked against Crossref (journal, volume, pages, DOI) and their abstracts:
   two reviews of deep learning for bearing and machinery diagnosis, wavelet-based and physics-informed
   noise-robust networks, an attention survey, a review of physics-informed machine learning for
   reliability, a noise-robust 1D-CNN evaluated on CWRU, and a study of post-hoc calibration under domain
   shift. 16 of the 23 references are now from 2021-2026. Seven are older or undated and were kept because the
   paper uses them directly and a newer source cannot replace them:
   - the two dataset pages (CWRU, Paderborn; undated web pages, accessed 2026);
   - the Paderborn dataset paper (Lessmeier et al. 2016), which defines protocols P3 and P4 and which the
     data licence asks users to cite;
   - Smith and Randall 2015, the reference analysis of the CWRU data and the only source for several facts
     used (4.2 kHz interference line, records that cannot be diagnosed);
   - Zhang et al. 2017 (WDCNN), whose architecture is reimplemented as a baseline;
   - Guo et al. 2017, which introduced the temperature scaling and the calibration error used here;
   - Nadeau and Bengio 2003, whose corrected t-test is applied to the Paderborn folds.

   Standard textbook methods (Wilcoxon signed-rank test, Holm-Bonferroni procedure, random forests, AdamW,
   envelope analysis) are now named without a citation, which removes the 1945 and 1979 references.
   *Effect on the project:* the newer references do not change what was measured, so no experiment had to be
   changed. The new noise-robust and physics-informed networks (WPConvNet, PIResNet) were not re-implemented
   as baselines; if the supervisor wants one of them compared, that is a new experiment.

4. **How many models and what each is for.** Section IV-C now opens by stating that 10 models are compared
   in three groups (the studied model, three ablations and six baselines). Table II lists each model's
   abbreviation, group, purpose (which question it answers and which model it is compared with), the
   protocols it is evaluated in, and its trainable parameters. The previous table had omitted the
   attention ablation (PAC-att), which is used only in P1.

5. **IEEE format.** The paper uses the current IEEE LaTeX template: IEEEtran class V1.8b and IEEEtran.bst
   1.14, the latest versions, in conference mode. Changes made to follow IEEE style more closely:
   - abstract within 150-250 words;
   - index terms in alphabetical order;
   - references in the format of the IEEE Reference Guide (abbreviated journal names, DOI at the end);
   - headings in IEEE title case, with third-level headings (e.g. "1) Pipeline Check:") instead of
     italic paragraph labels.

   The venue, page limit, double-blind review and authorship are still to be decided.

6. **Headings.** The headings now follow a conventional IEEE paper structure:
   - I. Introduction
   - II. Literature Review
   - III. Datasets and Evaluation Protocols
   - IV. Model Architectures
   - V. Experimental Setup
   - VI. Results and Discussion
   - VII. Threats to Validity
   - VIII. Conclusion and Future Work

   "The contributions are empirical:" now reads "The main contributions of this paper are as follows:".
