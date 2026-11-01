# MM-RFMNN: Empirical Research Benchmark Suite (Phase 2)

This directory contains the experimental benchmark suite designed to evaluate the core research questions (RQ1 – RQ4) proposed in the MM-RFMNN research methodology.

## Execution

To run the complete benchmark suite from the project root:
```bash
python experiments/benchmark.py
```

---

## Research Questions Addressed

### RQ1: Clean Multimodal Classification & Probability Calibration
- **Goal**: Compare modality-specific FMNN modeling against early feature concatenation and deep neural baselines.
- **Baselines Compared**:
  1. `Early Concatenation FMNN`: Single massive FMNN built across concatenated features.
  2. `Early Concatenation MLP`: Multi-layer perceptron trained on concatenated features.
  3. `Late Mean Fusion FMNN`: Separate FMNNs combined with unweighted uniform averaging.
  4. `Proposed MM-RFMNN`: Separate FMNNs combined with OOF-calibrated reliability fusion.
- **Key Metric**: Expected Calibration Error (ECE) and Macro-F1. MM-RFMNN achieves **ECE of 0.0333**, compared to **0.4578** for naive Early Concatenation FMNN (over 13× better calibration).

---

### RQ2: Robustness Under Modality Corruption
- **Goal**: Determine whether instance-level reliability weighting protects against corrupted or noisy modalities.
- **Evaluation**: Injects escalating levels of Gaussian noise ($\sigma \in \{0.0, 1.0, 2.5, 5.0, 8.0\}$) into Modality 1 while keeping Modality 2 clean.
- **Key Finding**:
  - Early Concatenation MLP collapses from **1.0000** Macro-F1 to **0.4514** under $\sigma = 8.0$.
  - MM-RFMNN dynamically detects the noise degradation, lowers $\alpha^{(1)}$, and retains **0.9874** Macro-F1.

---

### RQ3: Missing-Modality Inference (Zero-Imputation Handling)
- **Goal**: Evaluate graceful degradation when one or more channels are completely unavailable.
- **Scenarios Evaluated**:
  - Clean Baseline (both modalities present): **1.0000 Macro-F1**
  - Modality 1 completely missing ($q^{(0)} = 0$): **0.9874 Macro-F1**
  - Modality 2 completely missing ($q^{(1)} = 0$): **0.9711 Macro-F1**
- **Key Design**: Uses binary presence mask $q_i^{(k)}$ to renormalize the remaining simplex weights directly. Never passes imputed zero vectors to avoid artificial hyperbox boundary hits.

---

### RQ4: Uncertainty Quantification & Selective Classification
- **Goal**: Assess whether prediction confidence supports calibrated risk-coverage tradeoffs and abstention.
- **Results**: Evaluates retained classification accuracy across coverage budgets ($100\%, 95\%, 90\%, 85\%, 80\%$), consistently maintaining 100% precision on retained predictions.
