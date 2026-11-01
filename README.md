# MM-RFMNN: Multimodal Reliability-Aware Fuzzy Min-Max Neural Network

[![Tests](https://img.shields.io/badge/tests-7%2F7%20passed-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-ruff-black.svg)]()
[![Status](https://img.shields.io/badge/Phase%201%20%26%20Phase%202-Complete-blue.svg)]()

This repository contains the complete mathematical engine (Phase 1) and empirical research benchmark suite (Phase 2) for the **Multimodal Reliability-Aware Fuzzy Min-Max Neural Network (MM-RFMNN)**, designed for reliable classification under modality noise, corruption, and missingness.

---

## Key Highlights & Innovations

1. **Modality-Specific Hyperbox Quantization**:
   - Maintains dedicated geometric hyperbox banks $\mathcal{H}^{(k)}$ for each modality, preserving native feature scales and preventing high-dimensional cross-modal distortion.
2. **Leakage-Free Reliability Calibration**:
   - Generates unbiased Out-Of-Fold (OOF) correctness targets via 5-fold cross-validation, preventing overconfident memorization.
3. **Zero-Imputation Missingness Masking**:
   - Natively gates missing modalities via presence indicators ($q_i^{(k)} \cdot r_i^{(k)}$), setting $\alpha_i^{(k)} = 0.0$ and re-normalizing the simplex without hallucinated zero vectors.
4. **Calibrated Multi-Evidence Fusion**:
   - Converts non-probabilistic FMNN evidence into calibrated distributions via validation-optimized temperature scaling ($T_k$).
5. **Certifiable Selective Classification**:
   - Multi-criteria epistemic and aleatoric uncertainty bounds (predictive entropy, margin deficit, Jensen-Shannon modality disagreement, total reliability) for selective abstention.

---

## Project Structure

```
MM-RFMNN/
├── MM_RFMNN_Algorithm_Blueprint.md   # Mathematical blueprint & methodology
├── MM_RFMNN_Research_Paper.pdf       # Compiled publication-ready PDF document
├── MM_RFMNN_Research_Paper.docx      # Compiled Word document specification
├── README.md                         # Main project overview & index
├── pytest.ini                        # Pytest configuration file
├── GEMINI.md                         # Anti-slop engineering & code standards
│
├── mm_rfmnn/                         # Phase 1: Core Mathematical Engine
│   ├── README.md                     # Detailed engine documentation & equations
│   ├── __init__.py                   # Public API imports
│   ├── base/                         # Vectorized hyperboxes & Simpson expansion/contraction
│   ├── encoders/                     # Train-fitted normalizers (leakage prevention)
│   ├── reliability/                  # OOF-supervised reliability estimator
│   ├── fusion/                       # Temperature calibration & simplex evidence fuser
│   └── model.py                      # Scikit-learn compatible MMRFMNNClassifier
│
├── experiments/                      # Phase 2: Empirical Research Benchmark (RQ1 - RQ4)
│   ├── README.md                     # Research questions breakdown & experimental protocol
│   └── benchmark.py                  # End-to-end benchmark script
│
├── notebooks/                        # Phase 2: Interactive Demonstration
│   ├── README.md                     # Notebook documentation & usage instructions
│   ├── MM_RFMNN_Colab_Benchmark.ipynb # Ready-to-run Google Colab notebook
│   └── MM_RFMNN_Executed.ipynb       # Executed notebook with outputs & metrics
│
├── tests/                            # Automated Verification Suite
│   ├── README.md                     # Test coverage summary
│   └── test_mm_rfmnn.py              # 7 targeted unit and integration tests
│
└── .agents/                          # Custom AI Skills & Directives
    └── skills/                       # anti-slop-engineer, doc-pdf-creator, colab-notebook-runner, web-deep-researcher
```

---

## Quickstart

### 1. Run Unit Tests (7/7 passing)
```bash
pytest
# or
python -m unittest discover tests
```

### 2. Run Empirical Research Benchmark (RQ1 – RQ4)
```bash
python experiments/benchmark.py
```

### 3. Generate Docs & PDFs
```bash
python .agents/skills/doc-pdf-creator/scripts/md_to_pdf.py \
  --input "MM_RFMNN_Algorithm_Blueprint.md" \
  --output "MM_RFMNN_Research_Paper.pdf" \
  --title "MM-RFMNN Architecture Blueprint"
```

---

## Empirical Benchmark Findings (Phase 2 Summary)

| Research Question | Metric | Baseline Result | MM-RFMNN Result | Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **RQ1: Probability Calibration** | Expected Calibration Error (ECE) | 0.4578 (Early FMNN) | **0.0333** | **13.7× better calibration** |
| **RQ2: Corruption Robustness** | Macro-F1 at extreme noise ($\sigma = 8.0$) | 0.4514 (Deep MLP) | **0.9874** | **Maintains performance under noise** |
| **RQ3: Missing Modalities** | Macro-F1 with Modality 1 missing ($q^{(0)}=0$) | Fail / undefined in naive models | **0.9874** | **Graceful zero-imputation-free degradation** |
| **RQ4: Selective Prediction** | Retained accuracy at 80%–100% coverage | Uncalibrated baseline | **100.00%** | **Certifiable uncertainty abstention** |
