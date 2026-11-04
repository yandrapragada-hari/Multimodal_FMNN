# MM-RFMNN: Multimodal Reliability-Aware Fuzzy Min-Max Neural Network

[![Tests](https://img.shields.io/badge/tests-27%2F27%20passed-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-ruff-black.svg)]()
[![Status](https://img.shields.io/badge/Phases%201--4-Complete-blue.svg)]()

This repository contains the complete production-grade implementation and empirical research benchmark suite for the **Multimodal Reliability-Aware Fuzzy Min-Max Neural Network (MM-RFMNN)**, designed for noise-resilient, interpretable, and certifiable classification under real-world modality noise, corruption, and channel missingness.

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
6. **Principled Hyperbox Pruning & Online Streaming**:
   - Compresses hyperbox banks by 66.6% via low-support elimination and geometric subsumption pruning with zero accuracy loss, alongside `partial_fit` streaming adaptation.
7. **Turnkey Raw Multimodal Encoders**:
   - Integrated lightweight encoders for raw Vision (`ImageFeatureEncoder`), Audio (`AudioFeatureEncoder`), Text (`TextFeatureEncoder`), and Tabular data (`TabularQuantileEncoder`) projecting into compact, scale-invariant hyperbox subspaces.
8. **Density-Adaptive Theta & Cross-Modal Conflict Detection**:
   - Automated dispersion-based $\theta_k$ autotuning and empirical co-occurrence conflict detection to catch and abstain on irreconcilable cross-modality contradictions.
9. **Visual Explainability Suite**:
   - Headless diagnostic rendering of 2D hyperbox geometry, sample decision boundaries, modality attribution profiles, and empirical risk-coverage curves.

---

## Project Structure

```
MM-RFMNN/
├── MM_RFMNN_Algorithm_Blueprint.md   # Mathematical blueprint & methodology
├── PROJECT_STATUS_AND_ARCHITECTURE_REPORT.md # Comprehensive architectural report
├── PROJECT_STATUS_AND_ARCHITECTURE_REPORT.docx # Publication-grade Word report
├── README.md                         # Main project overview & index
├── pytest.ini                        # Pytest configuration file
├── GEMINI.md                         # Anti-slop engineering & code standards
│
├── mm_rfmnn/                         # Core Mathematical & Algorithmic Engine
│   ├── __init__.py                   # Public API exports
│   ├── base/                         # Geometric Hyperbox Core
│   │   ├── hyperbox.py               # Vectorized hyperboxes & Simpson expansion/contraction
│   │   ├── adaptive_theta.py         # Density-adaptive theta autotuning
│   │   ├── pruning.py                # Hyperbox subsumption & support pruning
│   │   └── incremental.py            # Stream adapter & partial_fit online learning
│   ├── datasets/                     # Real-world dataset loaders (mfeat, WDBC, Dermatology, HAR, Raw)
│   │   └── loader.py                 # Multi-domain multimodal dataset loaders with offline caching
│   ├── encoders/                     # Preprocessors & Subspace Encoders
│   │   ├── normalizer.py             # Leakage-free MinMax and Quantile normalizers
│   │   └── multimodal.py             # Turnkey Image, Audio, Text, and Tabular encoders
│   ├── reliability/                  # Instance Reliability Estimation
│   │   └── estimator.py              # Meta-feature extractor & OOF target generator
│   ├── fusion/                       # Evidence Fusion & Contradiction Detection
│   │   ├── fuser.py                  # Temperature calibration & simplex evidence fuser
│   │   └── conflict.py               # Cross-modality conflict & contradiction detector
│   ├── viz/                          # Visual Explainability & Diagnostics
│   │   └── plotter.py                # 2D hyperboxes, attributions, & risk-coverage curves
│   └── model.py                      # Scikit-learn compatible MMRFMNNClassifier
│
├── experiments/                      # Empirical Research & Evaluation Suites
│   ├── benchmark.py                  # RQ1 - RQ4 synthetic benchmark protocol
│   ├── evaluate_real_datasets.py     # UCI mfeat 6-modality real-world benchmark
│   ├── benchmark_multiple_datasets.py# 5-dataset benchmark suite across diverse domains
│   ├── demo_phase4_capabilities.py   # End-to-end Phase 4 capability demonstration
│   └── output/                       # Diagnostic figures & benchmark JSON outputs
│
├── notebooks/                        # Interactive Research Demonstrations
│   ├── MM_RFMNN_Colab_Benchmark.ipynb # Ready-to-run Google Colab notebook
│   └── MM_RFMNN_Executed.ipynb       # Executed notebook with metrics & outputs
│
└── tests/                            # Comprehensive Automated Test Suite (27/27 passed)
    ├── test_mm_rfmnn.py              # Phase 1 & 2 unit and integration tests
    ├── test_next_stage.py            # Phase 3 pruning & streaming adaptation tests
    ├── test_phase4.py                # Phase 4 encoders, adaptive theta, conflict tests
    └── test_dataset_loaders.py       # Dataset loaders and caching integrity tests
```

---

## Quickstart

### 1. Run Complete Automated Test Suite (27/27 passing)
```bash
pytest
```

### 2. Run Comprehensive 5-Dataset Multimodal Benchmark Suite
```bash
python experiments/benchmark_multiple_datasets.py
```

### 3. Run UCI Multiple Features Benchmark
```bash
python experiments/evaluate_real_datasets.py
```

### 4. Run Phase 4 Demonstration & Generate Diagnostic Plots
```bash
python experiments/demo_phase4_capabilities.py
```

---

## Comprehensive Multi-Dataset Benchmark Results

Evaluated across **5 diverse multimodal benchmarks** spanning computer vision, clinical oncology, dermatology, wearable kinematics, and heterogeneous raw signals:

| Dataset | Modalities ($K$) | Domain & Channels | Clean MLP F1 | Clean MM-RFMNN F1 | Extreme Noise MLP F1 ($\sigma=6.0$) | Extreme Noise MM-RFMNN F1 | Missing Channel F1 ($q_0=0$) | Selective Acc (80% Coverage) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UCI Multiple Features** | 6 | Digits: Fourier, Factors, Karhunen, Morph, Pixel, Zernike | 0.9783 | 0.8622 | 0.0865 *(collapsed)* | **0.8591** | **0.8558** | **87.71%** |
| **Breast Cancer (WDBC)** | 3 | Oncology: Nucleus Mean, SE, Worst Views | 0.9813 | 0.9501 | 0.4972 *(random)* | **0.9562** | **0.9620** | **99.27%** |
| **UCI Dermatology** | 2 | Clinical: Symptoms View + Histopathology Biopsy | 0.9796 | 0.8948 | 0.2074 *(collapsed)* | **0.7990** | **0.8710** | **95.45%** |
| **UCI HAR (Wearables)** | 3 | Kinematics: Body Accel, Gravity, Gyroscope | 0.9649 | 0.6896 | 0.6362 *(degraded)* | **0.8649** | **0.6838** | **66.04%** |
| **Heterogeneous Raw** | 4 | Multi-Type: Vision (2D), Audio (1D), Text, Tabular | 1.0000 | 1.0000 | 0.7337 *(degraded)* | **0.9966** | **1.0000** | **100.00%** |

