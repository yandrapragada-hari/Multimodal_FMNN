# MM-RFMNN: Core Mathematical Engine & Architecture (Phase 1)

This package contains the core implementation of the **Multimodal Reliability-Aware Fuzzy Min-Max Neural Network (MM-RFMNN)**.

## Package Architecture

```
mm_rfmnn/
├── __init__.py               # Public API exports
├── base/
│   ├── __init__.py
│   └── hyperbox.py           # HyperboxBank & FMNN expansion/contraction
├── encoders/
│   ├── __init__.py
│   └── normalizer.py         # Leakage-free modality normalizer
├── reliability/
│   ├── __init__.py
│   └── estimator.py          # Meta-feature extractor & OOF reliability estimator
├── fusion/
│   ├── __init__.py
│   └── fuser.py              # Temperature calibration, evidence fusion & abstention
└── model.py                  # High-level scikit-learn compatible MMRFMNNClassifier
```

---

## Module Descriptions

### 1. `base/hyperbox.py`
- **`HyperboxBank`**: Vectorized geometric container storing min points ($V \in \mathbb{R}^{M \times d}$), max points ($W \in \mathbb{R}^{M \times d}$), and class labels ($c \in \{1,\dots,C\}$) for a single modality. Computes batch Manhattan outside distances:
  $$D_j(z) = \sum_{r=1}^d \max(0, v_{jr} - z_r, z_r - w_{jr})$$
  and continuous fuzzy membership:
  $$\mu_j(z) = \exp(-\gamma D_j(z))$$
- **`FMNN`**: Implements Simpson's (1992) one-pass hyperbox expansion governed by maximum hyperbox size $\theta \in (0, 1]$, and 4-case dimensional overlap contraction.

### 2. `encoders/normalizer.py`
- **`ModalityNormalizer`**: Scales raw modality features to $[0, 1]$ based strictly on the training partition to prevent data leakage. Supports min-max scaling and quantile bounds. Outliers at inference time are preserved ($D_j(z) \gg 0$) to allow proper low-confidence falloff and abstention.

### 3. `reliability/estimator.py`
- **`MetaFeatureExtractor`**: Extracts instance-level meta-features $u_i^{(k)} = [m_i, \Delta_i, \text{ratio}_i, Q_i]$, capturing membership peak, class separation margin, and evidence concentration.
- **`ReliabilityEstimator`**: Models instance reliability $r_i^{(k)} = \sigma(g_{\phi_k}(u_i^{(k)}))$. Trained strictly on **Out-Of-Fold (OOF)** targets via 5-fold cross-validation ($t_i = \mathbb{I}(\hat{y}_{\text{OOF}} == y_i)$) to eliminate optimistic in-sample memorization.

### 4. `fusion/fuser.py`
- **`TemperatureCalibrator`**: Optimizes temperature scalars $T_k > 0$ on validation Negative Log-Likelihood (NLL) via L-BFGS-B, converting non-probabilistic FMNN evidence into calibrated distributions.
- **`ReliabilityFuser`**: Implements presence-masked simplex weighting:
  $$\tilde{r}_i^{(k)} = q_i^{(k)} \cdot r_i^{(k)}, \quad \alpha_i^{(k)} = \frac{\tilde{r}_i^{(k)}}{\sum_l \tilde{r}_i^{(l)}}$$
  Supports **Arithmetic Convex Combination** ($p_i(c) = \sum \alpha_k p^{(k)}(c)$) and **Log-Opinion Pool** ($\log p_i(c) \propto \sum \alpha_k \log p^{(k)}(c)$). Evaluates multi-criteria uncertainty (Entropy, Confidence Deficit, Jensen-Shannon Disagreement, Aggregate Reliability) for selective abstention.

### 5. `model.py`
- **`MMRFMNNClassifier`**: Unified scikit-learn compatible estimator providing `.fit()`, `.predict()`, `.predict_proba()`, and `.explain()`. Returns transparent audit payloads containing winning hyperbox coordinates, distance margins, and modality reliability weights.
