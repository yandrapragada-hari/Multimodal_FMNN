# MM-RFMNN: Complete Architecture, Encoders, and Pipeline Guide

This guide covers the technical architecture, data flow, mathematical operations, encoder options, and concrete numeric execution of the **Multimodal Reliability-Aware Fuzzy Min-Max Neural Network (MM-RFMNN)**.

---

## 1. Tech Stack

- **Core Language**: Python 3.10+ with strict typing (`typing.Optional`, `list`, `tuple`, `dict`, `np.ndarray`).
- **Mathematical Computation & Vectorization**: NumPy, SciPy (`scipy.optimize.minimize` with L-BFGS-B, `scipy.spatial.distance`).
- **Base Machine Learning Framework**: Scikit-Learn (`BaseEstimator`, `ClassifierMixin`, `KFold`, `LogisticRegression`, `MLPClassifier`, `MinMaxScaler`).
- **Verification & Code Quality**: Pytest (`pytest.ini`, 13 tests passing), Ruff (0 lint errors).
- **Data Loaders**: OpenML API (`sklearn.datasets.fetch_openml`), compressed `.npz` archive caching.

---

## 2. High-Level Concept: Problem & Architectural Fix

### The Problem in Standard Multimodal Deep Learning
Standard deep neural networks concatenate multimodal features into a single high-dimensional vector. When any single channel experiences severe noise (e.g. sensor corruption) or goes missing (e.g. disconnected probe):
1. **Entangled Representation**: Noise in one channel poisons the entire representation.
2. **Zero-Imputation Trap**: Imputing zeros for missing channels creates artificial out-of-distribution vectors, causing catastrophic misclassifications.
3. **Overconfidence**: Softmax outputs dangerously confident wrong predictions.

```
[Modality 1 (Clean)]    ──────┐
[Modality 2 (CORRUPT)]  ──────┼──> [Early Concatenation / Deep MLP] ──> Overconfident Error!
[Modality 3 (Missing)]  ──────┘
```

### The MM-RFMNN Architectural Fix
1. **Modality-Specific Hyperbox Banks**: Modalities stay isolated in their native spaces, preventing cross-channel distortion.
2. **Out-of-Fold (OOF) Reliability Calibration**: Uses 5-fold cross-validation to assess whether a modality's fuzzy evidence is trustworthy, preventing overconfident memorization.
3. **Temperature Calibration ($T_k$)**: Calibrates non-probabilistic fuzzy scores into genuine probabilities.
4. **Simplex Missingness Masking ($q^{(k)}$)**: Completely zeros out missing channels without corrupting the model with imputed zeros.
5. **Selective Abstention**: Directly measures uncertainty (predictive entropy and Jensen-Shannon modality disagreement) to safely abstain ("say I don't know") when inputs are ambiguous.

```
Modality 1 (Audio)  ──> [Normalizer 1] ──> [FMNN Bank 1] ──> [Temp Scale] ──> Calibrated Probs P_1 ──┐
                                                 │                                                     │
                                                 └──> [Meta-Features] ──> [Reliability Est 1] ──> r_1  ├──> [Simplex Fuser] ──> [Uncertainty Engine] ──> Class OR Abstain (-1)
                                                                                                       │
Modality 2 (Vision) ──> [Normalizer 2] ──> [FMNN Bank 2] ──> [Temp Scale] ──> Calibrated Probs P_2 ──┤
                                                 │                                                     │
                                                 └──> [Meta-Features] ──> [Reliability Est 2] ──> r_2  ┘
Presence Mask Q     ───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Encoders and Modalities

### 3.1 What Encoders Are Used Currently

| Data Channel | Current Feature Extraction | Transformation Encoder | Dimensions ($d_k$) |
| :--- | :--- | :--- | :---: |
| **Tabular / Sensor 1** | Synthesized Gaussian shift / overlap | `ModalityNormalizer(method="minmax")` | 6 |
| **Tabular / Sensor 2** | Synthesized non-linear shift | `ModalityNormalizer(method="minmax")` | 4 |
| **UCI Shape Modality** | Fourier contour harmonics | `ModalityNormalizer` + bounds guard | 76 |
| **UCI Profile Modality** | Horizontal/vertical projections | `ModalityNormalizer` + bounds guard | 216 |
| **UCI Eigen Modality** | Karhunen-Loève (PCA eigenvectors) | `ModalityNormalizer` + bounds guard | 64 |
| **UCI Geometry Modality**| Morphological (area, perimeter, circularity) | `ModalityNormalizer` + bounds guard | 6 |
| **UCI Spatial Modality** | 2×3 window pixel subsampling | `ModalityNormalizer` + bounds guard | 240 |
| **UCI Moment Modality** | Orthogonal Zernike polynomial moments | `ModalityNormalizer` + bounds guard | 47 |

---

### 3.2 What Each Encoder Gives at Last

**Final Output**: Exactly **one 1D normalized dense float vector per modality**:
$$Z^{(k)} \in [0, 1]^{d_k}$$

**Why**:
- FMNN hyperbox geometry strictly requires coordinates inside the unit hypercube $[0, 1]^{d_k}$.
- $\theta_k$ (max hyperbox size bound, e.g. `0.35`) defines maximum edge length relative to $[0, 1]$. Without scaling, coordinates on scale $[0, 1000]$ would explode box count; coordinates on scale $[0, 0.001]$ would merge all classes into one box.

---

### 3.3 Can We Use Better Encoders for Each Type?

**Yes**. But with one mathematical constraint: **FMNN hyperboxes suffer curse of dimensionality if $d > 128$** (empty space phenomenon). Encoders must output compact embeddings ($16 \le d \le 64$ optimal).

| Data Type | Current Encoder | Better Production Upgrade | Why Better |
| :--- | :--- | :--- | :--- |
| **Tabular / Numerical** | Linear Min-Max | **QuantileTransformer(output_distribution="uniform")** + Yeo-Johnson power transform | Warps skewed outliers uniformly into $[0, 1]$; prevents single outlier squashing all hyperboxes into corner. |
| **Images / Vision** | Raw Pixels / Moments (240 dims) | **MobileNetV3-Small / EfficientNet-B0 + PCA (32 dims)** | Semantic visual invariances (lighting, rotation, texture) instead of raw pixel grids. |
| **Audio / Acoustic** | Synthetic / MFCC | **Log-Mel Spectrogram + 1D-CNN / MobileNet-Audio (32 dims)** | Captures harmonic pitch and formant shifts rather than hand-tuned heuristics. |
| **Text / Notes** | None (Tabular only) | **MiniLM-L6-v2 + TruncatedSVD (32-48 dims)** | Semantic phrase understanding; maps synonyms to adjacent hyperbox coordinates. |
| **Time-Series / Sensors**| Raw point vectors | **MiniRocket / Catch22 (22 dims) or 1D-ResNet** | Scale/phase invariant temporal shapelets; captures frequency drifts. |

---

### 3.4 What Other Data Types Can Be Considered?

1. **Biosignals & Streaming Sensor Telemetry**:
   - ECG, EEG, PPG, IMU accelerometers, industrial vibration.
   - Encoders: Catch22 features, Wavelet packet transforms (DWT), Fast Fourier Transform (FFT) bands.
2. **Graph / Relational / Network Structures**:
   - Protein-protein interactions, social graphs, financial transaction networks, molecular structures.
   - Encoders: Node2Vec, Graph Convolution Network (GCN) bottleneck vectors, Weisfeiler-Lehman graph kernels.
3. **Categorical / Discrete Codes**:
   - Medical ICD-10 diagnosis codes, user IDs, event flags.
   - Encoders: Target encoding, Entity Embeddings (dense 8D-16D learned lookup tables), CatBoost encoding.
4. **Spatio-Temporal / Geospatial**:
   - GPS coordinates, latitude/longitude trajectories, weather grids.
   - Encoders: Spherical harmonics, Space2Vec, H3 hexagonal grid embeddings.
5. **Chemical / Genomic Sequences**:
   - SMILES strings, DNA/RNA k-mers.
   - Encoders: Morgan circular fingerprints (folded to 64 dims), ProtBERT embeddings + PCA.

---

### 3.5 Options, Reliability & Weight Tradeoffs

#### Modality A: Tabular / Numerical / Lab Tests
- **Current**: `ModalityNormalizer` (MinMax).
- **Options**:
  - *Option 1 (Recommended)*: `QuantileTransformer(n_quantiles=100)` $\to$ Robust against sensor spikes, zero GPU needed, 100% deterministic, runtime $< 1$ms.
  - *Option 2*: `SplineTransformer` (b-spline basis expansion) $\to$ captures non-linear trends.
- **Verdict**: Ultra-reliable, 0 MB memory overhead.

#### Modality B: Vision / Images
- **Current**: Spatial pixel pooling + Zernike moments.
- **Options**:
  - *Option 1 (Recommended - Lightweight)*: **MobileNetV3-Small** (pretrained on ImageNet, frozen) $\to$ pooled feature vector $\to$ Linear PCA to 32 dimensions $\to$ MinMax to $[0, 1]^{32}$.
    - Weight: **~9 MB total disk size**.
    - Speed: **~3ms on standard CPU** (no GPU required).
    - Reliability: High; frozen weights prevent catastrophic drift.
  - *Option 2 (Zero Neural Network)*: OpenCV Hu Moments + Histogram of Oriented Gradients (HOG) + PCA.
    - Weight: **0 MB** (pure C/NumPy math).
    - Speed: $< 1$ms.
  - *Heavy Option (Avoid)*: CLIP ViT-L/14 (900MB+, slow CPU inference, 768 dims causes FMNN hyperbox sparsity).

#### Modality C: Audio / Acoustic
- **Current**: Feature shift vectors.
- **Options**:
  - *Option 1 (Recommended - Lightweight)*: **Librosa MFCCs (13 coefficients) + Deltas + Spectral Centroid** (Total: 28 dims).
    - Weight: **0 MB** (pure digital signal processing / SciPy FFT).
    - Speed: $< 2$ms on CPU.
    - Reliability: Extremely robust to background gain/volume changes.
  - *Option 2 (Neural)*: **YAMNet** (MobileNet-based audio model, 15 MB) pruned to 32 dims.

#### Modality D: Text / Clinical Notes
- **Current**: None.
- **Options**:
  - *Option 1 (Recommended - Lightweight Neural)*: **`all-MiniLM-L6-v2`** via ONNX runtime $\to$ PCA down to 32 dims.
    - Weight: **~22 MB** (compact 6-layer transformer).
    - Speed: **~8-12ms on CPU** with ONNX runtime.
    - Reliability: Fixed sentence embeddings, zero generation risk.
  - *Option 2 (Zero Neural)*: TF-IDF + TruncatedSVD (LSA) to 32 dims.
    - Weight: **< 1 MB**.
    - Speed: $< 1$ms.

#### Modality E: Time-Series / Waveforms
- **Current**: None.
- **Options**:
  - *Option 1 (Recommended)*: **Catch22 (C-based 22 canonical time-series characteristics)**.
    - Captures autocorrelation, entropy, outlier frequency, fluctuation analysis.
    - Weight: **0 MB** (compiled C library).
    - Speed: $< 0.5$ms per sample.
    - Reliability: Scientifically calibrated; perfectly matches FMNN dimensional target ($d = 22$).

---

### 3.6 Standard Ingestion Architecture Rule

```
[Raw High-Dim Input] (Image: 224x224, Audio: 1sec PCM, Text: 100 words)
         │
         ▼
[Frozen Feature Extractor] (MobileNetV3, Librosa MFCC, MiniLM, Catch22)
         │
         ▼ Output: d = 128..384
[Dimensionality Compressor] (PCA / TruncatedSVD to d_k = 16..48)
         │
         ▼ Output: Dense vector in R^{d_k}
[ModalityNormalizer] (Quantile / Min-Max fitted strictly on X_train)
         │
         ▼
[Unit Hypercube Input] Z^(k) in [0, 1]^{d_k}  ──>  Feeds into FMNN Hyperbox Banks
```

---

## 4. End-to-End Pipeline: Raw Input to Final Output

### Step 1: Raw Inputs
- **Feature matrices**: List of $K$ arrays: $[X^{(1)}, X^{(2)}, \dots, X^{(K)}]$.
  - Sample $i$: $X_i^{(k)} \in \mathbb{R}^{d_k}$ (each modality can have different dimensions $d_k$).
- **Presence mask**: $Q \in \{0, 1\}^{N \times K}$.
  - $q_i^{(k)} = 1$: modality $k$ present.
  - $q_i^{(k)} = 0$: sensor unplugged / missing.

---

### Step 2: Modality Normalization ([`normalizer.py`](file:///home/whysooraj/Projects/Multimodal_FMNN/mm_rfmnn/encoders/normalizer.py))
- **Model**: `ModalityNormalizer(method="minmax")`.
- **Input**: $X^{(k)} \in \mathbb{R}^{N \times d_k}$.
- **Operation**:
  - Fits bounds $min_j, max_j$ strictly on $X_{train}$ where $q_i^{(k)} = 1$ (zero test leakage).
  - Transforms:
    $$Z_j = \frac{X_j - \min_j}{\max_j - \min_j + \epsilon}$$
  - Clips values to unit hypercube $[0, 1]^{d_k}$.
- **Output**: Normalized feature vectors $Z_i^{(k)} \in [0, 1]^{d_k}$.

---

### Step 3: Modality-Specific Hyperbox Quantization ([`hyperbox.py`](file:///home/whysooraj/Projects/Multimodal_FMNN/mm_rfmnn/base/hyperbox.py))
- **Model**: `FMNN(theta, gamma)` running dedicated `HyperboxBank` per modality.
- **Input**: $Z_i^{(k)} \in [0, 1]^{d_k}$, label $y_i \in \{0, \dots, C-1\}$.
- **Internal Mechanics (No Backpropagation, Fast 1-Pass Learning)**:
  1. **Hyperbox Structure**:
     - Box $B_j = (V_j, W_j, c_j)$ where $V_j \in [0, 1]^{d_k}$ (min corner), $W_j \in [0, 1]^{d_k}$ (max corner), $c_j$ (class).
  2. **Expansion**:
     - Check if existing box of class $y_i$ can expand to cover $Z_i^{(k)}$ without violating max size $\theta$:
       $$\max(W_{j,d}, Z_{i,d}^{(k)}) - \min(V_{j,d}, Z_{i,d}^{(k)}) \le \theta \quad \forall d$$
     - If yes $\to$ expand corners $V_j, W_j$.
     - If no $\to$ instantiate new box with $V = W = Z_i^{(k)}$.
  3. **Overlap Contraction (Simpson's 4 Cases)**:
     - If expanded box overlaps box of different class, shrink overlapping dimension using minimum adjustment rule $\to$ eliminates ambiguous decision regions.
  4. **Fuzzy Membership Evaluation (Simpson Exponential Form)**:
     - For query sample $Z$, distance outside box along dimension $d$:
       $$\Delta_d = \max(0, V_{j,d} - Z_d) + \max(0, Z_d - W_{j,d})$$
     - Continuous membership $\mu_j(Z) \in (0, 1]$:
       $$\mu_j(Z) = \exp(-\gamma \|\Delta\|_1) = \exp\left(-\gamma \sum_d \Delta_d\right)$$
       - Inside box $\to \Delta_d = 0 \to \mu_j = 1.0$.
       - Outside box $\to$ decays exponentially with Manhattan distance.
  5. **Class Evidence Aggregation**:
     - Raw evidence for class $c$:
       $$e_c^{(k)}(Z) = \max_{j : c_j = c} \mu_j(Z)$$
- **Output**:
  - Evidence vector: $\mathbf{e}^{(k)} = [e_0^{(k)}, \dots, e_{C-1}^{(k)}] \in (0, 1]^C$.
  - Best membership score: $m^{(k)} = \max_c e_c^{(k)}$.

---

### Step 4: Temperature Calibration ([`fuser.py`](file:///home/whysooraj/Projects/Multimodal_FMNN/mm_rfmnn/fusion/fuser.py))
- **Model**: `TemperatureCalibrator`.
- **Why Needed**: Raw fuzzy membership is not a true probability (often overconfident near box boundaries).
- **Input**: Evidence vector $\mathbf{e}^{(k)} \in \mathbb{R}^C$.
- **Operation**:
  - Validation set optimizes scalar temperature $T_k > 0$ via Scipy L-BFGS-B minimizing Negative Log-Likelihood (NLL):
    $$\mathcal{L}(T_k) = -\sum_{i} \log \left( \frac{\exp(e_{y_i}^{(k)} / T_k)}{\sum_c \exp(e_c^{(k)} / T_k)} \right)$$
  - Calibrate:
    $$\mathbf{p}^{(k)} = \text{Softmax}\left(\frac{\mathbf{e}^{(k)}}{T_k}\right)$$
- **Output**: Well-calibrated class probability distribution $\mathbf{p}^{(k)} \in \Delta^{C-1}$.

---

### Step 5: Meta-Feature Extraction & Reliability Estimation ([`estimator.py`](file:///home/whysooraj/Projects/Multimodal_FMNN/mm_rfmnn/reliability/estimator.py))
- **Models**: `MetaFeatureExtractor` + `ReliabilityEstimator` (`LogisticRegression` or `MLPClassifier`).
- **Why Needed**: Figure out if modality $k$ is clean or noisy for *this specific input*.
- **Operation**:
  1. **Meta-Feature Vector** $u^{(k)} \in \mathbb{R}^5$:
     - $u_1$: Peak membership $m^{(k)} = \max_c e_c^{(k)}$.
     - $u_2$: Margin deficit $e_{\text{top1}}^{(k)} - e_{\text{top2}}^{(k)}$ (ambiguity indicator).
     - $u_3$: Evidence entropy ratio $H(\mathbf{e}^{(k)}) / \log C$.
     - $u_4$: Outside boundary distance $\min_j \|\Delta_j\|_1$.
     - $u_5$: Box density / support count.
  2. **Out-of-Fold (OOF) Training**:
     - During `.fit()`, splits training data into 5 folds.
     - Fits FMNN on 4 folds, extracts meta-features on held-out 5th fold.
     - Target $t_i = 1$ if predicted class matches true label, else $0$.
     - *Result*: Zero memorization leakage $\to$ reliability model learns true failure patterns.
  3. **Inference**:
     $$r_i^{(k)} = \sigma(\mathbf{w}^T u_i^{(k)} + b) \in [0, 1]$$
- **Output**: Reliability scalar $r^{(k)} \in [0, 1]$ ($1.0 = \text{trustworthy}$, $0.0 = \text{noisy/failing}$).

---

### Step 6: Simplex Evidence Fusion ([`fuser.py`](file:///home/whysooraj/Projects/Multimodal_FMNN/mm_rfmnn/fusion/fuser.py))
- **Model**: `ReliabilityFuser`.
- **Input**:
  - Calibrated probability vectors $\{\mathbf{p}^{(1)}, \dots, \mathbf{p}^{(K)}\}$.
  - Reliability scalars $\{r^{(1)}, \dots, r^{(K)}\}$.
  - Modality presence mask $\{q^{(1)}, \dots, q^{(K)}\}$.
- **Operation**:
  1. **Gated Effective Reliability**:
     $$\tilde{r}^{(k)} = q^{(k)} \cdot r^{(k)}$$
     *(Missing modality has $q^{(k)}=0 \implies \tilde{r}^{(k)}=0$, eliminating imputation)*
  2. **Simplex Weight Normalization**:
     $$\alpha^{(k)} = \frac{\tilde{r}^{(k)}}{\sum_{m=1}^K \tilde{r}^{(m)} + \epsilon}, \quad \sum_{k=1}^K \alpha^{(k)} = 1.0$$
  3. **Fusion (Arithmetic Opinion Pool)**:
     $$\mathbf{P}_{\text{fused}} = \sum_{k=1}^K \alpha^{(k)} \mathbf{p}^{(k)}$$
- **Output**: Final ensemble probability vector $\mathbf{P}_{\text{fused}} \in \Delta^{C-1}$.

---

### Step 7: Uncertainty Quantification & Selective Classification ([`fuser.py`](file:///home/whysooraj/Projects/Multimodal_FMNN/mm_rfmnn/fusion/fuser.py))
- **Engine**: Evaluates 4 certifiable uncertainty criteria:
  1. **Confidence Deficit**: $\max_c P_{\text{fused}}(c) < \tau_{\text{conf}}$ (e.g. $< 0.40$).
  2. **Predictive Entropy**: $-\sum_c P_{\text{fused}}(c) \log P_{\text{fused}}(c) > \tau_{\text{entropy}}$ (e.g. $> 1.80$).
  3. **Jensen-Shannon Modality Disagreement**:
     $$D_{JS}(\mathbf{p}^{(1)}, \dots, \mathbf{p}^{(K)}) > \tau_{\text{disagreement}}$$
     *(Detects when Modality 1 says Class A but Modality 2 says Class B)*
  4. **Total Reliability Collapse**: $\sum_k \tilde{r}^{(k)} < \tau_{\text{min\_rel}}$ (all available channels are corrupt/offline).
- **Decision Rule**:
  - If any uncertainty trigger fires $\to$ **Abstain** (output `-1`: *"Uncertain, human expert needed"*).
  - Else $\to$ **Predict**: $\hat{y} = \arg\max_c P_{\text{fused}}(c)$.

---

## 5. Phase 3 Additions: Hyperbox Pruning & Online Stream Adaptation

1. **Hyperbox Pruning (`HyperboxPruner`)**:
   - Scans trained boxes:
     - Subsumption check: Box $A \subseteq$ Box $B$ with same class $\to$ delete $A$, add count to $B$.
     - Low-support check: Box sample count $< 2 \to$ prune unless sole representative of class.
   - *Result on real UCI dataset*: 3,610 boxes $\to$ 1,207 boxes (**66.6% reduction**), 0% accuracy drop.
2. **Online Stream Adaptation (`OnlineStreamAdapter` / `partial_fit`)**:
   - Consumes live data mini-batches without retraining from scratch.
   - Caps memory footprint by triggering auto-pruning every $N$ batches.

---

## 6. Concrete Numeric Walkthrough

Imagine patient diagnosis (Class 0: Healthy, Class 1: Arrhythmia).
- **Modality 1 (ECG Sensor)**: Noisy loose wire ($\sigma = 5.0$).
- **Modality 2 (Blood Pressure / Tabular)**: Clean reading.
- **Modality 3 (Patient Age / Metadata)**: Sensor missing ($q^{(3)} = 0$).

1. **Normalized Input**:
   - $Z^{(1)} = [0.85, 0.12]$ (corrupted)
   - $Z^{(2)} = [0.30, 0.45]$ (clean)
   - $Z^{(3)} = \text{None}$ ($q^{(3)} = 0$)
2. **FMNN Evidence**:
   - Modality 1 (ECG): Fuzzy membership weak, margin near zero $\to \mathbf{e}^{(1)} = [0.35, 0.38]$.
   - Modality 2 (BP): Inside hyperbox of Class 1 $\to \mathbf{e}^{(2)} = [0.10, 0.95]$.
3. **Calibrated Probabilities**:
   - $\mathbf{p}^{(1)} = [0.48, 0.52]$ (high entropy)
   - $\mathbf{p}^{(2)} = [0.03, 0.97]$ (decisive)
4. **Reliability Predictor**:
   - Modality 1 meta-features (low margin, high entropy) $\to r^{(1)} = 0.08$ (unreliable).
   - Modality 2 meta-features (high margin, inside box) $\to r^{(2)} = 0.92$ (reliable).
   - Modality 3 missing $\to \tilde{r}^{(3)} = 0 \cdot r^{(3)} = 0.00$.
5. **Simplex Fusion**:
   - Total reliability: $0.08 + 0.92 + 0.00 = 1.00$.
   - Weights: $\alpha^{(1)} = 0.08$, $\alpha^{(2)} = 0.92$, $\alpha^{(3)} = 0.00$.
   - Fused:
     $$\mathbf{P}_{\text{fused}} = 0.08 \cdot [0.48, 0.52] + 0.92 \cdot [0.03, 0.97] = [0.066, 0.934]$$
6. **Uncertainty Check**:
   - Confidence: $0.934 > 0.40$ (OK).
   - Entropy: low (OK).
   - Result: **Class 1 (Arrhythmia)** with 93.4% calibrated certainty. No hallucination, noise suppressed.
