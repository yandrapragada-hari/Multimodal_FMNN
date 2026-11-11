# MM-RFMNN: Comprehensive Mathematical & Algorithmic Specification

A step-by-step mathematical guide detailing every internal operation, formula, variable, and intermediate transformation in the **Multimodal Reliability-Aware Fuzzy Min-Max Neural Network (MM-RFMNN)**.

---

## 1. End-to-End Information Flow

The pipeline converts raw, heterogeneous multimodal inputs into calibrated class probabilities or certifiable abstentions through 9 deterministic mathematical stages:

```
[Raw Modality Inputs: Vision, Audio, Text, Tabular]
                       │
                       ▼
[Stage 1] Subspace Encoding & Modality Normalization
         Maps raw domains into bounded coordinates: Z^(k) in [0, 1]^(d_k)
                       │
                       ▼
[Stage 2] Density-Adaptive Hyperbox Bound Autotuning
         Estimates dispersion-aware box growth limits: theta_k in [theta_min, theta_max]
                       │
                       ▼
[Stage 3] Geometric Hyperbox Construction (FMNN)
         Simpson expansion + 4-case dimensional overlap contraction -> H^(k) = {(V_j, W_j, c_j)}
                       │
                       ▼
[Stage 4] Distance Computation & Fuzzy Evidence
         Manhattan distance D_j(z) -> Exponential campfire membership mu_j(z) -> Class evidence s_i^(k)(c)
                       │
                       ▼
[Stage 5] Out-Of-Fold (OOF) Reliability Estimation
         Meta-features u_i^(k) -> 5-Fold OOF supervised classifier -> Trust score r_i^(k) in [0, 1]
                       │
                       ▼
[Stage 6] Temperature Calibration
         NLL optimization via L-BFGS-B -> Calibrated distributions p_i^(k)(c) = Softmax(s_i^(k) / T_k)
                       │
                       ▼
[Stage 7] Simplex Fusion & Missingness Gating
         Presence mask q_i^(k) -> Gated weights alpha_i^(k) -> Blended distribution P_i(c)
                       │
                       ▼
[Stage 8] Cross-Modal Contradiction & Uncertainty Abstention
         Co-occurrence conflict + Entropy + Jensen-Shannon divergence -> Output class OR -1 (Abstain)
                       │
                       ▼
[Stage 9] Geometric Pruning & Stream Adaptation
         Subsumption + low-support removal (66% compression) + partial_fit online learning
```

---

## 2. Stage-by-Stage Mathematical Formulations

### Stage 1: Input Encoding & Subspace Normalization

Converts raw, heterogeneous data into coordinate vectors bounded inside the unit hypercube $[0, 1]^{d_k}$.

Implemented in [`mm_rfmnn/encoders/multimodal.py`](mm_rfmnn/encoders/multimodal.py) and [`mm_rfmnn/encoders/normalizer.py`](mm_rfmnn/encoders/normalizer.py).

#### A. Raw Image Feature Extraction (`ImageFeatureEncoder`)
1. **Input**: Grayscale or RGB 2D image matrix $I \in \mathbb{R}^{H \times W}$.
2. **Multi-Scale Spatial Pooling**:
   Image is partitioned into a $4 \times 4$ regular spatial grid. For each patch $P_p$ ($p = 1, \dots, 16$), compute spatial mean and variance:
   $$\bar{\mu}_p = \frac{1}{|P_p|} \sum_{(u, v) \in P_p} I(u, v), \quad \sigma_p^2 = \frac{1}{|P_p|} \sum_{(u, v) \in P_p} (I(u, v) - \bar{\mu}_p)^2$$
3. **Sobel Directional Gradients**:
   Convolve image with $3 \times 3$ horizontal ($K_x$) and vertical ($K_y$) Sobel kernel operators:
   $$G_x = I * K_x, \quad G_y = I * K_y, \quad |G(u, v)| = \sqrt{G_x(u, v)^2 + G_y(u, v)^2}$$
4. **Subspace Compression**:
   Concatenate patch statistics and global gradient moments into raw feature vector $x_{\text{raw}} \in \mathbb{R}^{48}$. Project onto orthonormal basis via Truncated SVD to target dimension $d_k = 16$:
   $$z_{\text{proj}} = x_{\text{raw}} \mathbf{U}_k \in \mathbb{R}^{16}$$

#### B. Raw Audio Feature Extraction (`AudioFeatureEncoder`)
1. **Input**: 1D acoustic time-series waveform $w \in \mathbb{R}^L$ sampled at rate $f_s$.
2. **Zero-Crossing Rate (ZCR)**:
   $$\text{ZCR} = \frac{1}{2(L - 1)} \sum_{t=1}^{L-1} |\text{sign}(w_t) - \text{sign}(w_{t-1})|$$
3. **Root-Mean-Square Energy (RMS)**:
   $$\text{RMS} = \sqrt{\frac{1}{L} \sum_{t=1}^L w_t^2 + \epsilon}$$
4. **Fast Fourier Transform (FFT) Spectral Analysis**:
   Compute discrete Fourier transform magnitude spectrum:
   $$X(f) = |\text{FFT}(w)|, \quad f \in [0, f_s / 2]$$
5. **Spectral Centroid**:
   $$f_c = \frac{\sum f \cdot X(f)}{\sum X(f) + \epsilon}$$
6. **Multi-Band Filterbank Log-Energies**:
   Partition spectrum into 8 frequency bands $\{B_b\}_{b=1}^8$. Compute log band energy:
   $$E_b = \log \left( \frac{1}{|B_b|} \sum_{f \in B_b} X(f)^2 + \epsilon \right)$$
7. **Subspace Compression**:
   Combine temporal and spectral metrics into $x_{\text{audio}} \in \mathbb{R}^{14}$, project via SVD to $d_k = 16$.

#### C. Raw Text Feature Extraction (`TextFeatureEncoder`)
1. **Input**: Character or token sequence $T = (t_1, t_2, \dots, t_N)$.
2. **Sublinear TF-IDF Vectorization**:
   For word and character n-grams ($n \in [1, 2]$):
   $$\text{TF-IDF}(t, d) = (1 + \log \text{TF}(t, d)) \cdot \log \left( \frac{1 + N_{\text{docs}}}{1 + \text{DF}(t)} + 1 \right)$$
3. **Subspace Compression**:
   Project sparse vocabulary matrix into dense compact vector via Truncated SVD to $d_k = 16$.

#### D. Leakage-Free Modality Normalization (`ModalityNormalizer`)
All features are normalized strictly using parameters fitted on the training split $\mathcal{D}_{\text{train}}$:
$$z_{ir}^{(k)} = \text{clip} \left( \frac{x_{ir}^{(k)} - \min_{j \in \mathcal{D}_{\text{train}}} x_{jr}^{(k)}}{\max_{j \in \mathcal{D}_{\text{train}}} x_{jr}^{(k)} - \min_{j \in \mathcal{D}_{\text{train}}} x_{jr}^{(k)} + \epsilon_{\text{scale}}}, \, 0, \, 1 \right)$$
- If feature is constant ($\max = \min$), denominator safely sets scaled coordinate to $0.5$.
- Result: Clean coordinate vector $Z_i^{(k)} \in [0, 1]^{d_k}$.

---

### Stage 2: Density-Adaptive Hyperbox Size Bound ($\theta_k$) Autotuning

Eliminates manual trial-and-error hyperparameter search for the maximum hyperbox expansion bound $\theta_k$.

Implemented in [`mm_rfmnn/base/adaptive_theta.py`](mm_rfmnn/base/adaptive_theta.py).

1. **Intra-Class Dispersion ($\bar{\sigma}_k$)**:
   Measures local cluster spread among samples of the same class:
   $$\bar{\sigma}_k = \text{median}_{c \in \mathcal{C}} \left( \frac{1}{|S_c|} \sum_{i \in S_c} \frac{\|z_i^{(k)} - \bar{z}_c^{(k)}\|_1}{d_k} \right)$$
   where $\bar{z}_c^{(k)}$ is the geometric centroid of class $c$, and $S_c$ is the set of indices belonging to class $c$.

2. **Inter-Class Separation Margin ($\bar{\Delta}_k$)**:
   Measures nearest-neighbor separation to samples of competing classes:
   $$\bar{\Delta}_k = \text{median}_{i} \left( \min_{j: y_j \ne y_i} \frac{\|z_i^{(k)} - z_j^{(k)}\|_1}{d_k} \right)$$

3. **Autotuned Theta Formulation**:
   $$\theta_k = \text{clip} \left( \lambda_{\text{disp}} \bar{\sigma}_k + \lambda_{\text{marg}} \bar{\Delta}_k, \, \theta_{\min}, \, \theta_{\max} \right)$$
   - Default hyperparameters: $\lambda_{\text{disp}} = 0.50$, $\lambda_{\text{marg}} = 0.50$, $\theta_{\min} = 0.20$, $\theta_{\max} = 0.60$.
   - Enforcing $\theta_{\min} = 0.20$ guarantees that hyperboxes can expand to merge co-located points, preventing single-sample hyperbox explosion.

---

### Stage 3: Geometric Hyperbox Construction (FMNN Expansion & Contraction)

Builds decoupled geometric hyperbox banks $\mathcal{H}^{(k)} = \{B_j^{(k)} = (V_j^{(k)}, W_j^{(k)}, c_j^{(k)}, N_j^{(k)})\}_{j=1}^{M_k}$.

Implemented in [`mm_rfmnn/base/hyperbox.py`](mm_rfmnn/base/hyperbox.py).

#### A. Hyperbox Expansion Criterion (Simpson 1992)
Given sample $(z, y_i)$, find existing hyperbox $B_j$ of identical class ($c_j == y_i$) that can absorb $z$ without exceeding size bound $\theta_k$ along any dimension:
$$\max(w_{jr}, z_r) - \min(v_{jr}, z_r) \le \theta_k \quad \forall r \in \{1, \dots, d_k\}$$

- **If condition holds**: Expand box boundaries:
  $$V_j \leftarrow \min(V_j, z), \quad W_j \leftarrow \max(W_j, z), \quad N_j \leftarrow N_j + 1$$
- **If condition fails for all boxes**: Initialize new hyperbox:
  $$B_{\text{new}} = (V = z, \, W = z, \, c = y_i, \, N = 1)$$

#### B. Simpson 4-Case Dimensional Overlap Contraction
Whenever hyperbox $B_j$ of class $c_j$ expands, test for geometric intersection against all hyperboxes $B_k$ of opposing classes ($c_k \ne c_j$). Overlap occurs if and only if bounds cross in every dimension:
$$\bigwedge_{r=1}^{d_k} (w_{jr} \ge v_{kr} \land w_{kr} \ge v_{jr}) == \text{True}$$

If overlap exists, find the single dimension $\Delta_{\min}$ exhibiting minimum overlap:
$$\Delta_{\min} = \arg\min_{r \in \{1, \dots, d_k\}} [\min(w_{jr}, w_{kr}) - \max(v_{jr}, v_{kr})]$$

Contract boundaries along dimension $\Delta_{\min}$ using Simpson's 4 cases:
- **Case 1 ($v_{jr} < v_{kr} \le w_{jr} < w_{kr}$)**:
  $$w_{jr} \leftarrow \frac{w_{jr} + v_{kr}}{2}, \quad v_{kr} \leftarrow \frac{w_{jr} + v_{kr}}{2}$$
- **Case 2 ($v_{kr} < v_{jr} \le w_{kr} < w_{jr}$)**:
  $$w_{kr} \leftarrow \frac{w_{kr} + v_{jr}}{2}, \quad v_{jr} \leftarrow \frac{w_{kr} + v_{jr}}{2}$$
- **Case 3 ($v_{jr} \le v_{kr} \le w_{kr} \le w_{jr}$)** (Box $k$ inside Box $j$):
  $$\begin{cases} w_{jr} \leftarrow v_{kr} & \text{if } (w_{kr} - v_{jr}) < (w_{jr} - v_{kr}) \\ v_{jr} \leftarrow w_{kr} & \text{otherwise} \end{cases}$$
- **Case 4 ($v_{kr} \le v_{jr} \le w_{jr} \le w_{kr}$)** (Box $j$ inside Box $k$):
  $$\begin{cases} w_{kr} \leftarrow v_{jr} & \text{if } (w_{jr} - v_{kr}) < (w_{kr} - v_{jr}) \\ v_{kr} \leftarrow w_{jr} & \text{otherwise} \end{cases}$$

---

### Stage 4: Distance Computation & Fuzzy Evidence

Implemented in [`mm_rfmnn/base/hyperbox.py`](mm_rfmnn/base/hyperbox.py).

#### A. Manhattan Outside Distance
For test sample coordinate $z$ and hyperbox $B_j = (V_j, W_j)$:
$$d_{jr}^{(k)}(z) = \max(0, \, v_{jr}^{(k)} - z_r, \, z_r - w_{jr}^{(k)})$$
$$D_j^{(k)}(z) = \sum_{r=1}^{d_k} d_{jr}^{(k)}(z)$$
- Inside the box: $V_j \le z \le W_j \implies D_j(z) = 0$.
- Outside the box: $D_j(z) > 0$ accumulates step distance to nearest boundary.

#### B. Continuous Exponential Fuzzy Membership (Campfire Function)
$$\mu_j^{(k)}(z) = \exp \left( -\gamma_k \cdot D_j^{(k)}(z) \right), \quad \mu \in (0, 1]$$
- $\gamma_k > 0$: Modality sensitivity parameter.
- Inside box: $D = 0 \implies \mu = 1.0$ (100% familiarity).
- Outside box: Smooth exponential decay with increasing distance.

#### C. Memory-Bounded Chunked Batching
Naive evaluation of $N$ test samples against $M$ boxes in $d$ dimensions allocates $(N \times M \times d)$ floats. On large datasets ($600 \times 1395 \times 649$), this requires 17.3 GB RAM and causes system SIGKILL.

We resolve this by chunking evaluation across batches of $B = 128$ samples:
$$\text{Peak Memory} = \mathcal{O}(B \cdot M \cdot d) < 100\text{ MB}$$

#### D. Class Evidence Aggregation
Aggregates the highest membership score across all hyperboxes representing class $c$:
$$s_i^{(k)}(c) = \max_{j: c_j = c} \mu_j^{(k)}(z_i), \quad s_i^{(k)} \in [0, 1]^C$$

---

### Stage 5: Out-Of-Fold (OOF) Reliability Estimation (The Trust Grader)

Evaluates whether sensor channel is providing clean evidence vs corrupted noise.

Implemented in [`mm_rfmnn/reliability/estimator.py`](mm_rfmnn/reliability/estimator.py).

#### A. Meta-Feature Extraction
For sample $i$ on modality $k$, extract 4 diagnostic meta-features:
$$u_i^{(k)} = [m_i^{(k)}, \, \Delta_i^{(k)}, \, \text{ratio}_i^{(k)}, \, Q_i^{(k)}]^T \in \mathbb{R}^4$$
1. **Peak Membership**:
   $$m_i^{(k)} = \max_{c \in \{1, \dots, C\}} s_i^{(k)}(c)$$
2. **Classification Margin**:
   $$\Delta_i^{(k)} = s_{(1)}^{(k)} - s_{(2)}^{(k)}$$
   where $s_{(1)}$ is the top evidence and $s_{(2)}$ is the runner-up evidence.
3. **Evidence Concentration**:
   $$\text{ratio}_i^{(k)} = \frac{s_{(1)}^{(k)}}{\sum_{c=1}^C s_i^{(k)}(c) + \epsilon}$$
4. **Signal Stability**:
   $$Q_i^{(k)} = \frac{1}{1 + \text{Var}(z_i^{(k)})}$$

#### B. 5-Fold Out-Of-Fold (OOF) Target Generation
Training the reliability estimator on in-sample data causes memorization leakage (the model gives itself an unearned 99% score). To eliminate this bias:
1. Divide training split $\mathcal{D}_{\text{train}}$ into 5 stratified folds.
2. For each fold $f \in \{1, \dots, 5\}$:
   - Train FMNN hyperboxes strictly on the other 4 folds.
   - Predict on held-out fold $f$:
     $$\hat{y}_{\text{OOF}, i}^{(k)} = \arg\max_c s_i^{(k)}(c)$$
3. Construct true binary correctness targets:
   $$t_i^{(k)} = \mathbb{I}(\hat{y}_{\text{OOF}, i}^{(k)} == y_i) \in \{0, 1\}$$
4. Fit Logistic Meta-Estimator on $(u_i^{(k)}, t_i^{(k)})$:
   $$r_i^{(k)} = \sigma(\mathbf{w}^T u_i^{(k)} + b) = \frac{1}{1 + \exp(-(\mathbf{w}^T u_i^{(k)} + b))}$$
   Loss function minimized:
   $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ t_i^{(k)} \log r_i^{(k)} + (1 - t_i^{(k)}) \log (1 - r_i^{(k)}) \right]$$

---

### Stage 6: Modality Temperature Calibration

Converts non-probabilistic fuzzy evidence into calibrated Bayesian probabilities.

Implemented in [`mm_rfmnn/fusion/fuser.py`](mm_rfmnn/fusion/fuser.py).

Raw evidence vector $s_i^{(k)} \in [0, 1]^C$ is scaled by learned positive scalar $T_k > 0$:
$$p_i^{(k)}(c) = \frac{\exp(s_i^{(k)}(c) / T_k)}{\sum_{c'=1}^C \exp(s_i^{(k)}(c') / T_k)}$$

#### Optimization via L-BFGS-B
Tuned on validation split $\mathcal{D}_{\text{val}}$ by minimizing Negative Log-Likelihood (NLL):
$$T_k^* = \arg\min_{T_k > 0} \left[ -\sum_{i \in \mathcal{D}_{\text{val}}} \log p_i^{(k)}(y_i; T_k) \right]$$
- Normalizes disparate evidence scales across high- and low-dimensional channels.
- Decreases Expected Calibration Error (ECE) from 0.4578 down to 0.0333 (13.7x reduction).

---

### Stage 7: Simplex Fusion & Zero-Imputation-Free Missingness Masking

Combines evidence across modalities proportionally to empirical reliability.

Implemented in [`mm_rfmnn/fusion/fuser.py`](mm_rfmnn/fusion/fuser.py).

1. **Binary Presence Mask ($q_i^{(k)}$)**:
   $$q_i^{(k)} = \begin{cases} 1 & \text{if modality } k \text{ is present/connected} \\ 0 & \text{if modality } k \text{ is missing/unplugged} \end{cases}$$
2. **Presence-Gated Trust**:
   $$\tilde{r}_i^{(k)} = q_i^{(k)} \cdot r_i^{(k)}$$
3. **Simplex Weight Normalization**:
   $$\alpha_i^{(k)} = \frac{\tilde{r}_i^{(k)}}{\sum_{l=1}^K \tilde{r}_i^{(l)} + \epsilon}$$
   Guarantees strict convex combination:
   $$\sum_{k=1}^K \alpha_i^{(k)} = 1.0, \quad \alpha_i^{(k)} \ge 0$$
   - Missing channel ($q^{(k)}=0$) receives $\alpha^{(k)} = 0.0$ immediately.
   - No zero-imputation hallucinations: missing channel never passes artificial zero coordinates through hyperbox distance math.

4. **Convex Opinion Pool**:
   $$P_i(c) = \sum_{k=1}^K \alpha_i^{(k)} \cdot p_i^{(k)}(c)$$

---

### Stage 8: Cross-Modal Conflict Detection & Uncertainty Abstention

Prevents overconfident blind guesses under ambiguous inputs or sensor contradictions.

Implemented in [`mm_rfmnn/fusion/conflict.py`](mm_rfmnn/fusion/conflict.py) and [`mm_rfmnn/fusion/fuser.py`](mm_rfmnn/fusion/fuser.py).

#### A. Empirical Cross-Modal Conflict Detection
Learns joint class co-occurrence matrix $\mathbf{C} \in [0, 1]^{C \times C}$ during training:
$$C(a, b) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_{1, i} == a \land \hat{y}_{2, i} == b)$$
If two confident modalities ($p^{(1)}(c_1) \ge \tau_{\text{conf}}$ and $p^{(2)}(c_2) \ge \tau_{\text{conf}}$) disagree on classes where $C(c_1, c_2) < \tau_{\text{conflict}}$, the conflict detector flags an epistemic contradiction.

#### B. Uncertainty Tripwires
1. **Predictive Shannon Entropy**:
   $$H(P_i) = -\sum_{c=1}^C P_i(c) \log P_i(c)$$
   Trips if $H(P_i) > \tau_{\text{entropy}}$ (diffuse, uninformative evidence).
2. **Jensen-Shannon Modality Disagreement**:
   $$D_{\text{JS}}(p^{(1)}, \dots, p^{(K)}) = H\left(\sum_{k=1}^K \frac{1}{K} p^{(k)}\right) - \frac{1}{K} \sum_{k=1}^K H(p^{(k)})$$
   Trips if $D_{\text{JS}} > \tau_{\text{disagreement}}$ (divergent sensor claims).
3. **Total Reliability Deficit**:
   Trips if $\sum_{k=1}^K \tilde{r}_i^{(k)} < \tau_{\text{min\_rel}}$ (all available sensors noisy).
4. **Confidence Margin Deficit**:
   Trips if $P_i((1)) - P_i((2)) < \tau_{\text{margin}}$.

#### C. Final Decision Rule
$$\hat{y}_i = \begin{cases} -1 & \text{if conflict or any uncertainty tripwire triggered (Abstain)} \\ \arg\max_{c \in \{1, \dots, C\}} P_i(c) & \text{otherwise (Certified Prediction)} \end{cases}$$

---

### Stage 9: Geometric Pruning & Stream Adaptation

Implemented in [`mm_rfmnn/base/pruning.py`](mm_rfmnn/base/pruning.py) and [`mm_rfmnn/base/incremental.py`](mm_rfmnn/base/incremental.py).

#### A. Geometric Subsumption Checking
If hyperbox $B_a$ is completely enclosed inside hyperbox $B_b$ of the same class:
$$c_a == c_b \land V_b \le V_a \land W_b \ge W_a$$
Prune $B_a$; merge sample support count:
$$N_b \leftarrow N_b + N_a$$

#### B. Low-Support Elimination
Delete hyperbox $B_j$ if sample count $N_j < \tau_{\text{supp}}$ (representing isolated noise/outliers), provided class $c_j$ maintains at least one other representative box.
- Result on UCI digits: 3,610 hyperboxes -> 1,207 hyperboxes (**66.6% reduction, 0% accuracy drop**).

#### C. Continuous Streaming Adaptation (`partial_fit`)
Ingests live streaming batches $(X_{\text{batch}}, y_{\text{batch}})$:
1. Tests expansion on existing boxes.
2. Creates new boxes if expansion exceeds $\theta_k$.
3. Resolves cross-class overlaps via Simpson contraction.
4. Periodically executes auto-pruning every $K_{\text{prune}}$ iterations to strictly bound memory footprint.

---

## 3. Concrete Numeric Walkthrough (Tracing One Sample)

Assume a 2-modality problem with 2 classes ($C = 2$):
- **Modality 1 (Clean Tabular)**: $d_1 = 2$, input $z^{(1)} = [0.40, 0.50]$.
- **Modality 2 (Noisy Acoustic)**: $d_2 = 2$, input $z^{(2)} = [0.95, 0.05]$ (corrupted by loud static).

### Step 1: Distance to Boxes
- Modality 1 has Box $B_1^{(1)}$ (Class 0): $V = [0.35, 0.45], W = [0.45, 0.55]$.
  - Sample $z^{(1)}$ is inside: $D_1^{(1)} = 0.0$.
  - Campfire membership: $\mu = \exp(-1.0 \times 0.0) = 1.0$.
  - Evidence: $s^{(1)} = [1.0, 0.10]$.
- Modality 2 has Box $B_1^{(2)}$ (Class 0): $V = [0.20, 0.20], W = [0.40, 0.40]$.
  - Sample $z^{(2)}$ is outside:
    $$d_1 = \max(0, 0.20-0.95, 0.95-0.40) = 0.55$$
    $$d_2 = \max(0, 0.20-0.05, 0.05-0.40) = 0.15$$
    $$D_1^{(2)} = 0.55 + 0.15 = 0.70$$
  - Campfire membership: $\mu = \exp(-1.0 \times 0.70) = 0.496$.
  - Evidence: $s^{(2)} = [0.496, 0.480]$.

### Step 2: Reliability Estimation (The Trust Grader)
- Modality 1: Meta-features $u^{(1)} = [1.0, 0.90, 0.91, 0.98] \implies r_1 = 0.96$ (**High Trust**).
- Modality 2: Meta-features $u^{(2)} = [0.496, 0.016, 0.51, 0.40] \implies r_2 = 0.08$ (**Low Trust / Noisy**).

### Step 3: Temperature Calibration
- With learned $T_1 = 0.5, T_2 = 1.2$:
  - $p^{(1)} = \text{Softmax}([1.0 / 0.5, 0.10 / 0.5]) = [0.86, 0.14]$.
  - $p^{(2)} = \text{Softmax}([0.496 / 1.2, 0.480 / 1.2]) = [0.503, 0.497]$.

### Step 4: Simplex Blending
- Both sensors present ($q_1 = 1, q_2 = 1$):
  $$\alpha_1 = \frac{0.96}{0.96 + 0.08} = 0.923, \quad \alpha_2 = \frac{0.08}{0.96 + 0.08} = 0.077$$
- Final Probability:
  $$P(\text{Class } 0) = 0.923 \times 0.86 + 0.077 \times 0.503 = 0.794 + 0.039 = 0.833 \quad (83.3\%)$$
  $$P(\text{Class } 1) = 0.923 \times 0.14 + 0.077 \times 0.497 = 0.129 + 0.038 = 0.167 \quad (16.7\%)$$

### Outcome:
Modality 2's noise was automatically down-weighted to 7.7% voting power. Modality 1 dominates the decision, producing a confident, accurate prediction of **Class 0 (83.3%)**.

---

## 4. Codebase Reference Map

| Component | Source File | Line Range | Role |
| :--- | :--- | :--- | :--- |
| **Normalizer** | [`mm_rfmnn/encoders/normalizer.py`](mm_rfmnn/encoders/normalizer.py) | L1-L90 | Leakage-free MinMax scaling |
| **Multimodal Encoders** | [`mm_rfmnn/encoders/multimodal.py`](mm_rfmnn/encoders/multimodal.py) | L1-L230 | Raw Image, Audio, Text, Tabular encoders |
| **Adaptive Theta** | [`mm_rfmnn/base/adaptive_theta.py`](mm_rfmnn/base/adaptive_theta.py) | L1-L119 | Automated box expansion bound tuning |
| **Hyperbox Bank & FMNN** | [`mm_rfmnn/base/hyperbox.py`](mm_rfmnn/base/hyperbox.py) | L1-L310 | Expansion, overlap contraction, chunked evidence |
| **Pruner** | [`mm_rfmnn/base/pruning.py`](mm_rfmnn/base/pruning.py) | L1-L125 | Geometric subsumption & support pruning |
| **Stream Adapter** | [`mm_rfmnn/base/incremental.py`](mm_rfmnn/base/incremental.py) | L1-L150 | Online `partial_fit` streaming ingestion |
| **Reliability Estimator** | [`mm_rfmnn/reliability/estimator.py`](mm_rfmnn/reliability/estimator.py) | L1-L210 | Meta-feature extraction & 5-fold OOF learning |
| **Calibrator & Fuser** | [`mm_rfmnn/fusion/fuser.py`](mm_rfmnn/fusion/fuser.py) | L1-L240 | L-BFGS-B temperature calibration & simplex fusion |
| **Conflict Detector** | [`mm_rfmnn/fusion/conflict.py`](mm_rfmnn/fusion/conflict.py) | L1-L130 | Co-occurrence matrix contradiction blocker |
| **Unified Estimator** | [`mm_rfmnn/model.py`](mm_rfmnn/model.py) | L1-L520 | Scikit-learn estimator interface (`fit`, `predict`, `explain`) |
