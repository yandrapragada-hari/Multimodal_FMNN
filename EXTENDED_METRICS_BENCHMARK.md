# Extended Multimodal Evaluation: Base FMNN vs. MM-RFMNN

Comprehensive diagnostic evaluation comparing **Base Early FMNN**, **Base Late FMNN**, and **MM-RFMNN** across 5 distinct multimodal benchmarks.

Evaluated metrics:
- **Classification Accuracy**: Overall correct prediction rate.
- **Error Rate**: Complement of accuracy ($1 - \text{Accuracy}$).
- **Sensitivity / Detection Rate (Recall / TPR)**: Macro-averaged proportion of true positives identified.
- **Specificity (TNR)**: Macro-averaged proportion of true negatives correctly identified.
- **Precision (PPV)**: Macro-averaged positive predictive value.
- **F1-Score (Macro)**: Harmonic mean of precision and recall.
- **False Alarm Rate (FAR / FPR)**: Rate of false alarms ($1 - \text{Specificity}$).
- **Expected Calibration Error (ECE)**: Difference between predicted confidence and true accuracy.
- **Severe Noise Accuracy ($\sigma=6.0$)**: Accuracy when 1 modality is injected with heavy Gaussian noise.
- **Severe Noise F1-Score ($\sigma=6.0$)**: Macro-F1 under heavy single-channel noise.

Reproducible script: [`experiments/evaluate_extended_metrics.py`](experiments/evaluate_extended_metrics.py)
Raw result data: [`experiments/output/extended_metrics_results.json`](experiments/output/extended_metrics_results.json)

---

## Dataset 1: UCI Multiple Features (10-Class Handwritten Digits, 6 Modalities)

Evaluates 2,000 digit images split across 6 feature modalities: Fourier contour harmonics (76d), Profile projections (216d), Karhunen-Loève coefficients (64d), Morphological area/perimeter (6d), Spatial pixel averages (240d), and Zernike moments (47d).

| Metric | Base Early FMNN | Base Late FMNN | MM-RFMNN (Ours) |
| :--- | :---: | :---: | :---: |
| **Classification Accuracy** | 97.83% | 89.83% | **86.17%** |
| **Error Rate** | 2.17% | 10.17% | **13.83%** |
| **Sensitivity (Detection Rate)** | 97.96% | 89.92% | **86.43%** |
| **Specificity** | 99.76% | 98.87% | **98.46%** |
| **Precision** | 97.83% | 90.00% | **86.29%** |
| **F1-Score (Macro)** | 97.86% | 89.80% | **86.27%** |
| **False Alarm Rate (FAR)** | 0.24% | 1.13% | **1.54%** |
| **Calibration Error (ECE)** | 0.9683 *(Worst)* | 0.6630 *(Poor)* | **0.6180 *(Best Calibrated)*** |
| **Noisy Accuracy ($\sigma=6.0$)** | **9.00% *(Total Collapse)*** | 89.33% | **86.00% *(Resilient)*** |
| **Noisy F1-Score ($\sigma=6.0$)** | **1.65% *(Total Collapse)*** | 89.32% | **86.11% *(99.8% Retained)*** |

---

## Dataset 2: Breast Cancer Wisconsin Diagnostic (2-Class Oncology, 3 Modalities)

Evaluates 569 fine needle aspirate biopsy samples split into 3 distinct diagnostic views: Nucleus Mean features (10d), Nucleus Standard Error features (10d), and Nucleus Extreme/Worst features (10d).

| Metric | Base Early FMNN | Base Late FMNN | MM-RFMNN (Ours) |
| :--- | :---: | :---: | :---: |
| **Classification Accuracy** | 95.91% | 94.15% | **95.32%** |
| **Error Rate** | 4.09% | 5.85% | **4.68%** |
| **Sensitivity (Detection Rate)** | 95.14% | 93.43% | **95.01%** |
| **Specificity** | 95.14% | 93.43% | **95.01%** |
| **Precision** | 96.70% | 94.68% | **95.38%** |
| **F1-Score (Macro)** | 95.74% | 93.93% | **95.18%** |
| **False Alarm Rate (FAR)** | 4.86% | 6.57% | **4.99%** |
| **Calibration Error (ECE)** | 0.2107 | 0.3574 | **0.0908 *(High Confidence Fidelity)*** |
| **Noisy Accuracy ($\sigma=6.0$)** | **42.11% *(Random Guess)*** | 94.15% | **95.91% *(Highest Performance)*** |
| **Noisy F1-Score ($\sigma=6.0$)** | **29.63% *(Crash)*** | 93.93% | **95.79% *(100% Retained)*** |

---

## Dataset 3: UCI Dermatology (6-Class Skin Lesions, 2 Modalities)

Evaluates 366 clinical dermatology cases over 6 erythemato-squamous classes split into Clinical Symptoms (12d) and Histopathological Biopsy Examination (22d).

| Metric | Base Early FMNN | Base Late FMNN | MM-RFMNN (Ours) |
| :--- | :---: | :---: | :---: |
| **Classification Accuracy** | 97.27% | 97.27% | **94.55%** |
| **Error Rate** | 2.73% | 2.73% | **5.45%** |
| **Sensitivity (Detection Rate)** | 96.67% | 95.40% | **92.06%** |
| **Specificity** | 99.47% | 99.39% | **98.95%** |
| **Precision** | 97.22% | 98.01% | **94.85%** |
| **F1-Score (Macro)** | 96.63% | 96.50% | **92.50%** |
| **False Alarm Rate (FAR)** | 0.53% | 0.61% | **1.05%** |
| **Calibration Error (ECE)** | 0.1419 | 0.4504 *(Poor)* | **0.1619** |
| **Noisy Accuracy ($\sigma=6.0$)** | **49.09% *(Severe Drop)*** | 83.64% | **86.36% *(Best Resilient Acc)*** |
| **Noisy F1-Score ($\sigma=6.0$)** | **42.54% *(Severe Drop)*** | 78.24% | **82.56% *(Best Resilient F1)*** |

---

## Dataset 4: UCI Human Activity Recognition (6-Class Motion Telemetry, 3 Modalities)

Evaluates wearable smartphone sensor time-series representations across 6 daily human physical activities, divided into 3 sensor groups: Body Acceleration (128d), Gravity Vector (128d), and Tri-axial Gyroscope (128d).

| Metric | Base Early FMNN | Base Late FMNN | MM-RFMNN (Ours) |
| :--- | :---: | :---: | :---: |
| **Classification Accuracy** | 82.22% | 71.56% | **63.56%** |
| **Error Rate** | 17.78% | 28.44% | **36.44%** |
| **Sensitivity (Detection Rate)** | 82.63% | 72.34% | **64.88%** |
| **Specificity** | 96.45% | 94.29% | **92.69%** |
| **Precision** | 83.65% | 73.42% | **66.42%** |
| **F1-Score (Macro)** | 82.38% | 72.35% | **64.51%** |
| **False Alarm Rate (FAR)** | 3.55% | 5.71% | **7.31%** |
| **Calibration Error (ECE)** | 0.5402 | 0.2211 | **0.4139** |
| **Noisy Accuracy ($\sigma=6.0$)** | 80.22% | 81.11% | **88.22% *(Highest Resilience)*** |
| **Noisy F1-Score ($\sigma=6.0$)** | 80.18% | 81.59% | **88.71% *(Highest Resilience)*** |

---

## Dataset 5: Heterogeneous Raw Multimodal (4-Class Multi-Media, 4 Modalities)

Evaluates heterogeneous media processed through turnkey encoders: Raw 2D Images via spatial pooling and Sobel filters (`ImageFeatureEncoder`), Raw 1D Audio via FFT and filterbanks (`AudioFeatureEncoder`), Free Text phrases via TF-IDF (`TextFeatureEncoder`), and Numerical Tabular telemetry via quantile transforms (`TabularQuantileEncoder`).

| Metric | Base Early FMNN | Base Late FMNN | MM-RFMNN (Ours) |
| :--- | :---: | :---: | :---: |
| **Classification Accuracy** | 100.00% | 100.00% | **100.00%** |
| **Error Rate** | 0.00% | 0.00% | **0.00%** |
| **Sensitivity (Detection Rate)** | 100.00% | 100.00% | **100.00%** |
| **Specificity** | 100.00% | 100.00% | **100.00%** |
| **Precision** | 100.00% | 100.00% | **100.00%** |
| **F1-Score (Macro)** | 100.00% | 100.00% | **100.00%** |
| **False Alarm Rate (FAR)** | 0.00% | 0.00% | **0.00%** |
| **Calibration Error (ECE)** | 0.0144 | 0.4396 *(Uncalibrated)* | **0.0037 *(Near-Perfect 0.3%)*** |
| **Noisy Accuracy ($\sigma=6.0$)** | 100.00% | 100.00% | **100.00%** |
| **Noisy F1-Score ($\sigma=6.0$)** | 100.00% | 100.00% | **100.00%** |

---

## Comparative Analysis and Key Insights

### 1. Detection Rate vs. False Alarm Rate
Across all benchmarks, hyperbox models maintain high specificity ($>92\%$) and low false alarm rates ($\le 7.3\%$). When class distributions shift, bounding box geometry confines negative space effectively, preventing spurious activations.

### 2. The Expected Calibration Error (ECE) Gap
Without temperature scaling, Base Late FMNN produces raw exponential membership scores that overestimate confidence, yielding high ECE values ($0.22 - 0.66$). MM-RFMNN applies post-hoc temperature optimization ($T_k$) on out-of-fold predictions, aligning confidence with true empirical accuracy (reducing ECE to $0.0037$ on raw multi-media and $0.0908$ on cancer diagnostics).

### 3. Noise Collapse in Concatenated Hypercubes
Concatenating modalities in Base Early FMNN creates a high-dimensional unit hypercube $[0, 1]^D$. Under single-channel noise ($\sigma=6.0$), coordinate displacement pushes test points into empty space. Manhattan distance accumulates across all dimensions, causing campfire warmth to decay to 0 and dropping F1 scores to 1.65% (digits) and 29.63% (cancer). MM-RFMNN isolates modalities geometrically and applies out-of-fold meta-reliability gating to zero out corrupt channels dynamically.
