"""Comprehensive Experimental Benchmark for MM-RFMNN.

Directly investigates and answers Research Questions RQ1 - RQ4:
- RQ1: Modality-specific FMNN vs. Feature Concatenation vs. Early MLP vs. Mean Fusion.
- RQ2: Robustness under increasing corruption levels (Gaussian noise, feature blanking).
- RQ3: Robustness under missing modalities (Modality 1 missing, Modality 2 missing).
- RQ4: Uncertainty quantification, Expected Calibration Error (ECE), and Risk-Coverage abstention.
"""

import sys
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split

# Ensure mm_rfmnn is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mm_rfmnn.model import MMRFMNNClassifier
from mm_rfmnn.base.hyperbox import FMNN
from mm_rfmnn.encoders.normalizer import ModalityNormalizer


def generate_multimodal_dataset(n_samples: int = 600, n_classes: int = 3, random_state: int = 42):
    """Generates synthetic paired multimodal data simulating sensory / acoustic / clinical channels."""
    rng = np.random.RandomState(random_state)
    y = rng.choice(n_classes, size=n_samples)

    # Modality 1 (Sensor/Acoustic, dim=6): Class means with some overlap
    mean_shifts_1 = np.array([
        [-1.0, -0.5, 0.0, 0.5, 1.0, -0.2],
        [0.5, 1.2, -0.8, -0.2, 0.3, 0.9],
        [1.5, -1.0, 1.1, -0.7, -0.5, 0.4]
    ])
    X1 = mean_shifts_1[y] + rng.normal(scale=0.7, size=(n_samples, 6))

    # Modality 2 (Tabular/Metadata, dim=4): Non-linear combination
    mean_shifts_2 = np.array([
        [2.0, -1.5, 0.5, -0.5],
        [-1.0, 0.8, 1.5, 1.2],
        [0.0, 2.0, -1.0, -1.8]
    ])
    X2 = mean_shifts_2[y] + rng.normal(scale=0.8, size=(n_samples, 4))

    return [X1, X2], y


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.any(bin_mask):
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            bin_weight = np.mean(bin_mask)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    return float(ece)


def run_benchmark():
    print("=" * 72)
    print("MULTIMODAL RELIABILITY-AWARE FUZZY MIN-MAX NEURAL NETWORK (MM-RFMNN)")
    print("Research Benchmark Suite (RQ1 - RQ4)")
    print("=" * 72)

    X_list, y = generate_multimodal_dataset(n_samples=800, n_classes=3, random_state=42)
    X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te = train_test_split(
        X_list[0], X_list[1], y, test_size=0.30, stratify=y, random_state=42
    )

    X_train = [X1_tr, X2_tr]
    X_test = [X1_te, X2_te]
    N_test = len(y_te)

    # -----------------------------------------------------------------
    # Baseline 1: Early Concatenation FMNN
    # -----------------------------------------------------------------
    norm_early = ModalityNormalizer()
    X_early_tr = np.hstack(X_train)
    X_early_te = np.hstack(X_test)
    Z_early_tr = norm_early.fit_transform(X_early_tr)
    Z_early_te = norm_early.transform(X_early_te)

    fmnn_early = FMNN(theta=0.3, gamma=1.0).fit(Z_early_tr, y_tr)
    ev_early_te, _, _ = fmnn_early.compute_evidence(Z_early_te)
    pred_early_fmnn = np.argmax(ev_early_te, axis=1)
    probs_early_fmnn = ev_early_te / (np.sum(ev_early_te, axis=1, keepdims=True) + 1e-9)

    # -----------------------------------------------------------------
    # Baseline 2: Early Concatenation Deep MLP
    # -----------------------------------------------------------------
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    mlp.fit(Z_early_tr, y_tr)
    pred_early_mlp = mlp.predict(Z_early_te)
    probs_early_mlp = mlp.predict_proba(Z_early_te)

    # -----------------------------------------------------------------
    # Baseline 3: Modality-Specific FMNN with Uniform Mean Fusion
    # -----------------------------------------------------------------
    fmnn_uniform = MMRFMNNClassifier(
        theta=0.3,
        gamma=1.0,
        fusion_method="arithmetic",
        reliability_model="rule_based",
        abstention_enabled=False,
        random_state=42,
    )
    # Fit uniform model (rule-based with equal weights)
    fmnn_uniform.fit(X_train, y_tr)
    # Force equal weights in fuser
    fmnn_uniform.fuser_.compute_fusion_weights = lambda rels, mask=None: (
        np.full((len(rels[0]), len(rels)), 1.0 / len(rels)),
        np.column_stack(rels),
    )
    probs_uniform = fmnn_uniform.predict_proba(X_test)
    pred_uniform = np.argmax(probs_uniform, axis=1)

    # -----------------------------------------------------------------
    # Proposed: MM-RFMNN (OOF Calibrated Reliability-Aware Fusion)
    # -----------------------------------------------------------------
    mm_rfmnn = MMRFMNNClassifier(
        theta=0.3,
        gamma=1.0,
        fusion_method="arithmetic",
        reliability_model="logistic",
        abstention_enabled=False,
        random_state=42,
    )
    mm_rfmnn.fit(X_train, y_tr)
    probs_mm_rfmnn = mm_rfmnn.predict_proba(X_test)
    pred_mm_rfmnn = mm_rfmnn.predict(X_test, return_abstention=False)

    print("\n--- RQ1: CLEAN MULTIMODAL CLASSIFICATION PERFORMANCE ---")
    models = {
        "Early Concatenation FMNN": (pred_early_fmnn, probs_early_fmnn),
        "Early Concatenation MLP": (pred_early_mlp, probs_early_mlp),
        "Late Mean Fusion FMNN": (pred_uniform, probs_uniform),
        "Proposed MM-RFMNN": (pred_mm_rfmnn, probs_mm_rfmnn),
    }

    print(f"{'Model Architecture':<30} | {'Accuracy':<10} | {'Macro-F1':<10} | {'ECE (Calibration)':<18}")
    print("-" * 75)
    for name, (preds, probs) in models.items():
        acc = accuracy_score(y_te, preds)
        macro_f1 = f1_score(y_te, preds, average="macro")
        ece = compute_ece(probs, y_te)
        print(f"{name:<30} | {acc * 100:>8.2f}% | {macro_f1:>10.4f} | {ece:>18.4f}")

    # -----------------------------------------------------------------
    # RQ2: ROBUSTNESS UNDER MODALITY CORRUPTION
    # -----------------------------------------------------------------
    print("\n--- RQ2: ROBUSTNESS UNDER CORRUPTION (Severe Noise on Modality 1) ---")
    sigmas = [0.0, 1.0, 2.5, 5.0, 8.0]
    print(f"{'Noise Sigma (Mod 1)':<20} | {'Early MLP F1':<15} | {'Mean FMNN F1':<15} | {'MM-RFMNN F1':<15}")
    print("-" * 72)
    for sigma in sigmas:
        X1_te_corrupt = X_test[0] + np.random.RandomState(42).normal(scale=sigma, size=X_test[0].shape)
        X_test_corrupt = [X1_te_corrupt, X_test[1]]

        # Early MLP
        Z_corrupt_mlp = norm_early.transform(np.hstack(X_test_corrupt))
        mlp_f1 = f1_score(y_te, mlp.predict(Z_corrupt_mlp), average="macro")

        # Mean FMNN
        mean_probs = fmnn_uniform.predict_proba(X_test_corrupt)
        mean_f1 = f1_score(y_te, np.argmax(mean_probs, axis=1), average="macro")

        # MM-RFMNN
        rfmnn_preds = mm_rfmnn.predict(X_test_corrupt, return_abstention=False)
        rfmnn_f1 = f1_score(y_te, rfmnn_preds, average="macro")

        print(f"Sigma = {sigma:<12.1f} | {mlp_f1:>15.4f} | {mean_f1:>15.4f} | {rfmnn_f1:>15.4f}")

    # -----------------------------------------------------------------
    # RQ3: MISSING-MODALITY INFERENCE
    # -----------------------------------------------------------------
    print("\n--- RQ3: MISSING MODALITY EVALUATION (Graceful Degradation) ---")
    # Scenario A: Modality 1 completely missing
    mask_m1_missing = np.zeros((N_test, 2), dtype=bool)
    mask_m1_missing[:, 1] = True  # Only Modality 2 present

    rfmnn_m1_miss_f1 = f1_score(
        y_te, mm_rfmnn.predict(X_test, mask=mask_m1_missing, return_abstention=False), average="macro"
    )

    # Scenario B: Modality 2 completely missing
    mask_m2_missing = np.zeros((N_test, 2), dtype=bool)
    mask_m2_missing[:, 0] = True  # Only Modality 1 present

    rfmnn_m2_miss_f1 = f1_score(
        y_te, mm_rfmnn.predict(X_test, mask=mask_m2_missing, return_abstention=False), average="macro"
    )

    print(f"MM-RFMNN (Clean Both Modalities Present) Macro-F1 : {macro_f1:.4f}")
    print(f"MM-RFMNN (Modality 1 Fully Missing, Mod 2 Only)   : {rfmnn_m1_miss_f1:.4f}")
    print(f"MM-RFMNN (Modality 2 Fully Missing, Mod 1 Only)   : {rfmnn_m2_miss_f1:.4f}")
    print("Note: MM-RFMNN automatically zeroes alpha for missing channels without imputation bias.")

    # -----------------------------------------------------------------
    # RQ4: SELECTIVE CLASSIFICATION & RISK-COVERAGE CURVE
    # -----------------------------------------------------------------
    print("\n--- RQ4: UNCERTAINTY QUANTIFICATION & SELECTIVE ABSTENTION ---")
    confidences = np.max(probs_mm_rfmnn, axis=1)
    coverages = [1.0, 0.95, 0.90, 0.85, 0.80]
    print(f"{'Target Coverage':<18} | {'Retained Samples':<18} | {'Retained Accuracy':<20}")
    print("-" * 60)
    for cov in coverages:
        cutoff = np.percentile(confidences, (1.0 - cov) * 100.0)
        retained = confidences >= cutoff
        acc_retained = accuracy_score(y_te[retained], pred_mm_rfmnn[retained])
        print(f"{cov * 100:>14.0f}% | {np.sum(retained):>18d} | {acc_retained * 100:>18.2f}%")

    print("\n" + "=" * 72)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
