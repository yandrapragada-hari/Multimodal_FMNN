"""Real-World Multimodal Benchmark on UCI Multiple Features Dataset (mfeat).

Evaluates MM-RFMNN against standard baselines across:
- RQ1: Clean Multi-Modality Classification & Probability Calibration (ECE).
- RQ2: Modality Corruption Robustness (severe noise on high-dimensional channels).
- RQ3: Missing Modality Inference (zero-imputation-free degradation across 1 to 5 missing channels).
- RQ4: Uncertainty-Aware Selective Abstention (Risk-Coverage profile).
- Next Stage (Phase 3): Hyperbox Pruning Efficiency and Streaming Online Adaptation.
"""

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mm_rfmnn.base.hyperbox import FMNN
from mm_rfmnn.base.incremental import OnlineStreamAdapter
from mm_rfmnn.datasets.loader import load_uci_mfeat
from mm_rfmnn.encoders.normalizer import ModalityNormalizer
from mm_rfmnn.model import MMRFMNNClassifier


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.any(bin_mask):
            bin_acc = float(np.mean(accuracies[bin_mask]))
            bin_conf = float(np.mean(confidences[bin_mask]))
            bin_weight = float(np.mean(bin_mask))
            ece += bin_weight * abs(bin_acc - bin_conf)
    return float(ece)


def run_real_dataset_benchmark() -> None:
    print("=" * 78)
    print("MM-RFMNN REAL-WORLD MULTIMODAL BENCHMARK: UCI MULTIPLE FEATURES (mfeat)")
    print("Heterogeneous 6-Modality Dataset: 2,000 samples, 10 digit classes")
    print("=" * 78)

    # 1. Load Real Dataset
    X_modalities, y, mod_names = load_uci_mfeat(return_modality_names=True)
    K = len(X_modalities)
    print(f"\nLoaded {K} modalities:")
    for name, X_k in zip(mod_names, X_modalities):
        print(f"  - Modality '{name:<14}': shape {X_k.shape}")

    # 2. Stratified Train / Test Partition (70% train, 30% test)
    # Prevent leakage: split indices first
    indices = np.arange(len(y))
    idx_tr, idx_te = train_test_split(indices, test_size=0.30, stratify=y, random_state=42)

    X_train: list[np.ndarray] = [X_k[idx_tr] for X_k in X_modalities]
    y_train = y[idx_tr]
    X_test: list[np.ndarray] = [X_k[idx_te] for X_k in X_modalities]
    y_test = y[idx_te]
    N_test = len(y_test)

    # -----------------------------------------------------------------
    # Baseline 1: Early Concatenation FMNN
    # -----------------------------------------------------------------
    norm_early = ModalityNormalizer(method="minmax")
    X_concat_tr = np.hstack(X_train)
    X_concat_te = np.hstack(X_test)
    Z_concat_tr = norm_early.fit_transform(X_concat_tr)
    Z_concat_te = norm_early.transform(X_concat_te)

    fmnn_early = FMNN(theta=0.35, gamma=1.0).fit(Z_concat_tr, y_train)
    ev_early_te, _, _ = fmnn_early.compute_evidence(Z_concat_te)
    pred_early_fmnn = np.argmax(ev_early_te, axis=1)
    probs_early_fmnn = ev_early_te / (np.sum(ev_early_te, axis=1, keepdims=True) + 1e-9)

    # -----------------------------------------------------------------
    # Baseline 2: Early Concatenation MLP
    # -----------------------------------------------------------------
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42)
    mlp.fit(Z_concat_tr, y_train)
    pred_early_mlp = mlp.predict(Z_concat_te)
    probs_early_mlp = mlp.predict_proba(Z_concat_te)

    # -----------------------------------------------------------------
    # Baseline 3: Late Mean Fusion FMNN (Equal Weights)
    # -----------------------------------------------------------------
    fmnn_uniform = MMRFMNNClassifier(
        theta=0.35,
        gamma=1.0,
        fusion_method="arithmetic",
        reliability_model="rule_based",
        abstention_enabled=False,
        random_state=42,
    )
    fmnn_uniform.fit(X_train, y_train)
    # Overwrite fusion weights to uniform 1/K
    fmnn_uniform.fuser_.compute_fusion_weights = lambda rels, mask=None: (
        np.full((len(rels[0]), len(rels)), 1.0 / len(rels)),
        np.column_stack(rels),
    )
    probs_uniform = fmnn_uniform.predict_proba(X_test)
    pred_uniform = np.argmax(probs_uniform, axis=1)

    # -----------------------------------------------------------------
    # Proposed: MM-RFMNN (OOF Calibrated Reliability-Aware Simplex Fusion)
    # -----------------------------------------------------------------
    mm_rfmnn = MMRFMNNClassifier(
        theta=0.35,
        gamma=1.0,
        fusion_method="arithmetic",
        reliability_model="logistic",
        abstention_enabled=False,
        random_state=42,
    )
    mm_rfmnn.fit(X_train, y_train)
    probs_mm_rfmnn = mm_rfmnn.predict_proba(X_test)
    pred_mm_rfmnn = mm_rfmnn.predict(X_test, return_abstention=False)

    # -----------------------------------------------------------------
    # RQ1: Clean Benchmark
    # -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("RQ1: CLEAN MULTIMODAL CLASSIFICATION PERFORMANCE ON UCI mfeat")
    print("-" * 78)
    models: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Early Concatenation FMNN": (pred_early_fmnn, probs_early_fmnn),
        "Early Concatenation MLP": (pred_early_mlp, probs_early_mlp),
        "Late Mean Fusion FMNN": (pred_uniform, probs_uniform),
        "Proposed MM-RFMNN": (pred_mm_rfmnn, probs_mm_rfmnn),
    }

    print(f"{'Model Architecture':<30} | {'Accuracy':<10} | {'Macro-F1':<10} | {'ECE (Calibration)':<18}")
    print("-" * 78)
    for name, (preds, probs) in models.items():
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        ece = compute_ece(probs, y_test)
        print(f"{name:<30} | {acc * 100:>8.2f}% | {macro_f1:>10.4f} | {ece:>18.4f}")

    # -----------------------------------------------------------------
    # RQ2: Corruption Robustness
    # -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("RQ2: CORRUPTION ROBUSTNESS (Severe Gaussian Noise on Pixel & Fourier)")
    print("-" * 78)
    sigmas = [0.0, 1.0, 2.5, 5.0, 8.0]
    print(f"{'Noise Sigma':<16} | {'Early MLP F1':<15} | {'Mean FMNN F1':<15} | {'MM-RFMNN F1':<15}")
    print("-" * 70)

    # Pixel is index 4, Fourier is index 0
    corrupt_indices = [0, 4]
    for sigma in sigmas:
        X_test_corrupt = [X_k.copy() for X_k in X_test]
        rng = np.random.RandomState(42)
        for c_idx in corrupt_indices:
            noise = rng.normal(scale=sigma * np.std(X_train[c_idx]), size=X_test[c_idx].shape)
            X_test_corrupt[c_idx] += noise

        # Early MLP
        Z_corrupt_mlp = norm_early.transform(np.hstack(X_test_corrupt))
        mlp_f1 = f1_score(y_test, mlp.predict(Z_corrupt_mlp), average="macro")

        # Mean FMNN
        mean_probs = fmnn_uniform.predict_proba(X_test_corrupt)
        mean_f1 = f1_score(y_test, np.argmax(mean_probs, axis=1), average="macro")

        # MM-RFMNN
        rfmnn_f1 = f1_score(
            y_test, mm_rfmnn.predict(X_test_corrupt, return_abstention=False), average="macro"
        )
        print(f"Sigma = {sigma:<8.1f} | {mlp_f1:>15.4f} | {mean_f1:>15.4f} | {rfmnn_f1:>15.4f}")

    # -----------------------------------------------------------------
    # RQ3: Missing Modalities
    # -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("RQ3: MISSING MODALITY EVALUATION (Zero-Imputation-Free Degradation)")
    print("-" * 78)

    # Scenario 1: All 6 channels present
    mask_full = np.ones((N_test, K), dtype=bool)
    f1_full = f1_score(
        y_test, mm_rfmnn.predict(X_test, mask=mask_full, return_abstention=False), average="macro"
    )

    # Scenario 2: Fourier and Factors dropped (4 modalities remaining)
    mask_drop_2 = np.ones((N_test, K), dtype=bool)
    mask_drop_2[:, [0, 1]] = False
    f1_drop_2 = f1_score(
        y_test, mm_rfmnn.predict(X_test, mask=mask_drop_2, return_abstention=False), average="macro"
    )

    # Scenario 3: 4 modalities dropped, only Karhunen & Morphological present
    mask_drop_4 = np.zeros((N_test, K), dtype=bool)
    mask_drop_4[:, [2, 3]] = True
    f1_drop_4 = f1_score(
        y_test, mm_rfmnn.predict(X_test, mask=mask_drop_4, return_abstention=False), average="macro"
    )

    # Scenario 4: Single modality only (Karhunen only)
    mask_single = np.zeros((N_test, K), dtype=bool)
    mask_single[:, 2] = True
    f1_single = f1_score(
        y_test, mm_rfmnn.predict(X_test, mask=mask_single, return_abstention=False), average="macro"
    )

    print(f"MM-RFMNN (All 6 Modalities Present)       Macro-F1: {f1_full:.4f}")
    print(f"MM-RFMNN (4 Modalities: Fourier & Factors Dropped): {f1_drop_2:.4f}")
    print(f"MM-RFMNN (2 Modalities: Karhunen & Morph Only)    : {f1_drop_4:.4f}")
    print(f"MM-RFMNN (1 Modality: Karhunen Only)              : {f1_single:.4f}")

    # -----------------------------------------------------------------
    # RQ4: Selective Classification
    # -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("RQ4: SELECTIVE CLASSIFICATION RISK-COVERAGE PROFILE")
    print("-" * 78)
    confidences = np.max(probs_mm_rfmnn, axis=1)
    coverages = [1.0, 0.95, 0.90, 0.85, 0.80]
    print(f"{'Target Coverage':<18} | {'Retained Samples':<18} | {'Retained Accuracy':<20}")
    print("-" * 62)
    for cov in coverages:
        cutoff = float(np.percentile(confidences, (1.0 - cov) * 100.0))
        retained = confidences >= cutoff
        acc_retained = accuracy_score(y_test[retained], pred_mm_rfmnn[retained])
        print(f"{cov * 100:>14.0f}% | {int(np.sum(retained)):>18d} | {acc_retained * 100:>18.2f}%")

    # -----------------------------------------------------------------
    # NEXT STAGE (Phase 3): Hyperbox Pruning & Stream Adaptation
    # -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("PHASE 3: NEXT STAGE CAPABILITIES (Hyperbox Pruning & Online Stream Adaptation)")
    print("-" * 78)

    # Part A: Pruning
    initial_stats = mm_rfmnn.get_hyperbox_stats()
    prune_report = mm_rfmnn.prune_hyperboxes(min_samples=2, overlap_threshold=0.95)
    final_stats = mm_rfmnn.get_hyperbox_stats()
    pruned_preds = mm_rfmnn.predict(X_test, return_abstention=False)
    pruned_acc = accuracy_score(y_test, pruned_preds)
    pruned_f1 = f1_score(y_test, pruned_preds, average="macro")

    compression_ratio = (1.0 - (final_stats["total_boxes"] / max(initial_stats["total_boxes"], 1))) * 100.0
    total_low_support = sum(m.get("pruned_low_support", 0) for m in prune_report["modalities"])
    total_redundant = sum(m.get("pruned_redundant", 0) for m in prune_report["modalities"])
    print("Hyperbox Pruning Analysis:")
    print(f"  - Initial Total Hyperboxes : {initial_stats['total_boxes']}")
    print(f"  - Pruned Low-Support Boxes : {total_low_support}")
    print(f"  - Pruned Redundant Boxes   : {total_redundant}")
    print(f"  - Final Compact Hyperboxes : {final_stats['total_boxes']} ({compression_ratio:.1f}% reduction)")
    print(f"  - Post-Pruning Accuracy    : {pruned_acc * 100:.2f}% (Macro-F1: {pruned_f1:.4f})")

    # Part B: Online Stream Adaptation
    stream_clf = MMRFMNNClassifier(
        theta=0.35,
        gamma=1.0,
        fusion_method="arithmetic",
        abstention_enabled=False,
        random_state=42,
    )
    adapter = OnlineStreamAdapter(
        max_boxes_per_modality=150,
        auto_prune_interval=3,
        prune_min_samples=2,
    )

    batch_size = 100
    n_batches = 5
    for b in range(n_batches):
        b_slice = slice(b * batch_size, (b + 1) * batch_size)
        X_batch = [X_tr[b_slice] for X_tr in X_train]
        y_batch = y_train[b_slice]
        adapter.adapt_batch(stream_clf, X_batch, y_batch)

    stream_summary = adapter.get_stream_summary()
    stream_preds = stream_clf.predict(X_test, return_abstention=False)
    stream_acc = accuracy_score(y_test, stream_preds)

    print("\nOnline Stream Adaptation Analysis:")
    print(f"  - Batches Streamed         : {stream_summary['batches_processed']}")
    print(f"  - Samples Processed        : {stream_summary['total_samples_seen']}")
    print(f"  - Auto-Prune Interventions : {stream_summary['pruning_events_count']}")
    print(f"  - Online Test Accuracy     : {stream_acc * 100:.2f}%")
    print("=" * 78)
    print("REAL-WORLD MULTIMODAL BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    run_real_dataset_benchmark()
