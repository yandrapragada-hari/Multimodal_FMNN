"""Multi-Dataset Benchmark Suite for MM-RFMNN.

Evaluates MM-RFMNN and competing baselines across 5 diverse multimodal datasets:
1. UCI Multiple Features (mfeat) - 6 modalities, 2,000 samples, 10 classes
2. Wisconsin Diagnostic Breast Cancer (WDBC) - 3 morphological views, 569 samples, 2 classes
3. UCI Dermatology - Clinical symptoms + Histopathology views, 366 samples, 6 classes
4. UCI Human Activity Recognition (HAR) - Multi-sensor streams (Accel, Gravity, Gyro), 2,000 samples, 6 classes
5. Raw Heterogeneous Multimodal - Vision (2D), Audio (1D), Text (strings), Tabular, 1,000 samples, 4 classes

For each dataset, measures:
- Clean multi-modality classification (Macro-F1, Accuracy, ECE)
- Extreme modality noise immunity (Macro-F1 under sigma = 6.0 noise on Modality 1)
- Missing modality inference (Macro-F1 with Modality 0 dropped)
- Selective classification certifiable accuracy (Accuracy at 80% coverage)
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mm_rfmnn.base.hyperbox import FMNN
from mm_rfmnn.datasets.loader import (
    load_dermatology_multiview,
    load_synthetic_multimodal_raw,
    load_uci_har,
    load_uci_mfeat,
    load_wdbc_multiview,
)
from mm_rfmnn.encoders.multimodal import (
    AudioFeatureEncoder,
    ImageFeatureEncoder,
    TabularQuantileEncoder,
    TextFeatureEncoder,
)
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


def benchmark_single_dataset(
    dataset_name: str,
    X_modalities: list[Any],
    y: np.ndarray,
    modality_names: list[str],
    is_raw: bool = False,
) -> dict[str, Any]:
    """Runs end-to-end benchmark on a single multimodal dataset."""
    print("\n" + "=" * 78)
    print(f"BENCHMARKING DATASET: {dataset_name.upper()}")
    print(f"Samples: {len(y)}, Classes: {len(np.unique(y))}, Modalities: {len(X_modalities)}")
    print("=" * 78)

    # Stratified split first to prevent leakage
    idx_tr, idx_te = train_test_split(
        np.arange(len(y)), test_size=0.30, stratify=y, random_state=42
    )

    if is_raw:
        print("Preprocessing raw heterogeneous modalities (Vision, Audio, Text, Tabular)...")
        enc_img = ImageFeatureEncoder(n_components=16)
        enc_aud = AudioFeatureEncoder(n_components=16)
        enc_txt = TextFeatureEncoder(n_components=16)
        enc_tab = TabularQuantileEncoder(n_quantiles=50)

        X_tr_raw = [
            X_k[idx_tr] if isinstance(X_k, np.ndarray) else [X_k[i] for i in idx_tr]
            for X_k in X_modalities
        ]
        X_te_raw = [
            X_k[idx_te] if isinstance(X_k, np.ndarray) else [X_k[i] for i in idx_te]
            for X_k in X_modalities
        ]

        X_train = [
            enc_img.fit_transform(X_tr_raw[0]),
            enc_aud.fit_transform(X_tr_raw[1]),
            enc_txt.fit_transform(X_tr_raw[2]),
            enc_tab.fit_transform(X_tr_raw[3]),
        ]
        X_test = [
            enc_img.transform(X_te_raw[0]),
            enc_aud.transform(X_te_raw[1]),
            enc_txt.transform(X_te_raw[2]),
            enc_tab.transform(X_te_raw[3]),
        ]
    else:
        for name, X_k in zip(modality_names, X_modalities):
            print(f"  - Modality '{name:<20}': shape {X_k.shape}")

        X_train = [X_k[idx_tr] for X_k in X_modalities]
        X_test = [X_k[idx_te] for X_k in X_modalities]

    y_train = y[idx_tr]
    y_test = y[idx_te]
    N_test = len(y_test)
    K = len(X_train)

    # -------------------------------------------------------------
    # 1. Early Concatenation Baseline: Deep MLP
    # -------------------------------------------------------------
    norm_early = ModalityNormalizer(method="minmax")
    X_concat_tr = np.hstack(X_train)
    X_concat_te = np.hstack(X_test)
    Z_concat_tr = norm_early.fit_transform(X_concat_tr)
    Z_concat_te = norm_early.transform(X_concat_te)

    mlp_early = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=250, random_state=42)
    mlp_early.fit(Z_concat_tr, y_train)
    mlp_preds = mlp_early.predict(Z_concat_te)
    mlp_probs = mlp_early.predict_proba(Z_concat_te)

    mlp_acc = float(accuracy_score(y_test, mlp_preds))
    mlp_f1 = float(f1_score(y_test, mlp_preds, average="macro"))
    mlp_ece = compute_ece(mlp_probs, y_test)

    # -------------------------------------------------------------
    # 2. Early Concatenation Baseline: Single Hyperbox FMNN
    # -------------------------------------------------------------
    fmnn_early = FMNN(theta=0.50, gamma=1.0).fit(Z_concat_tr, y_train)
    ev_early, _, _ = fmnn_early.compute_evidence(Z_concat_te)
    ev_early_norm = ev_early / np.maximum(np.sum(ev_early, axis=1, keepdims=True), 1e-8)
    fmnn_early_preds = np.argmax(ev_early, axis=1)

    fmnn_early_acc = float(accuracy_score(y_test, fmnn_early_preds))
    fmnn_early_f1 = float(f1_score(y_test, fmnn_early_preds, average="macro"))
    fmnn_early_ece = compute_ece(ev_early_norm, y_test)

    # -------------------------------------------------------------
    # 3. Late Mean Fusion FMNN (Equal unweighted average)
    # -------------------------------------------------------------
    late_evidence_list = []
    for k in range(K):
        norm_k = ModalityNormalizer(method="minmax")
        Z_tr_k = norm_k.fit_transform(X_train[k])
        Z_te_k = norm_k.transform(X_test[k])
        fmnn_k = FMNN(theta=0.30, gamma=1.0).fit(Z_tr_k, y_train)
        ev_k, _, _ = fmnn_k.compute_evidence(Z_te_k)
        late_evidence_list.append(ev_k)

    late_mean_ev = np.mean(late_evidence_list, axis=0)
    late_preds = np.argmax(late_mean_ev, axis=1)
    late_probs = late_mean_ev / np.maximum(np.sum(late_mean_ev, axis=1, keepdims=True), 1e-8)

    late_acc = float(accuracy_score(y_test, late_preds))
    late_f1 = float(f1_score(y_test, late_preds, average="macro"))
    late_ece = compute_ece(late_probs, y_test)

    # -------------------------------------------------------------
    # 4. Proposed MM-RFMNN (Adaptive Theta + OOF Reliability + Temperature Simplex)
    # -------------------------------------------------------------
    mm_rfmnn = MMRFMNNClassifier(
        theta="adaptive",
        gamma=1.0,
        fusion_method="arithmetic",
        reliability_model="logistic",
        abstention_enabled=False,
        tau_conf=0.40,
        tau_entropy=1.85,
        tau_disagreement=0.65,
        tau_min_reliability=0.15,
        random_state=42,
    )
    mm_rfmnn.fit(X_train, y_train)
    mm_preds = mm_rfmnn.predict(X_test, return_abstention=False)
    mm_probs = mm_rfmnn.predict_proba(X_test)

    mm_acc = float(accuracy_score(y_test, mm_preds))
    mm_f1 = float(f1_score(y_test, mm_preds, average="macro"))
    mm_ece = compute_ece(mm_probs, y_test)
    mm_total_boxes = sum(len(f.bank.V) for f in mm_rfmnn.fmnns_)

    # -------------------------------------------------------------
    # 5. Extreme Noise Robustness Benchmark (Noise on Modality 1)
    # -------------------------------------------------------------
    rng = np.random.default_rng(42)
    corrupted_k = 1 if K > 1 else 0
    X_test_noisy = [X_k.copy() for X_k in X_test]
    scale_k = float(np.std(X_train[corrupted_k])) + 1e-6
    X_test_noisy[corrupted_k] = X_test_noisy[corrupted_k] + rng.normal(
        0, 6.0 * scale_k, size=X_test[corrupted_k].shape
    )

    # Early MLP under noise
    X_concat_noisy = np.hstack(X_test_noisy)
    Z_concat_noisy = norm_early.transform(X_concat_noisy)
    mlp_noisy_preds = mlp_early.predict(Z_concat_noisy)
    mlp_noisy_f1 = float(f1_score(y_test, mlp_noisy_preds, average="macro"))

    # MM-RFMNN under noise
    mm_noisy_preds = mm_rfmnn.predict(X_test_noisy, return_abstention=False)
    mm_noisy_f1 = float(f1_score(y_test, mm_noisy_preds, average="macro"))

    # -------------------------------------------------------------
    # 6. Missing Modality Benchmark (Modality 0 Dropped)
    # -------------------------------------------------------------
    mask_missing = np.ones((N_test, K), dtype=bool)
    mask_missing[:, 0] = False
    mm_missing_preds = mm_rfmnn.predict(X_test, mask=mask_missing, return_abstention=False)
    mm_missing_f1 = float(f1_score(y_test, mm_missing_preds, average="macro"))

    # -------------------------------------------------------------
    # 7. Selective Classification at 80% Coverage
    # -------------------------------------------------------------
    confidences = np.max(mm_probs, axis=1)
    k_retain = int(np.ceil(0.80 * N_test))
    cutoff = np.partition(confidences, N_test - k_retain)[N_test - k_retain]
    retained_mask = confidences >= cutoff
    retained_acc = float(accuracy_score(y_test[retained_mask], mm_preds[retained_mask]))

    # Print summary results for dataset
    print("\nResults for", dataset_name)
    print(f"  {'Model':<22} | {'Macro-F1':<10} | {'Accuracy':<10} | {'ECE':<10}")
    print("  " + "-" * 60)
    print(f"  {'Early Concatenation MLP':<22} | {mlp_f1:<10.4f} | {mlp_acc:<10.4f} | {mlp_ece:<10.4f}")
    print(f"  {'Early FMNN':<22} | {fmnn_early_f1:<10.4f} | {fmnn_early_acc:<10.4f} | {fmnn_early_ece:<10.4f}")
    print(f"  {'Late Mean Fusion FMNN':<22} | {late_f1:<10.4f} | {late_acc:<10.4f} | {late_ece:<10.4f}")
    print(f"  {'MM-RFMNN (Proposed)':<22} | {mm_f1:<10.4f} | {mm_acc:<10.4f} | {mm_ece:<10.4f}")
    print("  " + "-" * 60)
    print(f"  Extreme Noise (sigma=6.0): MLP F1 = {mlp_noisy_f1:.4f}  -->  MM-RFMNN F1 = {mm_noisy_f1:.4f}")
    print(f"  Missing Channel (q0=0):    MM-RFMNN F1 = {mm_missing_f1:.4f}")
    print(f"  Selective Accuracy (80%):  Full = {mm_acc:.4f}  -->  Selective = {retained_acc:.4f}")

    return {
        "dataset": dataset_name,
        "n_samples": len(y),
        "n_classes": len(np.unique(y)),
        "n_modalities": K,
        "modality_names": modality_names,
        "clean": {
            "mlp": {"f1": mlp_f1, "acc": mlp_acc, "ece": mlp_ece},
            "early_fmnn": {"f1": fmnn_early_f1, "acc": fmnn_early_acc, "ece": fmnn_early_ece},
            "late_fmnn": {"f1": late_f1, "acc": late_acc, "ece": late_ece},
            "mm_rfmnn": {"f1": mm_f1, "acc": mm_acc, "ece": mm_ece, "hyperboxes": mm_total_boxes},
        },
        "robustness": {
            "noise_sigma": 6.0,
            "corrupted_modality": modality_names[corrupted_k] if corrupted_k < len(modality_names) else str(corrupted_k),
            "mlp_noisy_f1": mlp_noisy_f1,
            "mm_rfmnn_noisy_f1": mm_noisy_f1,
            "noise_resilience_gain": mm_noisy_f1 - mlp_noisy_f1,
        },
        "missingness": {
            "dropped_modality": modality_names[0],
            "mm_rfmnn_missing_f1": mm_missing_f1,
            "f1_retention_ratio": mm_missing_f1 / max(mm_f1, 1e-6),
        },
        "selective": {
            "coverage_target": 0.80,
            "full_accuracy": mm_acc,
            "selective_accuracy": retained_acc,
            "accuracy_boost": retained_acc - mm_acc,
        },
    }


def run_full_suite() -> None:
    print("=" * 80)
    print("MM-RFMNN MULTI-DATASET EMPIRICAL BENCHMARK SUITE")
    print("Testing 5 Distinct Multimodal Datasets Across Diverse Domains")
    print("=" * 80)

    results: list[dict[str, Any]] = []

    # 1. UCI Multiple Features (Digit Multi-View)
    X_mfeat, y_mfeat, mod_mfeat = load_uci_mfeat(return_modality_names=True)
    res_mfeat = benchmark_single_dataset(
        "UCI Multiple Features (mfeat)", X_mfeat, y_mfeat, mod_mfeat, is_raw=False
    )
    results.append(res_mfeat)

    # 2. Wisconsin Diagnostic Breast Cancer (WDBC)
    X_wdbc, y_wdbc, mod_wdbc = load_wdbc_multiview(return_modality_names=True)
    res_wdbc = benchmark_single_dataset(
        "Breast Cancer Multi-View (WDBC)", X_wdbc, y_wdbc, mod_wdbc, is_raw=False
    )
    results.append(res_wdbc)

    # 3. UCI Dermatology
    X_derm, y_derm, mod_derm = load_dermatology_multiview(return_modality_names=True)
    res_derm = benchmark_single_dataset(
        "UCI Dermatology Multi-View", X_derm, y_derm, mod_derm, is_raw=False
    )
    results.append(res_derm)

    # 4. UCI Human Activity Recognition (HAR Multi-Sensor)
    X_har, y_har, mod_har = load_uci_har(return_modality_names=True, max_samples=2000)
    res_har = benchmark_single_dataset(
        "UCI HAR Multi-Sensor", X_har, y_har, mod_har, is_raw=False
    )
    results.append(res_har)

    # 5. Raw Heterogeneous Multimodal (Vision + Audio + Text + Tabular)
    X_raw, y_raw, mod_raw = load_synthetic_multimodal_raw(
        n_samples=1000, n_classes=4, return_modality_names=True
    )
    res_raw = benchmark_single_dataset(
        "Heterogeneous Raw Multimodal", X_raw, y_raw, mod_raw, is_raw=True
    )
    results.append(res_raw)

    # -------------------------------------------------------------
    # Global Comparative Summary Table
    # -------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("COMPREHENSIVE MULTI-DATASET BENCHMARK SUMMARY")
    print("=" * 90)
    print(
        f"{'Dataset':<26} | {'K':<2} | {'MLP F1':<8} | {'MM-RFMNN F1':<11} | {'MM ECE':<7} | {'Noisy MLP':<9} | {'Noisy MM':<8} | {'Selective':<9}"
    )
    print("-" * 90)
    for r in results:
        name = r["dataset"][:25]
        k = r["n_modalities"]
        mlp_f1 = r["clean"]["mlp"]["f1"]
        mm_f1 = r["clean"]["mm_rfmnn"]["f1"]
        mm_ece = r["clean"]["mm_rfmnn"]["ece"]
        noisy_mlp = r["robustness"]["mlp_noisy_f1"]
        noisy_mm = r["robustness"]["mm_rfmnn_noisy_f1"]
        sel_acc = r["selective"]["selective_accuracy"]
        print(
            f"{name:<26} | {k:<2} | {mlp_f1:<8.4f} | {mm_f1:<11.4f} | {mm_ece:<7.4f} | {noisy_mlp:<9.4f} | {noisy_mm:<8.4f} | {sel_acc:<9.4f}"
        )
    print("=" * 90)

    # Save to JSON
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "multi_dataset_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved multi-dataset benchmark results to: {output_file}")


if __name__ == "__main__":
    run_full_suite()
