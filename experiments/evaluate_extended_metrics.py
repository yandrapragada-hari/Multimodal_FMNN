"""Comprehensive Multimodal Metrics Benchmark across Base Early FMNN, Base Late FMNN, and MM-RFMNN.

Computes exact multi-class metrics:
- Classification Accuracy
- Error Rate (1 - Accuracy)
- Sensitivity / Detection Rate / Recall
- Specificity (True Negative Rate)
- Precision (Positive Predictive Value)
- F1-Score (Macro)
- False Alarm Rate (FAR = 1 - Specificity)
- Expected Calibration Error (ECE)
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

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
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == y_true
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)


def compute_extended_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None
) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    acc = float(accuracy_score(y_true, y_pred))
    err = 1.0 - acc

    sensitivities = []
    specificities = []
    precisions = []
    fars = []

    for c in range(n_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = np.sum(cm) - tp - fn - fp

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        sensitivities.append(sens)
        specificities.append(spec)
        precisions.append(prec)
        fars.append(far)

    macro_sens = float(np.mean(sensitivities))
    macro_spec = float(np.mean(specificities))
    macro_prec = float(np.mean(precisions))
    macro_far = float(np.mean(fars))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    res = {
        "Accuracy": acc,
        "Error Rate": err,
        "Sensitivity (Detection Rate)": macro_sens,
        "Specificity": macro_spec,
        "Precision": macro_prec,
        "F1-Score": macro_f1,
        "False Alarm Rate (FAR)": macro_far,
    }
    if probs is not None:
        res["Expected Calibration Error (ECE)"] = compute_ece(probs, y_true)
    return res


def run_benchmark():
    datasets_to_run = [
        ("UCI Multiple Features (mfeat)", lambda: load_uci_mfeat(), False),
        ("Breast Cancer (WDBC)", lambda: load_wdbc_multiview(), False),
        ("UCI Dermatology", lambda: load_dermatology_multiview(), False),
        ("UCI Human Activity Recognition (HAR)", lambda: load_uci_har(max_samples=1500), False),
        ("Heterogeneous Raw Multimodal", lambda: load_synthetic_multimodal_raw(n_samples=800, n_classes=4), True),
    ]

    all_results = {}

    for name, loader, is_raw in datasets_to_run:
        print(f"\n--- Running: {name} ---")
        X_modalities, y = loader()
        n_samples = len(y)
        idx = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(idx)
        tr_len = int(0.7 * n_samples)
        idx_tr, idx_te = idx[:tr_len], idx[tr_len:]

        if is_raw:
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

            X_tr = [
                enc_img.fit_transform(X_tr_raw[0]),
                enc_aud.fit_transform(X_tr_raw[1]),
                enc_txt.fit_transform(X_tr_raw[2]),
                enc_tab.fit_transform(X_tr_raw[3]),
            ]
            X_te = [
                enc_img.transform(X_te_raw[0]),
                enc_aud.transform(X_te_raw[1]),
                enc_txt.transform(X_te_raw[2]),
                enc_tab.transform(X_te_raw[3]),
            ]
        else:
            X_tr = [X_k[idx_tr] for X_k in X_modalities]
            X_te = [X_k[idx_te] for X_k in X_modalities]

        y_tr = y[idx_tr]
        y_te = y[idx_te]
        K = len(X_tr)

        # 1. Base Early FMNN
        print("  Evaluating Base Early FMNN...")
        norm_early = ModalityNormalizer()
        Z_tr_early = norm_early.fit_transform(np.hstack(X_tr))
        Z_te_early = norm_early.transform(np.hstack(X_te))
        early_fmnn = FMNN(theta=0.35, gamma=1.0).fit(Z_tr_early, y_tr)
        early_ev, _, _ = early_fmnn.compute_evidence(Z_te_early)
        early_preds = np.argmax(early_ev, axis=1)
        early_probs = early_ev / np.maximum(np.sum(early_ev, axis=1, keepdims=True), 1e-8)
        early_metrics = compute_extended_metrics(y_te, early_preds, early_probs)

        # 2. Base Late FMNN (Uniform Mean Fusion)
        print("  Evaluating Base Late FMNN...")
        fmnns = []
        norms = []
        for k in range(K):
            nk = ModalityNormalizer()
            zk_tr = nk.fit_transform(X_tr[k])
            norms.append(nk)
            fmnns.append(FMNN(theta=0.35, gamma=1.0).fit(zk_tr, y_tr))

        late_evs = [fmnns[k].compute_evidence(norms[k].transform(X_te[k]))[0] for k in range(K)]
        late_mean_ev = np.mean(late_evs, axis=0)
        late_preds = np.argmax(late_mean_ev, axis=1)
        late_probs = late_mean_ev / np.maximum(np.sum(late_mean_ev, axis=1, keepdims=True), 1e-8)
        late_metrics = compute_extended_metrics(y_te, late_preds, late_probs)

        # 3. MM-RFMNN (Ours)
        print("  Evaluating MM-RFMNN...")
        mm_model = MMRFMNNClassifier(theta="adaptive", gamma=1.0, random_state=42)
        mm_model.fit(X_tr, y_tr)
        mm_preds = mm_model.predict(X_te, return_abstention=False)
        mm_probs = mm_model.predict_proba(X_te)
        mm_metrics = compute_extended_metrics(y_te, mm_preds, mm_probs)

        all_results[name] = {
            "Base Early FMNN": early_metrics,
            "Base Late FMNN": late_metrics,
            "MM-RFMNN (Ours)": mm_metrics,
        }

    out_path = Path("experiments/output/extended_metrics_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_path}")
    return all_results


if __name__ == "__main__":
    run_benchmark()
