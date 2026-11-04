"""End-to-End Demonstration of MM-RFMNN Phase 4 Capabilities:

1. Turnkey Raw Multimodal Encoders (Image, Audio, Text, Tabular).
2. Data-Driven Density-Adaptive Theta Estimation.
3. Cross-Modality Semantic Contradiction / Conflict Abstention.
4. Visual Hyperbox and Modality Attribution Plots.
"""

import os
import sys

import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mm_rfmnn.encoders.multimodal import (
    AudioFeatureEncoder,
    ImageFeatureEncoder,
    TabularQuantileEncoder,
    TextFeatureEncoder,
)
from mm_rfmnn.model import MMRFMNNClassifier
from mm_rfmnn.viz.plotter import (
    plot_hyperboxes_2d,
    plot_modality_attributions,
    plot_risk_coverage_curve,
)


def run_demo():
    print("=" * 76)
    print("MM-RFMNN: END-TO-END DEMO OF ADVANCED MULTIMODAL CAPABILITIES")
    print("=" * 76)
    np.random.seed(42)
    n_samples = 120
    classes = [0, 1, 2]
    y = np.tile(classes, n_samples // len(classes))

    # 1. Synthesize Raw Heterogeneous Multimodal Channels
    print("\n[Step 1] Synthesizing raw heterogeneous multimodal inputs...")
    # Modality 1: Raw 2D Grayscale Images (28x28) with class-specific spatial energy
    raw_images = [
        np.random.rand(28, 28) * 0.4 + (0.5 if c == 0 else (0.2 if c == 1 else 0.8))
        for c in y
    ]
    # Modality 2: Raw 1D Audio Waveforms (16kHz, 0.25s) with class-specific harmonic frequency
    sample_rate = 16000
    t = np.linspace(0, 0.25, int(sample_rate * 0.25))
    raw_audio = [
        np.sin(2 * np.pi * (300 + c * 500) * t) + 0.15 * np.random.randn(len(t))
        for c in y
    ]
    # Modality 3: Raw Text Notes (clinical / status summaries)
    vocab = {
        0: "patient vitals stable normal sinus rhythm breathing calm regular",
        1: "patient arrhythmia mild tachycardia elevated systolic pressure warning",
        2: "critical cardiac arrest emergency resuscitation ventricular fibrillation",
    }
    raw_text = [
        vocab[c] + f" patient id {i} observation record recorded"
        for i, c in enumerate(y)
    ]
    # Modality 4: Skewed Tabular Sensor Readings
    raw_tabular = np.random.exponential(scale=3.0, size=(n_samples, 6)) + y[:, None] * 2.0

    # 2. Encode Raw Data into Unit Hypercube [0, 1]^d
    print("[Step 2] Encoding raw data into unit hypercube [0, 1]^d via turnkey encoders...")
    enc_img = ImageFeatureEncoder(n_components=16, random_state=42)
    enc_aud = AudioFeatureEncoder(n_components=12, sample_rate=sample_rate, random_state=42)
    enc_txt = TextFeatureEncoder(n_components=16, random_state=42)
    enc_tab = TabularQuantileEncoder(n_quantiles=50, random_state=42)

    Z_img = enc_img.fit_transform(raw_images)
    Z_aud = enc_aud.fit_transform(raw_audio)
    Z_txt = enc_txt.fit_transform(raw_text)
    Z_tab = enc_tab.fit_transform(raw_tabular)

    X_modalities = [Z_img, Z_aud, Z_txt, Z_tab]
    print(f"  - Image Modality Shape   : {Z_img.shape} in [0, 1]")
    print(f"  - Audio Modality Shape   : {Z_aud.shape} in [0, 1]")
    print(f"  - Text Modality Shape    : {Z_txt.shape} in [0, 1]")
    print(f"  - Tabular Modality Shape : {Z_tab.shape} in [0, 1]")

    # 3. Train MMRFMNNClassifier with Adaptive Theta and Conflict Detection
    print("\n[Step 3] Fitting MMRFMNNClassifier with theta='adaptive' & conflict detection...")
    clf = MMRFMNNClassifier(
        theta="adaptive",
        gamma=1.0,
        fusion_method="arithmetic",
        conflict_detection_enabled=True,
        tau_conflict=0.45,
        random_state=42,
    )
    clf.fit(X_modalities, y)

    stats = clf.get_hyperbox_stats()
    print("  - Trained Model Hyperbox Banks:")
    print(f"    * Total Hyperboxes        : {stats['total_boxes']}")
    print(f"    * Boxes per Modality      : {stats['boxes_per_modality']}")

    # 4. In-Sample Predictions & Accuracy
    preds = clf.predict(X_modalities)
    valid_mask = preds != -1
    acc = np.mean(preds[valid_mask] == y[valid_mask])
    abstain_rate = np.mean(preds == -1) * 100.0
    print(f"  - Classification Accuracy   : {acc * 100:.2f}% (on retained samples)")
    print(f"  - Abstention Rate           : {abstain_rate:.2f}%")

    # 5. Semantic Contradiction / Conflict Test
    print("\n[Step 4] Testing Cross-Modal Contradiction / Conflict Abstention...")
    # Craft a synthetic conflicting sample:
    # Image indicates Class 2 (critical), but Audio indicates Class 0 (calm)
    conflicting_img = Z_img[y == 2][:1]
    conflicting_aud = Z_aud[y == 0][:1]
    conflicting_txt = Z_txt[y == 2][:1]
    conflicting_tab = Z_tab[y == 0][:1]
    X_conflict = [conflicting_img, conflicting_aud, conflicting_txt, conflicting_tab]

    conflict_pred = clf.predict(X_conflict, return_abstention=True)
    conflict_expl = clf.explain(X_conflict, sample_idx=0)

    print(f"  - Conflicting Input Prediction : {conflict_pred[0]} (-1 means safely abstained)")
    print(f"  - Should Abstain Flag          : {conflict_expl['should_abstain']}")
    print(f"  - Modality Disagreement        : {conflict_expl['uncertainty_metrics']['modality_disagreement']:.4f}")

    # 6. Generate Diagnostic Explainability Figures
    print("\n[Step 5] Generating diagnostic explainability plots...")
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    # Figure 1: 2D Hyperbox Slice for Tabular modality
    fig1 = plot_hyperboxes_2d(
        clf.fmnns_[3],
        Z=Z_tab,
        y=y,
        feature_indices=(0, 1),
        class_names=["Stable", "Warning", "Critical"],
        title="MM-RFMNN: Tabular Modality Hyperboxes",
    )
    p1 = os.path.join(out_dir, "hyperboxes_2d_tabular.png")
    fig1.savefig(p1, dpi=150)
    print(f"  - Saved 2D Hyperbox Plot       : {p1}")

    # Figure 2: Sample Modality Attributions
    fig2 = plot_modality_attributions(
        conflict_expl,
        modality_names=["Vision", "Acoustic", "Text", "Vitals"],
        class_names=["Stable", "Warning", "Critical"],
    )
    p2 = os.path.join(out_dir, "modality_attributions.png")
    fig2.savefig(p2, dpi=150)
    print(f"  - Saved Modality Attribution   : {p2}")

    # Figure 3: Risk-Coverage Curve
    probs = clf.predict_proba(X_modalities)
    confs = np.max(probs, axis=1)
    fig3 = plot_risk_coverage_curve(confs, y, preds)
    p3 = os.path.join(out_dir, "risk_coverage_curve.png")
    fig3.savefig(p3, dpi=150)
    print(f"  - Saved Risk-Coverage Curve    : {p3}")

    print("\n" + "=" * 76)
    print("ALL PHASE 4 CAPABILITIES DEMONSTRATED SUCCESSFULLY")
    print("=" * 76)


if __name__ == "__main__":
    run_demo()
