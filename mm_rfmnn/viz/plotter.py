"""Visual explanation and diagnostic plotting suite for MM-RFMNN.

Supports 2D hyperbox projections, per-modality attribution breakdowns,
and risk-coverage uncertainty curves. Uses headless Agg backend.
"""

from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from mm_rfmnn.base.hyperbox import FMNN

PALETTE = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED", "#DB2777"]


def plot_hyperboxes_2d(
    fmnn: FMNN,
    Z: np.ndarray | None = None,
    y: np.ndarray | None = None,
    feature_indices: tuple[int, int] = (0, 1),
    class_names: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plots 2D slice/projection of FMNN hyperboxes with optional data points."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    idx_x, idx_y = feature_indices
    bank = fmnn.bank
    assert bank is not None, "FMNN must be fitted before plotting hyperboxes."

    classes = np.unique(bank.classes)
    legend_handles = []

    # 1. Draw Hyperbox rectangles
    for j in range(bank.num_boxes):
        cls_idx = bank.classes[j]
        color = PALETTE[cls_idx % len(PALETTE)]
        v = bank.V[j]
        w = bank.W[j]

        x_min = v[idx_x]
        y_min = v[idx_y]
        width = max(w[idx_x] - v[idx_x], 0.005)
        height = max(w[idx_y] - v[idx_y], 0.005)

        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.25,
            linestyle="-",
        )
        ax.add_patch(rect)

    # 2. Scatter actual data points if provided
    if Z is not None and y is not None:
        for cls_idx in classes:
            mask = y == cls_idx
            color = PALETTE[cls_idx % len(PALETTE)]
            name = (
                class_names[cls_idx]
                if class_names and cls_idx < len(class_names)
                else f"Class {cls_idx}"
            )
            sc = ax.scatter(
                Z[mask, idx_x],
                Z[mask, idx_y],
                c=color,
                label=name,
                alpha=0.65,
                edgecolors="none",
                s=20,
            )
            legend_handles.append(sc)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(f"Normalized Feature {idx_x}")
    ax.set_ylabel(f"Normalized Feature {idx_y}")
    ax.set_title(title or f"FMNN Hyperbox Banks (Total Boxes: {bank.num_boxes})")
    ax.grid(True, linestyle=":", alpha=0.5)
    if legend_handles:
        ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    return fig


def plot_modality_attributions(
    explanation: dict[str, Any],
    modality_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    fig_size: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """Plots sample-level modality fusion weights and class evidence contributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)

    modalities = explanation["modalities"]
    K = len(modalities)
    names = (
        [modality_names[k] if modality_names else f"Modality {k}" for k in range(K)]
    )

    alphas = [m["fusion_weight"] for m in modalities]
    rels = [m["reliability"] for m in modalities]

    # Panel 1: Reliability & Fusion Weight Simplex
    x = np.arange(K)
    w = 0.35
    ax1.bar(x - w / 2, rels, width=w, label="Reliability r^(k)", color="#64748B", alpha=0.85)
    ax1.bar(x + w / 2, alphas, width=w, label="Simplex Weight alpha^(k)", color="#2563EB", alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Weight / Score [0, 1]")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title("Modality Attributions")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel 2: Fused Probability Distribution
    fused_probs = explanation["fused_probabilities"]
    C = len(fused_probs)
    c_names = [class_names[c] if class_names else f"Class {c}" for c in range(C)]
    pred_cls = explanation["predicted_class"]

    bar_colors = [
        "#059669" if c == pred_cls else "#94A3B8" for c in range(C)
    ]
    ax2.bar(range(C), fused_probs, color=bar_colors, alpha=0.9)
    ax2.set_xticks(range(C))
    ax2.set_xticklabels(c_names)
    ax2.set_ylabel("Probability p(c)")
    ax2.set_ylim(0.0, 1.05)

    decision_text = f"Predicted: {pred_cls}" if not explanation["abstained"] else "ABSTAINED (-1)"
    ax2.set_title(f"Fused Distribution ({decision_text})")
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_risk_coverage_curve(
    confidences: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coverages: Sequence[float] = (1.0, 0.95, 0.90, 0.85, 0.80, 0.70),
    ax: plt.Axes | None = None,
    title: str = "Selective Classification: Risk-Coverage Profile",
) -> plt.Figure:
    """Plots accuracy / error rate as a function of sample retention coverage."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
    else:
        fig = ax.figure

    retained_accs = []
    error_rates = []

    for cov in coverages:
        cutoff = float(np.percentile(confidences, (1.0 - cov) * 100.0))
        retained = confidences >= cutoff
        acc = float(np.mean(y_pred[retained] == y_true[retained]))
        retained_accs.append(acc * 100.0)
        error_rates.append((1.0 - acc) * 100.0)

    cov_percents = [c * 100.0 for c in coverages]
    ax.plot(cov_percents, retained_accs, marker="o", color="#2563EB", linewidth=2.0, label="Accuracy (%)")
    ax.plot(cov_percents, error_rates, marker="s", color="#DC2626", linewidth=1.5, linestyle="--", label="Error Rate (%)")

    ax.set_xlabel("Coverage (% Samples Retained)")
    ax.set_ylabel("Performance (%)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    return fig
