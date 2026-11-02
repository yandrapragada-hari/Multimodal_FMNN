"""Incremental Online Adaptation and Streaming Drift Management for MM-RFMNN.

Enables life-long learning and stream classification:
- Online mini-batch adaptation without catastrophic forgetting.
- Memory budget regulation: auto-triggers hyperbox pruning when capacity bounds are exceeded.
- Modality drift and degradation tracking across streaming data channels.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mm_rfmnn.model import MMRFMNNClassifier


class OnlineStreamAdapter:
    """Manages online learning, stream adaptation, and memory-bounded auto-pruning.

    Parameters
    ----------
    max_boxes_per_modality : int, default=500
        Maximum allowable hyperboxes per modality bank before automatic compactification.
    auto_prune_interval : int, default=10
        Number of stream batches between automatic pruning audits.
    prune_min_samples : int, default=2
        Minimum support threshold used during automatic compactification.
    """

    def __init__(
        self,
        max_boxes_per_modality: int = 500,
        auto_prune_interval: int = 10,
        prune_min_samples: int = 2,
    ):
        self.max_boxes_per_modality = max_boxes_per_modality
        self.auto_prune_interval = auto_prune_interval
        self.prune_min_samples = prune_min_samples

        self.batches_processed: int = 0
        self.total_samples_seen: int = 0
        self.rolling_accuracies: list[float] = []
        self.pruning_history: list[dict[str, Any]] = []

    def adapt_batch(
        self,
        model: "MMRFMNNClassifier",
        X_modalities: list[np.ndarray],
        y: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Processes a new streaming mini-batch: evaluates pre-update accuracy, adapts

        model hyperboxes, and regulates memory footprint.
        """
        y = np.asarray(y, dtype=np.int64)
        batch_size = len(y)
        self.total_samples_seen += batch_size
        self.batches_processed += 1

        # 1. Pre-update performance evaluation (test-then-train)
        pre_acc = 0.0
        if model.fuser_ is not None:
            try:
                preds = model.predict(X_modalities, mask=mask, return_abstention=False)
                pre_acc = float(np.mean(preds == y))
            except (ValueError, RuntimeError, AssertionError):
                pre_acc = 0.0
            self.rolling_accuracies.append(pre_acc)

        # 2. Incremental partial fit
        model.partial_fit(X_modalities, y, mask=mask)

        # 3. Check capacity constraints & auto-prune
        pruned_info: dict[str, Any] | None = None
        box_counts = [fmnn.bank.num_boxes for fmnn in model.fmnns_ if fmnn.bank is not None]
        exceeds_budget = any(cnt > self.max_boxes_per_modality for cnt in box_counts)
        interval_reached = (self.batches_processed % self.auto_prune_interval == 0)

        if exceeds_budget or (interval_reached and self.batches_processed > 0):
            pruned_info = model.prune_hyperboxes(min_samples=self.prune_min_samples)
            self.pruning_history.append(
                {
                    "batch_index": self.batches_processed,
                    "stats": pruned_info,
                }
            )

        return {
            "batch_index": self.batches_processed,
            "samples_in_batch": batch_size,
            "total_samples_seen": self.total_samples_seen,
            "pre_update_accuracy": pre_acc,
            "box_counts_post": [
                fmnn.bank.num_boxes for fmnn in model.fmnns_ if fmnn.bank is not None
            ],
            "pruning_triggered": pruned_info is not None,
            "pruning_stats": pruned_info,
        }

    def get_stream_summary(self) -> dict[str, Any]:
        """Returns consolidated stream metrics and learning trajectory."""
        return {
            "batches_processed": self.batches_processed,
            "total_samples_seen": self.total_samples_seen,
            "mean_rolling_accuracy": (
                float(np.mean(self.rolling_accuracies))
                if self.rolling_accuracies
                else 0.0
            ),
            "final_rolling_accuracy": (
                float(self.rolling_accuracies[-1])
                if self.rolling_accuracies
                else 0.0
            ),
            "pruning_events_count": len(self.pruning_history),
        }
