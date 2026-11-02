"""Hyperbox Pruning and Geometric Compactification Module for MM-RFMNN.

Implements principled geometric pruning:
1. Redundancy & Subsumption Elimination: Discards hyperboxes geometrically subsumed
   or near-fully overlapped by larger hyperboxes of the same class without loss of coverage.
2. Low-Support Outlier Pruning: Eliminates low-density singleton noise hyperboxes that
   fail to achieve minimum pattern support, guarding against noise overfitting.
3. Class Presence Preservation: Guarantees that at least one representative hyperbox
   is preserved for each valid class.
"""

from typing import Any

import numpy as np

from mm_rfmnn.base.hyperbox import HyperboxBank


class HyperboxPruner:
    """Principled geometric hyperbox pruner for FMNN hyperbox banks.

    Parameters
    ----------
    min_samples : int, default=1
        Minimum number of training samples absorbed by a hyperbox to avoid pruning.
        Hyperboxes with support < min_samples are pruned as isolated noise/outliers.
    overlap_threshold : float, default=0.98
        Geometric overlap ratio threshold above which a smaller hyperbox is considered
        subsumed and redundant relative to a co-class hyperbox.
    preserve_class_presence : bool, default=True
        Whether to ensure every class retains at least one representative hyperbox.
    """

    def __init__(
        self,
        min_samples: int = 1,
        overlap_threshold: float = 0.98,
        preserve_class_presence: bool = True,
    ):
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}.")
        if not (0.0 < overlap_threshold <= 1.0):
            raise ValueError(
                f"overlap_threshold must be in (0, 1], got {overlap_threshold}."
            )

        self.min_samples = min_samples
        self.overlap_threshold = overlap_threshold
        self.preserve_class_presence = preserve_class_presence

    def prune(self, bank: HyperboxBank) -> dict[str, Any]:
        """Prunes redundant and low-support hyperboxes in-place from bank.

        Parameters
        ----------
        bank : HyperboxBank
            The hyperbox bank to prune.

        Returns
        -------
        stats : dict[str, Any]
            Pruning summary report containing initial, final, and pruned counts.
        """
        initial_boxes = bank.num_boxes
        if initial_boxes <= 1:
            return {
                "initial_boxes": initial_boxes,
                "final_boxes": initial_boxes,
                "pruned_boxes": 0,
                "reduction_ratio": 0.0,
                "pruned_redundant": 0,
                "pruned_low_support": 0,
            }

        V_arr, W_arr, classes_arr = bank.get_arrays()
        counts_arr = np.array(bank.sample_counts, dtype=np.int64)
        M, _d = V_arr.shape

        keep_mask = np.ones(M, dtype=bool)
        pruned_redundant = 0

        # Stage 1: Geometric Subsumption & Redundancy Pruning (within identical class)
        unique_classes = np.unique(classes_arr)
        for c in unique_classes:
            c_indices = np.where(classes_arr == c)[0]
            if len(c_indices) <= 1:
                continue

            # Sort candidate boxes by span/volume descending so larger boxes absorb smaller ones
            spans = np.sum(W_arr[c_indices] - V_arr[c_indices], axis=1)
            sorted_order = np.argsort(-spans)
            sorted_indices = c_indices[sorted_order]

            for i_pos in range(len(sorted_indices)):
                idx_large = sorted_indices[i_pos]
                if not keep_mask[idx_large]:
                    continue

                v_large = V_arr[idx_large]
                w_large = W_arr[idx_large]

                for j_pos in range(i_pos + 1, len(sorted_indices)):
                    idx_small = sorted_indices[j_pos]
                    if not keep_mask[idx_small]:
                        continue

                    v_small = V_arr[idx_small]
                    w_small = W_arr[idx_small]

                    # Check exact or near-subsumption: is small covered by large?
                    dim_overlap = np.maximum(
                        0.0,
                        np.minimum(w_small, w_large) - np.maximum(v_small, v_large),
                    )
                    span_small = np.sum(w_small - v_small)

                    # Containment condition: overlap covers >= overlap_threshold of smaller box
                    if span_small <= 1e-9:
                        # Zero-volume point box: check exact containment
                        contained = bool(np.all((v_small >= v_large - 1e-9) & (w_small <= w_large + 1e-9)))
                    else:
                        contained = bool(
                            (np.sum(dim_overlap) / (span_small + 1e-9))
                            >= self.overlap_threshold
                        )

                    if contained:
                        # Subsume small into large
                        keep_mask[idx_small] = False
                        counts_arr[idx_large] += counts_arr[idx_small]
                        pruned_redundant += 1

        # Stage 2: Low-Support / Outlier Noise Pruning
        pruned_low_support = 0
        if self.min_samples > 1:
            for idx in range(M):
                if keep_mask[idx] and counts_arr[idx] < self.min_samples:
                    keep_mask[idx] = False
                    pruned_low_support += 1

        # Stage 3: Safeguard Class Presence
        if self.preserve_class_presence:
            for c in unique_classes:
                c_mask = (classes_arr == c) & keep_mask
                if not np.any(c_mask):
                    # Reinstate the highest-support original box of this class
                    orig_c_indices = np.where(classes_arr == c)[0]
                    best_orig = orig_c_indices[np.argmax(counts_arr[orig_c_indices])]
                    keep_mask[best_orig] = True
                    if counts_arr[best_orig] < self.min_samples:
                        pruned_low_support = max(0, pruned_low_support - 1)

        # Apply filtering in-place to bank
        bank.filter_boxes(keep_mask, updated_counts=counts_arr[keep_mask])
        final_boxes = bank.num_boxes
        total_pruned = initial_boxes - final_boxes

        return {
            "initial_boxes": int(initial_boxes),
            "final_boxes": int(final_boxes),
            "pruned_boxes": int(total_pruned),
            "reduction_ratio": float(total_pruned / initial_boxes) if initial_boxes > 0 else 0.0,
            "pruned_redundant": int(pruned_redundant),
            "pruned_low_support": int(pruned_low_support),
        }
