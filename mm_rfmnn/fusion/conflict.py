"""Cross-modality semantic conflict and contradiction detector.

Quantifies epistemic contradictions between modality evidence streams.
Prevents overconfident fusion when different sensory channels strongly disagree.
"""

import numpy as np


class CrossModalConflictDetector:
    """Detects semantic contradictions between paired modality predictions.

    Parameters
    ----------
    tau_conflict : float, default=0.55
        Conflict score threshold above which cross-modal contradiction is flagged.
    min_modality_confidence : float, default=0.45
        Minimum individual modality top-class confidence required to trigger conflict check.
    """

    def __init__(self, tau_conflict: float = 0.55, min_modality_confidence: float = 0.45):
        self.tau_conflict = tau_conflict
        self.min_modality_confidence = min_modality_confidence
        self.co_occurrence_matrix_: np.ndarray | None = None
        self.num_classes_: int = 0

    def fit(self, calibrated_probs: list[np.ndarray], y: np.ndarray) -> "CrossModalConflictDetector":
        """Fits empirical joint class agreement rates across modality pairs from training data."""
        y = np.asarray(y, dtype=np.int64)
        K = len(calibrated_probs)
        C = max(int(np.max(y)) + 1 if len(y) > 0 else 1, calibrated_probs[0].shape[1])
        self.num_classes_ = C

        # K x K x C x C joint conditional co-occurrence tensor
        # P(class_m = c_2 | class_k = c_1)
        co_occur = np.zeros((K, K, C, C), dtype=np.float64)

        for k in range(K):
            preds_k = np.argmax(calibrated_probs[k], axis=1)
            for m in range(K):
                preds_m = np.argmax(calibrated_probs[m], axis=1)
                for i in range(len(y)):
                    co_occur[k, m, preds_k[i], preds_m[i]] += 1.0

        # Laplace smoothing and conditional row normalization
        co_occur += 1.0
        row_sums = np.sum(co_occur, axis=3, keepdims=True)
        self.co_occurrence_matrix_ = co_occur / row_sums
        return self

    def compute_conflict_scores(
        self,
        calibrated_probs: list[np.ndarray],
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Computes sample-level cross-modal conflict score in [0, 1].

        Conflict is high when two present modalities are individually highly confident
        in classes that rarely or never co-occur.
        """
        K = len(calibrated_probs)
        N = len(calibrated_probs[0])
        conflict_scores = np.zeros(N, dtype=np.float64)

        if mask is None:
            mask = np.ones((N, K), dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)

        top_classes = np.array([np.argmax(p, axis=1) for p in calibrated_probs])  # (K, N)
        top_confs = np.array([np.max(p, axis=1) for p in calibrated_probs])       # (K, N)

        for i in range(N):
            active_k = [k for k in range(K) if mask[i, k]]
            if len(active_k) < 2:
                conflict_scores[i] = 0.0
                continue

            max_pairwise_conflict = 0.0
            for idx1, k in enumerate(active_k):
                for m in active_k[idx1 + 1 :]:
                    c_k = top_classes[k, i]
                    c_m = top_classes[m, i]
                    conf_k = top_confs[k, i]
                    conf_m = top_confs[m, i]

                    if c_k != c_m and conf_k >= self.min_modality_confidence and conf_m >= self.min_modality_confidence:
                        if (
                            self.co_occurrence_matrix_ is not None
                            and c_k < self.num_classes_
                            and c_m < self.num_classes_
                        ):
                            prior_co_occur = self.co_occurrence_matrix_[k, m, c_k, c_m]
                            disagreement_weight = 1.0 - prior_co_occur
                        else:
                            disagreement_weight = 1.0

                        pairwise_conflict = (conf_k * conf_m) * disagreement_weight
                        max_pairwise_conflict = max(max_pairwise_conflict, pairwise_conflict)

            conflict_scores[i] = max_pairwise_conflict

        return conflict_scores

    def check_conflict_abstention(self, conflict_scores: np.ndarray) -> np.ndarray:
        """Returns boolean mask where True indicates sample should be abstained due to conflict."""
        return conflict_scores >= self.tau_conflict
