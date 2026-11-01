"""Evidence calibration, reliability-weighted fusion, uncertainty estimation,

and risk-coverage abstention mechanisms.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon


class TemperatureCalibrator:
    """Calibrates modality-specific evidence scores into well-calibrated probabilities

    using temperature scaling fitted on validation data via NLL.
    """

    def __init__(self, init_temperature: float = 1.0):
        self.temperature: float = init_temperature
        self.is_fitted: bool = False

    def fit(self, evidence: np.ndarray, y_val: np.ndarray) -> "TemperatureCalibrator":
        evidence = np.asarray(evidence, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.int64)
        N, _C = evidence.shape

        def nll_obj(t_arr):
            t = max(float(t_arr[0]), 1e-4)
            scaled = evidence / t
            # Stable log-softmax
            max_s = np.max(scaled, axis=1, keepdims=True)
            log_sum_exp = max_s + np.log(
                np.sum(np.exp(scaled - max_s), axis=1, keepdims=True) + 1e-12
            )
            log_probs = scaled - log_sum_exp
            nll = -np.mean(log_probs[np.arange(N), y_val])
            return nll

        res = minimize(
            nll_obj, x0=[self.temperature], bounds=[(0.01, 20.0)], method="L-BFGS-B"
        )
        if res.success:
            self.temperature = float(res.x[0])
        self.is_fitted = True
        return self

    def calibrate(self, evidence: np.ndarray) -> np.ndarray:
        evidence = np.asarray(evidence, dtype=np.float64)
        scaled = evidence / max(self.temperature, 1e-4)
        max_s = np.max(scaled, axis=1, keepdims=True)
        exp_s = np.exp(scaled - max_s)
        return exp_s / (np.sum(exp_s, axis=1, keepdims=True) + 1e-12)


class ReliabilityFuser:
    """Fuses modality class distributions weighted by instance-level reliabilities

    and evaluates multi-criteria uncertainty for selective abstention.
    """

    def __init__(
        self,
        fusion_method: str = "arithmetic",
        tau_conf: float = 0.40,
        tau_entropy: float = 1.80,
        tau_disagreement: float = 0.60,
        tau_min_reliability: float = 0.15,
        eps: float = 1e-9,
    ):
        self.fusion_method = (
            fusion_method  # "arithmetic", "log_opinion_pool", "raw_score"
        )
        self.tau_conf = tau_conf
        self.tau_entropy = tau_entropy
        self.tau_disagreement = tau_disagreement
        self.tau_min_reliability = tau_min_reliability
        self.eps = eps

    def compute_fusion_weights(
        self,
        reliabilities: list[np.ndarray],  # K arrays of shape (N,)
        mask: np.ndarray | None = None,  # (N, K) boolean presence mask
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates normalized fusion weights alpha_i^(k) with explicit presence masking.

        Returns
        -------
        alphas : (N, K)
        masked_reliabilities : (N, K)
        """
        K = len(reliabilities)
        N = len(reliabilities[0])

        rel_matrix = np.column_stack(reliabilities)  # (N, K)

        if mask is not None:
            mask_arr = np.asarray(mask, dtype=np.float64)
        else:
            mask_arr = np.ones((N, K), dtype=np.float64)

        masked_rel = rel_matrix * mask_arr  # (N, K)
        rel_sums = np.sum(masked_rel, axis=1, keepdims=True)  # (N, 1)

        alphas = np.zeros((N, K), dtype=np.float64)
        for i in range(N):
            total_rel = rel_sums[i, 0]
            if total_rel > self.eps:
                alphas[i] = masked_rel[i] / total_rel
            else:
                # Fallback to uniform over present modalities
                present_count = np.sum(mask_arr[i])
                if present_count > 0:
                    alphas[i] = mask_arr[i] / present_count
                else:
                    # Total modality blackout
                    alphas[i] = np.ones(K) / K

        return alphas, masked_rel

    def fuse(
        self,
        modality_probs: list[np.ndarray],  # K arrays of shape (N, C)
        alphas: np.ndarray,  # (N, K)
        mask: np.ndarray | None = None,  # (N, K)
    ) -> np.ndarray:
        """Fuses class evidence across modalities using normalized reliability weights."""
        K = len(modality_probs)
        N, C = modality_probs[0].shape

        if self.fusion_method == "arithmetic":
            fused = np.zeros((N, C), dtype=np.float64)
            for k in range(K):
                fused += alphas[:, k : k + 1] * modality_probs[k]
            # Renormalize to guard against numeric precision
            fused = fused / (np.sum(fused, axis=1, keepdims=True) + self.eps)
            return fused

        elif self.fusion_method == "log_opinion_pool":
            log_fused = np.zeros((N, C), dtype=np.float64)
            for k in range(K):
                # Avoid log(0)
                safe_probs = np.clip(modality_probs[k], self.eps, 1.0)
                log_fused += alphas[:, k : k + 1] * np.log(safe_probs)
            max_log = np.max(log_fused, axis=1, keepdims=True)
            exp_log = np.exp(log_fused - max_log)
            return exp_log / (np.sum(exp_log, axis=1, keepdims=True) + self.eps)

        elif self.fusion_method == "raw_score":
            fused = np.zeros((N, C), dtype=np.float64)
            for k in range(K):
                fused += alphas[:, k : k + 1] * modality_probs[k]
            return fused / (np.sum(fused, axis=1, keepdims=True) + self.eps)

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def evaluate_uncertainty(
        self,
        fused_probs: np.ndarray,  # (N, C)
        modality_probs: list[np.ndarray],  # K arrays of shape (N, C)
        masked_reliabilities: np.ndarray,  # (N, K)
        mask: np.ndarray | None = None,  # (N, K)
    ) -> dict[str, np.ndarray]:
        """Calculates multi-criteria epistemic and aleatoric uncertainty signals."""
        N, C = fused_probs.shape
        K = len(modality_probs)

        # 1. Predictive entropy
        clipped_p = np.clip(fused_probs, self.eps, 1.0)
        entropy = -np.sum(fused_probs * np.log(clipped_p), axis=1)

        # 2. Confidence deficit
        max_probs = np.max(fused_probs, axis=1)
        conf_deficit = 1.0 - max_probs

        # 3. Class margin
        if C >= 2:
            sorted_p = np.sort(fused_probs, axis=1)
            margin = sorted_p[:, -1] - sorted_p[:, -2]
        else:
            margin = max_probs

        # 4. Total available reliability
        total_reliability = np.sum(masked_reliabilities, axis=1)

        # 5. Modality disagreement (Average pairwise Jensen-Shannon distance)
        if mask is None:
            mask = np.ones((N, K), dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)

        disagreement = np.zeros(N, dtype=np.float64)
        for i in range(N):
            active_k = np.where(mask[i])[0]
            if len(active_k) >= 2:
                dists = []
                for idx1 in range(len(active_k)):
                    for idx2 in range(idx1 + 1, len(active_k)):
                        p1 = modality_probs[active_k[idx1]][i]
                        p2 = modality_probs[active_k[idx2]][i]
                        dists.append(jensenshannon(p1, p2))
                disagreement[i] = np.mean(dists) if dists else 0.0

        return {
            "entropy": entropy,
            "max_probability": max_probs,
            "confidence_deficit": conf_deficit,
            "margin": margin,
            "total_reliability": total_reliability,
            "disagreement": disagreement,
        }

    def check_abstention(self, uncertainty: dict[str, np.ndarray]) -> np.ndarray:
        """Determines whether to abstain (True) or predict (False) for each sample."""
        N = len(uncertainty["max_probability"])
        abstain = np.zeros(N, dtype=bool)

        # Abstain if:
        # 1. Confidence is too low
        abstain |= uncertainty["max_probability"] < self.tau_conf
        # 2. Entropy is too high
        abstain |= uncertainty["entropy"] > self.tau_entropy
        # 3. Total active reliability is below safe threshold
        abstain |= uncertainty["total_reliability"] < self.tau_min_reliability
        # 4. Modality disagreement is excessive
        abstain |= uncertainty["disagreement"] > self.tau_disagreement

        return abstain
