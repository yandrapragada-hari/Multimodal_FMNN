"""High-level, scikit-learn compatible Multimodal Reliability-Aware Fuzzy Min-Max Neural Network."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from mm_rfmnn.base.hyperbox import FMNN
from mm_rfmnn.encoders.normalizer import ModalityNormalizer
from mm_rfmnn.fusion.fuser import ReliabilityFuser, TemperatureCalibrator
from mm_rfmnn.reliability.estimator import MetaFeatureExtractor, ReliabilityEstimator


class MMRFMNNClassifier(BaseEstimator, ClassifierMixin):
    """Multimodal Reliability-Aware Fuzzy Min-Max Neural Network (MM-RFMNN).

    Parameters
    ----------
    theta : float or List[float], default=0.2
        Maximum hyperbox size bound for FMNN expansion. Can be specified per modality.
    gamma : float or List[float], default=1.0
        Sensitivity decay parameter for continuous exponential membership.
    fusion_method : str, default='arithmetic'
        Fusion equation: 'arithmetic', 'log_opinion_pool', or 'raw_score'.
    reliability_model : str, default='logistic'
        Type of reliability estimator: 'logistic', 'mlp', or 'rule_based'.
    abstention_enabled : bool, default=True
        Whether to enable uncertainty-based abstention.
    tau_conf : float, default=0.40
        Minimum top-class probability required to prevent abstention.
    tau_entropy : float, default=1.80
        Maximum predictive entropy allowed before abstaining.
    tau_disagreement : float, default=0.60
        Maximum modality disagreement (Jensen-Shannon) allowed before abstaining.
    tau_min_reliability : float, default=0.15
        Minimum sum of available reliabilities required to predict.
    random_state : int, default=42
    """

    def __init__(
        self,
        theta: float | list[float] = 0.2,
        gamma: float | list[float] = 1.0,
        fusion_method: str = "arithmetic",
        reliability_model: str = "logistic",
        abstention_enabled: bool = True,
        tau_conf: float = 0.40,
        tau_entropy: float = 1.80,
        tau_disagreement: float = 0.60,
        tau_min_reliability: float = 0.15,
        random_state: int = 42,
    ):
        self.theta = theta
        self.gamma = gamma
        self.fusion_method = fusion_method
        self.reliability_model = reliability_model
        self.abstention_enabled = abstention_enabled
        self.tau_conf = tau_conf
        self.tau_entropy = tau_entropy
        self.tau_disagreement = tau_disagreement
        self.tau_min_reliability = tau_min_reliability
        self.random_state = random_state

        # Internal components (fitted during fit)
        self.num_modalities_: int = 0
        self.classes_: np.ndarray = np.array([])
        self.num_classes_: int = 0
        self.normalizers_: list[ModalityNormalizer] = []
        self.fmnns_: list[FMNN] = []
        self.reliability_estimators_: list[ReliabilityEstimator] = []
        self.temperature_calibrators_: list[TemperatureCalibrator] = []
        self.meta_extractor_ = MetaFeatureExtractor()
        self.fuser_: ReliabilityFuser | None = None

    def _resolve_param_list(self, param: float | list[float], K: int) -> list[float]:
        if isinstance(param, (int, float)):
            return [float(param)] * K
        elif isinstance(param, (list, tuple)):
            if len(param) != K:
                raise ValueError(
                    f"Parameter list length {len(param)} does not match modality count {K}."
                )
            return [float(p) for p in param]
        raise TypeError(f"Invalid parameter type: {type(param)}")

    def fit(
        self,
        X_modalities: list[np.ndarray],
        y: np.ndarray,
        mask: np.ndarray | None = None,
        val_data: tuple[list[np.ndarray], np.ndarray, np.ndarray | None] | None = None,
    ) -> "MMRFMNNClassifier":
        """Fits the complete MM-RFMNN pipeline following Algorithm 1.

        Parameters
        ----------
        X_modalities : List of K arrays, each of shape (N, d_k)
        y : array of shape (N,)
        mask : Optional array of shape (N, K) indicating presence (1) or missingness (0)
        val_data : Optional tuple (X_val_list, y_val, mask_val) for calibration
        """
        y = np.asarray(y, dtype=np.int64)
        N = len(y)
        K = len(X_modalities)
        self.num_modalities_ = K
        self.classes_ = np.unique(y)
        self.num_classes_ = int(np.max(y)) + 1

        thetas = self._resolve_param_list(self.theta, K)
        gammas = self._resolve_param_list(self.gamma, K)

        if mask is None:
            mask = np.ones((N, K), dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)

        # Split internal calibration set if not explicitly provided
        if val_data is None:
            # Hold out 15% for temperature calibration
            indices = np.arange(N)
            train_idx, calib_idx = train_test_split(
                indices, test_size=0.15, stratify=y, random_state=self.random_state
            )
            X_train = [X[train_idx] for X in X_modalities]
            y_train = y[train_idx]
            mask_train = mask[train_idx]

            X_calib = [X[calib_idx] for X in X_modalities]
            y_calib = y[calib_idx]
            mask_calib = mask[calib_idx]
        else:
            X_train = X_modalities
            y_train = y
            mask_train = mask
            X_calib, y_calib, mask_calib = val_data
            if mask_calib is None:
                mask_calib = np.ones((len(y_calib), K), dtype=bool)

        self.normalizers_ = []
        self.fmnns_ = []
        self.reliability_estimators_ = []
        self.temperature_calibrators_ = []

        # Stage 1 & 2: Fit Modality Normalizers, FMNN Hyperboxes, and OOF Reliability
        rule_fallback = self.reliability_model == "rule_based"
        model_type = "mlp" if self.reliability_model == "mlp" else "logistic"

        for k in range(K):
            # 1. Fit normalizer on training set
            norm_k = ModalityNormalizer(method="minmax")
            pres_k = mask_train[:, k]
            norm_k.fit(X_train[k], mask=pres_k)
            Z_train_k = norm_k.transform(X_train[k])
            self.normalizers_.append(norm_k)

            # 2. Fit modality FMNN on present training data
            fmnn_k = FMNN(theta=thetas[k], gamma=gammas[k])
            if np.any(pres_k):
                fmnn_k.fit(Z_train_k[pres_k], y_train[pres_k])
            self.fmnns_.append(fmnn_k)

            # 3. Fit reliability estimator using Out-Of-Fold targets
            rel_est = ReliabilityEstimator(
                model_type=model_type,
                rule_based_fallback=rule_fallback,
                random_state=self.random_state + k,
            )
            if np.any(pres_k) and not rule_fallback:
                rel_est.fit_oof(
                    Z_train_k[pres_k],
                    y_train[pres_k],
                    theta=thetas[k],
                    gamma=gammas[k],
                    n_splits=5,
                )
            self.reliability_estimators_.append(rel_est)

            # 4. Fit temperature calibration on held-out calibration split
            calibrator_k = TemperatureCalibrator()
            Z_calib_k = norm_k.transform(X_calib[k])
            calib_pres_k = mask_calib[:, k]

            if np.any(calib_pres_k):
                ev_calib, _, _ = fmnn_k.compute_evidence(Z_calib_k[calib_pres_k])
                calibrator_k.fit(ev_calib, y_calib[calib_pres_k])
            self.temperature_calibrators_.append(calibrator_k)

        # Stage 3: Initialize Fuser
        self.fuser_ = ReliabilityFuser(
            fusion_method=self.fusion_method,
            tau_conf=self.tau_conf,
            tau_entropy=self.tau_entropy,
            tau_disagreement=self.tau_disagreement,
            tau_min_reliability=self.tau_min_reliability,
        )

        return self

    def partial_fit(
        self,
        X_modalities: list[np.ndarray],
        y: np.ndarray,
        classes: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> "MMRFMNNClassifier":
        """Incrementally updates FMNN hyperboxes with a streaming multimodal batch.

        Parameters
        ----------
        X_modalities : list of K arrays, each of shape (N, d_k)
        y : array of shape (N,)
        classes : optional array of expected classes across streams
        mask : optional array of shape (N, K)
        """
        y = np.asarray(y, dtype=np.int64)
        N = len(y)
        K = len(X_modalities)

        if mask is None:
            mask = np.ones((N, K), dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)

        thetas = self._resolve_param_list(self.theta, K)
        gammas = self._resolve_param_list(self.gamma, K)

        if self.fuser_ is None or self.num_modalities_ == 0:
            # First initialization in streaming mode
            self.num_modalities_ = K
            if classes is not None:
                self.classes_ = np.asarray(classes, dtype=np.int64)
            else:
                self.classes_ = np.unique(y)
            self.num_classes_ = int(np.max(self.classes_)) + 1

            self.normalizers_ = [ModalityNormalizer(method="minmax") for _ in range(K)]
            self.fmnns_ = [FMNN(theta=thetas[k], gamma=gammas[k]) for k in range(K)]
            self.reliability_estimators_ = [
                ReliabilityEstimator(
                    model_type="mlp" if self.reliability_model == "mlp" else "logistic",
                    rule_based_fallback=self.reliability_model == "rule_based",
                    random_state=self.random_state + k,
                )
                for k in range(K)
            ]
            self.temperature_calibrators_ = [TemperatureCalibrator() for _ in range(K)]
            self.fuser_ = ReliabilityFuser(
                fusion_method=self.fusion_method,
                tau_conf=self.tau_conf,
                tau_entropy=self.tau_entropy,
                tau_disagreement=self.tau_disagreement,
                tau_min_reliability=self.tau_min_reliability,
            )
        else:
            if classes is not None:
                self.classes_ = np.unique(
                    np.concatenate([self.classes_, np.asarray(classes, dtype=np.int64)])
                )
            else:
                self.classes_ = np.unique(np.concatenate([self.classes_, y]))
            self.num_classes_ = int(np.max(self.classes_)) + 1

        for k in range(K):
            pres_k = mask[:, k]
            if np.any(pres_k):
                self.normalizers_[k].partial_fit(X_modalities[k], mask=pres_k)
                Z_k = self.normalizers_[k].transform(X_modalities[k])
                self.fmnns_[k].partial_fit(Z_k[pres_k], y[pres_k], classes=self.classes_)

        return self

    def prune_hyperboxes(
        self,
        min_samples: int = 1,
        overlap_threshold: float = 0.98,
        preserve_class_presence: bool = True,
    ) -> dict[str, Any]:
        """Prunes low-support and redundant hyperboxes across all modality banks.

        Parameters
        ----------
        min_samples : int, default=1
            Minimum sample support required to retain a hyperbox.
        overlap_threshold : float, default=0.98
            Overlap containment ratio for subsumption pruning.
        preserve_class_presence : bool, default=True
            Guarantees every class retains at least one representative hyperbox.

        Returns
        -------
        summary : dict[str, Any]
            Consolidated pruning statistics across all modalities.
        """
        assert self.num_modalities_ > 0, "Model must be fitted before pruning."

        per_modality_stats = []
        total_initial = 0
        total_final = 0

        for k in range(self.num_modalities_):
            stat_k = self.fmnns_[k].prune(
                min_samples=min_samples,
                overlap_threshold=overlap_threshold,
                preserve_class_presence=preserve_class_presence,
            )
            per_modality_stats.append(stat_k)
            total_initial += stat_k["initial_boxes"]
            total_final += stat_k["final_boxes"]

        total_pruned = total_initial - total_final
        reduction_ratio = (
            float(total_pruned / total_initial) if total_initial > 0 else 0.0
        )

        return {
            "total_initial_boxes": total_initial,
            "total_final_boxes": total_final,
            "total_pruned_boxes": total_pruned,
            "overall_reduction_ratio": reduction_ratio,
            "modalities": per_modality_stats,
        }

    def get_hyperbox_stats(self) -> dict[str, Any]:
        """Returns structural statistics for all modality hyperbox banks."""
        boxes_per_mod = [
            fmnn.bank.num_boxes if fmnn.bank is not None else 0 for fmnn in self.fmnns_
        ]
        return {
            "total_boxes": sum(boxes_per_mod),
            "boxes_per_modality": boxes_per_mod,
            "num_modalities": self.num_modalities_,
            "num_classes": self.num_classes_,
        }


    def _infer_evidence_and_reliabilities(
        self,
        X_modalities: list[np.ndarray],
        mask: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """Internal inference worker following Algorithm 2."""
        K = self.num_modalities_
        N = len(X_modalities[0])

        if mask is None:
            mask = np.ones((N, K), dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)

        calibrated_probs: list[np.ndarray] = []
        reliabilities: list[np.ndarray] = []

        for k in range(K):
            norm_k = self.normalizers_[k]
            fmnn_k = self.fmnns_[k]
            rel_est = self.reliability_estimators_[k]
            calibrator_k = self.temperature_calibrators_[k]

            Z_k = norm_k.transform(X_modalities[k])
            ev_k, best_m_k, _ = fmnn_k.compute_evidence(Z_k)

            # Calibrate evidence into probabilities
            probs_k = calibrator_k.calibrate(ev_k)

            # Extract meta-features and predict reliability
            u_k = self.meta_extractor_.extract(ev_k, best_m_k)
            r_k = rel_est.predict_reliability(u_k)

            calibrated_probs.append(probs_k)
            reliabilities.append(r_k)

        return calibrated_probs, reliabilities, mask

    def predict_proba(
        self,
        X_modalities: list[np.ndarray],
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Computes fused class probabilities p(c) for all samples."""
        assert self.fuser_ is not None, "Model must be fitted before predict_proba."
        calibrated_probs, reliabilities, mask_arr = (
            self._infer_evidence_and_reliabilities(X_modalities, mask)
        )
        alphas, _ = self.fuser_.compute_fusion_weights(reliabilities, mask_arr)
        fused = self.fuser_.fuse(calibrated_probs, alphas, mask_arr)
        return fused

    def predict(
        self,
        X_modalities: list[np.ndarray],
        mask: np.ndarray | None = None,
        return_abstention: bool = True,
    ) -> np.ndarray:
        """Returns class predictions.

        If return_abstention is True and abstention is enabled, abstained samples are marked as -1.
        """
        assert self.fuser_ is not None, "Model must be fitted before predict."
        calibrated_probs, reliabilities, mask_arr = (
            self._infer_evidence_and_reliabilities(X_modalities, mask)
        )
        alphas, masked_rel = self.fuser_.compute_fusion_weights(reliabilities, mask_arr)
        fused = self.fuser_.fuse(calibrated_probs, alphas, mask_arr)
        preds = np.argmax(fused, axis=1)

        if self.abstention_enabled and return_abstention:
            uncertainty = self.fuser_.evaluate_uncertainty(
                fused, calibrated_probs, masked_rel, mask_arr
            )
            abstain_mask = self.fuser_.check_abstention(uncertainty)
            preds[abstain_mask] = -1

        return preds

    def explain(
        self,
        X_modalities: list[np.ndarray],
        mask: np.ndarray | None = None,
        sample_idx: int = 0,
    ) -> dict[str, Any]:
        """Provides full geometric and evidence attribution for an individual sample."""
        assert self.fuser_ is not None, "Model must be fitted before explain."
        calibrated_probs, reliabilities, mask_arr = (
            self._infer_evidence_and_reliabilities(X_modalities, mask)
        )
        alphas, masked_rel = self.fuser_.compute_fusion_weights(reliabilities, mask_arr)
        fused = self.fuser_.fuse(calibrated_probs, alphas, mask_arr)
        uncertainty = self.fuser_.evaluate_uncertainty(
            fused, calibrated_probs, masked_rel, mask_arr
        )
        abstain = self.fuser_.check_abstention(uncertainty)

        i = sample_idx
        modality_details = []
        for k in range(self.num_modalities_):
            Z_k = self.normalizers_[k].transform(X_modalities[k][i : i + 1])
            _ev_k, best_m_k, win_boxes = self.fmnns_[k].compute_evidence(Z_k)
            modality_details.append(
                {
                    "modality_index": k,
                    "is_present": bool(mask_arr[i, k]),
                    "raw_reliability": float(reliabilities[k][i]),
                    "fusion_weight_alpha": float(alphas[i, k]),
                    "calibrated_distribution": calibrated_probs[k][i].tolist(),
                    "top_membership": float(best_m_k[0]),
                    "winning_hyperbox_per_class": win_boxes[0].tolist(),
                }
            )

        return {
            "sample_index": i,
            "predicted_class": int(np.argmax(fused[i])),
            "should_abstain": bool(abstain[i]),
            "fused_probabilities": fused[i].tolist(),
            "fusion_method": self.fusion_method,
            "uncertainty_metrics": {
                "predictive_entropy": float(uncertainty["entropy"][i]),
                "max_probability": float(uncertainty["max_probability"][i]),
                "class_margin": float(uncertainty["margin"][i]),
                "total_reliability": float(uncertainty["total_reliability"][i]),
                "modality_disagreement": float(uncertainty["disagreement"][i]),
            },
            "modalities": modality_details,
        }
