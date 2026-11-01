"""Modality-specific meta-feature extraction and out-of-fold calibrated reliability estimation."""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from mm_rfmnn.base.hyperbox import FMNN


class MetaFeatureExtractor:
    """Extracts instance-level reliability features from FMNN evidence and raw inputs.

    Features:
    1. Membership strength: max_c s_c(z)
    2. Class margin: s_(1) - s_(2)
    3. Hyperbox proximity: exp(-min_j D_j(z))
    4. Optional input quality indicator (e.g. signal-to-noise, missing feature ratio)
    """

    def __init__(self):
        pass

    def extract(
        self,
        class_evidence: np.ndarray,
        best_membership: np.ndarray,
        quality_indicator: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extracts meta-features u_i for a batch of samples.

        Parameters
        ----------
        class_evidence : (N, C)
        best_membership : (N,)
        quality_indicator : (N, d_q) or None

        Returns
        -------
        u : (N, feature_dim)
        """
        _N, C = class_evidence.shape
        # 1. Membership strength
        m = best_membership.reshape(-1, 1)

        # 2. Class margin: top score minus second top score
        if C >= 2:
            sorted_scores = np.sort(class_evidence, axis=1)
            margin = (sorted_scores[:, -1] - sorted_scores[:, -2]).reshape(-1, 1)
        else:
            margin = m.copy()

        # 3. Ratio of top evidence to sum of all evidence (entropy proxy)
        evidence_sum = np.sum(class_evidence, axis=1, keepdims=True) + 1e-9
        ratio = m / evidence_sum

        features = [m, margin, ratio]

        # 4. Optional quality features
        if quality_indicator is not None:
            q_arr = np.asarray(quality_indicator, dtype=np.float64)
            if q_arr.ndim == 1:
                q_arr = q_arr.reshape(-1, 1)
            features.append(q_arr)

        return np.hstack(features)


class ReliabilityEstimator:
    """Estimates instance-level modality reliability r_i in [0, 1].

    Trained strictly on Out-Of-Fold (OOF) predictions to prevent label leakage.
    Supports learned Logistic / MLP models or rule-based fallback.
    """

    def __init__(
        self,
        model_type: str = "logistic",
        rule_based_fallback: bool = False,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.rule_based_fallback = rule_based_fallback
        self.random_state = random_state
        self.model: Any | None = None
        self.is_fitted = False

    def fit_oof(
        self,
        X_normalized: np.ndarray,
        y: np.ndarray,
        theta: float,
        gamma: float,
        n_splits: int = 5,
        quality_indicators: np.ndarray | None = None,
    ) -> "ReliabilityEstimator":
        """Fits the reliability estimator using K-fold out-of-fold targets.

        Prevents optimistic in-sample evaluation leakage.
        """
        if self.rule_based_fallback:
            self.is_fitted = True
            return self

        X_norm = np.asarray(X_normalized, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        len(y)

        # 1. Generate OOF predictions and meta-features
        oof_meta_features: list[np.ndarray] = []
        oof_targets: list[int] = []

        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        meta_extractor = MetaFeatureExtractor()

        for train_idx, val_idx in skf.split(X_norm, y):
            X_tr, y_tr = X_norm[train_idx], y[train_idx]
            X_v, y_v = X_norm[val_idx], y[val_idx]

            # Fit fold FMNN
            fmnn_fold = FMNN(theta=theta, gamma=gamma).fit(X_tr, y_tr)
            evidence_v, best_m_v, _ = fmnn_fold.compute_evidence(X_v)
            y_pred_v = np.argmax(evidence_v, axis=1)

            # Ground-truth correctness target t_i
            t_v = (y_pred_v == y_v).astype(int)

            q_v = (
                quality_indicators[val_idx] if quality_indicators is not None else None
            )
            u_v = meta_extractor.extract(evidence_v, best_m_v, q_v)

            oof_meta_features.append(u_v)
            oof_targets.append(t_v)

        U_all = np.vstack(oof_meta_features)
        T_all = np.concatenate(oof_targets)

        # If all samples are correct or all incorrect, fallback to rule-based
        if len(np.unique(T_all)) < 2:
            self.rule_based_fallback = True
            self.is_fitted = True
            return self

        # 2. Train classifier on OOF targets
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
            )
        elif self.model_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                max_iter=500,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.model.fit(U_all, T_all)
        self.is_fitted = True
        return self

    def predict_reliability(
        self,
        meta_features: np.ndarray,
        clip_range: tuple[float, float] = (0.01, 0.99),
    ) -> np.ndarray:
        """Computes reliability scores r in [0, 1] for meta_features."""
        meta_features = np.asarray(meta_features, dtype=np.float64)
        if self.rule_based_fallback or self.model is None or not self.is_fitted:
            # Rule-based fallback: weighted combo of membership strength and margin
            m = meta_features[:, 0]
            margin = meta_features[:, 1]
            r = 0.5 * m + 0.5 * margin
            return np.clip(r, clip_range[0], clip_range[1])

        # Model predicted probability of class 1 (correctness)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(meta_features)[:, 1]
        else:
            decision = self.model.decision_function(meta_features)
            probs = 1.0 / (1.0 + np.exp(-decision))

        return np.clip(probs, clip_range[0], clip_range[1])
