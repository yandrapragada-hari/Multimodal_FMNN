"""Modality-specific normalization and preprocessing fitted strictly on training data."""

import numpy as np


class ModalityNormalizer:
    """Scales raw features to [0, 1] range based on training data.

    Fitted strictly on training partition to eliminate data leakage.
    Supports min-max scaling, outlier-resistant quantile bounds, and incremental partial_fit.
    """

    def __init__(
        self,
        method: str = "minmax",
        quantile_range: tuple = (1.0, 99.0),
        clip: bool = False,
        eps: float = 1e-7,
    ):
        self.method = method
        self.quantile_range = quantile_range
        self.clip = clip
        self.eps = eps
        self.min_vals: np.ndarray | None = None
        self.max_vals: np.ndarray | None = None

    def fit(
        self, X: np.ndarray, mask: np.ndarray | None = None
    ) -> "ModalityNormalizer":
        X = np.asarray(X, dtype=np.float64)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            X_valid = X[mask]
        else:
            X_valid = X

        if len(X_valid) == 0:
            raise ValueError("No valid samples provided to fit ModalityNormalizer.")

        if self.method == "minmax":
            self.min_vals = np.min(X_valid, axis=0)
            self.max_vals = np.max(X_valid, axis=0)
        elif self.method == "quantile":
            q_low, q_high = self.quantile_range
            self.min_vals = np.percentile(X_valid, q_low, axis=0)
            self.max_vals = np.percentile(X_valid, q_high, axis=0)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        # Guard against zero-variance features
        diff = self.max_vals - self.min_vals
        self.max_vals = np.where(diff < self.eps, self.min_vals + 1.0, self.max_vals)
        return self

    def partial_fit(
        self, X: np.ndarray, mask: np.ndarray | None = None
    ) -> "ModalityNormalizer":
        """Incrementally expands min/max bounds with streaming mini-batches."""
        X = np.asarray(X, dtype=np.float64)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            X_valid = X[mask]
        else:
            X_valid = X

        if len(X_valid) == 0:
            return self

        if self.min_vals is None or self.max_vals is None:
            return self.fit(X_valid)

        batch_min = np.min(X_valid, axis=0)
        batch_max = np.max(X_valid, axis=0)

        self.min_vals = np.minimum(self.min_vals, batch_min)
        self.max_vals = np.maximum(self.max_vals, batch_max)

        # Guard against zero-variance features
        diff = self.max_vals - self.min_vals
        self.max_vals = np.where(diff < self.eps, self.min_vals + 1.0, self.max_vals)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_vals is None or self.max_vals is None:
            raise RuntimeError("ModalityNormalizer must be fitted before transform.")
        X = np.asarray(X, dtype=np.float64)
        denom = self.max_vals - self.min_vals
        scaled = (X - self.min_vals) / denom
        if self.clip:
            return np.clip(scaled, 0.0, 1.0)
        return scaled

    def fit_transform(
        self, X: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        return self.fit(X, mask=mask).transform(X)
