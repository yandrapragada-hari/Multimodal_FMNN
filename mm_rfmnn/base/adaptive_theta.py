"""Density-adaptive hyperbox size bound (theta) estimation.

Estimates data-driven, scale-aware theta parameters per modality using
intra-class dispersion and inter-class nearest-neighbor boundary margins.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class AdaptiveThetaEstimator:
    """Estimates optimal hyperbox expansion bound (theta) from feature density.

    Parameters
    ----------
    k_neighbors : int, default=5
        Number of nearest neighbors to evaluate intra-class spread.
    margin_safety_factor : float, default=0.5
        Fraction of inter-class separation distance used as upper bound for theta
        to prevent cross-class hyperbox overlaps before contraction.
    min_theta : float, default=0.05
        Hard lower bound on theta to avoid single-sample hyperbox explosion.
    max_theta : float, default=0.60
        Hard upper bound on theta to prevent over-generalized hyperboxes.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        margin_safety_factor: float = 0.50,
        min_theta: float = 0.05,
        max_theta: float = 0.60,
    ):
        self.k_neighbors = k_neighbors
        self.margin_safety_factor = margin_safety_factor
        self.min_theta = min_theta
        self.max_theta = max_theta

    def estimate_theta(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        modality_idx: int | None = None,
    ) -> float:
        """Estimates scalar theta bound for a single normalized modality matrix Z in [0, 1]^d."""
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n_samples, n_dim = Z.shape

        if n_samples < 4:
            return float(np.clip(0.30, self.min_theta, self.max_theta))

        classes = np.unique(y)
        intra_distances = []
        inter_distances = []

        # 1. Intra-class nearest neighbor distance (local cluster radius)
        k_eff = min(self.k_neighbors, max(n_samples // len(classes) - 1, 1))
        for c in classes:
            Z_c = Z[y == c]
            if len(Z_c) > 1:
                nn_intra = NearestNeighbors(n_neighbors=min(k_eff + 1, len(Z_c)), metric="manhattan")
                nn_intra.fit(Z_c)
                distances, _ = nn_intra.kneighbors(Z_c)
                # Ignore self-distance at column 0
                intra_distances.extend(distances[:, 1:].flatten())

        # 2. Inter-class boundary margin (distance from sample to nearest opposing class)
        if len(classes) > 1:
            for c in classes:
                Z_c = Z[y == c]
                Z_other = Z[y != c]
                if len(Z_c) > 0 and len(Z_other) > 0:
                    nn_inter = NearestNeighbors(n_neighbors=1, metric="manhattan")
                    nn_inter.fit(Z_other)
                    dist_to_other, _ = nn_inter.kneighbors(Z_c)
                    inter_distances.extend(dist_to_other.flatten())

        # Normalize distances by dimension to match coordinate expansion delta per axis
        mean_intra = (
            float(np.median(intra_distances) / max(n_dim, 1))
            if intra_distances
            else 0.20
        )
        min_inter = (
            float(np.percentile(inter_distances, 15) / max(n_dim, 1))
            if inter_distances
            else 0.40
        )

        # Optimal theta: covers local intra-cluster spread while staying below inter-class boundary
        target_theta = max(mean_intra, min_inter * self.margin_safety_factor)

        # Apply dimensionality damping (higher dimensions need slightly tighter theta)
        dim_factor = 1.0 / (1.0 + 0.05 * np.log1p(n_dim))
        final_theta = target_theta * dim_factor

        return float(np.clip(final_theta, self.min_theta, self.max_theta))

    def estimate_multimodal_thetas(
        self,
        Z_modalities: list[np.ndarray],
        y: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> list[float]:
        """Estimates list of theta bounds for all K modalities."""
        thetas = []
        for k, Z_k in enumerate(Z_modalities):
            if mask is not None:
                pres_k = mask[:, k]
                if np.any(pres_k):
                    theta_k = self.estimate_theta(Z_k[pres_k], y[pres_k], modality_idx=k)
                else:
                    theta_k = 0.30
            else:
                theta_k = self.estimate_theta(Z_k, y, modality_idx=k)
            thetas.append(theta_k)
        return thetas
