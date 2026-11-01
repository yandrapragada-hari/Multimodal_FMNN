"""Fuzzy Min-Max Neural Network (FMNN) hyperbox representation, expansion, contraction,

and vectorized membership evaluation.
"""

import numpy as np


class HyperboxBank:
    """Manages hyperbox geometric bounds [V, W] and class labels for one modality."""

    def __init__(self, dimension: int, gamma: float = 1.0):
        self.dimension = dimension
        self.gamma = gamma
        self.V: list[np.ndarray] = []  # Min points, shape (d,)
        self.W: list[np.ndarray] = []  # Max points, shape (d,)
        self.classes: list[int] = []  # Class label per hyperbox

    @property
    def num_boxes(self) -> int:
        return len(self.classes)

    def add_box(self, v: np.ndarray, w: np.ndarray, class_idx: int):
        self.V.append(np.asarray(v, dtype=np.float64).copy())
        self.W.append(np.asarray(w, dtype=np.float64).copy())
        self.classes.append(int(class_idx))

    def get_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns V, W as (M, d) and classes as (M,)."""
        if self.num_boxes == 0:
            return (
                np.empty((0, self.dimension), dtype=np.float64),
                np.empty((0, self.dimension), dtype=np.float64),
                np.empty((0,), dtype=np.int64),
            )
        return (
            np.vstack(self.V),
            np.vstack(self.W),
            np.array(self.classes, dtype=np.int64),
        )

    def compute_memberships(self, Z: np.ndarray) -> np.ndarray:
        """Vectorized computation of membership scores for batch Z.

        Parameters
        ----------
        Z : np.ndarray of shape (N, d) in [0, 1]

        Returns
        -------
        memberships : np.ndarray of shape (N, M)
            Exponential Manhattan distance membership in (0, 1].
        """
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)

        N = Z.shape[0]
        M = self.num_boxes
        if M == 0:
            return np.zeros((N, 0), dtype=np.float64)

        V_arr, W_arr, _ = self.get_arrays()  # (M, d)

        # Broadcast Z: (N, 1, d) vs (1, M, d)
        Z_exp = Z[:, np.newaxis, :]
        V_exp = V_arr[np.newaxis, :, :]
        W_exp = W_arr[np.newaxis, :, :]

        d_lower = np.maximum(0.0, V_exp - Z_exp)
        d_upper = np.maximum(0.0, Z_exp - W_exp)
        outside_distance = np.sum(d_lower + d_upper, axis=-1)  # (N, M)

        return np.exp(-self.gamma * outside_distance)

    def compute_class_evidence(
        self, Z: np.ndarray, num_classes: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes class-level evidence, class margins, and winning box indices.

        Returns
        -------
        class_evidence : (N, num_classes)
        best_membership : (N,)
        winning_boxes : (N, num_classes)
        """
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)

        N = Z.shape[0]
        class_evidence = np.zeros((N, num_classes), dtype=np.float64)
        winning_boxes = np.full((N, num_classes), -1, dtype=np.int64)

        if self.num_boxes == 0:
            return class_evidence, np.zeros(N), winning_boxes

        memberships = self.compute_memberships(Z)  # (N, M)
        classes_arr = np.array(self.classes, dtype=np.int64)

        for c in range(num_classes):
            box_indices = np.where(classes_arr == c)[0]
            if len(box_indices) > 0:
                sub_mem = memberships[:, box_indices]  # (N, num_boxes_for_c)
                max_sub_idx = np.argmax(sub_mem, axis=1)
                class_evidence[:, c] = sub_mem[np.arange(N), max_sub_idx]
                winning_boxes[:, c] = box_indices[max_sub_idx]

        best_membership = np.max(class_evidence, axis=1)
        return class_evidence, best_membership, winning_boxes


class FMNN:
    """Modality-specific Fuzzy Min-Max Neural Network classifier with

    Simpson expansion, overlap contraction, and continuous membership.
    """

    def __init__(self, theta: float = 0.2, gamma: float = 1.0):
        self.theta = theta  # Max hyperbox size in [0, 1]
        self.gamma = gamma  # Distance decay coefficient
        self.bank: HyperboxBank | None = None
        self.classes_: np.ndarray = np.array([])
        self.num_classes_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FMNN":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        N, d = X.shape

        self.bank = HyperboxBank(dimension=d, gamma=self.gamma)
        self.classes_ = np.unique(y)
        self.num_classes_ = int(np.max(y)) + 1 if len(self.classes_) > 0 else 0

        for i in range(N):
            x_i = X[i]
            y_i = int(y[i])
            self._expand_or_create(x_i, y_i)

        return self

    def _can_expand(self, v: np.ndarray, w: np.ndarray, x: np.ndarray) -> bool:
        """Checks if expanding box (v, w) with point x satisfies dimension-wise size limit theta."""
        new_v = np.minimum(v, x)
        new_w = np.maximum(w, x)
        return bool(np.all((new_w - new_v) <= self.theta + 1e-9))

    def _expand_or_create(self, x: np.ndarray, class_idx: int):
        assert self.bank is not None
        # 1. Search for existing hyperbox of the same class that can accommodate x
        expanded = False
        for j in range(self.bank.num_boxes):
            if self.bank.classes[j] == class_idx:
                v = self.bank.V[j]
                w = self.bank.W[j]
                if self._can_expand(v, w, x):
                    # Expand box j
                    self.bank.V[j] = np.minimum(v, x)
                    self.bank.W[j] = np.maximum(w, x)
                    expanded = True
                    # Resolve overlaps with hyperboxes of other classes
                    self._resolve_overlaps(j)
                    break

        # 2. If no box can accommodate x without exceeding theta, create new box
        if not expanded:
            self.bank.add_box(x, x, class_idx)

    def _resolve_overlaps(self, j: int):
        """Simpson's overlap test and dimensional contraction for box j against other-class boxes."""
        assert self.bank is not None
        v_j = self.bank.V[j]
        w_j = self.bank.W[j]
        c_j = self.bank.classes[j]

        for k in range(self.bank.num_boxes):
            if k == j or self.bank.classes[k] == c_j:
                continue

            v_k = self.bank.V[k]
            w_k = self.bank.W[k]

            # Check overlap in all dimensions
            overlap = np.all((w_j >= v_k) & (w_k >= v_j))
            if not overlap:
                continue

            # Simpson 4-case dimensional contraction on dimension with minimal overlap
            overlaps_dim = np.minimum(w_j, w_k) - np.maximum(v_j, v_k)
            dim_min = int(np.argmin(overlaps_dim))

            # Case 1: v_j < v_k < w_j < w_k -> contract w_j and v_k
            if v_j[dim_min] < v_k[dim_min] <= w_j[dim_min] < w_k[dim_min]:
                mid = (w_j[dim_min] + v_k[dim_min]) / 2.0
                self.bank.W[j][dim_min] = mid
                self.bank.V[k][dim_min] = mid
            # Case 2: v_k < v_j < w_k < w_j -> contract w_k and v_j
            elif v_k[dim_min] < v_j[dim_min] <= w_k[dim_min] < w_j[dim_min]:
                mid = (w_k[dim_min] + v_j[dim_min]) / 2.0
                self.bank.W[k][dim_min] = mid
                self.bank.V[j][dim_min] = mid
            # Case 3: v_j < v_k <= w_k < w_j -> contract box j around box k
            elif v_j[dim_min] <= v_k[dim_min] and w_k[dim_min] <= w_j[dim_min]:
                if (w_k[dim_min] - v_j[dim_min]) < (w_j[dim_min] - v_k[dim_min]):
                    self.bank.V[j][dim_min] = w_k[dim_min]
                else:
                    self.bank.W[j][dim_min] = v_k[dim_min]
            # Case 4: v_k < v_j <= w_j < w_k -> contract box k around box j
            elif v_k[dim_min] <= v_j[dim_min] and w_j[dim_min] <= w_k[dim_min]:
                if (w_j[dim_min] - v_k[dim_min]) < (w_k[dim_min] - v_j[dim_min]):
                    self.bank.V[k][dim_min] = w_j[dim_min]
                else:
                    self.bank.W[k][dim_min] = v_j[dim_min]

    def compute_evidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.bank is not None
        return self.bank.compute_class_evidence(X, self.num_classes_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        evidence, _, _ = self.compute_evidence(X)
        return np.argmax(evidence, axis=1)
