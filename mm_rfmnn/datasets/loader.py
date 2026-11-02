"""Real-world multimodal dataset loaders.

Supports UCI Multiple Features (mfeat) digit recognition dataset across 6 heterogeneous modalities:
1. Fourier coefficients of character shapes (76 dimensions)
2. Profile correlations / factors (216 dimensions)
3. Karhunen-Loève coefficients (64 dimensions)
4. Morphological features (6 dimensions)
5. Pixel averages in 2x3 windows (240 dimensions)
6. Zernike moments (47 dimensions)

All modalities are aligned across 2,000 samples and 10 digit classes (0-9).
"""

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import fetch_openml

MODALITY_SPECS: list[tuple[str, str, int]] = [
    ("fourier", "mfeat-fourier", 76),
    ("factors", "mfeat-factors", 216),
    ("karhunen", "mfeat-karhunen", 64),
    ("morphological", "mfeat-morphological", 6),
    ("pixel", "mfeat-pixel", 240),
    ("zernike", "mfeat-zernike", 47),
]


def load_uci_mfeat(
    cache_dir: str | Path | None = None,
    return_modality_names: bool = False,
) -> tuple[list[np.ndarray], np.ndarray] | tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Loads the 6-modality UCI Multiple Features digit recognition dataset.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory where cached `.npz` file is stored or loaded. Defaults to
        `data/mfeat/` within the repository root.
    return_modality_names : bool, default=False
        Whether to return the list of modality names alongside X and y.

    Returns
    -------
    X_modalities : list of 6 np.ndarray
        Feature matrices for each modality:
        - fourier: shape (2000, 76)
        - factors: shape (2000, 216)
        - karhunen: shape (2000, 64)
        - morphological: shape (2000, 6)
        - pixel: shape (2000, 240)
        - zernike: shape (2000, 47)
    y : np.ndarray of shape (2000,)
        Zero-indexed class labels in {0, ..., 9}.
    modality_names : list of str, optional
        Names of the modalities, returned if return_modality_names is True.
    """
    if cache_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        cache_dir = project_root / "data" / "mfeat"
    else:
        cache_dir = Path(cache_dir)

    cache_file = cache_dir / "mfeat.npz"
    modality_names = [spec[0] for spec in MODALITY_SPECS]

    # Attempt loading from cached npz archive
    if cache_file.exists():
        archive = np.load(cache_file)
        X_modalities = [
            np.asarray(archive[spec[0]], dtype=np.float64) for spec in MODALITY_SPECS
        ]
        y = np.asarray(archive["y"], dtype=np.int64)
        _validate_mfeat_dimensions(X_modalities, y)
        if return_modality_names:
            return X_modalities, y, modality_names
        return X_modalities, y

    # Fetch from OpenML if cache file does not exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    X_dict: dict[str, np.ndarray] = {}
    target_ref: np.ndarray | None = None

    for short_name, openml_name, expected_dim in MODALITY_SPECS:
        bunch: Any = fetch_openml(openml_name, version=1, as_frame=False)
        X_mat = np.asarray(bunch.data, dtype=np.float64)
        if X_mat.shape[1] != expected_dim:
            raise ValueError(
                f"Modality {short_name} expected {expected_dim} dimensions, got {X_mat.shape[1]}."
            )
        X_dict[short_name] = X_mat

        target = np.asarray(bunch.target, dtype=np.int64)
        if target_ref is None:
            target_ref = target
        else:
            if not np.array_equal(target_ref, target):
                raise ValueError(
                    f"Target mismatch between modalities for {openml_name}."
                )

    assert target_ref is not None
    # Map classes from 1..10 to 0..9 if needed
    if np.min(target_ref) == 1 and np.max(target_ref) == 10:
        y = target_ref - 1
    else:
        y = target_ref.copy()

    X_modalities = [X_dict[spec[0]] for spec in MODALITY_SPECS]
    _validate_mfeat_dimensions(X_modalities, y)

    # Persist cache for offline reproducibility and fast retrieval
    np.savez_compressed(
        cache_file,
        fourier=X_dict["fourier"],
        factors=X_dict["factors"],
        karhunen=X_dict["karhunen"],
        morphological=X_dict["morphological"],
        pixel=X_dict["pixel"],
        zernike=X_dict["zernike"],
        y=y,
    )

    if return_modality_names:
        return X_modalities, y, modality_names
    return X_modalities, y


def _validate_mfeat_dimensions(X_modalities: list[np.ndarray], y: np.ndarray) -> None:
    """Defensive assertions for dataset shapes and alignment."""
    if len(X_modalities) != len(MODALITY_SPECS):
        raise ValueError(
            f"Expected {len(MODALITY_SPECS)} modalities, got {len(X_modalities)}."
        )

    n_samples = len(y)
    if n_samples != 2000:
        raise ValueError(f"Expected 2000 samples, got {n_samples}.")

    for (short_name, _, expected_dim), X_k in zip(MODALITY_SPECS, X_modalities):
        if X_k.shape != (n_samples, expected_dim):
            raise ValueError(
                f"Modality {short_name} has invalid shape {X_k.shape}, expected ({n_samples}, {expected_dim})."
            )

    unique_classes = np.unique(y)
    if not np.array_equal(unique_classes, np.arange(10)):
        raise ValueError(f"Classes must be integers 0..9, got {unique_classes}.")
