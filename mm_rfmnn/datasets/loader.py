"""Real-world and benchmark multimodal dataset loaders.

Supports heterogeneous multimodal datasets across domain spaces:
1. UCI Multiple Features (mfeat): 6 modalities, 2,000 samples, 10 digit classes.
2. Wisconsin Diagnostic Breast Cancer (WDBC Multi-View): 3 morphological views, 569 samples, 2 classes.
3. Dermatology Multi-View (UCI Dermatology): Clinical symptoms + Histopathology views, 366 samples, 6 classes.
4. UCI Human Activity Recognition (HAR Multi-Sensor): Body Accel + Gravity + Gyroscope, 10,299 samples, 6 classes.
5. Synthetic Heterogeneous Raw: Vision (2D patches) + Audio (1D signals) + Text (strings) + Tabular, 1,000 samples, 4 classes.
"""

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer

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
    """Loads the 6-modality UCI Multiple Features digit recognition dataset."""
    if cache_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        cache_dir = project_root / "data" / "mfeat"
    else:
        cache_dir = Path(cache_dir)

    cache_file = cache_dir / "mfeat.npz"
    modality_names = [spec[0] for spec in MODALITY_SPECS]

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
    if np.min(target_ref) == 1 and np.max(target_ref) == 10:
        y = target_ref - 1
    else:
        y = target_ref.copy()

    X_modalities = [X_dict[spec[0]] for spec in MODALITY_SPECS]
    _validate_mfeat_dimensions(X_modalities, y)

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
    """Defensive assertions for UCI mfeat shapes and alignment."""
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


def load_wdbc_multiview(
    cache_dir: str | Path | None = None,
    return_modality_names: bool = False,
) -> tuple[list[np.ndarray], np.ndarray] | tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Loads Wisconsin Diagnostic Breast Cancer (WDBC) divided into 3 morphological views.

    Modalities:
    1. nucleus_mean: 10 dimensions (mean values of cell nuclei characteristics)
    2. nucleus_se: 10 dimensions (standard error of characteristics)
    3. nucleus_worst: 10 dimensions (largest / 'worst' values of characteristics)
    """
    if cache_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        cache_dir = project_root / "data" / "wdbc"
    else:
        cache_dir = Path(cache_dir)

    cache_file = cache_dir / "wdbc.npz"
    modality_names = ["nucleus_mean", "nucleus_se", "nucleus_worst"]

    if cache_file.exists():
        archive = np.load(cache_file)
        X_modalities = [
            np.asarray(archive[m], dtype=np.float64) for m in modality_names
        ]
        y = np.asarray(archive["y"], dtype=np.int64)
        if return_modality_names:
            return X_modalities, y, modality_names
        return X_modalities, y

    cache_dir.mkdir(parents=True, exist_ok=True)
    bunch = load_breast_cancer()
    X_full = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target, dtype=np.int64)

    X_mean = X_full[:, :10]
    X_se = X_full[:, 10:20]
    X_worst = X_full[:, 20:30]
    X_modalities = [X_mean, X_se, X_worst]

    np.savez_compressed(
        cache_file,
        nucleus_mean=X_mean,
        nucleus_se=X_se,
        nucleus_worst=X_worst,
        y=y,
    )

    if return_modality_names:
        return X_modalities, y, modality_names
    return X_modalities, y


def load_dermatology_multiview(
    cache_dir: str | Path | None = None,
    return_modality_names: bool = False,
) -> tuple[list[np.ndarray], np.ndarray] | tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Loads UCI Dermatology multi-view dataset (OpenML did=35).

    Modalities:
    1. clinical_symptoms: 12 clinical evaluation features (erythema, scaling, borders, etc.)
    2. histopathology: 22 microscopic biopsy features (melanin, hyperkeratosis, acanthosis, etc.)
    """
    if cache_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        cache_dir = project_root / "data" / "dermatology"
    else:
        cache_dir = Path(cache_dir)

    cache_file = cache_dir / "dermatology.npz"
    modality_names = ["clinical_symptoms", "histopathology"]

    if cache_file.exists():
        archive = np.load(cache_file)
        X_modalities = [
            np.asarray(archive[m], dtype=np.float64) for m in modality_names
        ]
        y = np.asarray(archive["y"], dtype=np.int64)
        if return_modality_names:
            return X_modalities, y, modality_names
        return X_modalities, y

    cache_dir.mkdir(parents=True, exist_ok=True)
    df, y_raw = fetch_openml(data_id=35, as_frame=True, return_X_y=True)
    df_clean = df.copy()
    if "Age" in df_clean.columns:
        df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median()).astype(float)

    X_mat = df_clean.to_numpy(dtype=np.float64)
    y = np.asarray(y_raw, dtype=np.int64)
    if np.min(y) == 1:
        y = y - 1

    X_clinical = X_mat[:, :12]
    X_histo = X_mat[:, 12:]
    X_modalities = [X_clinical, X_histo]

    np.savez_compressed(
        cache_file,
        clinical_symptoms=X_clinical,
        histopathology=X_histo,
        y=y,
    )

    if return_modality_names:
        return X_modalities, y, modality_names
    return X_modalities, y


def load_uci_har(
    cache_dir: str | Path | None = None,
    return_modality_names: bool = False,
    max_samples: int | None = None,
    random_state: int = 42,
) -> tuple[list[np.ndarray], np.ndarray] | tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Loads UCI Human Activity Recognition (HAR) multi-sensor dataset (OpenML did=1478).

    Modalities:
    1. body_accelerometer: 200 dimensions (Body linear acceleration time & frequency components)
    2. gravity_sensor: 80 dimensions (Gravity vector and angle components)
    3. gyroscope: 281 dimensions (Body angular velocity and angular jerk components)
    """
    if cache_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        cache_dir = project_root / "data" / "har"
    else:
        cache_dir = Path(cache_dir)

    cache_file = cache_dir / "har.npz"
    modality_names = ["body_accelerometer", "gravity_sensor", "gyroscope"]

    if cache_file.exists():
        archive = np.load(cache_file)
        X_modalities = [
            np.asarray(archive[m], dtype=np.float64) for m in modality_names
        ]
        y = np.asarray(archive["y"], dtype=np.int64)
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        har_bunch = fetch_openml(data_id=1478, as_frame=False)
        X_all = np.asarray(har_bunch.data, dtype=np.float64)
        y_all = np.asarray(har_bunch.target, dtype=np.int64)
        if np.min(y_all) == 1:
            y_all = y_all - 1

        X_acc = X_all[:, :200]
        X_grav = X_all[:, 200:280]
        X_gyro = X_all[:, 280:]
        X_modalities = [X_acc, X_grav, X_gyro]
        y = y_all

        np.savez_compressed(
            cache_file,
            body_accelerometer=X_acc,
            gravity_sensor=X_grav,
            gyroscope=X_gyro,
            y=y,
        )

    if max_samples is not None and max_samples < len(y):
        rng = np.random.default_rng(random_state)
        indices: list[int] = []
        n_classes = len(np.unique(y))
        samples_per_class = max(1, max_samples // n_classes)
        for c in range(n_classes):
            idx_c = np.where(y == c)[0]
            chosen = rng.choice(idx_c, size=min(samples_per_class, len(idx_c)), replace=False)
            indices.extend(chosen.tolist())
        idx_arr = np.asarray(indices, dtype=np.int64)
        X_modalities = [X_k[idx_arr] for X_k in X_modalities]
        y = y[idx_arr]

    if return_modality_names:
        return X_modalities, y, modality_names
    return X_modalities, y


def load_synthetic_multimodal_raw(
    n_samples: int = 1000,
    n_classes: int = 4,
    random_state: int = 42,
    return_modality_names: bool = False,
) -> tuple[list[Any], np.ndarray] | tuple[list[Any], np.ndarray, list[str]]:
    """Generates a heterogeneous multimodal dataset with raw vision, audio, text, and tabular channels."""
    rng = np.random.default_rng(random_state)
    modality_names = ["vision_raw", "audio_raw", "text_raw", "tabular_raw"]

    y = rng.integers(0, n_classes, size=n_samples)

    # 1. Vision: 16x16 image patches with class-correlated spatial centers
    images = np.zeros((n_samples, 16, 16), dtype=np.float32)
    for i in range(n_samples):
        cls = y[i]
        center_r = 4 + (cls % 2) * 8
        center_c = 4 + (cls // 2) * 8
        rr, cc = np.ogrid[:16, :16]
        dist = np.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2)
        images[i] = np.exp(-dist / 3.0) + rng.normal(0, 0.05, size=(16, 16)).astype(np.float32)

    # 2. Audio: 1D waveform with harmonic frequencies
    audio_signals = np.zeros((n_samples, 256), dtype=np.float32)
    t = np.linspace(0, 1.0, 256, endpoint=False)
    for i in range(n_samples):
        base_freq = 5.0 * (y[i] + 1)
        audio_signals[i] = (
            np.sin(2 * np.pi * base_freq * t)
            + 0.5 * np.cos(4 * np.pi * base_freq * t)
            + rng.normal(0, 0.1, size=256).astype(np.float32)
        )

    # 3. Text: Semantic descriptions with class keywords
    vocab_map = {
        0: ["alpha", "beacon", "clarity", "dawn", "emerald"],
        1: ["bravo", "crimson", "delta", "flame", "garnet"],
        2: ["charlie", "indigo", "echo", "frost", "sapphire"],
        3: ["delta", "amber", "foxtrot", "glow", "topaz"],
    }
    texts: list[str] = []
    for i in range(n_samples):
        cls = int(y[i])
        chosen_words = rng.choice(vocab_map[cls], size=4, replace=True)
        noise_words = rng.choice(["sensor", "record", "reading", "channel"], size=2, replace=True)
        all_words = list(chosen_words) + list(noise_words)
        rng.shuffle(all_words)
        texts.append(" ".join(all_words))

    # 4. Tabular: 8 numerical sensor channels with non-linear shifts
    tabular = np.zeros((n_samples, 8), dtype=np.float32)
    for i in range(n_samples):
        cls = y[i]
        tabular[i] = rng.normal(loc=cls * 1.5, scale=0.8, size=8)

    X_modalities: list[Any] = [images, audio_signals, texts, tabular]

    if return_modality_names:
        return X_modalities, y, modality_names
    return X_modalities, y


def load_multimodal_dataset(
    name: str,
    **kwargs: Any,
) -> tuple[list[Any], np.ndarray] | tuple[list[Any], np.ndarray, list[str]]:
    """Unified dispatcher for multimodal datasets."""
    canonical = name.lower().strip()
    if canonical in ("mfeat", "uci_mfeat", "multiple_features"):
        return load_uci_mfeat(**kwargs)
    elif canonical in ("wdbc", "breast_cancer", "wdbc_multiview"):
        return load_wdbc_multiview(**kwargs)
    elif canonical in ("dermatology", "uci_dermatology"):
        return load_dermatology_multiview(**kwargs)
    elif canonical in ("har", "uci_har", "activity_recognition"):
        return load_uci_har(**kwargs)
    elif canonical in ("synthetic_raw", "raw_multimodal"):
        return load_synthetic_multimodal_raw(**kwargs)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Available options: 'mfeat', 'wdbc', 'dermatology', 'har', 'synthetic_raw'."
        )
