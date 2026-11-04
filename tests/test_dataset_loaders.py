"""Unit tests for multimodal dataset loaders."""

import numpy as np

from mm_rfmnn.datasets.loader import (
    load_dermatology_multiview,
    load_multimodal_dataset,
    load_synthetic_multimodal_raw,
    load_uci_har,
    load_uci_mfeat,
    load_wdbc_multiview,
)


def test_load_uci_mfeat():
    X, y, names = load_uci_mfeat(return_modality_names=True)
    assert len(X) == 6
    assert len(names) == 6
    assert len(y) == 2000
    assert set(np.unique(y)) == set(range(10))
    for x_k in X:
        assert x_k.shape[0] == 2000


def test_load_wdbc_multiview():
    X, y, names = load_wdbc_multiview(return_modality_names=True)
    assert len(X) == 3
    assert names == ["nucleus_mean", "nucleus_se", "nucleus_worst"]
    assert len(y) == 569
    assert set(np.unique(y)) == {0, 1}
    for x_k in X:
        assert x_k.shape == (569, 10)


def test_load_dermatology_multiview():
    X, y, names = load_dermatology_multiview(return_modality_names=True)
    assert len(X) == 2
    assert names == ["clinical_symptoms", "histopathology"]
    assert len(y) == 366
    assert len(np.unique(y)) == 6
    assert X[0].shape == (366, 12)
    assert X[1].shape == (366, 22)


def test_load_uci_har_subsampling():
    X, y, names = load_uci_har(return_modality_names=True, max_samples=300)
    assert len(X) == 3
    assert names == ["body_accelerometer", "gravity_sensor", "gyroscope"]
    assert len(y) == 300
    assert len(np.unique(y)) == 6
    assert X[0].shape == (300, 200)
    assert X[1].shape == (300, 80)
    assert X[2].shape == (300, 281)


def test_load_synthetic_multimodal_raw():
    X, y, names = load_synthetic_multimodal_raw(n_samples=100, n_classes=3, return_modality_names=True)
    assert len(X) == 4
    assert names == ["vision_raw", "audio_raw", "text_raw", "tabular_raw"]
    assert len(y) == 100
    assert X[0].shape == (100, 16, 16)
    assert X[1].shape == (100, 256)
    assert len(X[2]) == 100
    assert isinstance(X[2][0], str)
    assert X[3].shape == (100, 8)


def test_load_multimodal_dataset_dispatcher():
    X, y = load_multimodal_dataset("wdbc")
    assert len(X) == 3
    assert len(y) == 569

    X_s, y_s = load_multimodal_dataset("synthetic_raw", n_samples=50)
    assert len(X_s) == 4
    assert len(y_s) == 50
