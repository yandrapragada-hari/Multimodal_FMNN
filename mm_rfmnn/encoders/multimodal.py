"""Turnkey multimodal encoders for raw tabular, image, audio, and text inputs.

Strictly adheres to standard library, NumPy, SciPy, and Scikit-Learn dependencies.
Outputs compact normalized embeddings strictly in [0, 1]^d for FMNN hyperbox banks.
"""


import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import QuantileTransformer

from mm_rfmnn.encoders.normalizer import ModalityNormalizer


class TabularQuantileEncoder(BaseEstimator, TransformerMixin):
    """Robust tabular encoder using Quantile Transformation to [0, 1].

    Warping non-Gaussian / heavy-tailed numerical distributions uniformly into [0, 1].
    Prevents extreme outliers from collapsing hyperbox banks into corners.
    """

    def __init__(self, n_quantiles: int = 100, random_state: int = 42):
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self.transformer_ = QuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution="uniform",
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "TabularQuantileEncoder":
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        actual_quantiles = min(self.n_quantiles, max(n_samples, 2))
        self.transformer_.set_params(n_quantiles=actual_quantiles)
        self.transformer_.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Z = self.transformer_.transform(X)
        return np.clip(Z, 0.0, 1.0)


class ImageFeatureEncoder(BaseEstimator, TransformerMixin):
    """Compact image feature encoder for raw 2D/3D image arrays or PIL images.

    Extracts multi-scale spatial window statistics (mean, variance, Sobel gradients)
    and projects via TruncatedSVD/PCA into compact d-dimensional vectors in [0, 1]^d.
    """

    def __init__(self, n_components: int = 32, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd_: TruncatedSVD | None = None
        self.normalizer_ = ModalityNormalizer(method="minmax")

    def _extract_image_features(self, img: np.ndarray) -> np.ndarray:
        """Extracts spatial intensity, gradient, and block statistics from a single image."""
        arr = np.asarray(img, dtype=np.float64)
        if arr.ndim == 3:
            # Convert RGB to grayscale via standard luminance weights
            arr = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]

        H, W = arr.shape
        # Global statistical moments
        feats = [np.mean(arr), np.std(arr), np.percentile(arr, 10), np.percentile(arr, 90)]

        # Spatial 4x4 block pooling (16 blocks * 2 features = 32 features)
        grid_h, grid_w = 4, 4
        h_step = max(H // grid_h, 1)
        w_step = max(W // grid_w, 1)
        for r in range(grid_h):
            for c in range(grid_w):
                block = arr[r * h_step : (r + 1) * h_step, c * w_step : (c + 1) * w_step]
                if block.size > 0:
                    feats.extend([np.mean(block), np.std(block)])
                else:
                    feats.extend([0.0, 0.0])

        # Simple Sobel-like discrete gradient energy (edges)
        diff_y = np.abs(np.diff(arr, axis=0))
        diff_x = np.abs(np.diff(arr, axis=1))
        feats.extend([np.mean(diff_y), np.std(diff_y), np.mean(diff_x), np.std(diff_x)])

        return np.asarray(feats, dtype=np.float64)

    def fit(self, X: np.ndarray | list[np.ndarray], y: np.ndarray | None = None) -> "ImageFeatureEncoder":
        raw_feats = np.array([self._extract_image_features(img) for img in X])
        d_raw = raw_feats.shape[1]
        n_comp = min(self.n_components, d_raw, raw_feats.shape[0])

        if n_comp < d_raw:
            self.svd_ = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
            reduced = self.svd_.fit_transform(raw_feats)
        else:
            self.svd_ = None
            reduced = raw_feats

        self.normalizer_.fit(reduced)
        return self

    def transform(self, X: np.ndarray | list[np.ndarray]) -> np.ndarray:
        raw_feats = np.array([self._extract_image_features(img) for img in X])
        if self.svd_ is not None:
            reduced = self.svd_.transform(raw_feats)
        else:
            reduced = raw_feats
        return self.normalizer_.transform(reduced)


class AudioFeatureEncoder(BaseEstimator, TransformerMixin):
    """Compact audio feature encoder for 1D raw waveforms or audio signals.

    Computes short-time energy, zero-crossing rate, spectral centroid, spectral rolloff,
    and multi-band filterbank energies via SciPy FFT. Projects to [0, 1]^d.
    """

    def __init__(self, n_components: int = 24, sample_rate: int = 16000, random_state: int = 42):
        self.n_components = n_components
        self.sample_rate = sample_rate
        self.random_state = random_state
        self.svd_: TruncatedSVD | None = None
        self.normalizer_ = ModalityNormalizer(method="minmax")

    def _extract_audio_features(self, wav: np.ndarray) -> np.ndarray:
        sig = np.asarray(wav, dtype=np.float64).flatten()
        if len(sig) == 0:
            return np.zeros(self.n_components, dtype=np.float64)

        # 1. Zero crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(sig)))) / 2.0

        # 2. RMS energy
        rms = np.sqrt(np.mean(sig**2) + 1e-12)

        # 3. FFT Magnitude Spectrum
        fft_vals = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / self.sample_rate)

        total_energy = np.sum(fft_vals) + 1e-12
        # Spectral centroid
        centroid = np.sum(freqs * fft_vals) / total_energy

        # Spectral rolloff (85% energy frequency)
        cum_energy = np.cumsum(fft_vals)
        rolloff_idx = np.searchsorted(cum_energy, 0.85 * total_energy)
        rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

        # 4. Multi-band energy partitions (8 sub-bands)
        n_bands = 8
        band_splits = np.array_split(fft_vals, n_bands)
        band_energies = [np.sum(b**2) / (len(b) + 1e-12) for b in band_splits]

        # 5. Temporal envelope dynamics (peaks via hilbert transform approximation)
        analytic = np.abs(signal.hilbert(sig)) if len(sig) > 16 else np.abs(sig)
        env_mean = np.mean(analytic)
        env_std = np.std(analytic)

        features = [zcr, rms, centroid, rolloff, env_mean, env_std] + band_energies
        return np.asarray(features, dtype=np.float64)

    def fit(self, X: np.ndarray | list[np.ndarray], y: np.ndarray | None = None) -> "AudioFeatureEncoder":
        raw_feats = np.array([self._extract_audio_features(w) for w in X])
        d_raw = raw_feats.shape[1]
        n_comp = min(self.n_components, d_raw, raw_feats.shape[0])

        if n_comp < d_raw:
            self.svd_ = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
            reduced = self.svd_.fit_transform(raw_feats)
        else:
            self.svd_ = None
            reduced = raw_feats

        self.normalizer_.fit(reduced)
        return self

    def transform(self, X: np.ndarray | list[np.ndarray]) -> np.ndarray:
        raw_feats = np.array([self._extract_audio_features(w) for w in X])
        if self.svd_ is not None:
            reduced = self.svd_.transform(raw_feats)
        else:
            reduced = raw_feats
        return self.normalizer_.transform(reduced)


class TextFeatureEncoder(BaseEstimator, TransformerMixin):
    """Compact text encoder converting documents/strings into [0, 1]^d.

    Uses character/word n-gram TF-IDF vectorization followed by TruncatedSVD
    dimensionality compression and MinMax normalization into the unit hypercube.
    """

    def __init__(self, n_components: int = 32, max_features: int = 1000, random_state: int = 42):
        self.n_components = n_components
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.svd_: TruncatedSVD | None = None
        self.normalizer_ = ModalityNormalizer(method="minmax")

    def fit(self, X: list[str], y: np.ndarray | None = None) -> "TextFeatureEncoder":
        tfidf_mat = self.vectorizer_.fit_transform(X).toarray()
        n_comp = min(self.n_components, tfidf_mat.shape[1], tfidf_mat.shape[0])

        if n_comp < tfidf_mat.shape[1]:
            self.svd_ = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
            reduced = self.svd_.fit_transform(tfidf_mat)
        else:
            self.svd_ = None
            reduced = tfidf_mat

        self.normalizer_.fit(reduced)
        return self

    def transform(self, X: list[str]) -> np.ndarray:
        tfidf_mat = self.vectorizer_.transform(X).toarray()
        if self.svd_ is not None:
            reduced = self.svd_.transform(tfidf_mat)
        else:
            reduced = tfidf_mat
        return self.normalizer_.transform(reduced)
