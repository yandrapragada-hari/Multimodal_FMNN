"""Comprehensive verification test suite for Phase 4 extensions:

1. Turnkey multimodal encoders (Tabular, Image, Audio, Text).
2. Density-adaptive theta estimation.
3. Cross-modal conflict and semantic contradiction detection.
4. Visual explainability and diagnostic plotting suite.
5. End-to-end integration with MMRFMNNClassifier.
"""

import unittest

import numpy as np

from mm_rfmnn.base.adaptive_theta import AdaptiveThetaEstimator
from mm_rfmnn.base.hyperbox import FMNN
from mm_rfmnn.encoders.multimodal import (
    AudioFeatureEncoder,
    ImageFeatureEncoder,
    TabularQuantileEncoder,
    TextFeatureEncoder,
)
from mm_rfmnn.fusion.conflict import CrossModalConflictDetector
from mm_rfmnn.model import MMRFMNNClassifier
from mm_rfmnn.viz.plotter import (
    plot_hyperboxes_2d,
    plot_modality_attributions,
    plot_risk_coverage_curve,
)


class TestPhase4Extensions(unittest.TestCase):
    """Test suite for Phase 4 multimodal encoders, adaptive theta, and visualization."""

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 60
        self.y = np.tile([0, 1, 2], 20)

    def test_tabular_quantile_encoder(self):
        """Verifies TabularQuantileEncoder warps skewed distributions strictly to [0, 1]."""
        # Create heavy-tailed exponential distribution with extreme outliers
        raw_tabular = np.random.exponential(scale=5.0, size=(self.n_samples, 5))
        raw_tabular[0, 0] = 500.0  # massive outlier

        encoder = TabularQuantileEncoder(n_quantiles=30, random_state=42)
        Z = encoder.fit_transform(raw_tabular)

        self.assertEqual(Z.shape, (self.n_samples, 5))
        self.assertGreaterEqual(float(np.min(Z)), 0.0)
        self.assertLessEqual(float(np.max(Z)), 1.0)
        # Outlier should not squash the median to near 0
        self.assertGreater(float(np.median(Z[:, 0])), 0.10)

    def test_image_feature_encoder(self):
        """Verifies ImageFeatureEncoder processes 2D grayscale and 3D RGB images to [0, 1]^d."""
        images_2d = [np.random.rand(28, 28) for _ in range(self.n_samples)]
        encoder = ImageFeatureEncoder(n_components=16, random_state=42)
        Z_2d = encoder.fit_transform(images_2d)

        self.assertEqual(Z_2d.shape, (self.n_samples, 16))
        self.assertGreaterEqual(float(np.min(Z_2d)), 0.0)
        self.assertLessEqual(float(np.max(Z_2d)), 1.0)

        # 3D RGB test
        images_3d = [np.random.rand(32, 32, 3) for _ in range(10)]
        Z_3d = encoder.transform(images_3d)
        self.assertEqual(Z_3d.shape, (10, 16))

    def test_audio_feature_encoder(self):
        """Verifies AudioFeatureEncoder extracts spectral/temporal features into [0, 1]^d."""
        # 1-second sine wave bursts at 8000 Hz
        sample_rate = 8000
        t = np.linspace(0, 0.5, 4000)
        waveforms = [
            np.sin(2 * np.pi * (200 + label * 300) * t) + 0.1 * np.random.randn(len(t))
            for label in self.y
        ]

        encoder = AudioFeatureEncoder(n_components=12, sample_rate=sample_rate, random_state=42)
        Z_audio = encoder.fit_transform(waveforms)

        self.assertEqual(Z_audio.shape, (self.n_samples, 12))
        self.assertGreaterEqual(float(np.min(Z_audio)), 0.0)
        self.assertLessEqual(float(np.max(Z_audio)), 1.0)

    def test_text_feature_encoder(self):
        """Verifies TextFeatureEncoder vectorizes text strings into [0, 1]^d."""
        texts = [
            f"Sensor reading status normal operational state code {i % 3}"
            if (i % 2 == 0)
            else f"Critical emergency alarm vibration threshold exceeded code {i % 3}"
            for i in range(self.n_samples)
        ]

        encoder = TextFeatureEncoder(n_components=8, random_state=42)
        Z_text = encoder.fit_transform(texts)

        self.assertEqual(Z_text.shape, (self.n_samples, 8))
        self.assertGreaterEqual(float(np.min(Z_text)), 0.0)
        self.assertLessEqual(float(np.max(Z_text)), 1.0)

    def test_adaptive_theta_estimator(self):
        """Verifies AdaptiveThetaEstimator estimates scale-appropriate theta parameters."""
        estimator = AdaptiveThetaEstimator(k_neighbors=3, min_theta=0.08, max_theta=0.55)
        # Tightly clustered data
        tight_cluster = np.random.rand(40, 4) * 0.10
        y_tight = np.array([0] * 20 + [1] * 20)
        tight_cluster[20:] += 0.80  # clear separation

        theta_val = estimator.estimate_theta(tight_cluster, y_tight)
        self.assertGreaterEqual(theta_val, 0.08)
        self.assertLessEqual(theta_val, 0.55)

        # Multimodal estimation
        Z_list = [tight_cluster, np.random.rand(40, 6)]
        thetas = estimator.estimate_multimodal_thetas(Z_list, y_tight)
        self.assertEqual(len(thetas), 2)
        for t in thetas:
            self.assertTrue(0.08 <= t <= 0.55)

    def test_cross_modal_conflict_detector(self):
        """Verifies CrossModalConflictDetector identifies contradictory predictions."""
        detector = CrossModalConflictDetector(tau_conflict=0.40, min_modality_confidence=0.50)

        # Simulate 2 modalities on 3 classes
        # Modality 1 says Class 0 with 90% confidence
        p1 = np.tile([0.90, 0.05, 0.05], (10, 1))
        # Modality 2 agrees on first 5 samples, but completely contradicts on last 5 (says Class 2)
        p2 = np.tile([0.90, 0.05, 0.05], (10, 1))
        p2[5:] = [0.05, 0.05, 0.90]

        detector.fit([p1[:5], p2[:5]], np.zeros(5, dtype=int))
        scores = detector.compute_conflict_scores([p1, p2])

        self.assertEqual(len(scores), 10)
        # First 5 should have zero conflict (both agree Class 0)
        np.testing.assert_array_equal(scores[:5], np.zeros(5))
        # Last 5 should have high conflict
        self.assertTrue(np.all(scores[5:] > 0.40))
        abstain_mask = detector.check_conflict_abstention(scores)
        self.assertTrue(np.all(abstain_mask[5:]))

    def test_mm_rfmnn_adaptive_and_conflict_integration(self):
        """Tests MMRFMNNClassifier with theta='adaptive' and conflict_detection_enabled=True."""
        X1 = np.random.rand(self.n_samples, 6) + self.y[:, None] * 0.3
        X2 = np.random.rand(self.n_samples, 4) + self.y[:, None] * 0.4

        clf = MMRFMNNClassifier(
            theta="adaptive",
            gamma=1.0,
            conflict_detection_enabled=True,
            tau_conflict=0.50,
            random_state=42,
        )
        clf.fit([X1, X2], self.y)
        self.assertIsNotNone(clf.conflict_detector_)
        self.assertGreater(clf.get_hyperbox_stats()["total_boxes"], 0)

        preds = clf.predict([X1, X2])
        self.assertEqual(len(preds), self.n_samples)

        explanation = clf.explain([X1, X2], sample_idx=0)
        self.assertIn("fused_probabilities", explanation)
        self.assertIn("modalities", explanation)

    def test_visual_explainability_suite(self):
        """Verifies plotting functions execute headlessly without errors."""
        X = np.random.rand(30, 4)
        y = np.tile([0, 1, 2], 10)
        fmnn = FMNN(theta=0.3, gamma=1.0).fit(X, y)

        # 1. 2D Hyperbox plot
        fig1 = plot_hyperboxes_2d(fmnn, Z=X, y=y, feature_indices=(0, 1))
        self.assertIsNotNone(fig1)

        # 2. Risk-coverage plot
        confidences = np.random.uniform(0.5, 0.99, size=30)
        fig2 = plot_risk_coverage_curve(confidences, y, y)
        self.assertIsNotNone(fig2)

        # 3. Modality attributions plot
        clf = MMRFMNNClassifier(theta=0.3, random_state=42).fit([X[:, :2], X[:, 2:]], y)
        expl = clf.explain([X[:, :2], X[:, 2:]], sample_idx=0)
        fig3 = plot_modality_attributions(expl)
        self.assertIsNotNone(fig3)


if __name__ == "__main__":
    unittest.main()
