"""Unit and integration test suite for MM-RFMNN."""

import unittest

import numpy as np

from mm_rfmnn.base.hyperbox import FMNN, HyperboxBank
from mm_rfmnn.encoders.normalizer import ModalityNormalizer
from mm_rfmnn.fusion.fuser import ReliabilityFuser
from mm_rfmnn.model import MMRFMNNClassifier


class TestMMRFMNN(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # 2 modalities, 3 classes, 100 samples
        self.N = 120
        self.y = np.repeat([0, 1, 2], 40)
        # Modality 1: 4 features
        self.X1 = np.random.randn(self.N, 4) + self.y[:, np.newaxis] * 1.5
        # Modality 2: 3 features
        self.X2 = np.random.randn(self.N, 3) + self.y[:, np.newaxis] * 2.0

    def test_hyperbox_membership(self):
        bank = HyperboxBank(dimension=2, gamma=1.0)
        bank.add_box(np.array([0.2, 0.2]), np.array([0.5, 0.5]), class_idx=0)
        # Point inside
        p_in = np.array([[0.3, 0.4]])
        mem_in = bank.compute_memberships(p_in)
        self.assertAlmostEqual(float(mem_in[0, 0]), 1.0, places=5)

        # Point outside: distance = (0.7 - 0.5) + (0.8 - 0.5) = 0.5
        p_out = np.array([[0.7, 0.8]])
        mem_out = bank.compute_memberships(p_out)
        expected = np.exp(-1.0 * 0.5)
        self.assertAlmostEqual(float(mem_out[0, 0]), expected, places=5)

    def test_fmnn_expansion_and_fit(self):
        fmnn = FMNN(theta=0.3, gamma=1.0)
        X_toy = np.array([[0.1, 0.1], [0.15, 0.15], [0.8, 0.8]])
        y_toy = np.array([0, 0, 1])
        fmnn.fit(X_toy, y_toy)
        # Should have at least 2 boxes (1 for class 0, 1 for class 1)
        self.assertGreaterEqual(fmnn.bank.num_boxes, 2)
        preds = fmnn.predict(X_toy)
        self.assertEqual(len(preds), 3)

    def test_normalizer(self):
        norm = ModalityNormalizer(method="minmax", clip=True)
        X = np.array([[10.0, -5.0], [20.0, 5.0], [15.0, 0.0]])
        X_scaled = norm.fit_transform(X)
        self.assertTrue(np.all(X_scaled >= 0.0))
        self.assertTrue(np.all(X_scaled <= 1.0))
        # Check transform with out-of-bounds clips safely when clip=True
        X_test = np.array([[5.0, -10.0], [30.0, 10.0]])
        X_test_scaled = norm.transform(X_test)
        self.assertEqual(float(X_test_scaled[0, 0]), 0.0)
        self.assertEqual(float(X_test_scaled[1, 0]), 1.0)

    def test_missingness_masking(self):
        fuser = ReliabilityFuser(fusion_method="arithmetic")
        rel1 = np.array([0.9, 0.8])
        rel2 = np.array([0.7, 0.6])
        # Sample 0: both present; Sample 1: modality 1 missing
        mask = np.array([[True, True], [False, True]])
        alphas, _masked_rel = fuser.compute_fusion_weights([rel1, rel2], mask=mask)

        # In sample 1, modality 1 is absent, so alpha for modality 1 must be exactly 0.0
        self.assertAlmostEqual(alphas[1, 0], 0.0, places=5)
        self.assertAlmostEqual(alphas[1, 1], 1.0, places=5)

    def test_end_to_end_classifier(self):
        clf = MMRFMNNClassifier(
            theta=0.4,
            gamma=1.0,
            fusion_method="arithmetic",
            abstention_enabled=False,
            random_state=42,
        )
        clf.fit([self.X1, self.X2], self.y)
        probs = clf.predict_proba([self.X1, self.X2])
        self.assertEqual(probs.shape, (self.N, 3))
        # Probabilities sum to 1
        sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(sums, np.ones(self.N), rtol=1e-5)

        preds = clf.predict([self.X1, self.X2])
        acc = np.mean(preds == self.y)
        self.assertGreater(acc, 0.70)

    def test_explainability_payload(self):
        clf = MMRFMNNClassifier(theta=0.3, gamma=1.0, random_state=42)
        clf.fit([self.X1, self.X2], self.y)
        explanation = clf.explain([self.X1, self.X2], sample_idx=0)
        self.assertIn("winning_hyperbox_per_class", explanation["modalities"][0])
        self.assertIn("fusion_weight_alpha", explanation["modalities"][0])
        self.assertIn("uncertainty_metrics", explanation)

    def test_abstention_on_extreme_outlier(self):
        clf = MMRFMNNClassifier(
            theta=0.3,
            gamma=2.0,
            abstention_enabled=True,
            tau_conf=0.70,
            tau_min_reliability=0.30,
            random_state=42,
        )
        clf.fit([self.X1, self.X2], self.y)
        # Extreme outlier input far from training distribution
        outlier_X1 = np.ones((1, 4)) * 999.0
        outlier_X2 = np.ones((1, 3)) * 999.0
        pred = clf.predict([outlier_X1, outlier_X2], return_abstention=True)
        # Should abstain, emitting -1
        self.assertEqual(pred[0], -1)


if __name__ == "__main__":
    unittest.main()
