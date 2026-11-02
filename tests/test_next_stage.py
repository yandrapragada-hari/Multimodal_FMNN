"""Unit and integration test suite for Next-Stage Phase 3 capabilities:

1. UCI Multiple Features real multimodal dataset loading and alignment.
2. Principled hyperbox pruning (low-support outlier elimination and geometric redundancy subsumption).
3. Incremental online learning and streaming adaptation (partial_fit).
4. Memory-bounded stream adaptation with OnlineStreamAdapter.
"""

import unittest

import numpy as np

from mm_rfmnn.base.hyperbox import FMNN, HyperboxBank
from mm_rfmnn.base.incremental import OnlineStreamAdapter
from mm_rfmnn.base.pruning import HyperboxPruner
from mm_rfmnn.datasets.loader import load_uci_mfeat
from mm_rfmnn.model import MMRFMNNClassifier


class TestNextStage(unittest.TestCase):
    """Test suite for Phase 3 next-stage capabilities."""

    def setUp(self):
        np.random.seed(42)
        self.N = 150
        self.y = np.tile([0, 1, 2], 50)
        self.X1 = np.random.randn(self.N, 5) + self.y[:, np.newaxis] * 2.0
        self.X2 = np.random.randn(self.N, 4) + self.y[:, np.newaxis] * 2.5

    def test_uci_mfeat_loader(self):
        """Verifies loading, alignment, and shapes of real UCI Multiple Features dataset."""
        X_modalities, y, names = load_uci_mfeat(return_modality_names=True)
        self.assertEqual(len(X_modalities), 6)
        self.assertEqual(len(y), 2000)
        self.assertEqual(len(names), 6)

        # Expected dimensions
        expected_dims = [76, 216, 64, 6, 240, 47]
        for X_k, exp_dim in zip(X_modalities, expected_dims):
            self.assertEqual(X_k.shape, (2000, exp_dim))

        # Check classes: 10 classes, exactly 200 samples each
        unique_classes, counts = np.unique(y, return_counts=True)
        self.assertEqual(len(unique_classes), 10)
        np.testing.assert_array_equal(unique_classes, np.arange(10))
        np.testing.assert_array_equal(counts, np.full(10, 200))

    def test_hyperbox_pruner_redundancy_and_support(self):
        """Tests geometric subsumption and low-support outlier pruning on HyperboxBank."""
        bank = HyperboxBank(dimension=2, gamma=1.0)
        # Box 0: large box for class 0, support 10
        bank.add_box(np.array([0.1, 0.1]), np.array([0.5, 0.5]), class_idx=0, sample_count=10)
        # Box 1: strictly inside Box 0, class 0, support 2 -> redundant!
        bank.add_box(np.array([0.2, 0.2]), np.array([0.3, 0.3]), class_idx=0, sample_count=2)
        # Box 2: outlier box for class 0, support 1 -> low support
        bank.add_box(np.array([0.8, 0.8]), np.array([0.85, 0.85]), class_idx=0, sample_count=1)
        # Box 3: single box for class 1, support 1 -> low support but sole representative of class 1!
        bank.add_box(np.array([0.2, 0.7]), np.array([0.4, 0.9]), class_idx=1, sample_count=1)

        self.assertEqual(bank.num_boxes, 4)

        pruner = HyperboxPruner(min_samples=2, overlap_threshold=0.95, preserve_class_presence=True)
        stats = pruner.prune(bank)

        # Box 1 should be pruned (redundant), Box 2 should be pruned (low support).
        # Box 3 should be preserved because preserve_class_presence=True.
        # Box 0 should be retained and absorbed Box 1's count: 10 + 2 = 12.
        self.assertEqual(stats["initial_boxes"], 4)
        self.assertEqual(stats["final_boxes"], 2)
        self.assertEqual(stats["pruned_boxes"], 2)
        self.assertEqual(stats["pruned_redundant"], 1)
        self.assertEqual(stats["pruned_low_support"], 1)
        self.assertEqual(bank.num_boxes, 2)
        self.assertEqual(bank.sample_counts[0], 12)
        self.assertEqual(bank.classes, [0, 1])

    def test_fmnn_incremental_partial_fit(self):
        """Tests FMNN partial_fit on streaming chunks."""
        fmnn = FMNN(theta=0.3, gamma=1.0)
        X_chunk1 = np.array([[0.1, 0.1], [0.15, 0.15]])
        y_chunk1 = np.array([0, 0])
        fmnn.partial_fit(X_chunk1, y_chunk1)
        self.assertGreaterEqual(fmnn.bank.num_boxes, 1)

        X_chunk2 = np.array([[0.8, 0.8], [0.85, 0.85]])
        y_chunk2 = np.array([1, 1])
        fmnn.partial_fit(X_chunk2, y_chunk2)
        self.assertGreaterEqual(fmnn.bank.num_boxes, 2)
        self.assertEqual(len(fmnn.classes_), 2)

        preds = fmnn.predict(np.array([[0.12, 0.12], [0.82, 0.82]]))
        np.testing.assert_array_equal(preds, [0, 1])

    def test_mm_rfmnn_classifier_prune_hyperboxes(self):
        """Tests MMRFMNNClassifier prune_hyperboxes method."""
        clf = MMRFMNNClassifier(
            theta=0.15,  # smaller theta creates more granular hyperboxes
            gamma=1.0,
            fusion_method="arithmetic",
            abstention_enabled=False,
            random_state=42,
        )
        clf.fit([self.X1, self.X2], self.y)
        initial_stats = clf.get_hyperbox_stats()
        self.assertGreater(initial_stats["total_boxes"], 0)

        # Prune hyperboxes with min_samples=2
        prune_report = clf.prune_hyperboxes(min_samples=2, overlap_threshold=0.95)
        final_stats = clf.get_hyperbox_stats()

        self.assertEqual(prune_report["total_initial_boxes"], initial_stats["total_boxes"])
        self.assertEqual(prune_report["total_final_boxes"], final_stats["total_boxes"])
        self.assertLessEqual(final_stats["total_boxes"], initial_stats["total_boxes"])

        # Model should still predict accurately after pruning
        preds = clf.predict([self.X1, self.X2])
        acc = float(np.mean(preds == self.y))
        self.assertGreater(acc, 0.70)

    def test_mm_rfmnn_classifier_partial_fit(self):
        """Tests MMRFMNNClassifier streaming incremental adaptation via partial_fit."""
        clf = MMRFMNNClassifier(
            theta=0.3,
            gamma=1.0,
            fusion_method="arithmetic",
            abstention_enabled=False,
            random_state=42,
        )

        # Stream chunk 1: 50 samples
        clf.partial_fit([self.X1[:50], self.X2[:50]], self.y[:50], classes=np.array([0, 1, 2]))
        stats1 = clf.get_hyperbox_stats()
        self.assertGreater(stats1["total_boxes"], 0)

        # Stream chunk 2: next 50 samples
        clf.partial_fit([self.X1[50:100], self.X2[50:100]], self.y[50:100])
        stats2 = clf.get_hyperbox_stats()
        self.assertGreaterEqual(stats2["total_boxes"], stats1["total_boxes"])

        # Predict on remaining 50 samples
        preds = clf.predict([self.X1[100:], self.X2[100:]])
        self.assertEqual(len(preds), 50)
        acc = float(np.mean(preds == self.y[100:]))
        self.assertGreater(acc, 0.65)

    def test_online_stream_adapter(self):
        """Tests OnlineStreamAdapter managing stream chunks, rolling accuracy, and auto-pruning."""
        clf = MMRFMNNClassifier(
            theta=0.2,
            gamma=1.0,
            fusion_method="arithmetic",
            abstention_enabled=False,
            random_state=42,
        )

        adapter = OnlineStreamAdapter(
            max_boxes_per_modality=10,  # low threshold to trigger auto-prune
            auto_prune_interval=2,
            prune_min_samples=2,
        )

        # Feed 3 mini-batches
        batch_size = 30
        for i in range(3):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            info = adapter.adapt_batch(
                clf,
                [self.X1[start_idx:end_idx], self.X2[start_idx:end_idx]],
                self.y[start_idx:end_idx],
            )
            self.assertEqual(info["batch_index"], i + 1)
            self.assertEqual(info["samples_in_batch"], batch_size)

        summary = adapter.get_stream_summary()
        self.assertEqual(summary["batches_processed"], 3)
        self.assertEqual(summary["total_samples_seen"], 90)
        self.assertGreaterEqual(summary["pruning_events_count"], 1)


if __name__ == "__main__":
    unittest.main()
