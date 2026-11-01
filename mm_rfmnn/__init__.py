"""MM-RFMNN: Multimodal Reliability-Aware Fuzzy Min-Max Neural Network

A robust, interpretable multimodal classification framework under modality noise,
corruption, and missingness.
"""

from mm_rfmnn.base.hyperbox import FMNN, HyperboxBank
from mm_rfmnn.encoders.normalizer import ModalityNormalizer
from mm_rfmnn.fusion.fuser import ReliabilityFuser, TemperatureCalibrator
from mm_rfmnn.model import MMRFMNNClassifier
from mm_rfmnn.reliability.estimator import MetaFeatureExtractor, ReliabilityEstimator

__all__ = [
    "FMNN",
    "HyperboxBank",
    "MMRFMNNClassifier",
    "MetaFeatureExtractor",
    "ModalityNormalizer",
    "ReliabilityEstimator",
    "ReliabilityFuser",
    "TemperatureCalibrator",
]

__version__ = "1.0.0"
