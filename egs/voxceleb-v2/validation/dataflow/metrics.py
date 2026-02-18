from dataclasses import dataclass

from voicesdk.distillation.metrics import DFMetricsBase


@dataclass
class DFMetricsSpeakerVerification(DFMetricsBase):
    """Comprehensive speaker verification metrics."""
    eer: float = 0.0
    eer_threshold: float = 0.0
    min_dcf_001: float = 0.0  # minDCF with p_target=0.01
    min_dcf_01: float = 0.0   # minDCF with p_target=0.1
    num_target_trials: int = 0
    num_imposter_trials: int = 0


@dataclass
class DFMetricsClassification(DFMetricsBase):
    """Classification metrics."""
    accuracy: float = 0.0
    top3_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    loss: float = 0.0
    num_samples: int = 0


@dataclass
class DFMetricsEmbeddingQuality(DFMetricsBase):
    """Embedding quality metrics."""
    intra_class_distance: float = 0.0
    inter_class_distance: float = 0.0
    silhouette_score: float = 0.0
