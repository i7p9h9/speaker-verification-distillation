from dataclasses import dataclass, field
from ._types import DFMetricsBase


@dataclass
class DFMetricsEER(DFMetricsBase):
    """Equal Error Rate metrics."""
    eer: float = 0.0
    threshold: float = 0.0


@dataclass
class DFMetricsAccuracy(DFMetricsBase):
    """Accuracy-based metrics."""
    accuracy: float = 0.0
    top5_accuracy: float = 0.0


@dataclass
class DFMetricsVerification(DFMetricsBase):
    """Speaker verification metrics."""
    eer: float = 0.0
    min_dcf: float = 0.0
    threshold: float = 0.0


@dataclass
class DFMetricsLoss(DFMetricsBase):
    """Loss tracking metrics."""
    total_loss: float = 0.0
    logits_loss: float = 0.0
    embeddings_loss: float = 0.0
    target_loss: float = 0.0


@dataclass
class DFMetricsDetCurve(DFMetricsBase):
    """Detection error tradeoff curve metrics."""
    fpr: list = field(default_factory=list)
    fnr: list = field(default_factory=list)
    thresholds: list = field(default_factory=list)
