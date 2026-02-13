import typing as tp
from dataclasses import dataclass, field
from .dataflow import DFMetricsBase


@dataclass
class MetricsComposite(DFMetricsBase):
    """Composite metrics combining multiple metric types."""
    metrics: tp.Dict[str, DFMetricsBase] = field(default_factory=dict)

    def to_dict(self) -> tp.Dict[str, tp.Any]:
        result = {'timestamp': self.timestamp}
        for name, metric in self.metrics.items():
            metric_dict = metric.to_dict()
            for k, v in metric_dict.items():
                if k != 'timestamp':
                    result[f"{name}/{k}"] = v
        return result

    def to_tensorboard_dict(self) -> tp.Dict[str, float]:
        result = {}
        for name, metric in self.metrics.items():
            for k, v in metric.to_tensorboard_dict().items():
                result[f"{name}/{k}"] = v
        return result
