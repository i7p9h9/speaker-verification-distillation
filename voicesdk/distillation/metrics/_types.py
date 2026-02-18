import typing as tp
from datetime import datetime
from dataclasses import dataclass, asdict, field


@dataclass
class DFMetricsBase:
    """Base class for all metric outputs."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> tp.Dict[str, tp.Any]:
        """Convert metrics to dictionary."""
        return asdict(self)

    def to_line(self) -> str:
        """Convert metrics to single line string for logging."""
        items = [f"{k}={v}" for k, v in self.to_dict().items() if k != 'timestamp']
        return f"[{self.timestamp}] " + "\n".join(items)

    def __str__(self) -> str:
        return self.to_line()

    def to_tensorboard_dict(self) -> tp.Dict[str, float]:
        """Convert to dict suitable for TensorBoard logging."""
        result = {}
        for k, v in self.to_dict().items():
            if k == 'timestamp':
                continue
            if isinstance(v, (int, float)):
                result[k] = float(v)
        return result
