from dataclasses import dataclass


@dataclass
class ValidationTrial:
    """Data structure representing a single validation trial."""
    trial_left: str
    trial_right: str
    is_target: bool
