import typing as tp

import numpy as np
import torch
from torch import nn

from voicesdk.distillation.compute.helpers import cosine_similarity
from voicesdk.distillation.compute.metrics import compute_eer, compute_min_dcf

from ..dataflow.metrics import DFMetricsSpeakerVerification
from ._base import EmbeddingValidatorBase
from ._types import ValidationTrial


class TrialBasedValidator(EmbeddingValidatorBase):
    """
    Validator for trial-based evaluation (e.g., speaker verification).

    Expects trials in format: "target_label file1 file2"
    where target_label is 1 for target pairs and 0 for imposter pairs.
    """

    def __init__(
        self,
        name: str,
        trials: tp.List[ValidationTrial],
        file_to_idx: tp.Dict[str, int],
        **kwargs,
    ):
        """
        Args:
            name: Validator name
            trials: List of trial strings
            file_to_idx: Mapping from file path to embedding index
            **kwargs: Additional arguments for EmbeddingValidatorBase
        """
        super().__init__(name=name, **kwargs)
        self.trials = trials
        self.file_to_idx = file_to_idx

    def _evaluate_trials(
        self,
        embeddings: np.ndarray,
    ) -> tp.Tuple[tp.List[float], tp.List[float]]:
        """
        Evaluate trials and return target/imposter scores.

        Returns:
            Tuple of (scores_target, scores_imposter)
        """
        scores_target = []
        scores_imposter = []

        for trial in self.trials:
            if trial.trial_left not in self.file_to_idx or trial.trial_right not in self.file_to_idx:
                continue

            idx1 = self.file_to_idx[trial.trial_left]
            idx2 = self.file_to_idx[trial.trial_right]
            score = cosine_similarity(embeddings[idx1], embeddings[idx2])

            if trial.is_target:
                scores_target.append(score)
            else:
                scores_imposter.append(score)

        return scores_target, scores_imposter

    def validate(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> DFMetricsSpeakerVerification:
        """Run speaker verification evaluation."""
        embeddings = self._extract_embeddings(model, device)
        scores_target, scores_imposter = self._evaluate_trials(embeddings)

        if not scores_target or not scores_imposter:
            return DFMetricsSpeakerVerification()

        eer, eer_threshold = compute_eer(scores_target, scores_imposter)
        min_dcf_001 = compute_min_dcf(scores_target, scores_imposter, p_target=0.01)
        min_dcf_01 = compute_min_dcf(scores_target, scores_imposter, p_target=0.1)

        return DFMetricsSpeakerVerification(
            eer=eer,
            eer_threshold=eer_threshold,
            min_dcf_001=min_dcf_001,
            min_dcf_01=min_dcf_01,
            num_target_trials=len(scores_target),
            num_imposter_trials=len(scores_imposter),
        )
