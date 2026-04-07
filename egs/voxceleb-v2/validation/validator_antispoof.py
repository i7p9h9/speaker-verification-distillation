import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from voicesdk.distillation.compute.metrics import compute_eer
from voicesdk.distillation.data.collate import BatchSegments
from voicesdk.distillation.metrics import DFMetricsBase
from voicesdk.distillation.validation import ValidatorBase

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AntiSpoofingScores:
    """Raw scores produced during anti-spoofing evaluation."""

    scores_bonafide: tp.List[float] = field(default_factory=list)
    scores_spoof: tp.List[float] = field(default_factory=list)


@dataclass
class AntiSpoofingMetrics(DFMetricsBase):
    """
    Metrics for a single anti-spoofing evaluation.

    Higher score  -> model considers the utterance more likely bonafide.
    EER threshold separates bonafide (above) from spoof (below).

    Inherits from DFMetricsBase: to_dict(), to_line(), to_tensorboard_dict().

    Notes
    -----
    The ``scores`` field is intentionally excluded from ``to_tensorboard_dict``
    because it holds lists of floats, not scalar summaries.
    """

    eer: float = float("nan")
    eer_threshold: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    accuracy: float = float("nan")
    num_bonafide: int = 0
    num_spoof: int = 0

    # Populated only when the validator is called with return_scores=True.
    # Excluded from TensorBoard logging (non-scalar).
    scores: tp.Optional[AntiSpoofingScores] = None

    def to_tensorboard_dict(self) -> tp.Dict[str, float]:
        """
        Return scalar metrics only.

        Overrides the base implementation to skip the ``scores`` field, which
        is a nested dataclass containing lists and is not suitable for
        TensorBoard logging.
        """
        _SKIP = {"timestamp", "scores"}
        result: tp.Dict[str, float] = {}
        for k, v in self.to_dict().items():
            if k in _SKIP:
                continue
            if isinstance(v, (int, float)):
                result[k] = float(v)
        return result


@dataclass
class AggregatedAntiSpoofingMetrics(DFMetricsBase):
    """
    Aggregated metrics over a collection of anti-spoofing validators.

    Fields
    ------
    per_validator : metrics for every individual validator
    averaged      : simple average of per-validator scalar metrics
    joint         : metrics re-computed on the union of all scores

    TensorBoard layout
    ------------------
    Keys are prefixed to avoid collisions::

        per_validator/<name>/<metric>
        averaged/<metric>
        joint/<metric>
    """

    per_validator: tp.Dict[str, AntiSpoofingMetrics] = field(default_factory=dict)
    averaged: AntiSpoofingMetrics = field(default_factory=AntiSpoofingMetrics)
    joint: AntiSpoofingMetrics = field(default_factory=AntiSpoofingMetrics)

    def to_tensorboard_dict(self) -> tp.Dict[str, float]:
        """
        Flatten nested metrics into a prefixed dict suitable for TensorBoard.

        Example keys::

            "per_validator/asvspoof19/eer"
            "averaged/eer"
            "joint/precision"
        """
        result: tp.Dict[str, float] = {}

        for name, metrics in self.per_validator.items():
            for k, v in metrics.to_tensorboard_dict().items():
                result[f"per_validator/{name}/{k}"] = v

        for k, v in self.averaged.to_tensorboard_dict().items():
            result[f"averaged/{k}"] = v

        for k, v in self.joint.to_tensorboard_dict().items():
            result[f"joint/{k}"] = v

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_classification_metrics(
    scores_bonafide: tp.List[float],
    scores_spoof: tp.List[float],
    threshold: float,
) -> tp.Tuple[float, float, float]:
    """
    Compute precision, recall and accuracy given a decision threshold.

    Convention: score >= threshold -> predicted bonafide (positive class).

    Returns
    -------
    precision, recall, accuracy
    """
    tp_count = sum(s >= threshold for s in scores_bonafide)  # true positives
    fp_count = sum(s >= threshold for s in scores_spoof)     # false positives
    fn_count = sum(s < threshold for s in scores_bonafide)   # false negatives
    tn_count = sum(s < threshold for s in scores_spoof)      # true negatives

    total = tp_count + fp_count + fn_count + tn_count

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else float("nan")
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else float("nan")
    accuracy = (tp_count + tn_count) / total if total > 0 else float("nan")

    return precision, recall, accuracy


def _build_metrics(
    scores_bonafide: tp.List[float],
    scores_spoof: tp.List[float],
    return_scores: bool,
) -> AntiSpoofingMetrics:
    """Compute all metrics from raw score lists."""
    if not scores_bonafide or not scores_spoof:
        result = AntiSpoofingMetrics(
            num_bonafide=len(scores_bonafide),
            num_spoof=len(scores_spoof),
        )
        if return_scores:
            result.scores = AntiSpoofingScores(
                scores_bonafide=scores_bonafide,
                scores_spoof=scores_spoof,
            )
        return result

    eer, eer_threshold = compute_eer(scores_bonafide, scores_spoof)
    precision, recall, accuracy = _compute_classification_metrics(
        scores_bonafide, scores_spoof, eer_threshold
    )

    result = AntiSpoofingMetrics(
        eer=eer,
        eer_threshold=eer_threshold,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        num_bonafide=len(scores_bonafide),
        num_spoof=len(scores_spoof),
    )

    if return_scores:
        result.scores = AntiSpoofingScores(
            scores_bonafide=list(scores_bonafide),
            scores_spoof=list(scores_spoof),
        )

    return result


def _average_metrics(metrics_list: tp.List[AntiSpoofingMetrics]) -> AntiSpoofingMetrics:
    """Average scalar fields across a list of AntiSpoofingMetrics."""
    if not metrics_list:
        return AntiSpoofingMetrics()

    def _nanmean(values: tp.List[float]) -> float:
        finite = [v for v in values if not np.isnan(v)]
        return float(np.mean(finite)) if finite else float("nan")

    return AntiSpoofingMetrics(
        eer=_nanmean([m.eer for m in metrics_list]),
        eer_threshold=_nanmean([m.eer_threshold for m in metrics_list]),
        precision=_nanmean([m.precision for m in metrics_list]),
        recall=_nanmean([m.recall for m in metrics_list]),
        accuracy=_nanmean([m.accuracy for m in metrics_list]),
        num_bonafide=sum(m.num_bonafide for m in metrics_list),
        num_spoof=sum(m.num_spoof for m in metrics_list),
    )


# ---------------------------------------------------------------------------
# Single validator
# ---------------------------------------------------------------------------


class AntiSpoofingValidator(ValidatorBase):
    """
    Validator for anti-spoofing evaluation.

    The model is expected to output a scalar score per utterance — higher
    values indicate the utterance is more likely bonafide.

    Inherits checkpoint saving, metric history, and log writing from
    :class:`ValidatorBase`. Call :meth:`ValidatorBase.run` during training
    to get all of that for free; call :meth:`validate` directly when you
    only need the metrics object (e.g. inside
    :class:`AggregatedAntiSpoofingValidator`).

    Parameters
    ----------
    name : str
        Human-readable name used for logging and as a key in aggregated
        results.
    bonafide_loader : DataLoader
        Yields batches of genuine (bonafide) utterances.
    spoof_loader : DataLoader
        Yields batches of spoofed utterances.
    score_fn : callable, optional
        Maps raw model output (tensor or tuple) to a flat list of
        per-sample scalar scores. Defaults to treating the model output as
        a 1-D tensor of logits / log-likelihoods.
    **kwargs
        Forwarded to :class:`ValidatorBase` (``save_checkpoint``,
        ``checkpoint_dir``, ``log_dir``, ``metric_for_best``,
        ``metric_mode``).
    """

    def __init__(
        self,
        name: str,
        bonafide_loader: DataLoader,
        spoof_loader: DataLoader,
        score_fn: tp.Optional[tp.Callable[[tp.Any], tp.List[float]]] = None,
        **kwargs: tp.Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.bonafide_loader = bonafide_loader
        self.spoof_loader = spoof_loader
        self._score_fn = score_fn or self._default_score_fn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_score_fn(model_output: tp.Any) -> tp.List[float]:
        """Convert model output tensor to a list of floats."""
        if isinstance(model_output, (tuple, list)):
            model_output = model_output[0]
        t = model_output.detach().cpu().float()
        if t.dim() > 1:
            # Take the last logit / bonafide class score
            t = t[:, -1]
        return t.tolist()

    @torch.no_grad()
    def _collect_scores(
        self,
        model: nn.Module,
        device: torch.device,
        loader: DataLoader[BatchSegments],
    ) -> tp.List[float]:
        """
        Run the model over *loader* and collect one scalar score per sample.

        Expects the loader to yield :class:`BatchSegments` objects.
        ``batch.segments`` has shape ``(total_segments, channels, time)`` —
        since one segment == one file is guaranteed by the caller, this is
        equivalent to ``(batch_size, channels, time)``.
        """
        model.eval()
        all_scores: tp.List[float] = []

        for batch in loader:
            segments = batch.segments.to(device)
            output = model(segments)
            all_scores.extend(self._score_fn(output))

        return all_scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        return_scores: bool = False,
    ) -> AntiSpoofingMetrics:
        """
        Run anti-spoofing evaluation.

        Satisfies the :class:`ValidatorBase` abstract interface.
        The extra ``return_scores`` keyword-only argument does not break the
        base contract because :meth:`ValidatorBase.run` always calls
        ``validate(model, device)`` without additional arguments.

        Parameters
        ----------
        model : nn.Module
            Model to evaluate.
        device : torch.device
            Device to run inference on.
        return_scores : bool
            When ``True``, the returned :class:`AntiSpoofingMetrics` will
            also contain an :attr:`~AntiSpoofingMetrics.scores` field with
            the raw per-sample scores. Used internally by
            :class:`AggregatedAntiSpoofingValidator`.

        Returns
        -------
        AntiSpoofingMetrics
        """
        scores_bonafide = self._collect_scores(model, device, self.bonafide_loader)
        scores_spoof = self._collect_scores(model, device, self.spoof_loader)

        return _build_metrics(scores_bonafide, scores_spoof, return_scores)


# ---------------------------------------------------------------------------
# Aggregated validator
# ---------------------------------------------------------------------------


class AggregatedAntiSpoofingValidator(ValidatorBase):
    """
    Runs a collection of :class:`AntiSpoofingValidator` instances and
    consolidates their results.

    Inherits checkpoint saving, metric history, and log writing from
    :class:`ValidatorBase` — these operate on the
    :class:`AggregatedAntiSpoofingMetrics` object as a whole.

    Individual child validators are called via :meth:`validate` directly
    (not :meth:`ValidatorBase.run`) to avoid double-logging; if you want
    per-validator checkpointing, configure it on each child separately and
    call their :meth:`~ValidatorBase.run` methods before or after this one.

    In addition to per-validator metrics the class computes:

    * **averaged** — simple mean of each scalar metric across all validators.
    * **joint**    — metrics re-computed on the concatenation of *all* bonafide
      scores and *all* spoof scores (i.e. a single EER on the full pool).

    Parameters
    ----------
    name : str
        Human-readable name used for logging.
    validators : list of AntiSpoofingValidator
        Individual validators to run.
    return_scores : bool
        When ``True``, every per-validator result and the joint result will
        contain raw score lists. Configured at construction time so that
        :meth:`ValidatorBase.run` (which calls ``validate`` with no extra
        args) still honours this setting.
    **kwargs
        Forwarded to :class:`ValidatorBase`.
    """

    def __init__(
        self,
        name: str,
        validators: tp.List[AntiSpoofingValidator],
        return_scores: bool = False,
        **kwargs: tp.Any,
    ) -> None:
        if not validators:
            raise ValueError("At least one validator is required.")
        super().__init__(name=name, **kwargs)
        self.validators = validators
        self.return_scores = return_scores

    def validate(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> AggregatedAntiSpoofingMetrics:
        """
        Run all validators and aggregate results.

        Satisfies the :class:`ValidatorBase` abstract interface.
        Score-return behaviour is controlled by the ``return_scores``
        argument passed to :meth:`__init__`.

        Parameters
        ----------
        model : nn.Module
            Shared model passed to every child validator.
        device : torch.device
            Inference device.

        Returns
        -------
        AggregatedAntiSpoofingMetrics
        """
        per_validator: tp.Dict[str, AntiSpoofingMetrics] = {}

        # Always collect raw scores internally for joint pool computation.
        # Attach them to the output only when self.return_scores is True.
        all_bonafide: tp.List[float] = []
        all_spoof: tp.List[float] = []

        for validator in self.validators:
            metrics = validator.validate(model, device, return_scores=True)
            assert metrics.scores is not None  # guaranteed by return_scores=True

            all_bonafide.extend(metrics.scores.scores_bonafide)
            all_spoof.extend(metrics.scores.scores_spoof)

            if not self.return_scores:
                # Strip scores from the per-validator result if not requested
                metrics.scores = None

            per_validator[validator.name] = metrics

        averaged = _average_metrics(list(per_validator.values()))
        joint = _build_metrics(all_bonafide, all_spoof, self.return_scores)

        return AggregatedAntiSpoofingMetrics(
            per_validator=per_validator,
            averaged=averaged,
            joint=joint,
        )
