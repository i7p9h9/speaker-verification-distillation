from __future__ import annotations

import typing as tp

from ._base import WeightingStrategy

if tp.TYPE_CHECKING:
    from ..aggregated import AggregatedDataset


class StrategyMomentumLossWeighting(WeightingStrategy):
    """
    Curriculum sampling strategy that adjusts dataset weights based on
    per-source loss values using momentum-based updates.

    Intuition
    ---------
    Sources with **higher loss** get sampled **more often** — the model
    hasn't learned them well yet, so they carry more information per sample.

    Update formula applied at every ``update`` call::

        # 1. Optionally normalise raw per-sample losses → per-source means
        #    (norm_weights=True, default).

        # 2. Compute increment for source i:
        inc_i = scale * loss_i

        # 3. Blend with current weight via momentum:
        effective_momentum = (1 - momentum) [* lr if lr is given]
        w_i ← w_i * (1 - effective_momentum) + inc_i * effective_momentum

        # 4. Re-normalise all weights so they sum to 1.

    Parameters
    ----------
    dataset:
        The ``AggregatedDataset`` whose weights will be managed.
    scale:
        Multiplicative scale applied to each loss before blending.
        Acts as a gain factor: higher values make the weights track
        loss magnitudes more aggressively.  Default ``1.0``.
    momentum:
        Retention factor of the *old* weight.
        ``momentum=0.9`` → ``effective_momentum = 0.1``, so 90 % of the
        old weight is kept and 10 % is replaced by the loss-based increment.
        Must be in ``[0, 1)``.  Default ``0.9``.
    norm_weights:
        If ``True`` (default), losses supplied to ``update`` are first
        averaged per source (multiple samples may map to the same source)
        and re-normalised to sum to 1 before blending.
    init_weight:
        Initial weight assigned to every source.  All sources start equal;
        a subsequent ``normalise`` call makes them sum to 1.
        Default ``1.0``.

    Example
    -------
    ::

        dataset = build_train_dataset(reader_train)
        strategy = MomentumLossWeightingStrategy(
            dataset, scale=1.0, momentum=0.9, norm_weights=True
        )

        for step, batch in enumerate(dataloader):
            loss = model(batch)
            per_sample_losses = compute_per_sample_loss(loss)   # shape [B]
            source_ids = batch["source_id"]                      # shape [B]

            # losses dict: source_key → list-of-floats or single float
            strategy.update(dict(zip(source_ids, per_sample_losses.tolist())))

            if step % 100 == 0:
                print(strategy)
    """

    def __init__(
        self,
        dataset: AggregatedDataset,
        scale: float = 1.0,
        momentum: float = 0.9,
        norm_weights: bool = True,
        init_weight: float = 1.0,
    ) -> None:
        super().__init__(dataset)

        assert scale > 0,              "scale must be positive"
        assert 0.0 <= momentum < 1.0,  "momentum must be in [0, 1)"
        assert init_weight > 0,        "init_weight must be positive"

        self._scale = scale
        self._momentum = momentum
        self._norm_weights = norm_weights

        # Internal weight table: source_key → float (unnormalised).
        # Seeded from whatever the dataset currently holds; unknown keys will
        # be inserted with init_weight on first update.
        self._weights: tp.Dict[str, float] = {
            k: init_weight for k in self._keys
        }
        self._init_weight = init_weight

        # Synchronise the dataset with the (uniform) initial weights.
        self._apply_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        losses: tp.Dict[str, float],
        lr: tp.Optional[float] = None,
    ) -> None:
        """
        Receive per-source (or per-sample) losses and update weights.

        Parameters
        ----------
        losses:
            Mapping ``source_key → loss_value``.
            Keys must match :attr:`keys` (``"dataset_name/source_name"``).
            Sources absent from *losses* are not modified.
        lr:
            Optional external learning-rate multiplier.
            If given, ``effective_momentum = (1 - momentum) * lr``;
            otherwise ``effective_momentum = (1 - momentum)``.
        """
        per_source_losses, ordered_keys = self._aggregate_losses(losses)
        self._blend(per_source_losses, ordered_keys, lr=lr)
        self._apply_weights()

    def set_scale(self, scale: float) -> None:
        """Adjust scale on the fly."""
        assert scale > 0, "scale must be positive"
        self._scale = scale

    def set_momentum(self, momentum: float) -> None:
        """Adjust momentum on the fly."""
        assert 0.0 <= momentum < 1.0, "momentum must be in [0, 1)"
        self._momentum = momentum

    def get_weights(self) -> tp.Dict[str, float]:
        """Return the current normalised weight for every source."""
        normalised = self._normalise(list(self._weights.values()))
        return dict(zip(self._weights.keys(), normalised))

    def state_dict(self) -> tp.Dict[str, tp.Any]:
        """Serialisable state — persist with ``torch.save`` for checkpointing."""
        return {
            "weights":      dict(self._weights),
            "scale":        self._scale,
            "momentum":     self._momentum,
            "norm_weights": self._norm_weights,
            "init_weight":  self._init_weight,
        }

    def load_state_dict(self, state: tp.Dict[str, tp.Any]) -> None:
        """Restore strategy state from a checkpoint."""
        self._weights     = state["weights"]
        self._scale       = state["scale"]
        self._momentum    = state["momentum"]
        self._norm_weights = state["norm_weights"]
        self._init_weight  = state["init_weight"]
        self._apply_weights()

    def __repr__(self) -> str:
        current_w = self.get_weights()
        lines = [
            "MomentumLossWeightingStrategy(",
            f"  scale        = {self._scale}",
            f"  momentum     = {self._momentum}",
            f"  norm_weights = {self._norm_weights}",
        ]
        for k in self._keys:
            w = current_w.get(k, 0.0)
            raw = self._weights.get(k, 0.0)
            lines.append(f"  {k:<50s}  raw={raw:.6f}  w={w:.4f}")
        lines.append(")")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_losses(
        self,
        losses: tp.Dict[str, float],
    ) -> tp.Tuple[tp.List[float], tp.List[str]]:
        """
        Optionally aggregate per-sample losses into per-source means and
        (when norm_weights=True) re-normalise them to sum to 1.

        Returns a parallel pair (loss_values, source_keys).
        """
        if not self._norm_weights:
            keys = list(losses.keys())
            vals = [losses[k] for k in keys]
            return vals, keys

        # Group by source key → mean loss per source.
        grouped: tp.Dict[str, tp.List[float]] = {}
        for key, val in losses.items():
            grouped.setdefault(key, []).append(val)

        keys: tp.List[str] = list(grouped.keys())
        means: tp.List[float] = [
            sum(grouped[k]) / len(grouped[k]) for k in keys
        ]

        # Normalise means to sum to 1.
        total = sum(means)
        if total > 0:
            means = [m / total for m in means]

        return means, keys

    def _blend(
        self,
        loss_vals: tp.List[float],
        keys: tp.List[str],
        lr: tp.Optional[float],
    ) -> None:
        """
        Apply the momentum update::

            effective_momentum = (1 - momentum) [* lr]
            w_i ← w_i * (1 - eff_mom) + scale * loss_i * eff_mom
        """
        eff_momentum = (1.0 - self._momentum)
        if lr is not None:
            eff_momentum *= lr

        for key, loss_val in zip(keys, loss_vals):
            # Register previously unseen sources on the fly.
            if key not in self._weights:
                self._weights[key] = self._init_weight
                if key not in self._keys:
                    self._keys.append(key)

            increment = self._scale * loss_val
            if not _is_finite(increment):
                # Skip non-finite increments (NaN / Inf) to keep weights valid.
                continue

            old = self._weights[key]
            self._weights[key] = old * (1.0 - eff_momentum) + increment * eff_momentum

    def _normalise(self, values: tp.List[float]) -> tp.List[float]:
        """Return a copy of *values* normalised to sum to 1."""
        total = sum(values)
        if total <= 0:
            n = len(values)
            return [1.0 / n] * n
        return [v / total for v in values]

    def _apply_weights(self) -> None:
        """Push the current normalised weights into the underlying dataset."""
        normalised = self._normalise(list(self._weights.values()))
        self._dataset.set_weights(dict(zip(self._weights.keys(), normalised)))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _is_finite(value: float) -> bool:
    """Return True iff *value* is a finite real number."""
    return value == value and abs(value) != float("inf")  # NaN-safe
