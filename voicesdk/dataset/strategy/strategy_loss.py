from __future__ import annotations

import math
import typing as tp

from ._base import WeightingStrategy

if tp.TYPE_CHECKING:
    from ..aggregated import AggregatedDataset


class LossWeightingStrategy(WeightingStrategy):
    """
    Curriculum sampling strategy that adjusts dataset weights based on
    per-source loss values.

    Intuition
    ---------
    Sources with **higher loss** get sampled **more often** — the model
    hasn't learned them yet, so they carry more information per sample.

    Weight formula (applied at every ``update`` call)::

        w_i = softmax( (ema_loss_i + eps) / temperature )

    where ``eps`` keeps all weights strictly positive and the softmax is
    computed with the log-sum-exp trick to prevent overflow.  Underflow is
    handled by clamping shifted exponents to ``[-500, 0]`` so that every
    weight remains a strictly positive float even under extreme loss imbalance.

    Parameters
    ----------
    dataset:
        The ``AggregatedDataset`` whose weights will be managed.
    temperature:
        Controls sensitivity to loss differences.
        High T → weights stay close to uniform (less reactive).
        Low  T → high-loss source dominates (aggressive curriculum).
        Must be positive.
    ema_alpha:
        Smoothing factor for the exponential moving average of losses.
        ``1.0`` = no smoothing (raw loss applied immediately).
        ``0.1`` = heavy smoothing (slow, stable adaptation).
        Must be in ``(0, 1]``.
    eps:
        Small constant added to every EMA loss before softmax.
        Keeps all weights strictly positive.  Default ``1e-3``.

    Example
    -------
    ::

        dataset = build_train_dataset(reader_train)
        strategy = LossStrategy(dataset, temperature=1.0, ema_alpha=0.1)

        print(strategy.keys)
        # ['train_dataset/vox2', 'train_dataset/spgi', ..., 'train_dataset/codecs']

        for step, batch in enumerate(dataloader):
            loss = model(batch)
            per_source_loss = compute_per_source_loss(batch, loss)
            strategy.update(per_source_loss)

            if step % 100 == 0:
                print(strategy)
    """

    def __init__(
        self,
        dataset: AggregatedDataset,
        temperature: float = 1.0,
        ema_alpha: float = 0.1,
        eps: float = 1e-3,
    ) -> None:
        super().__init__(dataset)

        assert temperature > 0,        "temperature must be positive"
        assert 0.0 < ema_alpha <= 1.0, "ema_alpha must be in (0, 1]"
        assert eps > 0,                "eps must be positive"

        self._temperature = temperature
        self._ema_alpha = ema_alpha
        self._eps = eps

        # EMA state: None = not yet observed; seeded with raw value on first update
        self._ema: tp.Dict[str, tp.Optional[float]] = dataset.get_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, losses: tp.Dict[str, float]) -> None:
        """
        Receive per-source losses, update EMA, recompute and apply weights.

        Parameters
        ----------
        losses:
            Mapping ``source_path → loss_value``.
            Keys must match :attr:`keys` (``"dataset_name/source_name"``).
            Sources absent from *losses* keep their previous EMA value.
        """
        self._update_ema(losses)
        weights = self._compute_weights()
        self._dataset.set_weights(dict(zip(self._keys, weights)))

    def set_temperature(self, temperature: float) -> None:
        """Adjust temperature on the fly (e.g. for annealing schedules)."""
        assert temperature > 0, "temperature must be positive"
        self._temperature = temperature

    def set_ema_alpha(self, alpha: float) -> None:
        """Adjust EMA smoothing factor on the fly."""
        assert 0.0 < alpha <= 1.0, "ema_alpha must be in (0, 1]"
        self._ema_alpha = alpha

    def get_loss_ema(self) -> tp.Dict[str, tp.Optional[float]]:
        """Current smoothed loss per source (``None`` = not yet observed)."""
        return dict(self._ema)

    def get_weights(self) -> tp.Dict[str, float]:
        return dict(zip(self._keys, self._compute_weights()))

    def state_dict(self) -> tp.Dict[str, tp.Any]:
        """Serialisable state — persist with ``torch.save`` for checkpointing."""
        return {
            "ema":         dict(self._ema),
            "temperature": self._temperature,
            "ema_alpha":   self._ema_alpha,
            "eps":         self._eps,
        }

    def load_state_dict(self, state: tp.Dict[str, tp.Any]) -> None:
        """Restore strategy state from a checkpoint."""
        self._ema = state["ema"]
        self._temperature = state["temperature"]
        self._ema_alpha = state["ema_alpha"]
        self._eps = state["eps"]

    def __repr__(self) -> str:
        current_w = self.get_weights()
        lines = [
            "LossStrategy(",
            f"  temperature = {self._temperature}",
            f"  ema_alpha = {self._ema_alpha}",
            f"  eps = {self._eps}",
        ]
        for k in self._keys:
            w = current_w.get(k, 0.0)
            loss = self._ema.get(k)
            loss_str = f"{loss:.4f}" if loss is not None else "n/a"
            lines.append(f"  {k:<50s}  loss_ema={loss_str}  w={w:.4f}")
        lines.append(")")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_ema(self, losses: tp.Dict[str, float]) -> None:
        for key, loss in losses.items():
            if key not in self._ema:
                self._ema[key] = loss
                if key not in self._keys:
                    self._keys.append(key)
                continue

            if self._ema[key] is None:
                self._ema[key] = loss
            else:
                self._ema[key] = (
                    self._ema_alpha * loss
                    + (1.0 - self._ema_alpha) * self._ema[key]
                )

    def _compute_weights(self) -> tp.List[float]:
        """
        Numerically stable softmax over ``(ema_loss + eps) / temperature``.

        Underflow is prevented by clamping shifted exponents to [-500, 0]:
        ``exp(-500) ≈ 7e-218`` — negligibly small but strictly positive.
        """
        ema_vals = [
            (self._ema[k] if self._ema.get(k) is not None else 0.0) + self._eps
            for k in self._keys
        ]
        scaled = [v / self._temperature for v in ema_vals]
        max_val = max(scaled)
        raw = [math.exp(max(v - max_val, -500.0)) for v in scaled]
        total = sum(raw)
        return  [r / total for r in raw]
