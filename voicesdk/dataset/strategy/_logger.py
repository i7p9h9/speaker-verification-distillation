import csv
import typing as tp
from pathlib import Path

import matplotlib as plt
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger


class WeightLogger:
    """Logs dataset sampling weights to TensorBoard via matplotlib figures.

    TensorBoard outputs per call:
      - ``weights/heatmap``      — imshow figure, axes X=step, Y=dataset,
                                   color intensity proportional to weight.
                                   Accumulates history across calls so the
                                   heatmap grows rightward over training.
      - ``weights/snapshot``     — horizontal bar chart of current weights,
                                   useful as a quick readable snapshot.
      - ``weights/absolute/<n>`` — scalar per dataset (raw weight value).
      - ``weights/cumsum/<n>``   — scalar per dataset (cumulative sum).

    Both figures are created with ``add_figure`` so matplotlib handles
    colorbars, axis labels, and tick names — no manual pixel math.

    Args:
        tb_logger:          TensorBoardLogger attached to the trainer.
        csv_path:           Destination path for the CSV weight log.
        log_every_n_steps:  Write interval in global training steps.
        max_history:        Maximum number of steps kept in the heatmap
                            buffer (older entries are dropped, default 500).
    """

    def __init__(
        self,
        tb_logger: TensorBoardLogger,
        csv_path: Path,
        log_every_n_steps: int = 10,
        max_history: int = 500,
    ) -> None:
        self._tb = tb_logger
        self._path = csv_path
        self._every = log_every_n_steps
        self._max_history = max_history

        # Heatmap history: list of (step, {name: weight}) snapshots
        self._history: tp.List[tp.Tuple[int, tp.Dict[str, float]]] = []

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not csv_path.exists():
            with csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(["step", "dataset", "weight"])

    # ---------------------------------------------------------------------- #
    # Figure builders                                                          #
    # ---------------------------------------------------------------------- #

    def _make_heatmap(self) -> "plt.Figure":
        """Build an imshow heatmap from accumulated history.

        Returns:
            matplotlib Figure — X axis = training steps, Y axis = datasets,
            color = sampling weight (viridis colormap).
        """
        import matplotlib.pyplot as plt

        steps = [s for s, _ in self._history]
        names = sorted(self._history[0][1])          # stable Y-axis order
        n_ds = len(names)
        n_t = len(steps)

        # Build matrix [n_datasets, n_steps]
        matrix = np.zeros((n_ds, n_t), dtype=np.float32)
        for t_idx, (_, w_dict) in enumerate(self._history):
            for d_idx, name in enumerate(names):
                matrix[d_idx, t_idx] = np.log(w_dict.get(name, 0.0))

        fig, ax = plt.subplots(figsize=(max(12, n_t * 0.06 + 2), max(3, n_ds * 0.4 + 1)))
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="upper",
            cmap="viridis",
            interpolation="nearest",
        )
        # fig.colorbar(im, ax=ax, label="weight")

        ax.set_yticks(range(n_ds))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("training step")
        ax.set_title("dataset sampling weights")

        # Show only a reasonable number of X tick labels
        max_xticks = 10
        tick_indices = np.linspace(0, n_t - 1, min(max_xticks, n_t), dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([str(steps[i]) for i in tick_indices], rotation=45, ha="right", fontsize=7)

        # Compute left margin from the longest dataset name so ytick labels
        # never overflow — tight_layout raises warnings when label strings are
        # too long for it to resolve margins automatically.
        max_name_len = max(len(n) for n in names)
        left_margin = min(0.02 + max_name_len * 0.012, 0.55)
        fig.subplots_adjust(left=left_margin, right=0.88, top=0.92, bottom=0.18)
        return fig

    def _make_snapshot(
        self,
        names: tp.List[str],
        values: tp.List[float],
    ) -> "plt.Figure":
        """Build a horizontal bar chart of current weights.

        Args:
            names:  Dataset names (Y axis, sorted alphabetically).
            values: Corresponding weight values.

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, max(2, len(names) * 0.35 + 0.5)))
        y_pos = range(len(names))
        ax.barh(list(y_pos), np.log(1 + np.array(values) / np.max(values)), color="steelblue", height=0.6)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("weight")
        ax.set_title("dataset weights (current step)")
        ax.invert_yaxis()
        max_name_len = max((len(n) for n in names), default=10)
        left_margin = min(0.02 + max_name_len * 0.015, 0.6)
        fig.subplots_adjust(left=left_margin, right=0.95, top=0.90, bottom=0.18)
        return fig

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    def maybe_log(
        self,
        weights: tp.Dict[str, float],
        global_step: int,
    ) -> None:
        """Write a weight snapshot if the logging interval has been reached.

        Args:
            weights:     ``{dataset_name: weight}`` from
                         ``LabeledAggregatedDataset.get_weights()``.
            global_step: Current global training step.
        """
        if global_step % self._every != 0:
            return

        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend, safe in training loops

        writer = self._tb.experiment   # underlying SummaryWriter
        names = sorted(weights)
        values = [weights[n] for n in names]

        # ---- accumulate history for heatmap ------------------------------
        self._history.append((global_step, dict(weights)))
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # ---- heatmap (full history) --------------------------------------
        fig_heatmap = self._make_heatmap()
        writer.add_figure("weights/heatmap", fig_heatmap, global_step=global_step)

        # ---- snapshot (current step) -------------------------------------
        fig_snap = self._make_snapshot(names, values)
        writer.add_figure("weights/snapshot", fig_snap, global_step=global_step)

        # ---- scalars + CSV -----------------------------------------------
        cumsum = 0.0
        with self._path.open("a", newline="") as f:
            csv_w = csv.writer(f)
            for name, val in zip(names, values):
                cumsum += val
                writer.add_scalar(f"weights/absolute/{name}", val,    global_step)
                writer.add_scalar(f"weights/cumsum/{name}",   cumsum, global_step)
                csv_w.writerow([global_step, name, val])
