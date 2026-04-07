from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from scipy.special import digamma, gammaln


def matlab_eig(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eig(x)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    return eigenvectors, eigenvalues


@dataclass
class MetaEmbeddings:
    a: np.ndarray
    b: np.ndarray
    _logscal: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def logscal(self) -> Optional[np.ndarray]:
        return self._logscal

    @logscal.setter
    def logscal(self, x: np.ndarray) -> None:
        if self._logscal is None:
            self._logscal = -1.0 * x
        else:
            e = -1.0 * self._logscal + x
            self._logscal = e


@dataclass
class HTPLDAModelState:
    """
    Minimal state required for inference / compare / transform.
    """
    nu: float
    F: np.ndarray
    W: np.ndarray
    L: np.ndarray
    VP: np.ndarray
    G: np.ndarray


@dataclass
class HTPLDATrainingInfo:
    """
    Training-only metadata. Not needed for inference.
    """
    nsteps: int = 0
    objective_history: list[float] = field(default_factory=list)
    classes: Optional[int] = None


def labels_to_onehot(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    indices = np.array([label_to_index[label] for label in labels], dtype=np.int64)

    onehot = np.zeros((labels.shape[0], len(unique_labels)), dtype=np.float64)
    onehot[np.arange(labels.shape[0]), indices] = 1.0
    return onehot, label_to_index


def split_by_classes(
    x: np.ndarray,
    labels: np.ndarray,
    eval_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split by speaker/class identity, not by samples.

    Args:
        x: [N, D]
        labels: [N]
    Returns:
        x_train, y_train, x_eval, y_eval
    """
    x = np.asarray(x)
    labels = np.asarray(labels)

    unique_classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_classes)

    n_eval = max(1, int(round(len(unique_classes) * eval_ratio)))
    eval_classes = set(shuffled[:n_eval].tolist())

    eval_mask = np.isin(labels, list(eval_classes))
    train_mask = ~eval_mask

    return x[train_mask], labels[train_mask], x[eval_mask], labels[eval_mask]


class HTPLDABase:
    def __init__(self, model_state: HTPLDAModelState):
        self.model_state = model_state
        self._load_from_state()

    def _load_from_state(self) -> None:
        self.nu = float(self.model_state.nu)
        self.F = np.asarray(self.model_state.F, dtype=np.float64)
        self.W = np.asarray(self.model_state.W, dtype=np.float64)
        self.L = np.asarray(self.model_state.L, dtype=np.float64)
        self.VP = np.asarray(self.model_state.VP, dtype=np.float64)
        self.G = np.asarray(self.model_state.G, dtype=np.float64)

    def _update_model_state(self) -> None:
        self.model_state = HTPLDAModelState(
            nu=float(self.nu),
            F=self.F.copy(),
            W=self.W.copy(),
            L=self.L.copy(),
            VP=self.VP.copy(),
            G=self.G.copy(),
        )

    @staticmethod
    def _compute_cache(F: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        P = F.T @ W
        B0 = P @ F

        V, L = matlab_eig(B0)
        VP = V.T @ P
        G = W - VP.T @ (VP / L[:, None])

        return V, L, VP, G

    def compute_b(self, x: np.ndarray) -> np.ndarray:
        """
        x: [D, N]
        """
        if np.isinf(self.nu):
            return np.ones(x.shape[1], dtype=np.float64)

        q = np.sum(x * (self.G @ x), axis=0)
        return (self.nu + self.F.shape[0] - self.F.shape[1]) / (self.nu + q)

    def log_expectations(self, meta_embeddings: MetaEmbeddings) -> MetaEmbeddings:
        logbL1 = np.log1p(meta_embeddings.b[None, :] * self.L[:, None])
        logdets = np.sum(logbL1, axis=0)
        Q = np.sum(meta_embeddings.a ** 2 * np.exp(-1.0 * logbL1), axis=0)
        e = (Q - logdets) / 2.0

        meta_embeddings.logscal = e
        return meta_embeddings

    def normalization(self, meta_embeddings: MetaEmbeddings) -> MetaEmbeddings:
        if meta_embeddings.logscal is None:
            meta_embeddings = self.log_expectations(meta_embeddings)
        return meta_embeddings

    def extract(self, x: np.ndarray, is_norm: bool = True) -> MetaEmbeddings:
        """
        x: [D, N]
        """
        b = self.compute_b(x)
        a = b * (self.VP @ x)
        meta_embeddings = MetaEmbeddings(a=a, b=b)

        if is_norm:
            meta_embeddings = self.normalization(meta_embeddings)

        return meta_embeddings

    def transform(self, x: np.ndarray, normalize: bool = True) -> MetaEmbeddings:
        return self.extract(x, is_norm=normalize)

    def pool_embeddings(
        self,
        embeddings: MetaEmbeddings,
        flags: Optional[np.ndarray] = None,
        is_norm: bool = False,
    ) -> MetaEmbeddings:
        if flags is None:
            pooled = MetaEmbeddings(
                a=embeddings.a.copy(),
                b=embeddings.b.copy(),
                _logscal=None if embeddings.logscal is None else embeddings.logscal.copy(),
            )
        else:
            pooled = MetaEmbeddings(
                a=embeddings.a @ flags.T,
                b=embeddings.b @ flags.T,
            )

        if is_norm:
            pooled = self.normalization(pooled)

        return pooled

    def enroll(self, x: np.ndarray, y_flags: Optional[np.ndarray]) -> MetaEmbeddings:
        return self.pool_embeddings(self.extract(x, False), y_flags, True)

    def log_inner_products(
        self,
        left: MetaEmbeddings,
        right: MetaEmbeddings,
    ) -> np.ndarray:
        B = left.b[:, None] + right.b[None, :]
        X = np.zeros_like(B)

        for n in range(B.shape[1]):
            AA = left.a + right.a[:, n][:, None]
            me = MetaEmbeddings(a=AA, b=B[:, n])
            X[:, n] = -1.0 * self.log_expectations(me).logscal

        sl = left.logscal is not None
        sr = right.logscal is not None

        if sl and sr:
            X = X + (left.logscal[:, None] + right.logscal[None, :])
        elif sl and not sr:
            X = X + left.logscal[:, None]
        elif sr and not sl:
            X = X + right.logscal[None, :]
        else:
            raise RuntimeError("Both embeddings are unnormalized.")

        return X

    def score_cross(
        self,
        enroll: np.ndarray,
        test: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        enroll: [D, N_enroll]
        test: [D, N_test]
        """
        left = self.enroll(enroll, labels)
        right = self.extract(test)
        return self.log_inner_products(left=left, right=right)

    def score_trials(
        self,
        enroll: np.ndarray,
        test: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        enroll: [D, N]
        test: [D, N]
        """
        left = self.enroll(enroll, labels)
        right = self.extract(test)

        both = MetaEmbeddings(
            a=left.a + right.a,
            b=left.b + right.b,
        )
        both.logscal = left.logscal + right.logscal
        scores = -1.0 * self.log_expectations(both).logscal
        return scores

    def compare(
        self,
        enroll: np.ndarray,
        test: np.ndarray,
        labels: Optional[np.ndarray] = None,
        pairwise: bool = True,
    ) -> np.ndarray:
        if pairwise:
            return self.score_cross(enroll=enroll, test=test, labels=labels)
        return self.score_trials(enroll=enroll, test=test, labels=labels)

    def __call__(
        self,
        enroll: np.ndarray,
        test: np.ndarray,
        labels: Optional[np.ndarray] = None,
        pairwise: bool = True,
    ) -> np.ndarray:
        return self.compare(enroll=enroll, test=test, labels=labels, pairwise=pairwise)

    def save_model(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix != ".h5":
            raise ValueError("Model path must have .h5 extension")

        self._update_model_state()

        with h5py.File(path, "w") as hf:
            hf.create_dataset("nu", data=self.model_state.nu)
            hf.create_dataset("F", data=self.model_state.F)
            hf.create_dataset("W", data=self.model_state.W)
            hf.create_dataset("L", data=self.model_state.L)
            hf.create_dataset("VP", data=self.model_state.VP)
            hf.create_dataset("G", data=self.model_state.G)

    @classmethod
    def load_model_state(cls, path: str | Path) -> HTPLDAModelState:
        path = Path(path)
        with h5py.File(path, "r") as hf:
            return HTPLDAModelState(
                nu=float(hf["nu"][()]),
                F=hf["F"][()],
                W=hf["W"][()],
                L=hf["L"][()],
                VP=hf["VP"][()],
                G=hf["G"][()],
            )


class HTPLDATrainer(HTPLDABase):
    def __init__(
        self,
        model_state: Optional[HTPLDAModelState] = None,
        training_info: Optional[HTPLDATrainingInfo] = None,
        *,
        nu: Optional[float] = None,
        F: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
    ):
        if model_state is None:
            if any(v is None for v in [nu, F, W]):
                raise ValueError("Provide either model_state or all of: nu, F, W")

            F = np.asarray(F, dtype=np.float64)
            W = np.asarray(W, dtype=np.float64)

            _, L, VP, G = self._compute_cache(F, W)
            model_state = HTPLDAModelState(
                nu=float(nu),
                F=F.copy(),
                W=W.copy(),
                L=L.copy(),
                VP=VP.copy(),
                G=G.copy(),
            )

        if training_info is None:
            training_info = HTPLDATrainingInfo()

        self.training_info = training_info
        super().__init__(model_state=model_state)

    @classmethod
    def init_random(
        cls,
        nu: float = 20,
        rdim: int = 512,
        zdim: int = 100,
        seed: Optional[int] = None,
    ) -> HTPLDATrainer:
        rng = np.random.default_rng(seed)
        F = rng.standard_normal((rdim, zdim))
        W = np.eye(rdim, dtype=np.float64)
        return cls(nu=nu, F=F, W=W)

    def kl_gauss(
        self,
        logdets: np.ndarray,
        traces: np.ndarray,
        means: np.ndarray,
    ) -> np.ndarray:
        kl = (
            np.sum(traces, axis=0)
            - np.sum(logdets, axis=0)
            + np.sum(means ** 2)
            - means.shape[0] * len(logdets)
        ) / 2
        return kl

    def kl_gamma(self, b: np.ndarray) -> np.ndarray:
        a0 = b0 = self.nu / 2
        a = (self.nu + self.F.shape[0] - self.F.shape[1]) / 2
        b = a / b

        kl = np.sum(
            gammaln(a0)
            - gammaln(a)
            + a0 * np.log(b / b0)
            + digamma(a) * (a - a0)
            + a * (b0 - b) / b
        )
        return kl

    def step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        verbose: bool = False,
        scaling_mindiv: bool = True,
        z_mindiv: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        x: [D, N]
        y: [N, C]
        """
        self.training_info.classes = y.shape[1]

        P = self.F.T @ self.W
        B0 = P @ self.F

        V, L = matlab_eig(B0)
        VP = V.T @ P
        G = self.W - VP.T @ (VP / L[:, None])

        self.L = L
        self.VP = VP
        self.G = G

        b = self.compute_b(x)
        if weights is not None:
            b = b * weights

        bx = b[None, :] * x
        S = bx @ x.T

        f = bx @ y
        n = b @ y
        tot_n = np.sum(n)

        logLPP = np.log1p(n * self.L[:, None])
        LPC = np.exp(-1.0 * logLPP)
        logdetPP = np.sum(logLPP, axis=0)
        tracePC = np.sum(LPC, axis=0)

        Z = V @ (LPC * (self.VP @ f))
        T = Z @ f.T

        R = (n * Z) @ Z.T + V @ ((LPC @ n[:, None]) * V.T)
        C = (Z @ Z.T + V @ (np.sum(LPC, axis=1)[:, None] * V.T)) / y.shape[1]

        logdetW = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(self.W))))
        logLH = (
            (x.shape[1] / 2) * logdetW
            + (self.F.shape[0] / 2) * np.sum(np.log(b))
            - 0.5 * np.sum(self.W * S)
            + 1.0 * np.sum(T * P)
            - 0.5 * np.sum(B0 * R)
        )

        if np.isinf(self.nu):
            obj = logLH - self.kl_gauss(logdets=logdetPP, traces=tracePC, means=Z)
        else:
            obj = (
                logLH
                - self.kl_gauss(logdets=logdetPP, traces=tracePC, means=Z)
                - self.kl_gamma(b=b)
            )

        F_new = np.linalg.lstsq(R.T, T, rcond=None)[0].T
        FT = F_new @ T

        if scaling_mindiv:
            W_new = np.linalg.inv((S - (FT + FT.T) / 2) / tot_n)
        else:
            W_new = np.linalg.inv((S - (FT + FT.T) / 2) / x.shape[1])

        C_chol = np.linalg.cholesky(C)
        if z_mindiv:
            F_new = F_new @ C_chol

        _, L_new, VP_new, G_new = self._compute_cache(F_new, W_new)

        self.F = F_new
        self.W = W_new
        self.L = L_new
        self.VP = VP_new
        self.G = G_new

        self.training_info.nsteps += 1
        self.training_info.objective_history.append(float(obj))

        self._update_model_state()

        if verbose:
            print(
                f"step {self.training_info.nsteps:3d}: "
                f"cov(z): trace={np.trace(C):.6f}, "
                f"logdet={2 * np.sum(np.log(np.diag(C_chol))):.6f}, "
                f"object={obj:.6f}"
            )

        return self.F, self.W, float(obj)

    def fit(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        num_steps: int = 20,
        weights: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> HTPLDAModelState:
        """
        Public API:
            x: [N, D]
            labels: [N]
        Internal math:
            x_t: [D, N]
            y: [N, C]
        """
        x = np.asarray(x, dtype=np.float64)
        labels = np.asarray(labels)

        y, _ = labels_to_onehot(labels)
        self.training_info.classes = y.shape[1]

        x_t = x.T

        for _ in range(num_steps):
            self.step(
                x=x_t,
                y=y,
                weights=weights,
                verbose=verbose,
            )

        self._update_model_state()
        return self.model_state

    def save_training_info(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix != ".h5":
            raise ValueError("Training-info path must have .h5 extension")

        with h5py.File(path, "w") as hf:
            hf.create_dataset("nsteps", data=self.training_info.nsteps)
            hf.create_dataset(
                "objective_history",
                data=np.asarray(self.training_info.objective_history, dtype=np.float64),
            )
            if self.training_info.classes is not None:
                hf.create_dataset("classes", data=self.training_info.classes)

    @staticmethod
    def load_training_info(path: str | Path) -> HTPLDATrainingInfo:
        path = Path(path)
        with h5py.File(path, "r") as hf:
            return HTPLDATrainingInfo(
                nsteps=int(hf["nsteps"][()]) if "nsteps" in hf else 0,
                objective_history=hf["objective_history"][()].tolist()
                if "objective_history" in hf
                else [],
                classes=int(hf["classes"][()]) if "classes" in hf else None,
            )


class HTPLDAComparator(HTPLDABase):
    def __init__(self, model_state: HTPLDAModelState):
        super().__init__(model_state=model_state)

    @classmethod
    def from_file(cls, path: str | Path) -> HTPLDAComparator:
        model_state = cls.load_model_state(path)
        return cls(model_state=model_state)


def compute_eer(
    scores_target: np.ndarray,
    scores_impostor: np.ndarray,
    n: int = 1000,
) -> tuple[float, list[float], list[float], float]:
    n_imps = len(scores_impostor)
    n_tars = len(scores_target)

    start = min(float(np.min(scores_target)), float(np.min(scores_impostor))) - 1e-12
    stop = max(float(np.max(scores_target)), float(np.max(scores_impostor))) + 1e-12

    thresholds = np.linspace(start, stop, n)
    fars = []
    frrs = []

    min_gap = float("inf")
    best_thr = float(thresholds[0])
    eer = 100.0

    for thr in thresholds:
        far = np.mean(scores_impostor < thr)
        frr = np.mean(scores_target >= thr)

        fars.append(float(far))
        frrs.append(float(frr))

        gap = abs(far - frr)
        if gap < min_gap:
            min_gap = gap
            best_thr = float(thr)
            eer = (far + frr) / 2

    return eer * 100, fars, frrs, best_thr


def sample_ht_noise(nu: float, dim: int, n: int, W: np.ndarray) -> np.ndarray:
    cholW = np.linalg.cholesky(W).T
    if np.isinf(nu):
        precision = np.ones(n)
    else:
        precision = np.mean(np.random.randn(int(nu), n) ** 2, axis=0)

    std = 1.0 / np.sqrt(precision)
    x = np.linalg.lstsq(cholW, std * np.random.randn(dim, n), rcond=None)[0]
    return x


if __name__ == "__main__":
    np.random.seed(42)

    zdim = 2
    rdim = 20
    nu = 3
    fscal = 3.0

    F_true = np.random.randn(rdim, zdim) * fscal
    W_true = np.random.randn(rdim, 2 * rdim)
    W_true = W_true @ W_true.T
    W_true = (rdim / np.trace(W_true)) * W_true

    n_speakers = 1000
    recordings_per_speaker = 10

    speakers = np.arange(n_speakers)
    sample_labels = np.repeat(speakers, recordings_per_speaker)

    Z = np.random.randn(zdim, n_speakers)
    onehot_all, _ = labels_to_onehot(sample_labels)

    x_all = (
        F_true @ Z @ onehot_all.T
        + sample_ht_noise(nu=nu, dim=rdim, n=sample_labels.shape[0], W=W_true)
    ).T  # [N, D]

    x_train, y_train, x_eval, y_eval = split_by_classes(
        x_all,
        sample_labels,
        eval_ratio=0.2,
        seed=42,
    )

    trainer = HTPLDATrainer.init_random(
        nu=nu,
        rdim=rdim,
        zdim=zdim,
        seed=42,
    )

    model_state = trainer.fit(
        x=x_train,
        labels=y_train,
        num_steps=50,
        verbose=True,
    )

    trainer.save_model("htplda_model.h5")
    trainer.save_training_info("htplda_training.h5")

    comparator = HTPLDAComparator(model_state=model_state)
    comparator_loaded = HTPLDAComparator.from_file("htplda_model.h5")

    eval_classes = np.unique(y_eval)

    enroll_indices = []
    test_indices = []

    seen = set()
    for idx, label in enumerate(y_eval):
        if label not in seen:
            enroll_indices.append(idx)
            seen.add(label)
        else:
            test_indices.append(idx)

    enroll = x_eval[enroll_indices].T
    test = x_eval[test_indices].T

    enroll_labels = y_eval[enroll_indices]
    test_labels = y_eval[test_indices]

    trial_mask = enroll_labels[:, None] == test_labels[None, :]

    scores = -1.0 * comparator.compare(
        enroll=enroll,
        test=test,
        labels=None,
        pairwise=True,
    )
    target = scores[trial_mask]
    nontarget = scores[~trial_mask]
    eer = compute_eer(target, nontarget, n=1000)[0]
    print(f"EER: {eer:.4f}")

    scores_loaded = -1.0 * comparator_loaded(
        enroll=enroll,
        test=test,
        labels=None,
        pairwise=True,
    )
    target_loaded = scores_loaded[trial_mask]
    nontarget_loaded = scores_loaded[~trial_mask]
    eer_loaded = compute_eer(target_loaded, nontarget_loaded, n=1000)[0]
    print(f"EER loaded: {eer_loaded:.4f}")
