import os

import h5py
import numpy as np
from scipy.special import digamma, gammaln

try:
    from idrnd.tools.kaldi_io import write_mat, write_vec_flt
except:
    print("kaldi io not loaded - you will can not save in kaldi format")

def matlab_eig(x):
    D, V = np.linalg.eig(x)
    D = np.real(D)

    return V, D


class MetaEmbeddings(object):
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b
        self.__logscal = None

    @property
    def logscal(self):
        return self.__logscal

    @logscal.setter
    def logscal(self, x):
        if self.__logscal is None:
            self.__logscal = -1.0 * x
        else:
            e = -1.0 * self.__logscal + x
            self.__logscal = e


class HTPLDA(object):
    def __init__(self, nu=None, classes=None, F=None, W=None, scaling_mindiv=True, z_mindiv=True):
        self.__scaling_mindiv = scaling_mindiv
        self.__z_mindiv = z_mindiv
        self.__nsteps = 0
        self.temp_b = None

        self.pca = None
        self.mean = None
        if any(map(lambda x: x is None, [nu, classes, F, W])):
            self.nu = None
            self.classes = None
            self.F = None
            self.W = None
            self.build = False
        else:
            self.nu = nu
            self.classes = classes
            self.F = np.random.randn(*np.asarray(F).shape)
            self.W = np.eye(W.shape[0])
            self.build = True

        self.o = []

    def init(self, nu=20, rdim=512, zdim=100):
        self.W = np.eye(rdim)
        self.F = np.random.randn(rdim, zdim)
        self.nu = nu
        self.build = True

        return self

    def __compute_init_values(self):
        self.P = self.F.T @ self.W
        self.B0 = self.P @ self.F

        self.V, self.L = matlab_eig(self.B0)
        self.VP = self.V.T @ self.P

        self.G = self.W - self.VP.T @ (self.VP / self.L[:, None])

    def save(self, path:str):
        assert self.build, "can't save not built model"
        assert path.endswith(".h5"), "file format should be \"h5\""
        assert self.__nsteps > 0, "can't save until has been no steps optimization"

        with h5py.File(path, 'w') as hf:
            hf.create_dataset("F", data=self.F)
            hf.create_dataset("L", data=self.L)
            hf.create_dataset("VP", data=self.VP)
            hf.create_dataset("W", data=self.W)
            hf.create_dataset("G", data=self.G)
            hf.create_dataset("nu", data=self.nu)
            hf.create_dataset("classes", data=self.classes)
            hf.create_dataset("steps", data=self.__nsteps)

        print("{} was successful saved\n".format(path))

    def to_kaldi(self, path):
        write_mat(os.path.join(path, "F.mat"), self.F)
        write_vec_flt(os.path.join(path, "L.vec"), self.L)
        write_mat(os.path.join(path, "VP.mat"), self.VP)
        write_mat(os.path.join(path, "W.mat"), self.W)
        write_mat(os.path.join(path, "G.mat"), self.G)

        if self.pca is not None:
            write_mat(os.path.join(path, "pca.mat"), self.pca)
        if self.pca is not None:
            write_vec_flt(os.path.join(path, "mean.vec"), self.mean)

        # write_vec_int(os.path.join(path, "nu.vec"), np.asarray(self.nu).astype("int32"))

    def load(self, path):
        with h5py.File(path, 'r') as hf:
            self.F = hf.get("F").value
            self.L = hf.get("L").value
            self.VP = hf.get("VP").value
            self.W = hf.get("W").value
            self.G = hf.get("G").value
            self.classes = hf.get("classes").value
            self.nu = hf.get("nu").value
            self.__nsteps = hf.get("steps").value
            if "pca_matrix" in list(hf.keys()):
                self.pca = hf.get("pca_matrix").value

            if "M" in list(hf.keys()):
                self.mean = hf.get("M").value

        self.build = True
        print("{} was successful loaded".format(os.path.basename(path)))
        print("{} feature dim, {} meta dim".format(*self.F.shape))
        print("{} classes for one-hot-encoding".format(self.classes))

    def KLGauss(self, logdets, traces, means):
        kl = (np.sum(traces, axis=0) -
              np.sum(logdets, axis=0) +
              np.sum(means ** 2) -
              means.shape[0] * len(logdets)) / 2
        return kl

    def KLGamma(self, b):
        a0 = b0 = self.nu / 2

        a = (self.nu + self.F.shape[0] - self.F.shape[1]) / 2
        b = a / b

        kl = np.sum(gammaln(a0) -
                    gammaln(a) +
                    a0 * np.log(b / b0) +
                    digamma(a) * (a - a0) +
                    a * (b0 - b) / b
                    )

        return kl

    def compute_b(self, x):
        if self.nu is None or np.isinf(self.nu):
            b = np.ones(self.classes)
        else:
            q = np.sum(x * (self.G @ x), axis=0)
            b = (self.nu + self.F.shape[0] - self.F.shape[1]) / (self.nu + q)
            # b = (self.nu + self.F.shape[0] - self.F.shape[1]) / (self.nu + 10 * np.ones(x.shape[1]))

        return b

    def step(self, x, y, weights=None, verbose=False):
        assert self.build, "can't optimize not built model"
        if self.classes is None:
            self.classes = y.shape[0]

        self.__compute_init_values()

        b = self.compute_b(x)
        if weights is not None:
            b = b * weights

        bx = b[None, :] * x
        S = bx @ x.T

        f = bx @ y.T
        n = b @ y.T
        tot_n = np.sum(n)

        logLPP = np.log1p(n * self.L[:, None])
        LPC = np.exp(-1.0 * logLPP)
        logdetPP = np.sum(logLPP, axis=0)
        tracePC = np.sum(LPC, axis=0)

        Z = self.V @ (LPC * (self.VP @ f))
        T = Z @ f.T

        R = (n * Z) @ Z.T + self.V @ ((LPC @ n[:, None]) * self.V.T)
        C = (Z @ Z.T + self.V @ (np.sum(LPC, axis=1)[:, None] * self.V.T)) / y.shape[0]

        logdetW = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(self.W))))
        logLH = (y.shape[1] / 2) * logdetW + \
                (self.F.shape[0] / 2) * np.sum(np.log(b)) - \
                0.5 * np.sum(self.W * S) + \
                1.0 * np.sum(T * self.P) - \
                0.5 * np.sum(self.B0 * R)

        if self.nu is None or np.isinf(self.nu):
            o = logLH - self.KLGauss(logdets=logdetPP,
                                     traces=tracePC,
                                     means=Z)
        else:
            o = (logLH -
                 self.KLGauss(logdets=logdetPP,
                              traces=tracePC,
                              means=Z) -
                 self.KLGamma(b=b))

        F = np.linalg.lstsq(R.T, T)[0].T
        FT = F @ T

        if self.__scaling_mindiv:
            W = np.linalg.inv((S - (FT + FT.T) / 2) / tot_n)
        else:
            W = np.linalg.inv((S - (FT + FT.T) / 2) / y.shape[1])

        CC = np.linalg.cholesky(C)
        if self.__z_mindiv:
            F = F @ CC

        if verbose:
            print("step {:3d}: cov(z): trace: {:04f},"
                  " logdet = {:04f},"
                  " object: {:04f}".format(self.__nsteps,
                                           np.trace(C),
                                           2 * np.sum(np.log(np.diag(CC))),
                                           o))
        self.__nsteps += 1

        self.F = F
        self.W = W
        self.o.append(o)

        return F, W, o

    def log_expectations(self, meta_embeddings: MetaEmbeddings):
        logbL1 = np.log1p(meta_embeddings.b[None, :] * self.L[:, None])
        logdets = np.sum(logbL1, axis=0)
        Q = np.sum(meta_embeddings.a ** 2 * np.exp(-1.0 * logbL1), axis=0)
        e = (Q - logdets) / 2.0

        meta_embeddings.logscal = e
        return meta_embeddings

    def normalization(self, meta_embeddings: MetaEmbeddings):
        if meta_embeddings.logscal is None:
            meta_embeddings = self.log_expectations(meta_embeddings)

        return meta_embeddings

    def extract(self, x, is_norm=True):
        """
        :param x: [N, dim] np.array N - num vectors, dim - vectors dimension
        :param is_norm:
        :return:
        """
        b = self.compute_b(x)
        A = b * (self.VP @ x)
        meta_embeddings = MetaEmbeddings(a=A, b=b)

        if is_norm:
            meta_embeddings = self.normalization(meta_embeddings)

        return meta_embeddings

    def pool_embeddings(self, embeddings, flags=None, is_norm: bool=False):
        pooled_embeddings = embeddings
        if flags is not None:
            pooled_embeddings.a = embeddings.a @ flags.T
            pooled_embeddings.b = embeddings.b @ flags.T

        if is_norm:
            pooled_embeddings = self.normalization(pooled_embeddings)

        return pooled_embeddings

    def enroll(self, x, y_flags):
        return self.pool_embeddings(self.extract(x, False), y_flags, True)

    def log_inner_products(self, left: MetaEmbeddings, right: MetaEmbeddings):
        B = left.b[:, None] + right.b[None, :]
        X = np.zeros_like(B)

        for n in range(B.shape[1]):
            AA = left.a + right.a[:, n][:, None]
            me = MetaEmbeddings(a=AA, b=B[:, n])
            X[:, n] = -1.0 * self.log_expectations(me).logscal

        sl = left.logscal is not None
        sr = left.logscal is not None

        if sl and sr:
            X = X + (left.logscal[:, None] + right.logscal[None, :])
        elif sl and not sr:
            X = X + left.logscal[:, None]
        elif sr and not  sl:
            X = X + right.logscal
        else:
            NotImplementedError("oh, dear")

        return X

    def score_cross(self, enroll, test, labels=None):
        """
        scoring each-by-each enrolls and test pairs
        TODO: add support averaging enrolls (currently labels var don't have any impact)
        :param enroll: [N, dim] np.array N - num enroll vectors, dim - vectors dimension
        :param test: [M, dim] np.array M - num enroll vectors, dim - vectors dimension
        :param labels: [N, M] np.array N - num one-hot-encoded vectors, classes - numclasses
        :return:
        """
        assert self.build, "can't score not built model"

        left = self.enroll(enroll, labels)
        right = self.extract(test)
        return self.log_inner_products(left=left, right=right)

    def score_trials(self, enroll, test, labels=None):
        """
        scoring enroll to corresponding test
        TODO: add support averaging enrolls (currently labels var don't have any impact)
        :param enroll: [N, dim] np.array N - num enroll vectors, dim - vectors dimension
        :param test: [N, dim] np.array N - num test vectors, dim - vectors dimension
        :param labels: [N, classes] np.array N - num one-hot-encoded vectors, classes - numclasses
        :return: [N] scores for each pairs
        """
        assert self.build, "can't score not built model"

        left = self.enroll(enroll, labels)
        right = self.extract(test)

        both = MetaEmbeddings(a=left.a + right.a,
                              b=left.b + right.b)
        both.logscal = left.logscal + right.logscal
        scores = -1.0 * self.log_expectations(both).logscal

        return scores


# if __name__ == "__main__":
#     import numpy as np
#     ht = HTPLDA()
#     ht.load("/mnt/data_disk/user0/projects/TIVerification/voices/TDNN/result/htplda_a11/a11_eer.h5")
#     # ht.to_kaldi("/mnt/data_disk/user0/projects/TIVerification/voices/TDNN/result/htplda_a11/")
#
#     a = np.random.randn(180, 1) / 100
#     b = np.random.randn(180, 1) / 100
#
#     print(ht.score_trials(a, b))
#     print(ht.score_trials(b, a))
#     print("NU: {}".format(ht.nu))


if __name__ == "__main__":
    def compute_eer(scores_target, scores_impostor, n=100):
        n_imps = len(scores_impostor)
        n_tars = len(scores_target)

        start = min([min(scores_target), min(scores_impostor)]) - 1e-12
        stop = max([max(scores_target), max(scores_impostor)]) + 1e-12

        scores = np.linspace(start, stop, n)
        fars = []
        frrs = []
        dists = []
        min_gap = float("inf")
        min_dist = float("inf")
        eer = 100
        for i, dist in enumerate(scores):
            far = len(np.where(scores_impostor < dist)[0]) / n_imps
            frr = len(np.where(scores_target >= dist)[0]) / n_tars
            fars.append(far)
            frrs.append(frr)
            dists.append(dist)

            gap = np.abs(far - frr)

            if gap < min_gap:
                min_dist = dist
                min_gap = gap
                eer = (far + frr) / 2

        return eer * 100, fars, frrs, min_dist


    def sample_ht_noise(nu, dim, n, W):
        cholW = np.linalg.cholesky(W).T
        if np.isinf(nu):
            precision = np.ones(n)
        else:
            precision = np.mean(np.random.randn(nu, n) ** 2, axis=0)
        std = 1.0 / np.sqrt(precision)
        x, _, _, _ = np.linalg.lstsq(cholW, std * np.random.randn(dim, n))

        return x

    def example(is_big=False):
        if not is_big:
            zdim = 2
            rdim = 20
            nu = 3
            fscal = 3
        else:
            zdim = 100
            rdim = 512
            nu = 3
            fscal = 1 / 20

        F = np.random.randn(rdim, zdim) * fscal
        W = np.random.randn(rdim, 2 * rdim)
        W = W @ W.T
        W = (rdim / np.trace(W)) * W

        n_speakers = 1000
        recording_per_speaker = 10

        model = HTPLDA(nu=nu, F=F, W=W, classes=n_speakers)

        speakers = np.arange(0, n_speakers)
        ilabels = np.repeat(speakers, recording_per_speaker)
        hlabels = ilabels == speakers[:, None]

        Z = np.random.randn(zdim, n_speakers)

        train_x = F @ Z @ hlabels + sample_ht_noise(nu=nu, dim=rdim, n=ilabels.shape[0], W=W)

        for n in range(50):
            _ = model.step(x=train_x, y=hlabels)

        model.save("test.h5")

        # validation
        n_val_speakers = 300
        val_speakers = np.arange(0, n_val_speakers)
        Ztar = np.random.randn(zdim, n_val_speakers)
        enroll_1 = F @ Ztar + sample_ht_noise(nu=nu, dim=rdim, n=n_val_speakers, W=W)

        recording_per_speaker = 10
        N = n_val_speakers * recording_per_speaker
        ilabels = np.repeat(val_speakers, recording_per_speaker)
        hlabels = ilabels == val_speakers[:, None]
        test = F @ Ztar @ hlabels + sample_ht_noise(nu=nu, dim=rdim, n=ilabels.shape[0], W=W)

        scores = -1.0 * model.score_cross(enroll=enroll_1, test=test, labels=None)
        target = scores[hlabels]
        nontarget = scores[~hlabels]
        eer = compute_eer(target, nontarget, n=1000)[0]
        print("EER: {}".format(eer))

        model2 = HTPLDA()
        model2.load("test.h5")
        scores2 = -1.0 * model2.score_cross(enroll=enroll_1, test=test, labels=None)
        target2 = scores2[hlabels]
        nontarget2 = scores2[~hlabels]
        eer = compute_eer(target2, nontarget2, n=1000)[0]
        print("EER: {} for loaded model".format(eer))

    np.random.seed(42)
    example(is_big=False)
    # example(is_big=True)
