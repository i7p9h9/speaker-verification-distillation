import numpy as np


def compute_frr_far(tar, imp):
    tar_unique, tar_counts = np.unique(tar, return_counts=True)
    imp_unique, imp_counts = np.unique(imp, return_counts=True)
    thresholds = np.unique(np.hstack((tar_unique, imp_unique)))

    pt = np.hstack((tar_counts, np.zeros(len(thresholds) - len(tar_counts), dtype=np.int32)))
    pi = np.hstack((np.zeros(len(thresholds) - len(imp_counts), dtype=np.int32), imp_counts))

    pt = pt[np.argsort(np.hstack((tar_unique, np.setdiff1d(imp_unique, tar_unique))))]
    pi = pi[np.argsort(np.hstack((np.setdiff1d(tar_unique, imp_unique), imp_unique)))]

    fr = np.zeros(pt.shape[0] + 1, dtype=np.int32)
    fa = np.zeros(pi.shape[0] + 1, dtype=np.int32)

    for i in range(1, len(pt) + 1):
        fr[i] = fr[i - 1] + pt[i - 1]

    for i in range(len(pt) - 1, -1, -1):
        fa[i] = fa[i + 1] + pi[i]

    frr = fr / len(tar)
    far = fa / len(imp)

    thresholds = np.hstack((thresholds, thresholds[-1] + 1e-6))

    return thresholds, frr, far


def far_fix_frr(tar, imp, frr_point):
    tar_unique, tar_counts = np.unique(tar, return_counts=True)
    imp_unique, imp_counts = np.unique(imp, return_counts=True)
    thresholds = np.unique(np.hstack((tar_unique, imp_unique)))

    pt = np.hstack((tar_counts, np.zeros(len(thresholds) - len(tar_counts), dtype=np.int32)))
    pi = np.hstack((np.zeros(len(thresholds) - len(imp_counts), dtype=np.int32), imp_counts))

    pt = pt[np.argsort(np.hstack((tar_unique, np.setdiff1d(imp_unique, tar_unique))))]
    pi = pi[np.argsort(np.hstack((np.setdiff1d(tar_unique, imp_unique), imp_unique)))]

    fr = np.zeros(pt.shape[0] + 1, dtype=np.int32)
    fa = np.zeros(pi.shape[0] + 1, dtype=np.int32)

    for i in range(1, len(pt) + 1):
        fr[i] = fr[i - 1] + pt[i - 1]

    for i in range(len(pt) - 1, -1, -1):
        fa[i] = fa[i + 1] + pi[i]

    frr = fr / len(tar)
    far = fa / len(imp)

    n_point = np.argmin(np.abs(frr - frr_point))
    return far[n_point]


def htr(tar, imp):
    tar_unique, tar_counts = np.unique(tar, return_counts=True)
    imp_unique, imp_counts = np.unique(imp, return_counts=True)
    thresholds = np.unique(np.hstack((tar_unique, imp_unique)))

    pt = np.hstack((tar_counts, np.zeros(len(thresholds) - len(tar_counts), dtype=np.int32)))
    pi = np.hstack((np.zeros(len(thresholds) - len(imp_counts), dtype=np.int32), imp_counts))

    pt = pt[np.argsort(np.hstack((tar_unique, np.setdiff1d(imp_unique, tar_unique))))]
    pi = pi[np.argsort(np.hstack((np.setdiff1d(tar_unique, imp_unique), imp_unique)))]

    fr = np.zeros(pt.shape[0] + 1, dtype=np.int32)
    fa = np.zeros(pi.shape[0] + 1, dtype=np.int32)

    for i in range(1, len(pt) + 1):
        fr[i] = fr[i - 1] + pt[i - 1]

    for i in range(len(pt) - 1, -1, -1):
        fa[i] = fa[i + 1] + pi[i]

    frr = fr / len(tar)
    far = fa / len(imp)

    n_point = np.argmin(np.abs(thresholds - 0.5))
    return (frr[n_point] + far[n_point]) / 2


def compute_frr_far_old(tar, imp):
    pt = np.concatenate((np.ones(len(tar)), np.zeros(len(imp))), axis=0)
    pi = np.concatenate((np.zeros(len(tar)), np.ones(len(imp))), axis=0)

    tar_imp = np.hstack((tar, imp))
    arg_sort = np.argsort(tar_imp)

    pt = pt[arg_sort]
    pi = pi[arg_sort]
    tar_imp = tar_imp[arg_sort]

    fr = np.zeros(pt.shape[0])
    fa = np.zeros(pi.shape[0])

    for i in range(1, len(pt)):
        fr[i] = fr[i - 1] + pt[i - 1]

    for i in range(len(pt) - 2, -1, -1):
        fa[i] = fa[i + 1] + pi[i]

    frr = fr / len(tar)
    far = fa / len(imp)

    return tar_imp, frr, far


def compute_eer(tar, imp):
    tar_imp, fr, fa = compute_frr_far(tar, imp)

    index_min = np.argmin(np.abs(fr - fa))
    eer = 100.0 * np.mean((fr[index_min], fa[index_min]))
    threshold = tar_imp[index_min]

    return eer, threshold


def compute_eer_fast(scores_target, scores_impostor, n=1000):
    n_imps = len(scores_impostor)
    n_tars = len(scores_target)

    scores_target = np.sort(scores_target)
    scores_impostor = np.sort(scores_impostor)

    start = min([min(scores_target), min(scores_impostor)]) - 1e-7
    stop = max([max(scores_target), max(scores_impostor)]) + 1e-7

    scores = np.linspace(start, stop, n)
    fars = []
    frrs = []
    dists = []
    min_gap = float("inf")
    eer = 100
    th = None
    for i, dist in enumerate(scores):
        # far = sum(scores_impostor < dist) / n_imps
        # frr = sum(scores_target >= dist) / n_tars
        far = np.searchsorted(scores_impostor, dist, "left") / n_imps
        frr = (n_tars - np.searchsorted(scores_target, dist, "left")) / n_tars

        fars.append(far)
        frrs.append(frr)
        dists.append(dist)

        gap = np.abs(far - frr)

        if gap < min_gap:
            th = dist
            min_gap = gap
            eer = (far + frr) / 2

    return eer * 100, th



def compute_min_c(tar, imp, c_miss=1, c_fa=1, p_target=0.01):
    tar_imp, fnr, fpr = compute_frr_far(tar, imp)

    beta = c_fa * (1 - p_target) / (c_miss * p_target)
    log_beta = np.log(beta)
    act_c = fnr + beta * fpr
    index_min = np.argmin(act_c)
    min_c = act_c[index_min]
    threshold = tar_imp[index_min]

    return min_c, threshold, log_beta


def compute_act_c(tar, imp, c_miss=1, c_fa=1, p_target=0.01):
    beta = c_fa * (1 - p_target) / (c_miss * p_target)
    log_beta = np.log(beta)

    f_tar = list(filter(lambda t: t < log_beta, tar))
    f_imp = list(filter(lambda i: i > log_beta, imp))

    fnr = len(f_tar) / len(tar)
    fpr = len(f_imp) / len(imp)

    act_c = fnr + beta * fpr

    return act_c, fpr, fnr


def compute_min_dcf(tar, imp, c_miss=1, c_fa=1, p_target=0.01):
    min_c, threshold, log_beta = compute_min_c(tar, imp, c_miss, c_fa, p_target)

    # Normalization factor: cost of naive system
    c_default = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c / c_default

    return min_dcf, threshold, log_beta


def compute_all_costs(tar, imp, c_miss=1, c_fa=1, p_target=0.01):
    beta = c_fa * (1 - p_target) / (c_miss * p_target)
    c_default = min(c_miss * p_target, c_fa * (1 - p_target))

    log_beta = np.log(beta)
    fnr_act = np.mean(tar < log_beta)
    fpr_act = np.mean(imp > log_beta)
    act_c = fnr_act + beta * fpr_act

    thresholds = np.sort(np.concatenate([tar, imp]))
    costs = []
    for t in thresholds:
        fnr = np.mean(tar < t)
        fpr = np.mean(imp > t)
        costs.append(fnr + beta * fpr)
    min_c = np.min(costs)

    act_dcf = act_c / c_default
    min_dcf = min_c / c_default

    return act_c, min_c, act_dcf, min_dcf


def compute_llr_c(tar, imp):
    sum_tar = np.sum([np.log(1. + 1. / np.exp(score)) for score in tar])
    sum_imp = np.sum([np.log(1. + np.exp(score)) for score in imp])

    c_llr = 1 / (2 * np.log(2)) * (sum_tar / len(tar) + sum_imp / len(imp))

    return c_llr


def get_eer(tar, imp):
    return compute_eer(tar, imp)[0]


def get_min_c(tar, imp, c_miss=1, c_fa=1, p_target=0.01):
    if not hasattr(p_target, '__iter__'):
        p_target = [p_target]

    values = list(map(lambda pt: compute_min_c(tar, imp, c_miss, c_fa, pt)[0], p_target))

    return sum(values) / len(values)


def get_act_c(p_target, tar, imp, c_miss=1, c_fa=1):
    if not hasattr(p_target, '__iter__'):
        p_target = [p_target]

    values = list(map(lambda pt: compute_act_c(tar, imp, c_miss, c_fa, pt)[0], p_target))

    return sum(values) / len(values)


def get_llr_c(tar, imp):
    return compute_llr_c(tar, imp)


def get_fr_fa_at_threshold(tar, imp, threshold=0.5):
    fr = len(np.where(tar < threshold)[0])
    fa = len(np.where(imp > threshold)[0])
    fr = fr * 100. / len(tar)
    fa = fa * 100. / len(imp)
    return fr, fa


def get_acer_at_threshold(tar, imp, threshold=0.5):
    fr, fa = get_fr_fa_at_threshold(tar, imp, threshold=threshold)
    return (fr + fa) / 2.0


def get_bpcer_at_apcer(tar, imp, apcer=1):
    tar_imp, fr, fa = compute_frr_far(tar, imp)
    return 100.0 * fr[np.argmax(fa <= (apcer / 100.))]


def get_apcer_at_bpcer(tar, imp, bpcer=1):
    tar_imp, fr, fa = compute_frr_far(tar, imp)
    return 100.0 * fa[np.argmin(fr <= (bpcer / 100.))]


def test_by_protocol(file, c_miss, c_fa, protocol):
    target_scores = []
    impostor_scores = []

    predict = {}
    with open(file, "r") as f:
        for line in f:
            enroll, test, score = line.split(" ")
            predict["{}_{}".format(enroll, test)] = float(score)

    gdound_truth = {}
    with open(protocol, "r") as f:
        for line in f:
            enroll, test, target = line.split(" ")
            gdound_truth["{}_{}".format(enroll, test)] = target == "target\n"

    target_scores = []
    impostor_scores = []
    for trial, score in predict.items():
        if gdound_truth[trial]:
            target_scores.append(score)
        else:
            impostor_scores.append(score)

    p_target = [0.01, 0.005]
    minc = get_min_c(target_scores, impostor_scores, c_miss=c_miss, c_fa=c_fa, p_target=p_target)
    eer = get_eer(target_scores, impostor_scores)

    return eer, minc


def compute_metrics(file, c_miss, c_fa):
    target_scores = []
    impostor_scores = []
    with open(file, "r") as f:
        for line in f:
            score, target_type = line.split(" ")
            if target_type == "target\n":
                target_scores.append(float(score))
            else:
                impostor_scores.append(float(score))

    p_target = [0.01, 0.005]
    minc = get_min_c(target_scores, impostor_scores, c_miss=c_miss, c_fa=c_fa, p_target=p_target)
    eer = get_eer(target_scores, impostor_scores)

    return eer, minc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--scores_file', help='path source files', required=True)
    parser.add_argument('--protocol', default=None, help='path to test protocol', required=False)
    parser.add_argument('--c_miss', default=1, type=int, help='path source files', required=False)
    parser.add_argument('--c_fa', default=1, type=int, help='path source files', required=False)

    args = parser.parse_args()
    scores_file = args.scores_file
    protocol = args.protocol
    c_miss = args.c_miss
    c_fa = args.c_fa

    if protocol is not None:
        eer, minc = test_by_protocol(file=scores_file, c_miss=c_miss, c_fa=c_fa, protocol=protocol)
    else:
        eer, minc = compute_metrics(file=scores_file, c_miss=c_miss, c_fa=c_fa)

    print(f"EER: {eer}")
    print(f"minC: {minc}")
