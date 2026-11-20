import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def eer_threshold(y, s):
    """EER and threshold where FAR ~= FRR. Calibrate on VALIDATION data only."""
    if len(set(np.asarray(y).astype(int).tolist())) < 2:
        raise ValueError("need both genuine (1) and impostor (0) scores")
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    t = float(thr[i])
    if not np.isfinite(t):                      # sklearn >= 1.3 puts +inf at index 0
        t = float(np.max(s)) + 1e-9
    return float((fpr[i] + fnr[i]) / 2), t


def report(y, s, thr):
    y = np.asarray(y); pred = (np.asarray(s) >= thr).astype(int)
    if len(set(y.astype(int).tolist())) < 2:
        raise ValueError("need both genuine (1) and impostor (0) scores")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    far = fp / max(fp + tn, 1)   # false acceptances / impostor attempts
    frr = fn / max(fn + tp, 1)   # false rejections / genuine attempts
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"threshold": thr, "accuracy": (tp + tn) / len(y), "precision": prec,
            "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "FAR": far, "FRR": frr, "AUC": roc_auc_score(y, s),
            "balanced_accuracy": (1 - far + 1 - frr) / 2,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn}


def far_frr_curve(y, s, n_points: int = 400):
    """FAR(t) and FRR(t) on a score grid (decision rule: accept if score >= t)."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    ts = np.unique(np.concatenate([np.quantile(s, np.linspace(0, 1, n_points)), [s.min() - 1e-9, s.max() + 1e-9]]))
    gen, imp = s[y == 1], s[y == 0]
    far = np.array([(imp >= t).mean() if len(imp) else np.nan for t in ts])
    frr = np.array([(gen < t).mean() if len(gen) else np.nan for t in ts])
    return ts, far, frr


def threshold_at_far(y, s, target_far: float):
    """Smallest threshold whose FAR on (validation) data is <= target_far."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    imp = np.sort(s[y == 0])
    if len(imp) == 0:
        raise ValueError("no impostor scores")
    k = int(np.floor(target_far * len(imp)))      # allowed number of false accepts
    if k >= len(imp):
        return float(imp.min()) - 1e-9
    return float(np.nextafter(imp[len(imp) - 1 - k], np.inf))


def bootstrap_ci(y, s, groups, thr, n_boot: int = 500, seed: int = 0, alpha: float = 0.05):
    """Cluster bootstrap (resample PROBES, not trials: the trials of one probe are dependent).
    Returns {metric: (lo, hi)} for EER, AUC, FAR and FRR at the fixed validation threshold."""
    y = np.asarray(y); s = np.asarray(s, dtype=float); groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    acc = {"EER": [], "AUC": [], "FAR": [], "FRR": []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        ix = np.concatenate([idx_by[g] for g in pick])
        yb, sb = y[ix], s[ix]
        if yb.min() == yb.max():
            continue
        acc["EER"].append(eer_threshold(yb, sb)[0]); acc["AUC"].append(roc_auc_score(yb, sb))
        pr = sb >= thr
        acc["FAR"].append(pr[yb == 0].mean()); acc["FRR"].append(1 - pr[yb == 1].mean())
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if v else (np.nan, np.nan) for k, v in acc.items()}


def liveness_rates(is_attack, passed):
    """ISO/IEC 30107-3 style: APCER = attacks wrongly passed, BPCER = bona-fide wrongly rejected."""
    a = np.asarray(is_attack).astype(bool); p = np.asarray(passed).astype(bool)
    apcer = float(p[a].mean()) if a.any() else float("nan")
    bpcer = float((~p[~a]).mean()) if (~a).any() else float("nan")
    return {"APCER": apcer, "BPCER": bpcer, "ACER": float(np.nanmean([apcer, bpcer]))}
