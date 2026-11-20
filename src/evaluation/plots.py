"""Simple, readable evaluation plots (matplotlib, Agg backend)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from src.evaluation.metrics import eer_threshold, far_frr_curve


def _safe(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def plot_roc(curves: dict, path, title="ROC (test)"):
    plt.figure(figsize=(6, 5))
    for k, (y, s) in curves.items():
        f, t, _ = roc_curve(y, s); plt.plot(f, t, label=k)
    plt.plot([0, 1], [0, 1], "k--", lw=.8); plt.xlabel("FAR"); plt.ylabel("1 - FRR (genuine accept rate)")
    plt.title(title); plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_far_frr(name, y, s, thr, path):
    ts, far, frr = far_frr_curve(y, s)
    eer, t_eer = eer_threshold(y, s)
    plt.figure(figsize=(6, 4))
    plt.plot(ts, far, label="FAR"); plt.plot(ts, frr, label="FRR")
    plt.scatter([t_eer], [eer], c="k", zorder=5, label=f"EER (test, oracle) = {eer:.3f}")
    plt.axvline(thr, color="r", ls="--", lw=1, label="operating threshold (from validation)")
    plt.xlabel("threshold"); plt.ylabel("error rate"); plt.title(f"FAR / FRR vs threshold - {name}", fontsize=9)
    plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_score_dist(name, y, s, thr, path):
    y = np.asarray(y); s = np.asarray(s)
    plt.figure(figsize=(6, 4))
    bins = np.linspace(min(s), max(s), 40)
    plt.hist(s[y == 0], bins=bins, alpha=.6, density=True, label="impostor")
    plt.hist(s[y == 1], bins=bins, alpha=.6, density=True, label="genuine")
    plt.axvline(thr, color="r", ls="--", lw=1, label="threshold")
    plt.xlabel("score"); plt.ylabel("density"); plt.title(f"Genuine vs impostor scores (test) - {name}", fontsize=9)
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_confusions(summary: dict, path):
    n = len(summary); cols = min(3, n); rows = -(-n // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.3 * rows), squeeze=False)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    for ax, (k, r) in zip(axs.ravel(), summary.items()):
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]]); ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center")
        ax.set_xticks([0, 1], ["Rejected", "Accepted"]); ax.set_yticks([0, 1], ["Impostor", "Genuine"]); ax.set_title(k, fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_model_comparison(rows, path):
    """rows: list of (label, EER, lo, hi). Horizontal bars with 95% bootstrap CI (test EER at oracle threshold)."""
    rows = [r for r in rows if np.isfinite(r[1])]
    if not rows:
        return
    labels = [r[0] for r in rows]; v = np.array([r[1] for r in rows])
    lo = np.array([r[2] for r in rows]); hi = np.array([r[3] for r in rows])
    plt.figure(figsize=(7, 0.45 * len(rows) + 1.2))
    plt.barh(range(len(rows)), v, xerr=[np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)], color="#4c78a8", capsize=3)
    plt.yticks(range(len(rows)), labels, fontsize=7); plt.gca().invert_yaxis()
    plt.xlabel("test EER (lower is better), 95% cluster-bootstrap CI"); plt.grid(axis="x", alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()
