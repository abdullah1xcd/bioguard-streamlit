"""Genuine / impostor trial construction.

   GENUINE : probe of person P  vs  enrolment references of P            (label 1)
   IMPOSTOR: probe of person P  vs  enrolment references of another Q    (label 0)  ("P claims to be Q")

Enrolment references always come from the TRAIN split; probes from VAL (threshold / calibration / weights) or
TEST (final report).  Probe k of a person is the SAME capture event for every modality (virtual multimodal
user), and the reference subset for a (claimed person, trial) is derived from a per-trial seed so every model is
scored on exactly the same trials."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Trials:
    pid: np.ndarray      # true identity of the probe
    k: np.ndarray        # probe index inside (pid, split)
    claim: np.ndarray    # claimed identity
    y: np.ndarray        # 1 genuine / 0 impostor
    seed: np.ndarray     # per-trial seed -> reference subset
    group: np.ndarray    # probe id string, for the cluster bootstrap / grouped CV

    def __len__(self):
        return len(self.y)


def make_trials(probe_counts: dict, train_counts: dict, rng, n_imp: int = 10) -> Trials:
    pids = [p for p in sorted(probe_counts) if probe_counts[p] > 0 and train_counts.get(p, 0) > 0]
    if len(pids) < 2:
        raise ValueError("need >=2 participants with probe and enrolment data to form genuine AND impostor trials")
    rows = []
    for p in pids:
        others = [q for q in pids if q != p]
        for k in range(probe_counts[p]):
            claims = [p] + [str(q) for q in rng.choice(others, min(n_imp, len(others)), replace=False)]
            for q in claims:
                rows.append((p, k, q, int(q == p), int(rng.integers(0, 2 ** 31 - 1)), f"{p}#{k}"))
    a = list(zip(*rows))
    return Trials(np.array(a[0]), np.array(a[1]), np.array(a[2]), np.array(a[3]), np.array(a[4]), np.array(a[5]))


def ref_indices(n_train: int, seed: int, n_refs: int):
    return np.random.default_rng(int(seed)).choice(n_train, min(n_refs, n_train), replace=False)


def trial_lists(trials: Trials, probes: dict, refs: dict, n_refs: int):
    """[(probe_vec, refs_matrix), ...] for PairVerifier/CosineVerifier.score_trials."""
    out = []
    for p, k, q, sd in zip(trials.pid, trials.k, trials.claim, trials.seed):
        R = refs[q]
        out.append((probes[p][k], R[ref_indices(len(R), sd, n_refs)]))
    return out


def align_probes(counts_by_group: dict, rng):
    """For one split: choose, per participant, a common probe count n=min over modality groups and a random
    pairing permutation per group.  Returns {group: {pid: index_array}} (all arrays of equal length per pid)."""
    pids = set.intersection(*[set(c) for c in counts_by_group.values()])
    out = {g: {} for g in counts_by_group}
    for p in sorted(pids):
        n = min(c[p] for c in counts_by_group.values())
        if n == 0:
            continue
        for g, c in counts_by_group.items():
            out[g][p] = rng.permutation(c[p])[:n]
    return out
