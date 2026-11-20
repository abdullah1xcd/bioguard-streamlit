import numpy as np
import pytest
from src.evaluation.metrics import (bootstrap_ci, eer_threshold, far_frr_curve, liveness_rates, report, threshold_at_far)

rng = np.random.default_rng(3)
Y = np.r_[np.ones(300), np.zeros(900)]
S = np.r_[rng.normal(.7, .12, 300), rng.normal(.35, .12, 900)]


def test_perfect_separation_gives_zero_eer_and_auc_one():
    e, t = eer_threshold([1, 1, 0, 0], [.9, .8, .2, .1]); r = report([1, 1, 0, 0], [.9, .8, .2, .1], t)
    assert e == 0 and r["AUC"] == 1 and r["FAR"] == 0 and r["FRR"] == 0


def test_eer_point_has_far_close_to_frr():
    e, t = eer_threshold(Y, S); r = report(Y, S, t)
    assert abs(r["FAR"] - r["FRR"]) < 0.03 and abs(e - (r["FAR"] + r["FRR"]) / 2) < 0.03


def test_report_fields_and_confusion_consistency():
    r = report(Y, S, 0.5)
    assert r["TP"] + r["FN"] == 300 and r["TN"] + r["FP"] == 900
    assert r["FAR"] == pytest.approx(r["FP"] / 900) and r["FRR"] == pytest.approx(r["FN"] / 300)
    assert r["balanced_accuracy"] == pytest.approx(1 - (r["FAR"] + r["FRR"]) / 2)


def test_threshold_at_far_respects_target_on_the_data_it_was_chosen_on():
    for target in (0.05, 0.01, 0.0):
        t = threshold_at_far(Y, S, target)
        assert (S[Y == 0] >= t).mean() <= target + 1e-12
    with pytest.raises(ValueError): threshold_at_far([1, 1], [.1, .2], .01)


def test_far_frr_curve_is_monotone():
    ts, far, frr = far_frr_curve(Y, S)
    assert (np.diff(far) <= 1e-12).all() and (np.diff(frr) >= -1e-12).all()


def test_bootstrap_ci_brackets_point_estimate_and_is_seeded():
    g = np.arange(len(Y)) // 4; e, t = eer_threshold(Y, S)
    a = bootstrap_ci(Y, S, g, t, n_boot=100, seed=5); b = bootstrap_ci(Y, S, g, t, n_boot=100, seed=5)
    assert a == b and a["EER"][0] <= e <= a["EER"][1]


def test_single_class_is_an_error_not_a_silent_number():
    with pytest.raises(ValueError): eer_threshold([1, 1, 1], [.1, .2, .3])


def test_liveness_rates():
    r = liveness_rates([1, 1, 0, 0], [0, 1, 1, 1]); assert r == {"APCER": .5, "BPCER": 0.0, "ACER": .25}
