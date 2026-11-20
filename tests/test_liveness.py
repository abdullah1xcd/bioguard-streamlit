import numpy as np
import pytest
from src.challenges.liveness import (Challenge, evaluate_challenge, evaluate_liveness_clips, new_challenge, observation)
from src.config import LIVENESS_CFG


def lm(yaw):          # landmarks whose yaw proxy equals `yaw`: eyes x=40/80 (dist 40, mid 60)
    return [[40, 50], [80, 50], [60 + yaw * 40, 70], [45, 90], [75, 90]]


def turn(deltas, base=0.0, w=100):
    return [observation((0, 0, w, w), lm(base + d)) for d in deltas]


def closer(widths):
    return [observation((0, 0, w, w), lm(0)) for w in widths]


CH = lambda n: Challenge(n, 0.0, 20.0)
UP = [0, 0, 0, .1, .2, .35, .45, .5, .5, .45]
NOW = 5.0


def test_turn_left_passes_with_positive_yaw_and_fails_for_right():
    r = evaluate_challenge(CH("turn_left"), turn(UP), now=NOW); assert r["passed"] and r["reason"] == "movement detected"
    r = evaluate_challenge(CH("turn_right"), turn(UP), now=NOW); assert not r["passed"] and r["reason"] == "turned the wrong way"
    assert evaluate_challenge(CH("turn_right"), turn([-d for d in UP]), now=NOW)["passed"]


def test_static_face_fails_every_challenge():
    still = turn([0] * 12)
    for n in ("turn_left", "turn_right"):
        assert not evaluate_challenge(CH(n), still, now=NOW)["passed"]
    assert not evaluate_challenge(CH("move_closer"), closer([100] * 12), now=NOW)["passed"]


def test_single_noisy_frame_does_not_pass():
    spike = [0, 0, 0, 0, .8, 0, 0, 0, 0, 0]
    assert not evaluate_challenge(CH("turn_left"), turn(spike), now=NOW)["passed"]


def test_must_start_frontal():
    r = evaluate_challenge(CH("turn_left"), turn(UP, base=.4), now=NOW); assert not r["passed"] and "frontal" in r["reason"]


def test_move_closer():
    assert evaluate_challenge(CH("move_closer"), closer([100, 100, 100, 110, 125, 135, 140, 140, 140]), now=NOW)["passed"]
    assert not evaluate_challenge(CH("move_closer"), closer([100] * 5 + [110] * 5), now=NOW)["passed"]


def test_expired_few_frames_and_lost_face_fail_with_reasons():
    assert evaluate_challenge(CH("turn_left"), turn(UP), now=100.0)["reason"] == "challenge expired"
    assert "too few frames" in evaluate_challenge(CH("turn_left"), turn(UP[:3]), now=NOW)["reason"]
    assert evaluate_challenge(CH("turn_left"), [], now=NOW)["passed"] is False
    lost = turn(UP); lost[2:8] = [observation()] * 6
    assert evaluate_challenge(CH("turn_left"), lost, now=NOW)["reason"] == "face lost during challenge"
    nolm = [observation((0, 0, 100, 100), None)] * 10
    assert "landmarks" in evaluate_challenge(CH("turn_left"), nolm, now=NOW)["reason"]


def test_challenges_are_random_unique_and_from_config():
    cs = [new_challenge() for _ in range(300)]
    assert {c.name for c in cs} == set(LIVENESS_CFG["challenges"]) and len({c.nonce for c in cs}) == 300
    assert all(c.instruction for c in cs)


def test_clip_evaluation_gives_apcer_bpcer():
    clips = [("turn_left", turn(UP), False), ("turn_left", turn(UP), False), ("turn_left", turn([0] * 12), True), ("turn_left", turn(UP), True)]
    r = evaluate_liveness_clips(clips); assert r["BPCER"] == 0.0 and r["APCER"] == .5


def test_platform_rejects_without_liveness_and_does_not_run_matching(plat):
    plat.enrollment.enroll("alice", consent=True); plat.db.set_setting("require_liveness", True)
    img = np.zeros((64, 64), np.uint8)
    for lv in (None, {"passed": False, "reason": "no head turn"}, {"passed": "yes"}):
        r = plat.authenticate("alice", img, None, "open_dashboard", liveness=lv)
        assert r["final_status"] == "rejected" and "liveness" in r["reason"]
    r = plat.authenticate("alice", img, None, "open_dashboard", liveness={"passed": True})   # a passed liveness proceeds to matching
    assert r["final_status"] == "rejected" and "liveness" not in r["reason"]
