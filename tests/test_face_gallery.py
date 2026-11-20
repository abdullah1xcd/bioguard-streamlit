"""Face gallery / platform flow with a deterministic FAKE engine (no model download needed).  Identity is encoded in the image's top row;
real-model behaviour (SCRFD + ArcFace) was smoke-tested separately on a real face photo."""
import cv2
import numpy as np
import pytest
from src.face.engine import l2n
from src.face.gallery import FaceGallery

LM = np.array([[70, 80], [130, 80], [100, 100], [80, 130], [120, 130]], np.float32)


class FakeEngine:
    backend = "fake"

    def detect(self, img):
        if img[0, 0, 0] == 0:                                   # marker: "no face in this image"
            return None
        return {"box": (20.0, 20.0, 180.0, 180.0), "landmarks": LM, "det_score": 0.99, "detector": "fake"}

    def embed(self, img, detection=None):
        return l2n(img[0, :16, 1].astype(np.float32) - 127.5), detection

    def info(self):
        return {"backend": "fake", "insightface": True, "yolo": False, "detector_setting": "auto"}


def photo(identity, k=0, face=True, blurry=False):
    rng = np.random.default_rng(1000 * identity + k)
    img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    if blurry:
        img = cv2.GaussianBlur(img, (0, 0), 25)
    v = np.zeros(16); v[identity] = 1.0                      # identities are mutually orthogonal -> impostor cosine ~ 0
    img[0, :16, 1] = np.clip(127.5 + 100 * v + rng.normal(0, 3, 16), 0, 255).astype(np.uint8)   # person-specific code + small per-photo noise
    img[0, 0, 0] = 255 if face else 0
    return img


def enrol(tmp_path, pid, identity, n=5, **kw):
    d = tmp_path / pid / "face"; d.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        cv2.imwrite(str(d / f"{k}.png"), photo(identity, k, **kw))


@pytest.fixture
def gal(tmp_path):
    enrol(tmp_path, "p1", 1); enrol(tmp_path, "p2", 2)
    g = FaceGallery(tmp_path, FakeEngine(), cache_path=tmp_path / "g.joblib")
    g.refresh({"alice": "p1", "bob": "p2"})
    return g


def test_genuine_accepted_impostor_rejected_without_training(gal):
    assert gal.verify("alice", photo(1, 99))["accepted"] is True
    r = gal.verify("alice", photo(2, 99))
    assert r["accepted"] is False and r["score"] < gal.threshold


def test_too_few_photos_gives_clear_reason(tmp_path):
    enrol(tmp_path, "p1", 1, n=2)
    g = FaceGallery(tmp_path, FakeEngine(), cache_path=tmp_path / "g.joblib"); g.refresh({"alice": "p1"})
    r = g.verify("alice", photo(1, 9)); assert not r["accepted"] and "enrol" in r["reason"].lower() and g.ready_users() == []


def test_no_face_and_blurry_rejected_with_reasons(gal):
    r = gal.verify("alice", photo(1, 7, face=False)); assert not r["accepted"] and "no_face_detected" in r["reason"]
    r = gal.verify("alice", photo(1, 8, blurry=True)); assert not r["accepted"] and "too_blurry" in r["reason"]
    assert gal.verify("alice", photo(1, 8, blurry=True), enforce_quality=False)["face_detected"] is True


def test_mislabeled_enrolment_photo_is_dropped(tmp_path):
    enrol(tmp_path, "p1", 1, n=6)
    cv2.imwrite(str(tmp_path / "p1" / "face" / "wrong.png"), photo(2, 0))          # someone else's face in alice's folder
    g = FaceGallery(tmp_path, FakeEngine(), cache_path=tmp_path / "g.joblib")
    st = g.refresh({"alice": "p1"})["alice"]
    assert st["photos"] == 7 and st["usable"] == 6
    assert not g.verify("alice", photo(2, 50))["accepted"]


def test_bad_enrolment_photos_are_reported(tmp_path):
    enrol(tmp_path, "p1", 1, n=4); enrol(tmp_path / "x", "p", 1, n=0)
    cv2.imwrite(str(tmp_path / "p1" / "face" / "b.png"), photo(1, 40, blurry=True))
    g = FaceGallery(tmp_path, FakeEngine(), cache_path=tmp_path / "g.joblib")
    st = g.refresh({"alice": "p1"})["alice"]; assert st["usable"] == 4 and st["rejected_reasons"].get("too_blurry") == 1


def test_cache_is_incremental_and_follows_deletions(tmp_path):
    enrol(tmp_path, "p1", 1)
    g = FaceGallery(tmp_path, FakeEngine(), cache_path=tmp_path / "g.joblib"); g.refresh({"alice": "p1"})
    calls = []; orig = g.assess; g.assess = lambda im: (calls.append(1), orig(im))[1]
    g.refresh({"alice": "p1"}); assert calls == []                                  # nothing re-embedded
    (tmp_path / "p1" / "face" / "0.png").unlink(); g.refresh({"alice": "p1"}); assert len(g._emb["alice"]) == 4
    g2 = FaceGallery(tmp_path, FakeEngine(), cache_path=tmp_path / "g.joblib"); g2.assess = g.assess
    g2.refresh({"alice": "p1"}); assert calls == []                                 # persisted cache reused after restart
    g3 = FaceGallery(tmp_path, FakeEngine(), cache_path=tmp_path / "g.joblib"); g3.refresh({}); assert g3._load()["files"] == {}


def test_calibrate_needs_two_users_then_sets_threshold(tmp_path, gal):
    one = FaceGallery(tmp_path / "solo", FakeEngine(), cache_path=tmp_path / "s.joblib"); enrol(tmp_path / "solo", "p1", 1); one.refresh({"alice": "p1"})
    assert one.calibrate()["calibrated"] is False
    r = gal.calibrate(); assert r["calibrated"] and r["FAR"] == 0 and gal.threshold == pytest.approx(r["threshold"])
    gal.set_threshold(None); assert gal.threshold == gal.cfg["threshold"]


def test_platform_face_only_flow_end_to_end(plat, tmp_path):
    plat.face.engine = FakeEngine()
    ua = plat.enrollment.enroll("alice", consent=True); ub = plat.enrollment.enroll("bob", consent=True)
    ok, _ = cv2.imencode(".png", photo(1, 0))
    for k in range(5):
        plat.enrollment.add_samples(ua["id"], "face", [(f"a{k}.png", cv2.imencode(".png", photo(1, k))[1].tobytes())])
        plat.enrollment.add_samples(ub["id"], "face", [(f"b{k}.png", cv2.imencode(".png", photo(2, k))[1].tobytes())])
    assert plat.face_users() == ["alice", "bob"]
    assert plat.authenticate("alice", photo(1, 77), None, "open_dashboard")["final_status"] == "authenticated"
    assert plat.authenticate("alice", photo(2, 77), None, "open_dashboard")["final_status"] == "rejected"
    plat.enrollment.delete_user(ub["id"]); assert plat.face_users() == ["alice"]


def test_sensitive_action_requires_voice_and_never_crashes_without_model(plat):
    from src.security.permissions import COMMANDS
    sensitive = next(k for k, v in COMMANDS.items() if v["sensitivity"] in ("high", "critical"))
    plat.face.engine = FakeEngine(); u = plat.enrollment.enroll("alice", consent=True)
    for k in range(5):
        plat.enrollment.add_samples(u["id"], "face", [(f"a{k}.png", cv2.imencode(".png", photo(1, k))[1].tobytes())])
    r = plat.authenticate("alice", photo(1, 50), None, sensitive); assert r["final_status"] == "rejected" and "voice" in r["reason"]
    r = plat.authenticate("alice", photo(1, 51), np.zeros(16000, np.float32), sensitive); assert r["final_status"] == "rejected" and "not trained" in r["reason"]
