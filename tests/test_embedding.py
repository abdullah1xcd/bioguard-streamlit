import numpy as np
import pytest
from src.face.embedding import FaceEmbedder, FaceNotFound, MAX_SIDE, to_bgr


def row(x=10, y=10, w=80, h=80, score=.9):
    return np.array([x, y, w, h, 30, 30, 70, 30, 50, 50, 35, 70, 65, 70, score], dtype=np.float32)


class Det:
    def __init__(self, faces): self.faces, self.size = faces, None
    def setInputSize(self, s): self.size = s
    def detect(self, img): return 1, self.faces


class Rec:
    def __init__(self, vec=None): self.vec = np.arange(1, 129, dtype=np.float32).reshape(1, 128) if vec is None else vec
    def alignCrop(self, img, r): return img[:4, :4]
    def feature(self, a): return self.vec


def emb(faces, vec=None):
    return FaceEmbedder(detector=Det(faces), recognizer=Rec(vec))


def test_embedding_is_unit_norm_128d_with_landmarks():
    v, info = emb(np.array([row()])).embed(np.zeros((100, 100, 3), np.uint8))
    assert v.shape == (128,) and abs(np.linalg.norm(v) - 1) < 1e-5
    assert info["landmarks"].shape == (5, 2) and info["box"] == (10, 10, 80, 80) and info["det_score"] == pytest.approx(.9, abs=1e-6)


def test_no_face_raises_FaceNotFound():
    for faces in (None, np.zeros((0, 15), np.float32)):
        with pytest.raises(FaceNotFound): emb(faces).embed(np.zeros((100, 100, 3), np.uint8))


def test_degenerate_embedding_rejected():
    with pytest.raises(FaceNotFound): emb(np.array([row()]), vec=np.zeros((1, 128), np.float32)).embed(np.zeros((100, 100, 3), np.uint8))
    with pytest.raises(FaceNotFound): emb(np.array([row()]), vec=np.full((1, 128), np.nan, np.float32)).embed(np.zeros((100, 100, 3), np.uint8))


def test_best_face_is_largest_confident_one():
    faces = np.array([row(w=30, h=30, score=.99), row(w=90, h=90, score=.85), row(w=90, h=90, score=.3)])
    _, r = emb(faces).detect(np.zeros((200, 200, 3), np.uint8)); assert r[2] == 90 and r[-1] == pytest.approx(.85, abs=1e-6)


def test_gray_input_converted_and_large_images_downscaled():
    assert to_bgr(np.zeros((10, 10), np.uint8)).shape == (10, 10, 3)
    d = Det(np.array([row()])); e = FaceEmbedder(detector=d, recognizer=Rec())
    bgr, _ = e.detect(np.zeros((3000, 2000), np.uint8)); assert max(bgr.shape[:2]) == MAX_SIDE and d.size == (bgr.shape[1], bgr.shape[0])


@pytest.mark.parametrize("bad", [None, np.zeros((0, 0), np.uint8), np.zeros((5, 5, 4), np.uint8), np.zeros((2, 2, 2, 2))])
def test_invalid_images(bad):
    with pytest.raises(ValueError): to_bgr(bad)


def test_available_is_false_without_model_files(tmp_path):
    assert not FaceEmbedder.available(tmp_path / "a.onnx", tmp_path / "b.onnx")
