import numpy as np

from src.face.engine import FaceEngine


def _fake_detection(scale=1.0, include_yunet_row=False):
    detection = {
        "box": (20.0 * scale, 30.0 * scale, 100.0 * scale, 130.0 * scale),
        "landmarks": np.array(
            [[40, 60], [80, 60], [60, 80], [45, 105], [75, 105]],
            dtype=np.float32,
        ) * scale,
        "det_score": 0.91,
        "detector": "scrfd",
    }
    if include_yunet_row:
        detection["_row"] = np.arange(15, dtype=np.float32) * scale
    return detection


def test_detect_retries_small_grayscale_image_and_maps_coordinates(monkeypatch):
    engine = FaceEngine()
    calls = []

    def detect_once(image):
        calls.append(image.shape)
        if len(calls) == 1:
            return None
        return _fake_detection(scale=2.0)

    monkeypatch.setattr(engine, "_detect_once", detect_once)
    image = np.zeros((200, 180), dtype=np.uint8)

    result = engine.detect(image)

    assert len(calls) == 2
    assert calls[0] == (200, 180, 3)  # grayscale was converted to BGR
    assert calls[1] == (400, 360, 3)
    assert result["box"] == (20.0, 30.0, 100.0, 130.0)
    np.testing.assert_allclose(
        result["landmarks"],
        [[40, 60], [80, 60], [60, 80], [45, 105], [75, 105]],
    )
    assert result["detector"] == "scrfd+upscaled"


def test_detect_preserves_original_detection_without_upscaling(monkeypatch):
    engine = FaceEngine()
    calls = []
    original = _fake_detection()

    def detect_once(image):
        calls.append(image.shape)
        return original

    monkeypatch.setattr(engine, "_detect_once", detect_once)

    result = engine.detect(np.zeros((200, 180, 3), dtype=np.uint8))

    assert len(calls) == 1
    assert result is original


def test_upscaled_yunet_row_is_mapped_for_embedding(monkeypatch):
    engine = FaceEngine()
    calls = []

    def detect_once(image):
        calls.append(image.shape)
        if len(calls) == 1:
            return None
        return _fake_detection(scale=2.0, include_yunet_row=True)

    monkeypatch.setattr(engine, "_detect_once", detect_once)

    result = engine.detect(np.zeros((200, 180), dtype=np.uint8))

    np.testing.assert_allclose(result["_row"], np.arange(15, dtype=np.float32))
