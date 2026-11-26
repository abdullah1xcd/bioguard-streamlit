import numpy as np

from src.face.quality import assess_face_quality


def _permissive_cfg():
    return {
        "min_image_px": 60,
        "min_face_px": 60,
        "min_sharpness": 0,
        "brightness": (0, 255),
        "min_contrast": 0,
        "max_abs_yaw": 1.0,
    }


def _textured_gray_image():
    rng = np.random.default_rng(7)
    return rng.integers(30, 220, size=(80, 70), dtype=np.uint8)


def test_tight_crop_70x80_with_60px_face_passes_size_checks():
    image = _textured_gray_image()

    result = assess_face_quality(
        image,
        box=(5, 5, 60.0, 60.0),
        cfg=_permissive_cfg(),
    )

    assert "low_resolution_image" not in result["reasons"]
    assert "face_too_small" not in result["reasons"]
    assert result["metrics"]["image_w"] == 70
    assert result["metrics"]["image_h"] == 80
    assert result["metrics"]["face_px"] == 60.0


def test_fractional_face_box_below_60px_is_not_rounded_up():
    image = _textured_gray_image()

    result = assess_face_quality(
        image,
        box=(5, 5, 59.9, 60.0),
        cfg=_permissive_cfg(),
    )

    assert "face_too_small" in result["reasons"]
