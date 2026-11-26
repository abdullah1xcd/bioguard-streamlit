import cv2
import numpy as np


def test_bmp_fingerprint_upload_is_saved_and_discoverable(plat):
    user = plat.enrollment.enroll(
        "bmp_test_user",
        role="standard_user",
        consent=True,
    )

    image = np.zeros((48, 48), dtype=np.uint8)
    cv2.line(image, (4, 24), (44, 24), 255, 2)
    ok, encoded = cv2.imencode(".bmp", image)
    assert ok

    saved_count = plat.enrollment.add_samples(
        user["id"],
        "fingerprint",
        [("100__M_Left_index_finger_CR.bmp", encoded.tobytes())],
        actor="test",
        session=1,
    )

    assert saved_count == 1
    samples = plat._samples("bmp_test_user", "fingerprint")
    assert len(samples) == 1
    assert samples[0].suffix.lower() == ".bmp"

    loaded = cv2.imread(str(samples[0]), cv2.IMREAD_GRAYSCALE)
    assert loaded is not None
    np.testing.assert_array_equal(loaded, image)
