import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
FINGERPRINT_UPLOAD_TYPES = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")


@st.cache_resource
def get_platform():
    from src.platform import Platform
    return Platform()


def df(rows, cols=None):
    d = pd.DataFrame(rows)
    return d[[c for c in cols if c in d.columns]] if cols and not d.empty else d


def operator():
    return st.session_state.get("op_name", "local_admin"), st.session_state.get("op_role", "admin")


def show_safe_error(exc, message="The operation could not be completed. Please check the input and try again."):
    st.error(message)
    with st.expander("Technical details (for troubleshooting)"):
        st.code(f"{type(exc).__name__}: {exc}")


def _quality_guidance(img):
    import cv2
    import numpy as np
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    notes = []
    if brightness < 45:
        notes.append("Image may be too dark; move to brighter, even lighting.")
    elif brightness > 220:
        notes.append("Image may be overexposed; avoid strong light behind you.")
    if sharpness < 35:
        notes.append("Image may be blurry; hold the camera steady.")
    return notes


def gray_input(label="Face image"):
    import cv2
    import numpy as np

    st.caption("Use the camera or upload a clear image. Keep the face centered, well lit, and in focus.")
    cam = st.camera_input("Capture with webcam", key=f"{label}_camera")
    up = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"], key=f"{label}_upload")
    f = cam or up
    if not f:
        return None
    img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.warning("We couldn't read that image. Please try another JPG or PNG file.")
        return None
    st.image(img, channels="BGR", caption="Image preview · retake or upload another image if needed.", use_container_width=True)
    notes = _quality_guidance(img)
    if notes:
        for note in notes:
            st.warning(note)
    else:
        st.success("Basic image check looks good. Final face quality is assessed by the biometric engine.")
    return img


def fingerprint_input(label="Fingerprint"):
    import cv2
    import numpy as np

    st.caption("Upload a clear fingerprint image. Keep the ridge pattern visible and avoid glare or heavy cropping.")
    up = st.file_uploader("Upload fingerprint (BMP/PNG/JPG/TIFF)", type=list(FINGERPRINT_UPLOAD_TYPES), key=f"{label}_upload")
    if not up:
        return None
    img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.warning("We couldn't read that fingerprint image. Please try another file.")
        return None
    st.image(img, caption="Fingerprint preview", use_container_width=True)
    return img


def wave_input(label="Voice"):
    import io
    from src.voice.preprocessing import load_audio

    st.caption("Record or upload a WAV sample in a quiet place. Speak naturally for a few seconds.")
    rec = st.audio_input("Record with microphone", key=f"{label}_microphone") if hasattr(st, "audio_input") else None
    up = st.file_uploader("Or upload a WAV recording", type=["wav"], key=f"{label}_upload")
    f = rec or up
    if not f:
        return None
    st.audio(f.getvalue(), format="audio/wav")
    try:
        return load_audio(io.BytesIO(f.getvalue()))
    except Exception as exc:
        show_safe_error(exc, "We couldn't process this audio. Please use a valid WAV recording.")
        return None


def show_verification_result(result, modality, score_label="Match score", score_format=".3f"):
    accepted = bool(result.get("accepted"))
    if accepted:
        st.success(f"{modality} verification passed")
        st.markdown("**Identity check:** The submitted sample matched the enrolled profile.")
    else:
        st.error(f"{modality} verification not passed")
        st.markdown("**Identity check:** The sample was not accepted. Check the guidance below and try again.")
    cols = st.columns(3)
    score = result.get("score")
    threshold = result.get("threshold")
    latency = result.get("ms")
    cols[0].metric(score_label, format(score, score_format) if isinstance(score, (int, float)) else "—")
    cols[1].metric("Decision threshold", format(threshold, score_format) if isinstance(threshold, (int, float)) else "—")
    cols[2].metric("Processing time", f"{latency:.0f} ms" if isinstance(latency, (int, float)) else "—")
    reason = result.get("reason")
    if reason:
        st.info(f"Result details: {reason.replace('_', ' ').capitalize()}")
    quality = result.get("quality")
    if isinstance(quality, dict) and quality.get("ok") is False:
        st.warning("Sample quality check failed. Please capture or upload a clearer sample.")
    with st.expander("Technical details"):
        st.json({k: v for k, v in result.items() if k not in {"box"}})
