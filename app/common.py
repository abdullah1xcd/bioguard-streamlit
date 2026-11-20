import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st


@st.cache_resource
def get_platform():
    from src.platform import Platform
    return Platform()


def df(rows, cols=None):
    d = pd.DataFrame(rows)
    return d[[c for c in cols if c in d.columns]] if cols and not d.empty else d


def operator():
    return st.session_state.get("op_name", "local_admin"), st.session_state.get("op_role", "admin")


def gray_input(label="Face image"):
    import cv2
    import numpy as np
    cam = st.camera_input(label + " (webcam)")
    up = st.file_uploader(label + " (upload)", type=["jpg", "jpeg", "png"], key=label)
    f = cam or up
    if not f:
        return None
    img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not decode image")
    return img


def fingerprint_input(label="Fingerprint"):
    up = st.file_uploader(label + " (PNG/JPG/TIFF)", type=["png", "jpg", "jpeg", "tif", "tiff"], key=label)
    if not up:
        return None
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.error("Could not decode fingerprint image")
    return img


def wave_input(label="Voice"):
    import io
    from src.voice.preprocessing import load_audio
    rec = st.audio_input(label + " (microphone)") if hasattr(st, "audio_input") else None
    up = st.file_uploader(label + " (upload WAV)", type=["wav"], key=label)
    f = rec or up
    return load_audio(io.BytesIO(f.getvalue())) if f else None
