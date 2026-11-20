import importlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from app.common import get_platform

st.set_page_config(page_title="BioGuard-X - Multimodal Biometrics", page_icon="🛡️", layout="wide")
PAGES = [
    ("Home / Overview", "home"),
    ("Biometric Enrollment", "enrollment"),
    ("Face Verification", "face_verification"),
    ("Fingerprint Verification", "fingerprint_verification"),
    ("Voice Verification", "voice_verification"),
    ("Multimodal Authentication", "multimodal"),
    ("Evaluation", "evaluation"),
    ("Audit Logs", "audit_logs"),
    ("Settings", "settings"),
]


def make(mod):
    def page():
        try:
            importlib.import_module(f"app.views.{mod}").render(get_platform())
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
    return page

P = get_platform()
st.sidebar.title("🛡️ BioGuard-X")
st.sidebar.caption("Face + Fingerprint + Speaker authentication")
st.session_state["op_name"] = st.sidebar.text_input("Operator name", st.session_state.get("op_name", "local_admin"))
st.sidebar.info("Accuracy is determined by validation/test data, not by the UI threshold alone.")
st.navigation([st.Page(make(m), title=t, url_path=m) for t, m in PAGES]).run()
