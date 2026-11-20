import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from app.common import get_platform, show_safe_error

st.set_page_config(
    page_title="BioGuard-X | Biometric Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_CSS = """
<style>
:root { --bg: #071426; --panel: #10233d; --line: #203b5d; --blue: #4f8cff; }
.stApp { background: radial-gradient(ellipse at 12% 0%, #122d50 0%, #071426 48%); }
[data-testid="stHeader"] { background: rgba(7,20,38,.82); }
[data-testid="stSidebar"] { background: #091a2e; border-right: 1px solid #203b5d; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px; }
h1, h2, h3 { letter-spacing: -.025em; color: #edf4ff; }
p, label, [data-testid="stCaptionContainer"] { color: #b8c9df; }
[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(20,48,78,.95), rgba(13,32,54,.95));
  border: 1px solid #274666; border-radius: 14px; padding: 16px 18px;
}
[data-testid="stMetricLabel"] { color: #a9bfd9; }
[data-testid="stMetricValue"] { color: #f2f7ff; }
div.stButton > button, div.stFormSubmitButton > button {
  border-radius: 10px; border: 1px solid #477fc7; font-weight: 650;
  min-height: 2.65rem; transition: all .15s ease;
}
div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
  background: linear-gradient(100deg, #2464c5, #397fe8); color: white;
}
div.stButton > button:hover { border-color: #83b3ff; transform: translateY(-1px); }
[data-testid="stExpander"] { border: 1px solid #203b5d; border-radius: 12px; background: rgba(12,31,53,.55); }
[data-testid="stFileUploader"], [data-testid="stCameraInput"], [data-testid="stAudioInput"] {
  border: 1px dashed #355779; border-radius: 12px; padding: 8px;
}
[data-testid="stDataFrame"] { border: 1px solid #203b5d; border-radius: 12px; overflow: hidden; }
hr { border-color: #203b5d; }
.small-muted { color: #9fb4ce; font-size: .9rem; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

PAGES = [
    ("Overview", "home"),
    ("User Enrollment", "enrollment"),
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
        except Exception as exc:
            show_safe_error(exc, "This page could not be loaded. Please try again.")
    return page


P = get_platform()
with st.sidebar:
    st.markdown("## 🛡️ BioGuard-X")
    st.caption("MULTIMODAL BIOMETRIC SECURITY")
    st.divider()
    st.session_state["op_name"] = st.text_input(
        "Operator name · audit identity",
        st.session_state.get("op_name", "local_admin"),
        help="This name is recorded in audit logs. It is not a login or proof of identity.",
    )
    st.caption("Operator name is an audit label only; authentication and role enforcement are not provided by this field.")
    st.divider()
    st.caption("FACE  ·  FINGERPRINT  ·  VOICE")
    st.caption("Use the Evaluation page and a separate test set to measure biometric performance.")

st.navigation([st.Page(make(m), title=t, url_path=m) for t, m in PAGES]).run()
