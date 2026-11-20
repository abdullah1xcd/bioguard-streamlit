import streamlit as st
from app.common import df


def render(P):
    st.title("BioGuard-X")
    st.caption("Multimodal biometric authentication: Face + Fingerprint + Speaker")
    s = P.dashboard_stats()
    c = st.columns(4)
    for col, (k, lab) in zip(c, [("users", "Enrolled users"), ("attempts", "Attempts"), ("successes", "Accepted"), ("failures", "Rejected")]):
        col.metric(lab, s[k])
    st.subheader("Biometric stack")
    st.write({"Face": "SCRFD / optional YOLO face detection + ArcFace", "Fingerprint": "AFIS (`afis`)", "Voice": "SpeechBrain ECAPA-TDNN", "Fusion": "calibrated score-level fusion"})
    st.subheader("Recent authentication")
    st.dataframe(df(P.db.fetch("SELECT id,claimed,mode,accepted,face_score,voice_score,fused_score,reason,created_at FROM auth_attempts ORDER BY id DESC LIMIT 20")), use_container_width=True)
