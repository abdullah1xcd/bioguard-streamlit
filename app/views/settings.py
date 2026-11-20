import streamlit as st


def render(P):
    st.title("Biometric Settings")
    st.subheader("Thresholds")
    face = st.number_input("Face cosine threshold", 0.30, 0.80, float(P.face.threshold), 0.01)
    fp = st.number_input("Fingerprint threshold", 1.0, 200.0, float(P.fingerprint.threshold), 1.0)
    voice = st.number_input("Voice cosine threshold", 0.30, 0.95, float(P.db.get_setting("voice_threshold", 0.60)), 0.01)
    fusion = st.number_input("Fusion threshold", 0.30, 1.0, float(P.fusion.decision_threshold), 0.01)
    if st.button("Save thresholds"):
        P.face.set_threshold(face); P.db.set_setting("fingerprint_threshold", fp); P.db.set_setting("voice_threshold", voice); P.db.set_setting("fusion_threshold", fusion)
        st.success("Saved. Recalibrate on validation data before treating these as final production thresholds.")
    st.subheader("Installed backends")
    st.json(P.authentication_capabilities())
