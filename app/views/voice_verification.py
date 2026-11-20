import streamlit as st
from app.common import wave_input, show_safe_error, show_verification_result


def render(P):
    st.title("Voice Verification")
    st.caption("Verify a claimed identity using a voice recording.")
    st.info("For a reliable sample, record in a quiet place, speak naturally, and avoid background music.")
    users = [u for u in P.enrollment.list_users() if u["status"] == "active"]
    if not users:
        st.info("No active profiles are enrolled yet. Start with User Enrollment.")
        return

    user = st.selectbox("Who are you verifying?", [u["username"] for u in users])
    count = len(P._samples(user, "voice"))
    st.caption(f"Enrolled voice samples: {count}")
    if count < 1:
        st.warning("This profile has no voice samples. Add WAV recordings in User Enrollment.")
        return

    st.caption("Speaker model: pretrained SpeechBrain ECAPA-TDNN with cosine similarity.")
    wave = wave_input()
    if wave is not None and st.button("Verify speaker", type="primary", use_container_width=True):
        try:
            r = P.verify_voice(user, wave)
            show_verification_result(r, "Voice", "Speaker similarity", ".3f")
            P.record_attempt(
                user, "voice-only", r["accepted"],
                voice_score=r.get("score"), reason=r.get("reason", ""),
                vresults=[{
                    "modality": "voice", "model": r.get("backend", "ecapa-tdnn"),
                    "score": r.get("score"), "threshold": r.get("threshold"),
                    "accepted": int(r["accepted"]), "latency_ms": r.get("ms"),
                }],
            )
        except Exception as exc:
            show_safe_error(exc, "Voice verification could not be completed. Check the recording and try again.")
