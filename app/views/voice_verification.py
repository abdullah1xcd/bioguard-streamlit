import streamlit as st
from app.common import wave_input


def render(P):
    st.title("Voice / Speaker Verification")
    st.caption("Primary model: pretrained SpeechBrain ECAPA-TDNN speaker embeddings + cosine similarity. No personal voice classifier training is required.")
    users = [u for u in P.enrollment.list_users() if u["status"] == "active"]
    if not users:
        st.info("Enroll a user first."); return
    user = st.selectbox("Claimed identity", [u["username"] for u in users])
    w = wave_input()
    if w is not None and st.button("Verify speaker", type="primary"):
        try:
            r = P.verify_voice(user, w)
            (st.success if r["accepted"] else st.error)("ACCEPTED" if r["accepted"] else "REJECTED")
            if r.get("score") is not None:
                st.metric("Speaker cosine score", f"{r['score']:.3f}")
            st.json(r)
            P.record_attempt(user, "voice-only", r["accepted"], voice_score=r.get("score"), reason=r.get("reason", ""),
                             vresults=[{"modality": "voice", "model": r.get("backend", "ecapa-tdnn"), "score": r.get("score"),
                                        "threshold": r.get("threshold"), "accepted": int(r["accepted"]), "latency_ms": r.get("ms")}])
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
