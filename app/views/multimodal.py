import streamlit as st
from app.common import gray_input, fingerprint_input, wave_input


def render(P):
    st.title("Multimodal Authentication")
    st.caption("Face + fingerprint + speaker verification. All three are evaluated independently and then fused at score level.")
    users = [u for u in P.enrollment.list_users() if u["status"] == "active"]
    if not users:
        st.info("Enroll a user first."); return
    user = st.selectbox("Claimed identity", [u["username"] for u in users])
    face = gray_input("Face")
    fp = fingerprint_input("Fingerprint")
    voice = wave_input("Voice")
    if face is not None and fp is not None and voice is not None and st.button("Authenticate with all 3", type="primary"):
        try:
            r = P.verify_multimodal(user, face, fp, voice)
            (st.success if r["accepted"] else st.error)("AUTHENTICATED" if r["accepted"] else "REJECTED")
            if r.get("fused_score") is not None:
                st.metric("Fused confidence", f"{r['fused_score']:.3f}")
            cols = st.columns(3)
            for col, key, label in [(cols[0], "face", "Face"), (cols[1], "fingerprint", "Fingerprint"), (cols[2], "voice", "Voice")]:
                part = r.get(key, {})
                col.metric(label, "PASS" if part.get("accepted") else "FAIL", f"score {part.get('score', '—')}")
            st.json(r)
            P.record_attempt(user, "face+fingerprint+voice", r["accepted"],
                             face_score=r.get("face", {}).get("score"), voice_score=r.get("voice", {}).get("score"),
                             fused_score=r.get("fused_score"), reason=r.get("reason", ""),
                             vresults=[
                                 {"modality": "face", "model": r.get("face", {}).get("backend", "arcface"), "score": r.get("face", {}).get("score"), "threshold": r.get("face", {}).get("threshold"), "accepted": int(r.get("face", {}).get("accepted", False)), "latency_ms": r.get("face", {}).get("ms")},
                                 {"modality": "fingerprint", "model": r.get("fingerprint", {}).get("backend", "afis"), "score": r.get("fingerprint", {}).get("score"), "threshold": r.get("fingerprint", {}).get("threshold"), "accepted": int(r.get("fingerprint", {}).get("accepted", False)), "latency_ms": r.get("fingerprint", {}).get("ms")},
                                 {"modality": "voice", "model": r.get("voice", {}).get("backend", "ecapa-tdnn"), "score": r.get("voice", {}).get("score"), "threshold": r.get("voice", {}).get("threshold"), "accepted": int(r.get("voice", {}).get("accepted", False)), "latency_ms": r.get("voice", {}).get("ms")},
                             ])
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

    st.subheader("Fusion configuration")
    st.write({"weights": P.fusion.weights.tolist(), "decision_threshold": P.fusion.decision_threshold,
              "face_threshold": P.fusion.thresholds[0], "fingerprint_threshold": P.fusion.thresholds[1],
              "voice_threshold": P.fusion.thresholds[2]})
    st.caption("These are engineering defaults until you calibrate them on a leakage-safe validation set. Do not report them as accuracy results.")
