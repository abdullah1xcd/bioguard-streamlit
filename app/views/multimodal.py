import streamlit as st
from app.common import gray_input, fingerprint_input, wave_input, show_safe_error


def render(P):
    st.title("Multimodal Authentication")
    st.caption("Run face, fingerprint, and voice checks together, then apply the existing score-fusion decision.")
    users = [u for u in P.enrollment.list_users() if u["status"] == "active"]
    if not users:
        st.info("No active profiles are enrolled yet. Start with User Enrollment.")
        return

    user = st.selectbox("Who are you verifying?", [u["username"] for u in users])
    st.markdown("### 1 · Capture all three samples")
    c1, c2, c3 = st.columns(3)
    with c1:
        face = gray_input("Face")
    with c2:
        fp = fingerprint_input("Fingerprint")
    with c3:
        voice = wave_input("Voice")

    ready = face is not None and fp is not None and voice is not None
    if not ready:
        st.info("Provide all three samples to enable multimodal authentication.")
    if ready and st.button("Authenticate with all 3", type="primary", use_container_width=True):
        try:
            r = P.verify_multimodal(user, face, fp, voice)
            st.markdown("### 2 · Authentication decision")
            if r.get("accepted"):
                st.success("Authentication successful · all required checks passed.")
            else:
                st.error("Authentication not successful.")
                if r.get("reason"):
                    st.info(f"Reason: {str(r['reason']).replace('_', ' ').capitalize()}")

            parts = [
                ("Face", r.get("face")),
                ("Fingerprint", r.get("fingerprint")),
                ("Voice", r.get("voice")),
            ]
            cols = st.columns(3)
            for col, (label, part) in zip(cols, parts):
                with col:
                    if part is None:
                        st.metric(label, "Not run")
                    else:
                        st.metric(label, "Passed" if part.get("accepted") else "Not passed",
                                  f"Score: {part.get('score', '—')}")
                        if part.get("reason"):
                            st.caption(str(part["reason"]).replace("_", " "))

            fused = r.get("fused_score")
            threshold = r.get("threshold", P.fusion.decision_threshold)
            c1, c2 = st.columns(2)
            c1.metric("Combined score", f"{fused:.3f}" if isinstance(fused, (int, float)) else "—")
            c2.metric("Fusion threshold", f"{threshold:.3f}" if isinstance(threshold, (int, float)) else "—")

            P.record_attempt(
                user, "face+fingerprint+voice", r["accepted"],
                face_score=r.get("face", {}).get("score"),
                fingerprint_score=r.get("fingerprint", {}).get("score"),
                voice_score=r.get("voice", {}).get("score"),
                fused_score=r.get("fused_score"), reason=r.get("reason", ""),
                vresults=[
                    {"modality": modality, "model": part.get("backend", default),
                     "score": part.get("score"), "threshold": part.get("threshold"),
                     "accepted": int(part.get("accepted", False)), "latency_ms": part.get("ms")}
                    for modality, part, default in [
                        ("face", r.get("face", {}), "arcface"),
                        ("fingerprint", r.get("fingerprint", {}), "afis"),
                        ("voice", r.get("voice", {}), "ecapa-tdnn"),
                    ] if part
                ],
            )
            with st.expander("Technical verification details"):
                st.json(r)
        except Exception as exc:
            show_safe_error(exc, "Multimodal authentication could not be completed. Check all three samples and try again.")

    with st.expander("Fusion configuration"):
        c1, c2 = st.columns(2)
        c1.metric("Decision threshold", f"{P.fusion.decision_threshold:.3f}")
        c2.write("Weights and modality thresholds are advanced configuration values.")
        st.json({
            "weights": P.fusion.weights.tolist(),
            "face_threshold": P.fusion.thresholds[0],
            "fingerprint_threshold": P.fusion.thresholds[1],
            "voice_threshold": P.fusion.thresholds[2],
        })
        st.caption("These are engineering defaults until calibrated on a leakage-safe validation set. They are not accuracy results.")
