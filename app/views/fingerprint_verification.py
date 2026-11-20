import streamlit as st
from app.common import fingerprint_input, show_safe_error, show_verification_result


def render(P):
    st.title("Fingerprint Verification")
    st.caption("Compare a submitted fingerprint with the selected enrolled profile.")
    users = [u for u in P.enrollment.list_users() if u["status"] == "active"]
    if not users:
        st.info("No active profiles are enrolled yet. Start with User Enrollment.")
        return

    user = st.selectbox("Who are you verifying?", [u["username"] for u in users])
    count = len(P._samples(user, "fingerprint"))
    st.caption(f"Enrolled fingerprint samples: {count}")
    if count < 1:
        st.warning("This profile has no fingerprint samples. Add samples in User Enrollment.")
        return

    st.caption("Matching backend: AFIS / minutiae matching. The legacy HOG/LBP classifier is not used.")
    fp = fingerprint_input()
    if fp is not None and st.button("Verify fingerprint", type="primary", use_container_width=True):
        try:
            r = P.verify_fingerprint(user, fp)
            show_verification_result(r, "Fingerprint", "Match score", ".2f")
            P.record_attempt(
                user, "fingerprint-only", r["accepted"],
                fingerprint_score=r.get("score"), reason=r.get("reason", ""),
                vresults=[{
                    "modality": "fingerprint", "model": r.get("backend", "afis"),
                    "score": r.get("score"), "threshold": r.get("threshold"),
                    "accepted": int(r["accepted"]), "latency_ms": r.get("ms"),
                }],
            )
        except Exception as exc:
            show_safe_error(exc, "Fingerprint verification could not be completed. Check the uploaded image and try again.")
