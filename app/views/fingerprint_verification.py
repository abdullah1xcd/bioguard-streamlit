import streamlit as st
from app.common import fingerprint_input


def render(P):
    st.title("Fingerprint Verification")
    users = [u for u in P.enrollment.list_users() if u["status"] == "active"]
    if not users:
        st.info("Enroll a user first."); return
    user = st.selectbox("Claimed identity", [u["username"] for u in users])
    st.caption("Primary matcher: AFIS/SourceAFIS-style minutiae matching via the `afis` package. The older HOG/LBP classifier is no longer used.")
    fp = fingerprint_input()
    if fp is not None and st.button("Verify fingerprint", type="primary"):
        try:
            r = P.verify_fingerprint(user, fp)
            (st.success if r["accepted"] else st.error)("ACCEPTED" if r["accepted"] else "REJECTED")
            if r.get("score") is not None:
                st.metric("Match score", f"{r['score']:.2f}")
            st.json(r)
            P.record_attempt(user, "fingerprint-only", r["accepted"], fingerprint_score=r.get("score"), reason=r.get("reason", ""),
                             vresults=[{"modality": "fingerprint", "model": r.get("backend", "afis"), "score": r.get("score"),
                                        "threshold": r.get("threshold"), "accepted": int(r["accepted"]), "latency_ms": r.get("ms")}])
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
