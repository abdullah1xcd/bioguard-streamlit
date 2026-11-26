import streamlit as st
from app.common import df


def render(P):
    st.title("BioGuard-X")
    st.caption("A unified workspace for face, fingerprint, and speaker verification.")
    s = P.dashboard_stats()

    st.markdown("### System overview")
    metrics = st.columns(4)
    for col, (key, label) in zip(metrics, [
        ("users", "Enrolled users"), ("attempts", "Verification attempts"),
        ("successes", "Accepted"), ("failures", "Not accepted"),
    ]):
        col.metric(label, s[key])

    st.markdown("### Biometric capabilities")
    cards = st.columns(3)
    for col, title, description, detail in [
        (cards[0], "Face", "Face detection and identity matching", "ArcFace embeddings · SCRFD / optional YOLO"),
        (cards[1], "Fingerprint", "Fingerprint minutiae matching", "AFIS / SAFIS backend"),
        (cards[2], "Voice", "Speaker identity matching", "SpeechBrain ECAPA-TDNN"),
    ]:
        with col:
            st.markdown(f"#### {title}")
            st.write(description)
            st.caption(detail)

    st.info("A successful app launch does not by itself prove biometric accuracy. Use separate enrollment, validation, and test data to evaluate FAR, FRR, EER, and latency.")

    st.markdown("### Verification data management")
    with st.expander("Clear verification history", expanded=False):
        st.warning(
            "This permanently deletes verification attempts and their per-modality results. "
            "Enrolled users, biometric samples, and audit logs will be preserved."
        )
        st.caption(
            "This app currently has no authenticated admin roles. The confirmation below "
            "prevents accidental clicks but is not an access-control security boundary."
        )
        confirmation = st.text_input(
            'Type CLEAR VERIFICATION DATA to enable deletion',
            key="clear_verification_data_phrase",
        )
        acknowledged = st.checkbox(
            "I understand that verification history will be permanently deleted.",
            key="clear_verification_data_ack",
        )
        if st.button(
            "Delete verification history",
            type="primary",
            disabled=(confirmation != "CLEAR VERIFICATION DATA" or not acknowledged),
            key="clear_verification_data_button",
        ):
            deleted = P.clear_verification_data(actor=st.session_state.get("op_name", "local_operator"))
            st.session_state["clear_verification_data_phrase"] = ""
            st.session_state["clear_verification_data_ack"] = False
            st.success(
                f"Cleared {deleted['attempts']} verification attempts and "
                f"{deleted['results']} modality result records. Enrolled users and biometric samples were kept."
            )
            st.rerun()

    st.markdown("### Recent verification activity")
    rows = P.db.fetch(
        "SELECT id,claimed,mode,accepted,face_score,voice_score,fused_score,reason,created_at "
        "FROM auth_attempts ORDER BY id DESC LIMIT 20"
    )
    if rows:
        display = df(rows, ["id", "claimed", "mode", "accepted", "face_score", "voice_score", "fused_score", "created_at"])
        display = display.rename(columns={
            "id": "Attempt", "claimed": "Claimed user", "mode": "Method",
            "accepted": "Accepted (1/0)", "face_score": "Face score",
            "voice_score": "Voice score", "fused_score": "Combined score",
            "created_at": "Time",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.caption("No verification attempts have been recorded yet.")
