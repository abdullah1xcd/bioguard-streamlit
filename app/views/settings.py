import streamlit as st


BIOMETRIC_MODALITIES = {
    "Face": "face",
    "Voice": "voice",
    "Fingerprint": "fingerprint",
}


def _delete_password_is_valid():
    configured = st.secrets.get("BIOMETRIC_DELETE_PASSWORD", "")
    if not configured:
        st.warning(
            "Biometric deletion is locked. Configure BIOMETRIC_DELETE_PASSWORD "
            "in the Streamlit app Secrets before using this feature."
        )
        return False
    entered = st.text_input(
        "Administrator password",
        type="password",
        key="biometric_delete_admin_password",
    )
    return bool(entered) and entered == configured


def _refresh_after_face_delete(P):
    # Rebuild the face gallery so deleted embeddings disappear from memory and disk cache.
    P.sync_faces()


def render(P):
    st.title("Biometric Settings")
    st.subheader("Thresholds")
    face = st.number_input("Face cosine threshold", 0.30, 0.80, float(P.face.threshold), 0.01)
    fp = st.number_input("Fingerprint threshold", 1.0, 200.0, float(P.fingerprint.threshold), 1.0)
    voice = st.number_input("Voice cosine threshold", 0.30, 0.95, float(P.db.get_setting("voice_threshold", 0.60)), 0.01)
    fusion = st.number_input("Fusion threshold", 0.30, 1.0, float(P.fusion.decision_threshold), 0.01)
    if st.button("Save thresholds"):
        P.face.set_threshold(face)
        P.db.set_setting("fingerprint_threshold", fp)
        P.db.set_setting("voice_threshold", voice)
        P.db.set_setting("fusion_threshold", fusion)
        st.success("Saved. Recalibrate on validation data before treating these as final production thresholds.")

    st.divider()
    st.header("Biometric Data Management")
    st.caption(
        "Delete enrolled face photos, voice recordings, fingerprint images, and their "
        "enrollment records. User profiles and the application remain in place."
    )
    delete_password_ok = _delete_password_is_valid()
    users = P.enrollment.list_users()
    if not users:
        st.info("No enrolled users are available.")
    else:
        st.subheader("Delete data for one participant")
        selected = st.selectbox(
            "Select participant",
            users,
            format_func=lambda user: f'{user["username"]} · {user["participant_id"]}',
            key="biometric_delete_participant",
        )
        sample_rows = P.db.fetch(
            "SELECT modality, COUNT(*) AS total FROM enrollment_samples "
            "WHERE user_id=? GROUP BY modality",
            (selected["id"],),
        )
        sample_counts = {row["modality"]: row["total"] for row in sample_rows}
        cols = st.columns(3)
        for col, (label, modality) in zip(cols, BIOMETRIC_MODALITIES.items()):
            with col:
                st.metric(f"{label} samples", sample_counts.get(modality, 0))
                confirmed = st.checkbox(
                    f"Confirm {label} deletion",
                    key=f"confirm_delete_{modality}_{selected['id']}",
                )
                if st.button(
                    f"Delete {label} data",
                    key=f"delete_{modality}_{selected['id']}",
                    type="secondary",
                    disabled=not confirmed,
                    use_container_width=True,
                ):
                    if not delete_password_ok:
                        st.error("Administrator password is required.")
                    else:
                        try:
                            result = P.enrollment.delete_biometric_data(
                                selected["id"], modality=modality, actor="admin"
                            )
                            if modality == "face":
                                _refresh_after_face_delete(P)
                            st.success(
                                f'Deleted {result["files_deleted"]} {label.lower()} files '
                                f'and {result["sample_records_deleted"]} enrollment records '
                                f'for {result["username"]}.'
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not delete {label.lower()} data: {exc}")

        st.subheader("Delete all three biometric modalities for one participant")
        confirm_one = st.checkbox(
            f'I understand this permanently removes face, voice, and fingerprint data for {selected["username"]}.',
            key=f'confirm_delete_all_one_{selected["id"]}',
        )
        if st.button(
            "Delete ALL biometric data for selected participant",
            type="primary",
            disabled=not confirm_one,
            key=f'delete_all_one_{selected["id"]}',
        ):
            if not delete_password_ok:
                st.error("Administrator password is required.")
            else:
                try:
                    result = P.enrollment.delete_biometric_data(
                        selected["id"], actor="admin"
                    )
                    _refresh_after_face_delete(P)
                    st.success(
                        f'Deleted {result["files_deleted"]} biometric files and '
                        f'{result["sample_records_deleted"]} enrollment records for '
                        f'{result["username"]}. The participant profile was kept.'
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not delete biometric data: {exc}")

        st.divider()
        st.subheader("Delete biometric data for EVERY participant")
        st.warning(
            "This permanently deletes all enrolled face, voice, and fingerprint samples "
            "for every participant. Participant profiles, settings, audit history, "
            "application code, and pretrained AI model weights are kept."
        )
        confirm_everyone = st.checkbox(
            "I understand this deletes biometric samples for ALL participants.",
            key="confirm_delete_all_participants",
        )
        if st.button(
            "Delete ALL participants' biometric data",
            type="primary",
            disabled=not confirm_everyone,
            key="delete_all_participants_biometrics",
        ):
            if not delete_password_ok:
                st.error("Administrator password is required.")
            else:
                try:
                    totals = {"files": 0, "records": 0}
                    for user in P.enrollment.list_users():
                        result = P.enrollment.delete_biometric_data(
                            user["id"], actor="admin"
                        )
                        totals["files"] += result["files_deleted"]
                        totals["records"] += result["sample_records_deleted"]
                    _refresh_after_face_delete(P)
                    st.success(
                        f'Deleted {totals["files"]} biometric files and '
                        f'{totals["records"]} enrollment records across all participants. '
                        "Participant profiles were kept."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Biometric cleanup was incomplete: {exc}")

    st.divider()
    st.subheader("Installed backends")
    st.json(P.authentication_capabilities())
