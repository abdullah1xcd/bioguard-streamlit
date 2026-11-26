import streamlit as st
from app.common import FINGERPRINT_UPLOAD_TYPES, df, operator, show_safe_error
from src.security.permissions import ROLES


ROLE_GUIDANCE = {
    "admin": "System administration and full management permissions (when permission enforcement is connected).",
    "manager": "Operational oversight, reports, and approvals.",
    "standard_user": "Normal application usage and personal verification.",
    "guest": "Limited, read-oriented access.",
}


def show_status(P, name):
    fs = P.sync_faces().get(name)
    if fs:
        c1, c2 = st.columns(2)
        c1.metric("Usable face photos", fs["usable"])
        c2.metric("Uploaded face photos", fs["photos"])
        if fs["ready"]:
            st.success("Face enrollment is ready for verification.")
        else:
            st.warning("More usable face photos are required before face verification.")
        if fs["rejected_reasons"]:
            with st.expander("Why some face photos were rejected"):
                st.write(", ".join(f"{k.replace('_', ' ')} × {v}" for k, v in fs["rejected_reasons"].items()))
    c1, c2 = st.columns(2)
    c1.metric("Fingerprint samples", len(P._samples(name, "fingerprint")))
    c2.metric("Voice samples", len(P._samples(name, "voice")))


def render(P):
    st.title("User Enrollment")
    st.caption("Create a profile, record informed consent, then add biometric samples.")
    op, _ = operator()

    st.markdown("### 1 · Create user profile")
    with st.form("enroll"):
        name = st.text_input("Username", help="Use 2–32 letters, numbers, underscores, or hyphens.")
        role = st.selectbox(
            "Assigned role",
            ROLES,
            index=min(2, len(ROLES) - 1),
            format_func=lambda value: value.replace("_", " ").title(),
            help="This role is stored with the profile. The current Streamlit app does not yet provide login-based role enforcement.",
        )
        st.caption(ROLE_GUIDANCE.get(role, ""))
        consent = st.checkbox("I confirm informed consent was obtained and the user agrees to local biometric data storage.")
        st.markdown("### 2 · Add biometric samples")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("FACE")
            faces = st.file_uploader("Face photos · 5+ recommended", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        with c2:
            st.caption("FINGERPRINT")
            fingerprints = st.file_uploader("Fingerprint images · 2+ recommended", type=list(FINGERPRINT_UPLOAD_TYPES), accept_multiple_files=True)
        with c3:
            st.caption("VOICE")
            voices = st.file_uploader("Voice WAVs · 2+ recommended", type=["wav"], accept_multiple_files=True)
        st.caption("For better evaluation, collect samples in more than one session. The system does not train a personal classifier from these samples.")
        ok = st.form_submit_button("Create profile and save samples", type="primary", use_container_width=True)

    if ok:
        try:
            u = P.enrollment.enroll(name, role, consent, actor=op)
            if faces:
                P.enrollment.add_samples(u["id"], "face", [(f.name, f.getvalue()) for f in faces], actor=op, session=1)
            if fingerprints:
                P.enrollment.add_samples(u["id"], "fingerprint", [(f.name, f.getvalue()) for f in fingerprints], actor=op, session=1)
            if voices:
                P.enrollment.add_samples(u["id"], "voice", [(f.name, f.getvalue()) for f in voices], actor=op, session=1)
            P.sync_faces()
            st.success(f"Profile created successfully for {name}.")
            if not any([faces, fingerprints, voices]):
                st.info("The profile is ready. Add biometric samples from the management section when available.")
            show_status(P, name)
        except Exception as exc:
            show_safe_error(exc, "We couldn't create this profile. Check the username, consent, and uploaded files.")

    st.divider()
    st.markdown("### 3 · Manage enrolled users")
    users = P.enrollment.list_users()
    if not users:
        st.info("No enrolled users yet.")
        return
    st.dataframe(
        df(users, ["id", "username", "participant_id", "role", "status", "created_at"]).rename(columns={
            "id": "ID", "username": "Username", "participant_id": "Participant ID",
            "role": "Role", "status": "Status", "created_at": "Created",
        }),
        use_container_width=True,
        hide_index=True,
    )
    sel = st.selectbox("Select a user", [u["username"] for u in users])
    u = P.enrollment.get_user(sel)
    show_status(P, sel)

    st.markdown("#### Add more samples")
    c1, c2, c3 = st.columns(3)
    more_f = c1.file_uploader("Additional face photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="add_face")
    more_p = c2.file_uploader("Additional fingerprints", type=list(FINGERPRINT_UPLOAD_TYPES), accept_multiple_files=True, key="add_fp")
    more_v = c3.file_uploader("Additional voice WAVs", type=["wav"], accept_multiple_files=True, key="add_voice")
    if st.button("Save additional samples", type="primary"):
        try:
            added = 0
            if more_f:
                added += P.enrollment.add_samples(u["id"], "face", [(f.name, f.getvalue()) for f in more_f], actor=op, session=2)
            if more_p:
                added += P.enrollment.add_samples(u["id"], "fingerprint", [(f.name, f.getvalue()) for f in more_p], actor=op, session=2)
            if more_v:
                added += P.enrollment.add_samples(u["id"], "voice", [(f.name, f.getvalue()) for f in more_v], actor=op, session=2)
            if added:
                st.success(f"Saved {added} additional sample(s).")
                st.rerun()
            else:
                st.info("Choose at least one file first.")
        except Exception as exc:
            show_safe_error(exc, "We couldn't save the additional samples.")

    with st.expander("Role reference"):
        for role_name in ROLES:
            st.markdown(f"**{role_name.replace('_', ' ').title()}** — {ROLE_GUIDANCE[role_name]}")
        st.warning("Roles are profile metadata at this stage. Do not use them as a security boundary until login and server-side permission checks are implemented.")

    st.warning("Deleting a user permanently removes their stored biometric files and enrollment records.")
    confirm = st.checkbox(f"I understand and confirm deleting {sel}.")
    if st.button("Delete user permanently", disabled=not confirm):
        try:
            result = P.enrollment.delete_user(u["id"], actor=op)
            st.success(f"User deleted. {result['files_deleted']} file(s) removed.")
            st.rerun()
        except Exception as exc:
            show_safe_error(exc, "We couldn't delete this user.")
