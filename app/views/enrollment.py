import streamlit as st
from app.common import df, operator
from src.security.permissions import ROLES


def show_status(P, name):
    fs = P.sync_faces().get(name)
    if fs:
        (st.success if fs["ready"] else st.warning)(f"Face: {fs['usable']} usable / {fs['photos']} uploaded")
        if fs["rejected_reasons"]:
            st.caption("Face rejects: " + ", ".join(f"{k} x{v}" for k, v in fs["rejected_reasons"].items()))
    st.write({"Fingerprint samples": len(P._samples(name, "fingerprint")),
              "Voice samples": len(P._samples(name, "voice"))})


def render(P):
    st.title("Biometric Enrollment")
    st.caption("Enroll all three modalities. Use multiple sessions and clean samples; the system does not train a classifier on your personal data.")
    op, _ = operator()
    with st.form("enroll"):
        name = st.text_input("Username")
        role = st.selectbox("Role", ROLES, index=min(2, len(ROLES)-1))
        consent = st.checkbox("I confirm informed consent was obtained and the biometric data will be stored locally.")
        faces = st.file_uploader("Face photos (5+ recommended)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        fingerprints = st.file_uploader("Fingerprint images (2+ recommended)", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)
        voices = st.file_uploader("Voice recordings (2+ WAV, 2–8 seconds each)", type=["wav"], accept_multiple_files=True)
        ok = st.form_submit_button("Enroll user", type="primary")
    if ok:
        try:
            u = P.enrollment.enroll(name, role, consent, actor=op)
            if faces: P.enrollment.add_samples(u["id"], "face", [(f.name, f.getvalue()) for f in faces], actor=op, session=1)
            if fingerprints: P.enrollment.add_samples(u["id"], "fingerprint", [(f.name, f.getvalue()) for f in fingerprints], actor=op, session=1)
            if voices: P.enrollment.add_samples(u["id"], "voice", [(f.name, f.getvalue()) for f in voices], actor=op, session=1)
            P.sync_faces()
            st.success(f"Enrolled {name} as {u['participant_id']}")
            show_status(P, name)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

    users = P.enrollment.list_users()
    st.subheader("Users")
    st.dataframe(df(users, ["id", "username", "participant_id", "role", "status", "created_at"]), use_container_width=True)
    if users:
        sel = st.selectbox("Manage user", [u["username"] for u in users])
        u = P.enrollment.get_user(sel)
        show_status(P, sel)
        c1, c2, c3 = st.columns(3)
        more_f = c1.file_uploader("Add face photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="add_face")
        more_p = c2.file_uploader("Add fingerprints", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True, key="add_fp")
        more_v = c3.file_uploader("Add voice WAVs", type=["wav"], accept_multiple_files=True, key="add_voice")
        if st.button("Add samples"):
            if more_f: P.enrollment.add_samples(u["id"], "face", [(f.name, f.getvalue()) for f in more_f], actor=op, session=2)
            if more_p: P.enrollment.add_samples(u["id"], "fingerprint", [(f.name, f.getvalue()) for f in more_p], actor=op, session=2)
            if more_v: P.enrollment.add_samples(u["id"], "voice", [(f.name, f.getvalue()) for f in more_v], actor=op, session=2)
            st.rerun()
        st.warning("Deleting a user permanently removes the stored biometric files and enrollment records.")
        if st.checkbox(f"I confirm deleting {sel}") and st.button("Delete user permanently"):
            st.json(P.enrollment.delete_user(u["id"], actor=op)); st.rerun()
