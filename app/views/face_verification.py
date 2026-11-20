import cv2
import streamlit as st
from app.common import gray_input, show_safe_error, show_verification_result


def engine_banner(P):
    from src.config import SFACE_PATH, YUNET_PATH
    e = P.face.engine.info()
    if not e["insightface"] and not (SFACE_PATH.is_file() and YUNET_PATH.is_file()):
        st.error("The face recognition model is not available in this deployment.")
        st.caption("Check the model installation and deployment logs, then restart the app.")
        return False
    det = "YOLOv8-face" if e["yolo"] and e["detector_setting"] != "scrfd" else "SCRFD"
    st.caption(f"Detection: {det} · Recognition: ArcFace")
    return True


def render(P):
    st.title("Face Verification")
    st.caption("Confirm a claimed identity using a live camera capture or an uploaded face image.")
    if not engine_banner(P):
        return

    users = [u for u in P.enrollment.list_users() if u["status"] == "active"]
    if not users:
        st.info("No active profiles are enrolled yet. Start with User Enrollment.")
        return

    status = P.sync_faces()
    names = [u["username"] for u in users]
    user = st.selectbox(
        "Who are you verifying?",
        names,
        format_func=lambda n: f"{n} · {status[n]['usable']} usable face photos",
    )
    if not status[user]["ready"]:
        st.warning(
            f"This profile has {status[user]['usable']} usable face photos; "
            f"{P.face.cfg['min_enroll_images']} or more are required."
        )
        st.caption("Add more clear face photos in User Enrollment, then return here.")
        return

    img = gray_input("Face")
    if img is not None and st.button("Verify identity", type="primary", use_container_width=True):
        try:
            r = P.verify_face(user, img)
            if r.get("box"):
                vis = img.copy()
                x1, y1, x2, y2 = [int(v) for v in r["box"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0) if r["accepted"] else (0, 0, 220), 3)
                st.image(vis, channels="BGR", caption="Face detection result", width=420)
            show_verification_result(r, "Face", "Similarity", ".3f")
            P.record_attempt(
                user, "face-only", r["accepted"], 1,
                face_score=r.get("score"), reason=r.get("reason") or "face-only",
                face_detected=r.get("face_detected"),
            )
        except Exception as exc:
            show_safe_error(exc, "Face verification could not be completed. Check the image and try again.")

    with st.expander("Advanced threshold controls"):
        st.write(f"Current cosine-similarity threshold: **{P.face.threshold:.3f}**. Higher values are stricter.")
        c1, c2 = st.columns(2)
        if c1.button("Calibrate from enrolled photos"):
            try:
                st.json(P.face.calibrate())
            except Exception as exc:
                show_safe_error(exc, "Threshold calibration could not be completed.")
        if c2.button("Reset threshold"):
            P.face.set_threshold(None)
            st.rerun()
        st.caption("Calibration on enrollment photos is only a sanity check, not an independent accuracy evaluation.")
