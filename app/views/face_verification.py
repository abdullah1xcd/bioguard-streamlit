import cv2
import numpy as np
import streamlit as st
from app.common import gray_input


def engine_banner(P):
    from src.config import SFACE_PATH, YUNET_PATH
    e = P.face.engine.info()
    if not e["insightface"] and not (SFACE_PATH.is_file() and YUNET_PATH.is_file()):
        st.error("No face model installed. Run `pip install -r requirements.txt` (InsightFace + ONNX Runtime), then restart the app.")
        return False
    det = "YOLOv8-face" if e["yolo"] and e["detector_setting"] != "scrfd" else "SCRFD"
    st.caption(f"Detector: {det} (falls back to SCRFD) | recogniser: ArcFace | no training needed" + ("" if e["yolo"] else " | YOLO not installed: `pip install ultralytics`"))
    return True


def render(P):
    st.title("Face Verification (YOLO / SCRFD detection + ArcFace embeddings)")
    if not engine_banner(P):
        return
    users = [u for u in P.enrollment.list_users() if u["status"] == "active"]
    if not users:
        st.info("No users yet. Open **User Enrollment**, enrol someone with 3+ clear face photos, then come back - no training step is required.")
        return
    status = P.sync_faces()
    names = [u["username"] for u in users]
    user = st.selectbox("Claimed identity", names, format_func=lambda n: f"{n}  ({status[n]['usable']} usable photos)" + ("" if status[n]["ready"] else "  - needs more photos"))
    if not status[user]["ready"]:
        st.warning(f"'{user}' has {status[user]['usable']} usable face photos (need {P.face.cfg['min_enroll_images']}+). "
                   f"Rejected photos: {status[user]['rejected_reasons'] or 'none'}. Add photos on **User Enrollment**.")
        return
    img = gray_input()
    if img is not None and st.button("Verify face", type="primary"):
        try:
            r = P.verify_face(user, img)
            if r.get("box"):
                vis = img.copy(); x1, y1, x2, y2 = [int(v) for v in r["box"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0) if r["accepted"] else (0, 0, 220), 3)
                st.image(vis, channels="BGR", width=360)
            (st.success if r["accepted"] else st.error)("ACCEPTED" if r["accepted"] else "REJECTED")
            if "score" in r:
                c = st.columns(3); c[0].metric("Similarity", f"{r['score']:.3f}"); c[1].metric("Threshold", f"{r['threshold']:.3f}"); c[2].metric("Time", f"{r['ms']:.0f} ms")
            if r.get("reason"):
                st.write(r["reason"])
            with st.expander("Details"):
                st.json({k: v for k, v in r.items() if k != "box"})
            P.record_attempt(user, "face-only", r["accepted"], 1, face_score=r.get("score"), reason=r.get("reason") or "face-only",
                             face_detected=r.get("face_detected"), face_quality_ok=r["quality"].get("ok", True))
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
    with st.expander("Threshold"):
        st.write(f"Current threshold: **{P.face.threshold:.3f}** (cosine similarity; higher = stricter).")
        c1, c2 = st.columns(2)
        if c1.button("Auto-calibrate from enrolled photos"):
            st.json(P.face.calibrate())
        if c2.button("Reset to default"):
            P.face.set_threshold(None); st.rerun()
        st.caption("Calibration needs 2+ enrolled users. It uses the enrolment photos themselves, so treat it as a sanity check, not as a test result.")
