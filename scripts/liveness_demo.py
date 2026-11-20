"""Webcam liveness demo (needs the YuNet/SFace ONNX files and a camera; runs on YOUR machine).
    python -m scripts.download_face_models && python -m scripts.liveness_demo
NOT validated against real attacks - see src/challenges/liveness.py for limits."""
import time
import cv2
from src.challenges.liveness import evaluate_challenge, new_challenge, observations_from_frames
from src.face.embedding import FaceEmbedder


def main(seconds=4.0, fps=8):
    if not FaceEmbedder.available():
        raise SystemExit("YuNet/SFace models missing: python -m scripts.download_face_models")
    emb, cap = FaceEmbedder(), cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("camera unavailable")
    try:
        for _ in range(5): cap.read()
        ch = new_challenge(); print(">>>", ch.instruction, f"(you have {seconds:.0f} s)")
        frames, t0 = [], time.time()
        while time.time() - t0 < seconds:
            ok, fr = cap.read()
            if ok: frames.append(fr)
            time.sleep(1 / fps)
    finally:
        cap.release()
    print(evaluate_challenge(ch, observations_from_frames(frames, emb)))


if __name__ == "__main__":
    main()
