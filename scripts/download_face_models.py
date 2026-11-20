"""One-time model download (optional - the app also does this automatically on first use).
    python -m scripts.download_face_models
Fetches: InsightFace buffalo_l pack (SCRFD detector + ArcFace recogniser) and the YOLOv8-face weights (if `ultralytics` is installed).
Check the model licences / training-data terms before any use beyond coursework (InsightFace pretrained models: non-commercial research)."""
import urllib.request
from src.config import FACE_CFG, YOLO_FACE_PATH, YOLO_FACE_URL


def main():
    from src.face.engine import FaceEngine
    eng = FaceEngine()
    if eng.has_insightface():
        print("loading InsightFace pack (downloads ~280 MB on first run) ...")
        eng._load_insight(); print(f"[ok] {FACE_CFG['model_pack']} ready (SCRFD + ArcFace)")
    else:
        print("[!] insightface/onnxruntime missing -> pip install -r requirements.txt")
    if eng.has_yolo():
        if not YOLO_FACE_PATH.is_file():
            YOLO_FACE_PATH.parent.mkdir(parents=True, exist_ok=True)
            print("downloading YOLOv8n-face ..."); urllib.request.urlretrieve(YOLO_FACE_URL, YOLO_FACE_PATH)
        print(f"[ok] YOLO face weights: {YOLO_FACE_PATH}")
    else:
        print("[i] ultralytics not installed -> SCRFD is used for detection (pip install ultralytics to enable YOLO)")


if __name__ == "__main__":
    main()
