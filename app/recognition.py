"""
Détection et reconnaissance faciale.

Priorité :
  1. Modèle YOLO entraîné (models/face_yolo.pt) — détection + identification en 1 passe
  2. Fallback DeepFace/ArcFace — si le modèle YOLO n'existe pas encore
"""

import os
import cv2
import numpy as np
from pathlib import Path

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/face_yolo.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# Fallback DeepFace
DEEPFACE_MODEL = os.getenv("DEEPFACE_MODEL", "ArcFace")
DEEPFACE_DETECTOR = os.getenv("DEEPFACE_DETECTOR", "opencv")
FACES_DB_PATH = os.getenv("FACES_DB_PATH", "data/faces")
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.68"))

_yolo_model = None
_use_yolo = False


def _load_yolo():
    global _yolo_model, _use_yolo
    if Path(YOLO_MODEL_PATH).exists():
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL_PATH)
        _use_yolo = True
        print(f"✅ Modèle YOLO chargé : {YOLO_MODEL_PATH}")
    else:
        _use_yolo = False
        print(f"⚠️  Modèle YOLO non trouvé ({YOLO_MODEL_PATH}). Fallback DeepFace.")


_load_yolo()


# ── YOLO pipeline ─────────────────────────────────────────────────────────────

def _detect_and_recognize_yolo(frame: np.ndarray) -> list[dict]:
    """
    Passe YOLO unique : détecte ET identifie chaque visage.
    Retourne une liste de résultats avec bbox + identité + confiance.
    """
    results = _yolo_model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        identity = results.names[cls_id]

        detections.append({
            "bbox": (x1, y1, x2 - x1, y2 - y1),
            "identity": identity,
            "confidence": round(conf * 100, 1),
            "verified": conf >= CONFIDENCE_THRESHOLD,
        })
    return detections


# ── DeepFace fallback ─────────────────────────────────────────────────────────

def _detect_and_recognize_deepface(frame: np.ndarray) -> list[dict]:
    """Fallback DeepFace : détection + embeddings ArcFace."""
    try:
        from deepface import DeepFace
        import tempfile

        faces_info = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=DEEPFACE_DETECTOR,
            enforce_detection=False,
        )
    except Exception:
        return []

    detections = []
    for f in faces_info:
        region = f["facial_area"]
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        face_img = frame[y: y + h, x: x + w]

        result = _recognize_deepface(face_img)
        detections.append({
            "bbox": (x, y, w, h),
            "identity": result["identity"],
            "confidence": result["confidence"],
            "verified": result["verified"],
        })
    return detections


def _recognize_deepface(face_img: np.ndarray) -> dict:
    from deepface import DeepFace
    import tempfile

    if face_img is None or face_img.size == 0:
        return {"identity": "Inconnu", "confidence": 0.0, "verified": False}

    if not os.path.exists(FACES_DB_PATH) or not os.listdir(FACES_DB_PATH):
        return {"identity": "Base vide", "confidence": 0.0, "verified": False}

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, face_img)
            tmp_path = tmp.name

        results = DeepFace.find(
            img_path=tmp_path,
            db_path=FACES_DB_PATH,
            model_name=DEEPFACE_MODEL,
            detector_backend=DEEPFACE_DETECTOR,
            distance_metric="cosine",
            enforce_detection=False,
            silent=True,
        )
        os.unlink(tmp_path)

        if results and len(results[0]) > 0:
            top = results[0].iloc[0]
            distance = top[f"{DEEPFACE_MODEL}_cosine"]
            confidence = round((1 - distance) * 100, 1)
            identity = os.path.splitext(os.path.basename(top["identity"]))[0]
            verified = distance < RECOGNITION_THRESHOLD
            return {"identity": identity, "confidence": confidence, "verified": verified}

    except Exception:
        pass

    return {"identity": "Inconnu", "confidence": 0.0, "verified": False}


# ── Interface publique ────────────────────────────────────────────────────────

def detect_and_recognize(frame: np.ndarray) -> list[dict]:
    """
    Détecte et identifie tous les visages dans une frame.

    Retourne une liste de :
        {
            "bbox":       (x, y, w, h),
            "identity":   str,
            "confidence": float,   # 0–100
            "verified":   bool,
        }
    """
    if _use_yolo:
        return _detect_and_recognize_yolo(frame)
    return _detect_and_recognize_deepface(frame)


def draw_results(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Dessine les bounding boxes et labels sur la frame."""
    for det in detections:
        x, y, w, h = det["bbox"]
        verified = det["verified"]
        color = (0, 200, 0) if verified else (0, 0, 220)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label = det["identity"].replace("_", " ")
        if verified:
            label += f"  {det['confidence']}%"

        # Fond du label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 4, y), color, -1)
        cv2.putText(
            frame, label,
            (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    return frame


def using_yolo() -> bool:
    return _use_yolo
