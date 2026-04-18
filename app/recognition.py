"""
Pipeline deux modèles :

  Modèle 1 — Détecteur  : face_detector.pt  (yolov8n-face, pré-entraîné WiderFace)
               → trouve et croppe tous les visages dans le frame

  Modèle 2 — Classificateur : face_classifier.pt  (yolov8m-cls, LFW Top-20)
               → identifie la personne sur chaque crop

  Fallback  — DeepFace / ArcFace si face_classifier.pt absent
               → embeddings cosinus sur data/faces/
"""

import os
import cv2
import numpy as np
import tempfile
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DETECTOR_PATH        = os.getenv("DETECTOR_PATH",     "models/face_detector.pt")
CLASSIFIER_PATH      = os.getenv("CLASSIFIER_PATH",   "models/face_classifier.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

DEEPFACE_MODEL        = os.getenv("DEEPFACE_MODEL",    "ArcFace")
DEEPFACE_DETECTOR     = os.getenv("DEEPFACE_DETECTOR", "retinaface")
FACES_DB_PATH         = os.getenv("FACES_DB_PATH",     "data/faces")
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.68"))


# ── Chargement des modèles ────────────────────────────────────────────────────
_detector    = None
_classifier  = None


def _load_models():
    global _detector, _classifier
    from ultralytics import YOLO

    # Modèle 1 : détecteur de visages
    if Path(DETECTOR_PATH).exists():
        _detector = YOLO(DETECTOR_PATH)
        print(f"[Détecteur]     {DETECTOR_PATH}")
    else:
        # Téléchargement automatique du pré-entraîné Ultralytics
        _detector = YOLO("yolov8n-face.pt")
        print("[Détecteur]     yolov8n-face.pt (pré-entraîné, téléchargé)")

    # Modèle 2 : classificateur d'identité
    if Path(CLASSIFIER_PATH).exists():
        _classifier = YOLO(CLASSIFIER_PATH)
        print(f"[Classificateur] {CLASSIFIER_PATH}")
    else:
        _classifier = None
        print(f"[Classificateur] absent ({CLASSIFIER_PATH}) — fallback DeepFace/ArcFace")


_load_models()


def using_classifier() -> bool:
    return _classifier is not None


# ── Étape 1 : détection des visages ──────────────────────────────────────────

def _detect_faces(frame: np.ndarray) -> list[dict]:
    """
    Détecte tous les visages dans le frame.
    Retourne [ {"bbox": (x, y, w, h), "crop": ndarray} ]
    """
    results = _detector.predict(source=frame, conf=0.4, verbose=False)[0]
    faces = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Marge légère pour inclure le contour du visage
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(frame.shape[1], x2 + pad)
        y2 = min(frame.shape[0], y2 + pad)
        w, h = x2 - x1, y2 - y1
        if w < 20 or h < 20:
            continue
        crop = frame[y1:y2, x1:x2]
        faces.append({"bbox": (x1, y1, w, h), "crop": crop})
    return faces


# ── Étape 2a : classification YOLO ───────────────────────────────────────────

def _classify_crop(crop: np.ndarray) -> dict:
    """
    Identifie la personne sur un crop de visage via le classificateur YOLO.
    """
    results = _classifier.predict(source=crop, verbose=False)[0]
    top1_idx  = int(results.probs.top1)
    top1_conf = float(results.probs.top1conf)
    identity  = results.names[top1_idx]
    verified  = top1_conf >= CONFIDENCE_THRESHOLD
    return {
        "identity":   identity,
        "confidence": round(top1_conf * 100, 1),
        "verified":   verified,
    }


# ── Étape 2b : fallback ArcFace ───────────────────────────────────────────────

def _classify_arcface(crop: np.ndarray) -> dict:
    """
    Fallback : compare le crop avec data/faces/ via ArcFace embeddings.
    """
    from deepface import DeepFace

    if not os.path.exists(FACES_DB_PATH) or not any(
        f.lower().endswith((".jpg", ".jpeg", ".png"))
        for f in os.listdir(FACES_DB_PATH)
    ):
        return {"identity": "Base vide", "confidence": 0.0, "verified": False}

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, crop)
            tmp_path = tmp.name

        results = DeepFace.find(
            img_path=tmp_path,
            db_path=FACES_DB_PATH,
            model_name=DEEPFACE_MODEL,
            detector_backend="skip",
            distance_metric="cosine",
            enforce_detection=False,
            silent=True,
        )
        os.unlink(tmp_path)

        if results and len(results[0]) > 0:
            top      = results[0].iloc[0]
            distance = top[f"{DEEPFACE_MODEL}_cosine"]
            conf     = round((1 - distance) * 100, 1)
            identity = os.path.splitext(os.path.basename(top["identity"]))[0]
            return {
                "identity":   identity,
                "confidence": conf,
                "verified":   distance < RECOGNITION_THRESHOLD,
            }
    except Exception:
        pass

    return {"identity": "Inconnu", "confidence": 0.0, "verified": False}


# ── Interface publique ────────────────────────────────────────────────────────

def detect_and_recognize(frame: np.ndarray) -> list[dict]:
    """
    Pipeline complet :
      1. Détecte les visages (yolov8n-face)
      2. Classifie chaque crop (classificateur YOLO si dispo, sinon ArcFace)

    Retourne : [ {"bbox": (x,y,w,h), "identity": str, "confidence": float, "verified": bool} ]
    """
    faces = _detect_faces(frame)
    if not faces:
        return []

    detections = []
    for face in faces:
        crop = face["crop"]
        if _classifier is not None:
            result = _classify_crop(crop)
        else:
            result = _classify_arcface(crop)

        detections.append({
            "bbox":       face["bbox"],
            "identity":   result["identity"],
            "confidence": result["confidence"],
            "verified":   result["verified"],
        })

    return detections


def draw_results(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    for det in detections:
        x, y, w, h = det["bbox"]
        color = (0, 200, 0) if det["verified"] else (0, 100, 220)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label = det["identity"].replace("_", " ")
        if det["verified"]:
            label += f"  {det['confidence']}%"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 4, y), color, -1)
        cv2.putText(frame, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


# Alias conservé pour rétrocompatibilité avec l'UI
def using_yolo() -> bool:
    return using_classifier()
