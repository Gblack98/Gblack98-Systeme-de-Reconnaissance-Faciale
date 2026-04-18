"""
Pipeline de reconnaissance faciale — deux étapes :

  1. Détection  : RetinaFace (généraliste, robuste sur webcam/vidéo réelle)
  2. Reconnaissance :
       a. ArcFace (DeepFace) — compare le crop avec data/faces/
       b. YOLO   (optionnel) — classificateur rapide sur le crop si le modèle existe
                               et si la confiance ArcFace est insuffisante

Avantage : le détecteur RetinaFace trouve les visages dans n'importe quel contexte.
           ArcFace fonctionne pour toute identité dont une photo est dans data/faces/.
"""

import os
import cv2
import numpy as np
import tempfile
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH        = os.getenv("YOLO_MODEL_PATH", "models/face_yolo.pt")
CONFIDENCE_THRESHOLD   = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

DEEPFACE_MODEL         = os.getenv("DEEPFACE_MODEL", "ArcFace")
DEEPFACE_DETECTOR      = os.getenv("DEEPFACE_DETECTOR", "retinaface")
FACES_DB_PATH          = os.getenv("FACES_DB_PATH", "data/faces")
RECOGNITION_THRESHOLD  = float(os.getenv("RECOGNITION_THRESHOLD", "0.68"))

# ── YOLO (optionnel) ──────────────────────────────────────────────────────────
_yolo_model = None

def _load_yolo():
    global _yolo_model
    if Path(YOLO_MODEL_PATH).exists():
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"[YOLO] Modèle chargé : {YOLO_MODEL_PATH}")
    else:
        print(f"[YOLO] Modèle absent ({YOLO_MODEL_PATH}) — YOLO désactivé.")

_load_yolo()


def using_yolo() -> bool:
    return _yolo_model is not None


# ── Étape 1 : détection des visages ──────────────────────────────────────────

def _detect_faces(frame: np.ndarray) -> list[dict]:
    """
    Détecte tous les visages dans le frame avec RetinaFace.
    Retourne une liste de { "bbox": (x, y, w, h), "crop": np.ndarray }.
    """
    try:
        from deepface import DeepFace
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=DEEPFACE_DETECTOR,
            enforce_detection=False,
            align=True,
        )
    except Exception:
        return []

    results = []
    for f in faces:
        if f.get("confidence", 0) < 0.5:
            continue
        r = f["facial_area"]
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        # Garde uniquement les crops de taille raisonnable
        if w < 20 or h < 20:
            continue
        crop = frame[max(0, y):y + h, max(0, x):x + w]
        results.append({"bbox": (x, y, w, h), "crop": crop})
    return results


# ── Étape 2a : reconnaissance ArcFace ────────────────────────────────────────

def _recognize_arcface(crop: np.ndarray) -> dict:
    """
    Compare le crop avec data/faces/ via ArcFace + distance cosinus.
    """
    from deepface import DeepFace

    if crop is None or crop.size == 0:
        return {"identity": "Inconnu", "confidence": 0.0, "verified": False}

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
            detector_backend="skip",   # crop déjà extrait — pas besoin de re-détecter
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


# ── Étape 2b : classification YOLO sur crop (optionnel) ──────────────────────

def _classify_yolo(crop: np.ndarray) -> dict | None:
    """
    Passe le crop au classificateur YOLO.
    Retourne un résultat si confiance >= seuil, sinon None.
    """
    if _yolo_model is None or crop is None or crop.size == 0:
        return None
    try:
        results = _yolo_model.predict(source=crop, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        if results.boxes:
            box = results.boxes[0]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            identity = results.names[cls_id]
            return {"identity": identity, "confidence": round(conf * 100, 1), "verified": True}
    except Exception:
        pass
    return None


# ── Interface publique ────────────────────────────────────────────────────────

def detect_and_recognize(frame: np.ndarray) -> list[dict]:
    """
    Détecte et identifie tous les visages dans une frame BGR.

    Pipeline :
      RetinaFace (détection) → ArcFace (reconnaissance)
                             → YOLO en complément si confiance ArcFace < seuil

    Retourne une liste de :
        {
            "bbox"      : (x, y, w, h),
            "identity"  : str,
            "confidence": float,   # 0–100
            "verified"  : bool,
        }
    """
    faces = _detect_faces(frame)
    if not faces:
        return []

    detections = []
    for face in faces:
        crop = face["crop"]

        # Essayer YOLO en premier (rapide) si disponible
        result = _classify_yolo(crop)

        # Sinon (ou si confiance insuffisante) → ArcFace
        if result is None:
            result = _recognize_arcface(crop)

        detections.append({
            "bbox":       face["bbox"],
            "identity":   result["identity"],
            "confidence": result["confidence"],
            "verified":   result["verified"],
        })

    return detections


def draw_results(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Dessine les bounding boxes et labels sur la frame."""
    for det in detections:
        x, y, w, h = det["bbox"]
        verified = det["verified"]
        color = (0, 200, 0) if verified else (0, 100, 220)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label = det["identity"].replace("_", " ")
        if verified:
            label += f"  {det['confidence']}%"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 4, y), color, -1)
        cv2.putText(
            frame, label,
            (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    return frame
