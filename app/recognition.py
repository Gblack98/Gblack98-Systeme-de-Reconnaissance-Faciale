import cv2
import numpy as np
import tempfile
import os
from deepface import DeepFace

MODEL_NAME = os.getenv("DEEPFACE_MODEL", "ArcFace")
DETECTOR = os.getenv("DEEPFACE_DETECTOR", "opencv")
DISTANCE_METRIC = "cosine"
THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.68"))
DB_PATH_FACES = os.getenv("FACES_DB_PATH", "data/faces")


def build_face_db_if_needed():
    """Pré-calcule les embeddings du dossier faces/ au démarrage."""
    if not os.path.exists(FACES_DB_PATH):
        os.makedirs(FACES_DB_PATH, exist_ok=True)


def recognize_face(face_img: np.ndarray) -> dict:
    """
    Identifie un visage extrait (numpy array BGR).
    Retourne {"identity": str, "confidence": float, "verified": bool}.
    """
    if face_img is None or face_img.size == 0:
        return {"identity": "Inconnu", "confidence": 0.0, "verified": False}

    if not os.path.exists(DB_PATH_FACES) or not os.listdir(DB_PATH_FACES):
        return {"identity": "Base vide", "confidence": 0.0, "verified": False}

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, face_img)
            tmp_path = tmp.name

        results = DeepFace.find(
            img_path=tmp_path,
            db_path=DB_PATH_FACES,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False,
            silent=True,
        )
        os.unlink(tmp_path)

        if results and len(results[0]) > 0:
            top = results[0].iloc[0]
            distance = top[f"{MODEL_NAME}_{DISTANCE_METRIC}"]
            confidence = round((1 - distance) * 100, 1)
            identity = os.path.splitext(os.path.basename(top["identity"]))[0]
            verified = distance < THRESHOLD
            return {"identity": identity, "confidence": confidence, "verified": verified}

    except Exception:
        pass

    return {"identity": "Inconnu", "confidence": 0.0, "verified": False}


def detect_faces(frame: np.ndarray) -> list[dict]:
    """
    Détecte tous les visages dans une frame.
    Retourne une liste de {"bbox": (x,y,w,h), "face_img": np.ndarray}.
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=DETECTOR,
            enforce_detection=False,
        )
        result = []
        for f in faces:
            region = f["facial_area"]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            face_img = frame[y : y + h, x : x + w]
            result.append({"bbox": (x, y, w, h), "face_img": face_img})
        return result
    except Exception:
        return []


def draw_result(frame: np.ndarray, bbox: tuple, result: dict) -> np.ndarray:
    """Dessine le rectangle et le label sur la frame."""
    x, y, w, h = bbox
    verified = result.get("verified", False)
    color = (0, 200, 0) if verified else (0, 0, 220)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label = result["identity"]
    if verified:
        label += f"  {result['confidence']}%"

    cv2.putText(
        frame, label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
    )
    return frame
