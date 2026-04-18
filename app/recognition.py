"""
Pipeline de reconnaissance faciale — InsightFace (état de l'art)

  Détection    : RetinaFace  (det_10g.onnx  — buffalo_l)
  Reconnaissance : ArcFace ResNet50 (w600k_r50.onnx — buffalo_l)

  Fonctionnement :
    1. InsightFace détecte tous les visages et calcule un embedding ArcFace (512 dim)
       pour chacun en une seule passe.
    2. Chaque embedding est comparé par similarité cosinus à la base de référence
       construite à partir des images dans data/faces/.
    3. Si la similarité dépasse le seuil → identité reconnue.

  Ajout d'un visage : déposer une photo dans data/faces/ et supprimer
                      data/faces/embeddings.npz — la base se reconstruit au prochain appel.
"""

import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ────────────────────────────────────────────────────────────────────
FACES_DB_PATH         = os.getenv("FACES_DB_PATH",     "data/faces")
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.45"))
EMBEDDINGS_CACHE      = os.path.join(FACES_DB_PATH, "embeddings.pkl")

# ── Chargement InsightFace ────────────────────────────────────────────────────
_face_app  = None
_face_db   = {}   # { "Nom_Prénom": np.ndarray (512,) }
_db_mtime  = 0.0  # timestamp de dernière construction


def _load_insightface():
    global _face_app
    try:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("[InsightFace] buffalo_l chargé (RetinaFace + ArcFace ResNet50)")
    except Exception as e:
        print(f"[InsightFace] Erreur chargement : {e}")
        _face_app = None


_load_insightface()


# ── Base d'embeddings ─────────────────────────────────────────────────────────

def _faces_db_mtime() -> float:
    """Retourne le timestamp de modification le plus récent dans data/faces/."""
    if not os.path.exists(FACES_DB_PATH):
        return 0.0
    mtimes = [
        os.path.getmtime(os.path.join(FACES_DB_PATH, f))
        for f in os.listdir(FACES_DB_PATH)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return max(mtimes) if mtimes else 0.0


def _build_face_db():
    """
    Construit la base d'embeddings à partir des images dans data/faces/.
    Une image par identité ; le nom du fichier (sans extension) = identité.
    Sauvegarde dans embeddings.pkl pour éviter de recalculer à chaque démarrage.
    """
    global _face_db, _db_mtime

    if _face_app is None:
        return

    current_mtime = _faces_db_mtime()

    # Cache valide ?
    if os.path.exists(EMBEDDINGS_CACHE) and current_mtime <= _db_mtime:
        with open(EMBEDDINGS_CACHE, "rb") as f:
            _face_db = pickle.load(f)
        print(f"[Face DB] Cache chargé — {len(_face_db)} identités")
        return

    if not os.path.exists(FACES_DB_PATH):
        return

    db = {}
    imgs = [
        f for f in os.listdir(FACES_DB_PATH)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for fname in imgs:
        path  = os.path.join(FACES_DB_PATH, fname)
        name  = os.path.splitext(fname)[0]
        img   = cv2.imread(path)
        if img is None:
            continue
        faces = _face_app.get(img)
        if not faces:
            print(f"[Face DB] Aucun visage détecté dans {fname} — ignoré")
            continue
        # Garder le visage le plus grand (le plus probable)
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        db[name] = face.embedding / np.linalg.norm(face.embedding)

    _face_db   = db
    _db_mtime  = current_mtime

    os.makedirs(FACES_DB_PATH, exist_ok=True)
    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump(db, f)

    print(f"[Face DB] Construite — {len(db)} identités")


_build_face_db()


# ── Reconnaissance ────────────────────────────────────────────────────────────

def _recognize_embedding(embedding: np.ndarray) -> dict:
    """
    Compare un embedding ArcFace à la base de référence.
    Retourne l'identité la plus proche si similarité >= seuil.
    """
    if not _face_db:
        return {"identity": "Base vide", "confidence": 0.0, "verified": False}

    emb_norm = embedding / np.linalg.norm(embedding)
    names    = list(_face_db.keys())
    ref_embs = np.stack(list(_face_db.values()))

    scores = cosine_similarity([emb_norm], ref_embs)[0]
    best_idx  = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    verified = best_score >= RECOGNITION_THRESHOLD
    return {
        "identity":   names[best_idx] if verified else "Inconnu",
        "confidence": round(best_score * 100, 1),
        "verified":   verified,
    }


# ── Interface publique ────────────────────────────────────────────────────────

def detect_and_recognize(frame: np.ndarray) -> list[dict]:
    """
    Détecte et identifie tous les visages dans une frame BGR.

    Retourne :
        [ {"bbox": (x, y, w, h), "identity": str, "confidence": float, "verified": bool} ]
    """
    # Reconstruire la DB si data/faces/ a changé
    if _faces_db_mtime() > _db_mtime:
        _build_face_db()

    if _face_app is None:
        return []

    faces = _face_app.get(frame)
    if not faces:
        return []

    detections = []
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        w, h = x2 - x1, y2 - y1
        result = _recognize_embedding(face.embedding)
        detections.append({
            "bbox":       (x1, y1, w, h),
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


def using_yolo() -> bool:
    """Alias conservé pour l'UI — retourne True si InsightFace est chargé."""
    return _face_app is not None


def rebuild_face_db():
    """Invalide le cache et reconstruit la base d'embeddings."""
    global _db_mtime
    if os.path.exists(EMBEDDINGS_CACHE):
        os.remove(EMBEDDINGS_CACHE)
    _db_mtime = 0.0
    _build_face_db()
