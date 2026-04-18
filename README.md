# Système de Reconnaissance Faciale

> **Projet de Fin d'Études — Sonatel Academy (Orange Digital Center, Dakar)**  
> Auteur : **Ibrahima Gabar Diop** · [GitHub](https://github.com/Gblack98)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8.4.37-orange)](https://ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![mAP@50](https://img.shields.io/badge/mAP%4050-97.15%25-brightgreen)](#résultats)

Pipeline de reconnaissance faciale en temps réel basé sur **InsightFace buffalo_l** — RetinaFace pour la détection, ArcFace ResNet50 pour la reconnaissance par embeddings. Aucun entraînement requis, open-set : fonctionne pour toute personne dont une photo est disponible.

---

## Architecture

```
Entrée (webcam / image / vidéo)
        │
        ▼
┌─────────────────────────────┐
│  InsightFace buffalo_l      │
│                             │
│  1. RetinaFace              │  ← détecte + aligne les visages
│     (det_10g.onnx)          │
│  2. ArcFace ResNet50        │  ← embedding 512-dim par visage
│     (w600k_r50.onnx)        │
└────────────┬────────────────┘
             │  similarité cosinus vs data/faces/
             ▼
┌────────────────────────┐
│  Streamlit + SQLite    │
│  (auth bcrypt + logs)  │
└────────────────────────┘
```

## Pourquoi InsightFace

| Approche | Précision | Limites |
|---|---|---|
| CNN / ResNet classifieur | 33–53 % | Classes fixes, réentraînement requis |
| YOLOv8 classificateur | ~97 % sur LFW | Ne généralise pas au flux réel |
| **InsightFace ArcFace** | **99,77 % (LFW benchmark)** | **Open-set, zéro réentraînement** |

## Démarrage rapide

```bash
git clone https://github.com/Gblack98/Gblack98-Systeme-de-Reconnaissance-Faciale.git
cd Gblack98-Systeme-de-Reconnaissance-Faciale

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

streamlit run streamlit_app.py
# InsightFace buffalo_l (~300 MB) se télécharge automatiquement au 1er démarrage
```

Documentation complète : [`docs/face-recognition-system.md`](docs/face-recognition-system.md)

## Sécurité

- Mots de passe hashés **bcrypt** (jamais en clair)
- Pas de credential dans le code (`.env` exclu du versioning)
- `face_yolo.pt` et `data/users.db` non versionnés — voir [SECURITY.md](SECURITY.md)

## Licence

[MIT](LICENSE) — Ibrahima Gabar Diop, 2024
