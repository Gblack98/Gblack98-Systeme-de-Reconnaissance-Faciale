# Système de Reconnaissance Faciale

> **Projet de Fin d'Études — Sonatel Academy (Orange Digital Center, Dakar)**  
> Auteur : **Ibrahima Gabar Diop** · [GitHub](https://github.com/Gblack98)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8.4.37-orange)](https://ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![mAP@50](https://img.shields.io/badge/mAP%4050-97.15%25-brightgreen)](#résultats)

Pipeline de reconnaissance faciale en temps réel combinant **YOLOv8** (détection + classification en une passe) et **DeepFace/ArcFace** en fallback. Entraîné sur les 20 personnes les plus représentées du dataset LFW — **97,15 % de mAP@50** en 96 époques (~9 min sur Tesla T4).

---

## Architecture

```
Entrée (webcam / image / vidéo)
        │
        ▼
┌───────────────────┐
│   YOLOv8 26m      │  ← pipeline principal
│  détection +      │
│  classification   │
│  en 1 passe       │
└────────┬──────────┘
         │ confiance ≥ seuil → identité (20 classes)
         │ confiance < seuil
         ▼
┌───────────────────┐
│  DeepFace/ArcFace │  ← fallback nouvelles identités
└────────┬──────────┘
         ▼
┌────────────────────────┐
│  Streamlit + SQLite    │
│  (auth bcrypt + logs)  │
└────────────────────────┘
```

## Résultats

| Approche | Précision | Durée |
|---|---|---|
| CNN from scratch | 33 % | ~30 min |
| ResNet-50 Transfer Learning | 35 % | ~45 min |
| ResNet-50 Fine-tuning | 40 % | ~1 h |
| ResNet-50 Fine-tuning optimisé | 53 % | ~2 h 30 |
| **YOLOv8 26m (final)** | **97,15 % mAP@50** | **~9 min** |

## Démarrage rapide

```bash
git clone https://github.com/Gblack98/Gblack98-Systeme-de-Reconnaissance-Faciale.git
cd Gblack98-Systeme-de-Reconnaissance-Faciale

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Télécharger face_yolo.pt depuis les releases et le placer dans models/
streamlit run streamlit_app.py
```

Documentation complète : [`docs/face-recognition-system.md`](docs/face-recognition-system.md)

## Sécurité

- Mots de passe hashés **bcrypt** (jamais en clair)
- Pas de credential dans le code (`.env` exclu du versioning)
- `face_yolo.pt` et `data/users.db` non versionnés — voir [SECURITY.md](SECURITY.md)

## Licence

[MIT](LICENSE) — Ibrahima Gabar Diop, 2024
