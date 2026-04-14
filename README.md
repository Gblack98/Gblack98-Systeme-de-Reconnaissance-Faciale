# 👁️ Système de Reconnaissance Faciale

Projet de Fin d'Études réalisé à la **Sonatel Academy** (Orange Digital Center, Dakar).  
Système de reconnaissance faciale en temps réel pour des applications de **sécurité et de pointage**.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI                        │
│   Login / Register │ Reconnaissance │ Ajout visage  │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │      app/recognition    │
        │  DeepFace · ArcFace     │  ← Détection + Embeddings
        │  Détection OpenCV       │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │      app/database       │
        │  SQLite · bcrypt        │  ← Auth + Logs
        └─────────────────────────┘
```

| Composant       | Technologie                        |
|-----------------|------------------------------------|
| Détection       | DeepFace (OpenCV backend)          |
| Reconnaissance  | **ArcFace** (state-of-the-art)     |
| Interface       | Streamlit                          |
| Base de données | SQLite                             |
| Authentification| bcrypt (passwords hashés)          |
| Dataset origine | LFW — Labeled Faces in the Wild    |

---

## Fonctionnalités

- **Reconnaissance en temps réel** — webcam, image ou vidéo uploadée
- **Base de visages dynamique** — ajout de nouveaux visages sans réentraînement
- **Authentification sécurisée** — mots de passe hashés bcrypt, jamais stockés en clair
- **Logs des reconnaissances** — historique par utilisateur en base SQLite
- **Configurable** — modèle, détecteur et seuil de confiance via variables d'environnement

---

## Installation

```bash
git clone https://github.com/Gblack98/Gblack98-Systeme-de-Reconnaissance-Faciale.git
cd Gblack98-Systeme-de-Reconnaissance-Faciale

python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
```

---

## Lancer l'application

```bash
streamlit run streamlit_app.py
```

Ouvrir [http://localhost:8501](http://localhost:8501) dans le navigateur.

---

## Ajouter des visages à reconnaître

1. Lancer l'app et créer un compte
2. Aller dans **➕ Ajouter un visage**
3. Uploader une photo claire avec le nom complet (ex : `Ibrahima_Diop`)
4. Le visage est immédiatement disponible pour la reconnaissance

---

## Configuration `.env`

| Variable                | Défaut          | Description                                   |
|-------------------------|-----------------|-----------------------------------------------|
| `DB_PATH`               | `data/users.db` | Chemin vers la base SQLite                    |
| `FACES_DB_PATH`         | `data/faces`    | Dossier des images de référence               |
| `DEEPFACE_MODEL`        | `ArcFace`       | Modèle de reconnaissance (`Facenet512`, etc.) |
| `DEEPFACE_DETECTOR`     | `opencv`        | Détecteur de visages (`retinaface`, `mtcnn`)  |
| `RECOGNITION_THRESHOLD` | `0.68`          | Seuil de distance cosinus (plus bas = strict) |

---

## Dataset

Entraîné sur **[Labeled Faces in the Wild (LFW)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)** — 13 000+ images de personnalités publiques.

Le dataset n'est pas inclus dans le repo en raison de sa taille.  
Pour l'utiliser comme base de référence, télécharger et placer les images dans `data/faces/`.

---

## Sécurité

- Aucun credential hardcodé — tout passe par variables d'environnement
- Mots de passe hashés avec **bcrypt** (jamais stockés en clair)
- `data/` et `.env` exclus du versioning via `.gitignore`

---

## Auteur

**Ibrahima Gabar Diop** — Data & AI Engineer  
[github.com/Gblack98](https://github.com/Gblack98) · [linkedin.com/in/ibrahima-gabarda](https://linkedin.com/in/ibrahima-gabarda)
