import os
import cv2
import time
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

from app.database import init_db, register_user, authenticate_user, log_recognition, get_recognition_logs
from app.recognition import detect_and_recognize, draw_results, using_yolo, FACES_DB_PATH

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reconnaissance Faciale",
    page_icon="👁️",
    layout="wide",
)

init_db()


# ── Auth ──────────────────────────────────────────────────────────────────────
def login_page():
    st.title("👁️ Système de Reconnaissance Faciale")
    st.caption("Développé par Ibrahima Gabar Diop — PFE Sonatel Academy")
    st.divider()

    tab_login, tab_register = st.tabs(["Se connecter", "Créer un compte"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password")
            submitted = st.form_submit_button("Se connecter", use_container_width=True)
        if submitted:
            user = authenticate_user(username, password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Identifiants incorrects.")

    with tab_register:
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            first_name = col1.text_input("Prénom")
            last_name = col2.text_input("Nom")
            new_username = st.text_input("Nom d'utilisateur")
            new_password = st.text_input("Mot de passe", type="password")
            confirm = st.text_input("Confirmer le mot de passe", type="password")
            submitted = st.form_submit_button("Créer le compte", use_container_width=True)
        if submitted:
            if new_password != confirm:
                st.error("Les mots de passe ne correspondent pas.")
            elif len(new_password) < 6:
                st.error("Mot de passe trop court (6 caractères minimum).")
            elif register_user(first_name, last_name, new_username, new_password):
                st.success("Compte créé ! Connectez-vous.")
            else:
                st.error("Ce nom d'utilisateur est déjà pris.")


# ── Main app ──────────────────────────────────────────────────────────────────
def main_app():
    user = st.session_state.user

    with st.sidebar:
        st.markdown(f"### 👤 {user['first_name']} {user['last_name']}")
        st.caption(f"@{user['username']}")
        st.divider()
        page = st.radio(
            "Navigation",
            ["🔍 Reconnaissance", "📷 Webcam", "➕ Ajouter un visage", "📋 Historique", "ℹ️ À propos"],
            label_visibility="collapsed",
        )
        st.divider()
        if st.button("Se déconnecter", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if page == "🔍 Reconnaissance":
        recognition_page(user)
    elif page == "📷 Webcam":
        webcam_page(user)
    elif page == "➕ Ajouter un visage":
        add_face_page()
    elif page == "📋 Historique":
        history_page(user)
    else:
        about_page()


# ── Recognition page (image / vidéo) ─────────────────────────────────────────
def recognition_page(user):
    st.header("🔍 Reconnaissance faciale")

    source = st.radio("Source", ["🖼️ Image", "🎬 Vidéo"], horizontal=True)
    st.divider()

    if source == "🖼️ Image":
        uploaded = st.file_uploader("Uploader une image", type=["jpg", "jpeg", "png"])
        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            with st.spinner("Analyse en cours…"):
                process_frame_and_display(frame, user)

    elif source == "🎬 Vidéo":
        uploaded = st.file_uploader("Uploader une vidéo", type=["mp4", "mov", "avi", "mkv"])
        if uploaded:
            run_video(uploaded, user)


def process_frame_and_display(frame, user):
    detections = detect_and_recognize(frame)
    if not detections:
        st.warning("Aucun visage détecté.")
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        return

    for det in detections:
        if det["verified"]:
            log_recognition(user["id"], det["identity"], det["confidence"])

    annotated = draw_results(frame.copy(), detections)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("Résultats")
    for det in detections:
        if det["verified"]:
            st.success(f"✅ **{det['identity'].replace('_', ' ')}** — {det['confidence']}%")
        else:
            st.warning(f"❓ Inconnu — confiance insuffisante ({det['confidence']}%)")


# ── Webcam page (st.camera_input — fonctionne partout) ───────────────────────
def webcam_page(user):
    st.header("📷 Reconnaissance en temps réel")
    st.info("Prenez une photo depuis votre webcam. Le système analyse immédiatement.")

    img_file = st.camera_input("Capturer un visage")
    if img_file:
        file_bytes = np.frombuffer(img_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        with st.spinner("Analyse en cours…"):
            process_frame_and_display(frame, user)


# ── Video ─────────────────────────────────────────────────────────────────────
def run_video(uploaded, user):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    stframe = st.empty()
    progress = st.progress(0)
    stop = st.button("⏹ Arrêter")
    frame_idx = 0

    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detect_and_recognize(frame)
        for det in detections:
            if det["verified"]:
                log_recognition(user["id"], det["identity"], det["confidence"])
        draw_results(frame, detections)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        if total_frames > 0:
            progress.progress(min(frame_idx / total_frames, 1.0))
        frame_idx += 1
        time.sleep(1 / fps)

    cap.release()
    progress.empty()
    os.unlink(tmp_path)


# ── Add face page ─────────────────────────────────────────────────────────────
def add_face_page():
    st.header("➕ Ajouter un visage à la base")
    st.info(
        "Ajoutez une photo claire d'une personne. "
        "Le système utilisera cette image pour l'identifier lors des prochaines reconnaissances. "
        "_(Mode DeepFace/ArcFace uniquement — ne modifie pas le modèle YOLO.)_"
    )

    if using_yolo():
        st.warning(
            "Le pipeline YOLO est actif. Les visages ajoutés ici seront utilisés "
            "uniquement en cas de **fallback** (identité hors des 20 classes YOLO)."
        )

    with st.form("add_face_form"):
        name = st.text_input("Nom complet (ex : Jean_Dupont)", help="Utiliser underscore à la place des espaces")
        photo = st.file_uploader("Photo du visage", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Ajouter", use_container_width=True)

    if submitted:
        if not name or not photo:
            st.error("Veuillez remplir tous les champs.")
        elif " " in name:
            st.error("Remplacez les espaces par des underscores : ex. `Jean_Dupont`")
        else:
            os.makedirs(FACES_DB_PATH, exist_ok=True)
            dest = os.path.join(FACES_DB_PATH, f"{name}.jpg")
            img = Image.open(photo).convert("RGB")
            img.save(dest)
            # Invalider le cache DeepFace
            pkl_path = os.path.join(FACES_DB_PATH, "representations_arcface.pkl")
            if os.path.exists(pkl_path):
                os.remove(pkl_path)
            st.success(f"✅ **{name.replace('_', ' ')}** ajouté. Il sera reconnu lors du prochain scan.")
            st.image(img, width=200)


# ── History page ──────────────────────────────────────────────────────────────
def history_page(user):
    st.header("📋 Historique des reconnaissances")

    logs = get_recognition_logs(user["id"])
    if not logs:
        st.info("Aucune reconnaissance enregistrée pour ce compte.")
        return

    st.caption(f"{len(logs)} événement(s) enregistré(s)")

    import pandas as pd
    df = pd.DataFrame(logs, columns=["Identité", "Confiance (%)", "Date"])
    df["Identité"] = df["Identité"].str.replace("_", " ")
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%d/%m/%Y %H:%M")
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── About page ────────────────────────────────────────────────────────────────
def about_page():
    st.header("ℹ️ À propos du projet")
    backend = "YOLOv8 26m (détection + identification en 1 passe)" if using_yolo() else "DeepFace / ArcFace (fallback)"
    st.markdown(f"""
## Système de Reconnaissance Faciale

Projet de Fin d'Études réalisé à la **Sonatel Academy** (Orange Digital Center, Dakar).

### Backend actif
**{backend}**

### Architecture

| Composant | Technologie |
|---|---|
| Détection & Reconnaissance | **{backend}** |
| Interface | Streamlit |
| Base de données | SQLite |
| Auth | bcrypt |

### Pipeline
- **YOLO 26m** — modèle entraîné sur 20 classes LFW, 97,15 % mAP@50, ~9 min sur Tesla T4
- **DeepFace / ArcFace** — fallback pour les identités hors des 20 classes

### Dataset
Top-20 du **Labeled Faces in the Wild (LFW)** — ~1 900 images, 70/15/15 split.

---
**Développé par** Ibrahima Gabar Diop | [GitHub](https://github.com/Gblack98)
""")


# ── Entry point ───────────────────────────────────────────────────────────────
if "user" not in st.session_state:
    login_page()
else:
    main_app()
