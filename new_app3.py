import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input
import numpy as np
import os
import cv2
import time
import tempfile
import mediapipe as mp

import psycopg2

# Connexion à la base de données PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="Oeil_Celeste",
    user="gabar",
    password="Gabardiop1998"
)

# Création de la table pour les utilisateurs (à exécuter une seule fois)
def create_table():
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS utilisateurs (
            id SERIAL PRIMARY KEY,
            "Prenom" VARCHAR(50) NOT NULL,
            "Nom" VARCHAR(50) NOT NULL,
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(100) NOT NULL
        )
    """)
    conn.commit()
    cursor.close()

# Appel de la fonction pour créer la table utilisateurs
create_table()

# Fonction pour vérifier les informations de connexion
def authenticate(username, password):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM utilisateurs WHERE username = %s AND password = %s", (username, password))
    user = cursor.fetchone()
    cursor.close()
    return user

# Fonction pour inscrire un nouvel utilisateur
def register_user(prenom, nom, username, password):
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO utilisateurs (\"Prenom\", \"Nom\", username, password) VALUES (%s, %s, %s, %s)", (prenom, nom, username, password))
        conn.commit()
        cursor.close()
        return True
    except psycopg2.IntegrityError:
        return False  # L'utilisateur existe déjà

# Fonction principale de l'interface utilisateur
def main():
    st.title("Système de reconnaissance faciale")
    st.sidebar.header("Connexion")
    choix = st.sidebar.radio("Choisissez une option :", ("Se connecter", "S'inscrire"))

    if choix == "Se connecter":
        username = st.sidebar.text_input("Nom d'utilisateur")
        password = st.sidebar.text_input("Mot de passe", type="password")
        if st.sidebar.button("Se connecter"):
            user = authenticate(username, password)
            if user:
                st.sidebar.success("Connexion réussie!")
                st.session_state.logged_in = True
            else:
                st.sidebar.error("Nom d'utilisateur ou mot de passe incorrect")

    elif choix == "S'inscrire":
        prenom = st.sidebar.text_input("Prénom")
        nom = st.sidebar.text_input("Nom")
        new_username = st.sidebar.text_input("Nouveau nom d'utilisateur")
        new_password = st.sidebar.text_input("Nouveau mot de passe", type="password")
        confirm_password = st.sidebar.text_input("Confirmer le mot de passe", type="password")

        if st.sidebar.button("S'inscrire"):
            if new_password == confirm_password:
                if register_user(prenom, nom, new_username, new_password):
                    st.sidebar.success("Inscription réussie! Connectez-vous maintenant.")
                else:
                    st.sidebar.error("Ce nom d'utilisateur existe déjà.")
            else:
                st.sidebar.error("Les mots de passe est incorrect")

    if st.session_state.get("logged_in", False):
        st.sidebar.header("Sélection de la source vidéo")
        selected_video_source = st.sidebar.selectbox("Source vidéo", ["Caméra", "Upload vidéo"])  

        if selected_video_source == "Caméra":
            use_webcam = st.sidebar.button("Utiliser la webcam")
            if use_webcam:
                display_video("webcam")
        elif selected_video_source == "Upload vidéo":
            video_file_buffer = st.sidebar.file_uploader("Uploader une vidéo", type=["mp4","jpg","png", "mov", "avi", "asf", "m4v"])
            if video_file_buffer:
                display_video(video_file_buffer)

# Fonction pour afficher la vidéo
def display_video(source):
    stframe = st.empty()
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    model = load_face_recognition_model()
    classes_to_use = load_known_classes()

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
         mp_face_mesh.FaceMesh() as face_mesh:
        if source == "webcam":
            vid = cv2.VideoCapture(0)
        else:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(source.read())
            vid = cv2.VideoCapture(temp_file.name)

        while vid.isOpened():
            ret, image = vid.read()

            if not ret:
                break

            image.flags.writeable = False
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.detections:
                max_confidence = 0
                max_person_name = "Personne inconnue"
                
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Dessiner un rectangle autour du visage
                    mp_drawing.draw_detection(image, detection)

                    # Reconnaissance des célébrités
                    person_name = recognize_celebrities(image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]], model, classes_to_use)

                    cv2.putText(image, person_name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Affichage du nom de la personne avec la plus grande probabilité
                cv2.putText(image, max_person_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            stframe.image(image, channels="BGR", use_column_width=True)
            time.sleep(0.1)

        vid.release()

# Chargement du modèle de reconnaissance faciale
def load_face_recognition_model():
    model_path = "/home/gblack98/CNN_Model/archiveGA/Model_de_face_recognition_Gblack98AAA.h5"
    model = load_model(model_path)
    return model

# Fonction pour prétraiter l'image de visage
def preprocess_face_image(face_image):
    img_array = image.img_to_array(face_image)
    img_array = preprocess_input(img_array)
    preprocessed_image = np.expand_dims(img_array, axis=0)
    return preprocessed_image

# Reconnaître les célébrités dans l'image de visage donnée
def recognize_celebrities(face_image, model, classes_to_use):
    preprocessed_image = preprocess_face_image(face_image)
    prediction = model.predict(preprocessed_image)[0]
    predicted_class_index = np.argmax(prediction)
    if predicted_class_index < len(classes_to_use):
        predicted_person = classes_to_use[predicted_class_index]
    else:
        predicted_person = "Inconnu"
    return predicted_person

# Charger les classes connues
def load_known_classes(images_dir="/home/gblack98/Images/python/travail_IGD/archive (1)/lfw-deepfunneled/lfw-deepfunneled"):
    image_counts = {}
    for person in os.listdir(images_dir):
        person_dir = os.path.join(images_dir, person)
        if os.path.isdir(person_dir):
            image_counts[person] = len(os.listdir(person_dir))

    top_1000_persons = sorted(image_counts, key=image_counts.get, reverse=True)[:1000]
    return top_1000_persons    

if __name__ == "__main__":
    main()
