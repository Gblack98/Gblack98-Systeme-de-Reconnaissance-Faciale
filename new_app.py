import streamlit as st
import cv2
import time
import tempfile
import mediapipe as mp
import psycopg2
import numpy as np
import tensorflow as tf
import os

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
    choix = st.sidebar.radio("Choisissez une option", ("Se connecter", "S'inscrire"))

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
    # Préchargez les images des personnes avec le plus grand nombre d'images pour la reconnaissance
    images_dir = "/home/gblack98/Images/python/travail_IGD/archive (1)/lfw-deepfunneled/lfw-deepfunneled"
    # Sélectionner les noms des mille personnes avec le plus grand nombre d'images
    # Compter le nombre d'images par personne
    image_counts = {}
    for person in os.listdir(images_dir):
        person_dir = os.path.join(images_dir, person)
        if os.path.isdir(person_dir):
            image_counts[person] = len(os.listdir(person_dir))
        top_1000_persons = sorted(image_counts, key=image_counts.get, reverse=True)[:1000]

    # Liste des classes à utiliser pour l'entraînement et la validation
    classes_to_use = top_1000_persons

    top_persons_images = {}
    for person in os.listdir(images_dir):
        person_dir = os.path.join(images_dir, person)
        if os.path.isdir(person_dir) and person in classes_to_use:
            images = [os.path.join(person_dir, img) for img in os.listdir(person_dir)]
            top_persons_images[person] = images

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
                max_person_name = "inconnu"
                
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Dessiner un rectangle autour du visage
                    mp_drawing.draw_detection(image, detection)

                    # Récupérer les landmarks du visage
                    image.flags.writeable = True
                    landmarks = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).multi_face_landmarks

                    if landmarks:
                        for face_landmarks in landmarks:
                            for point in face_landmarks.landmark:
                                x, y = int(point.x * iw), int(point.y * ih)
                                cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

                    # Reconnaissance des célébrités
                    person_name = reconnaitre_celebrites(image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]], model, classes_to_use)

                    
                    # Mise à jour du nom de la personne avec la plus grande probabilité
                    if person_name != "inconnu":
                        confidence = get_confidence(image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]], model)
                        if confidence > max_confidence:
                            max_confidence = confidence
                            max_person_name = person_name

                    cv2.putText(image, person_name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Affichage du nom de la personne avec la plus grande probabilité
                cv2.putText(image, max_person_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            stframe.image(image, channels="BGR", use_column_width=True)
            time.sleep(0.1)

        vid.release()

   

# Chargement du modèle de reconnaissance faciale
def load_face_recognition_model():
    model_path = "/home/gblack98/CNN_Model/archiveGA/Model_de_face_recognition_Gblack98AAA.h5"
    model = tf.keras.models.load_model(model_path)
    return model

# Pré-traiter l'image de visage pour le modèle
def pretraiter_image_visage(image_visage):
    """
    Pré-traite l'image de visage pour la rendre compatible avec le modèle.

    Args:
        image_visage (numpy.ndarray): L'image de visage en format NumPy.

    Retourne:
        numpy.ndarray: L'image de visage pré-traitée.
    """
    if image_visage.ndim == 3 and image_visage.shape[2] == 4:
        image_visage = cv2.cvtColor(image_visage, cv2.COLOR_BGRA2RGB)
    image_redimensionnee = cv2.resize(image_visage, (224, 224))
    image_normalisee = image_redimensionnee / 255.0
    image_pretraitee = np.expand_dims(image_normalisee, axis=0)
    return image_pretraitee

# Reconnaître la célébrité dans l'image de visage donnée
def reconnaitre_celebrites(image_visage, modele, classes_a_utiliser, seuil_confiance=0.5):
    """
    Reconnaît la célébrité dans l'image de visage donnée en utilisant le modèle et les classes fournies.

    Args:
        image_visage (numpy.ndarray): L'image de visage en format NumPy.
        modele (tf.keras.Model): Le modèle de reconnaissance faciale.
        classes_a_utiliser (list): La liste des classes (noms des célébrités).
        seuil_confiance (float, optional): Le seuil de confiance minimum pour considérer une prédiction comme fiable. Defaults à 0.5.

    Retourne:
        str: Le nom de la célébrité reconnue ou "Inconnu" si non reconnu.
    """
    image_pretraitee = pretraiter_image_visage(image_visage)
    prediction = modele.predict(image_pretraitee)[0]
    indice_classe_predite = np.argmax(prediction)
    if indice_classe_predite < len(classes_a_utiliser):
        personne_predite = classes_a_utiliser[indice_classe_predite]
    else:
        personne_predite = "inconnu"
    confiance = prediction[0]
    if confiance >= seuil_confiance:
        return personne_predite
    else:
        return "inconnu"
def load_known_classes(images_dir="/home/gblack98/Images/python/travail_IGD/archive (1)/lfw-deepfunneled/lfw-deepfunneled"):
    image_counts = {}
    for person in os.listdir(images_dir):
        person_dir = os.path.join(images_dir, person)
        if os.path.isdir(person_dir):
            image_counts[person] = len(os.listdir(person_dir))

    top_1000_persons = sorted(image_counts, key=image_counts.get, reverse=True)[:1000]
    return top_1000_persons    

# Fonction pour obtenir la confiance de la prédiction
def get_confidence(face_img, model):
    # Prétraitement de l'image pour le modèle
    resized_image = cv2.resize(face_img, (224, 224))
    normalized_image = resized_image / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)

    # Prédiction du modèle
    prediction = model.predict(input_image)[0]

    # Retourne la confiance de la prédiction
    return prediction[0]

if __name__ == "__main__":
    main()
