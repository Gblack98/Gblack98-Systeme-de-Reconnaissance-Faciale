import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input
import mediapipe as mp
import os
import tempfile  # Importer le module tempfile

# Chargement du modèle de reconnaissance faciale
@st.cache(allow_output_mutation=True)
def load_face_recognition_model():
    model_path = "/home/gblack98/CNN_Model/archiveGA/Model_de_face_recognition_Gblack98AAA.h5"
    model = load_model(model_path)
    return model

# Chargement des données des personnes connues
@st.cache(allow_output_mutation=True)
def load_known_classes(images_dir):
    image_counts = {}
    for person in os.listdir(images_dir):
        person_dir = os.path.join(images_dir, person)
        if os.path.isdir(person_dir):
            image_counts[person] = len(os.listdir(person_dir))
    top_1000_persons = sorted(image_counts, key=image_counts.get, reverse=True)[:1000]
    return top_1000_persons

# Interface utilisateur
def main():
    model = load_face_recognition_model()
    images_dir = "/home/gblack98/Images/python/travail_IGD/archive (1)/lfw-deepfunneled/lfw-deepfunneled"
    known_classes = load_known_classes(images_dir)

    # Sélection de la source vidéo
    st.sidebar.header("Sélection de la source vidéo")
    selected_video_source = st.sidebar.selectbox("Source vidéo", ["Caméra", "Upload vidéo"])

    if selected_video_source == "Caméra":
        use_webcam = st.sidebar.button("Utiliser la webcam")
        if use_webcam:
            display_video("webcam", model, known_classes)
    elif selected_video_source == "Upload vidéo":
        video_file_buffer = st.sidebar.file_uploader("Uploader une vidéo", type=["mp4", "mov", "avi", "asf", "m4v"])
        if video_file_buffer:
            display_video(video_file_buffer, model, known_classes)

# Fonction pour afficher la vidéo et effectuer la reconnaissance faciale
def display_video(source, model, known_classes):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
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

            # Convertir l'image en RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Détecter les visages dans l'image
            results = face_detection.process(image_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Dessiner un rectangle autour du visage
                    mp_drawing.draw_detection(image, detection)

                    # Récupérer les coordonnées du rectangle du visage
                    x, y, w, h = bbox

                    # Dessiner des pointsillés autour du visage
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)

                    # Récupérer le visage pour la reconnaissance faciale
                    face_img = image[y:y+h, x:x+w]

                    # Prétraiter le visage pour la reconnaissance faciale
                    face_img = cv2.resize(face_img, (224, 224))
                    face_img = preprocess_input(face_img)

                    # Prédiction avec le modèle de reconnaissance faciale
                    prediction = model.predict(np.expand_dims(face_img, axis=0))[0]
                    predicted_class_index = np.argmax(prediction)

                    # Trouver le nom de la personne prédite
                    if predicted_class_index < len(known_classes):
                        predicted_person = known_classes[predicted_class_index]
                    else:
                        predicted_person = "Inconnu"

                    # Afficher le nom de la personne prédite
                    cv2.putText(image, predicted_person, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Afficher la vidéo avec les visages détectés et les noms des personnes prédites
            st.image(image, channels="BGR", use_column_width=True)

        vid.release()

# Exécution de l'interface utilisateur
if __name__ == "__main__":
    main()
