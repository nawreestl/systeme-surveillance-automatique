import cv2
from ultralytics import YOLO

# Charger le modèle YOLO
model = YOLO('models/yolo11n.pt')  
# Charger la vidéo
video_path = 'videos/video.mp4'  
cap = cv2.VideoCapture(video_path)

# Vérifier si la vidéo s'ouvre correctement
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo. Vérifie le chemin du fichier.")
    exit()

# Obtenir les dimensions de la vidéo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Définir le codec et créer un objet VideoWriter
out = cv2.VideoWriter('output_detected.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Appliquer YOLO pour la détection des objets
    results = model(frame)

    # 🔥 Correction ici : utiliser .plot() au lieu de .render()
    annotated_frame = results[0].plot()  # ✅ Utilisation correcte

    # Sauvegarder la vidéo annotée
    out.write(annotated_frame)

    # Afficher la vidéo avec les objets détectés
    cv2.imshow('Detected Objects', annotated_frame)

    # Quitter si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()
