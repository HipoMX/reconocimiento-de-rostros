import cv2
import face_recognition
import numpy as np

# Carga las imágenes de tus compañeros
leonardo_image = face_recognition.load_image_file("leonardo.jpg")
isaac_image = face_recognition.load_image_file("isaac.jpg")
edward_image = face_recognition.load_image_file("edward.jpg")
hipolito_image = face_recognition.load_image_file("hipolito.jpg")
omar_image = face_recognition.load_image_file("omar.jpg")

# Genera los embeddings de los rostros de tus compañeros
leonardo_face_encoding = face_recognition.face_encodings(leonardo_image)[0]
isaac_face_encoding = face_recognition.face_encodings(isaac_image)[0]
edward_face_encoding = face_recognition.face_encodings(edward_image)[0]
hipolito_face_encoding = face_recognition.face_encodings(hipolito_image)[0]
omar_face_encoding = face_recognition.face_encodings(omar_image)[0]

# Crea las listas de embeddings y etiquetas
known_face_encodings = [
    leonardo_face_encoding,
    isaac_face_encoding,
    edward_face_encoding,
    hipolito_face_encoding,
    omar_face_encoding
]
known_face_labels = [
    "Leonardo",
    "Isaac",
    "Edward",
    "Hipolito",
    "Omar"
]

# Inicializa algunas variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Captura el video de la cámara
video_capture = cv2.VideoCapture(0)

while True:
    # Toma un frame del video
    ret, frame = video_capture.read()

    # Redimensiona el frame para que sea más pequeño (más rápido)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convierte el color del frame de BGR a RGB (que es lo que face_recognition espera)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Procesa cada frame sólo una vez para ahorrar tiempo
    if process_this_frame:
        # Encuentra todas las caras y sus embeddings en el frame actual del video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Intenta hacer coincidir cada cara en el frame actual con las caras conocidas
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Si se encuentra una coincidencia, usa el nombre correspondiente
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_labels[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Muestra los resultados
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Escala los valores de las ubicaciones de las caras (ya que se redimensionó el frame)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Dibuja un rectángulo alrededor de la cara
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Dibuja el nombre de la cara debajo del rectángulo
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Muestra el resultado final del frame
        cv2.imshow('Video', frame)

        # Sale del programa si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()