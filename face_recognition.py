import cv2
import os
import datetime

# Cargar el clasificador pre-entrenado de detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Cargar las imágenes de las personas registradas
registered_images = []
registered_names = []
for i in range(1, 6):
    name = input(f"Ingrese el nombre de la persona {i}: ")
    registered_names.append(name)
    image = cv2.imread(f"{name}.jpg")
    registered_images.append(image)

# Definir una función que tome una imagen como entrada y devuelva una lista de las coordenadas de los rostros detectados
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
    return faces

# Definir una función que tome una imagen como entrada y devuelva True si la imagen es de una persona viva y False si la imagen es una foto o un video
def is_live_person(image):
    # Aquí iría el código para determinar si la imagen es de una persona viva
    return True  # Placeholder para el ejemplo

# Definir una función que tome una imagen como entrada y devuelva el nombre de la persona detectada
def recognize_person(image):
    # Aquí iría el código para reconocer el rostro de las personas registradas
    return "Desconocido"

# Capturar la imagen de la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Iniciar un bucle que capture imágenes de la cámara y realice el reconocimiento facial
while True:
    ret, frame = cap.read()

    if ret:
        # Detectar rostros en la imagen
        faces = detect_faces(frame)

        # Recorrer los rostros detectados y realizar el reconocimiento facial
        for (x, y, w, h) in faces:
            # Recortar la región de interés (ROI) correspondiente al rostro
            roi = frame[y:y+h, x:x+w]

            # Verificar si la imagen es de una persona viva
            if is_live_person(roi):
                # Reconocer el rostro de la persona
                name = recognize_person(roi)

                # Comparar la imagen del rostro con las imágenes registradas
                max_matches = 0
                for i, registered_image in enumerate(registered_images):
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    gray_registered_image = cv2.cvtColor(registered_image, cv2.COLOR_BGR2GRAY)
                    sift = cv2.SIFT_create()
                    kp1, des1 = sift.detectAndCompute(gray_roi, None)
                    kp2, des2 = sift.detectAndCompute(gray_registered_image, None)

                    # Configurar el algoritmo de emparejamiento
                    index_params = dict(algorithm=0, trees=5)
                    search_params = dict()
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(des1, des2, k=2)

                    # Aplicar el test de relación de distancia
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                    
                    # Verificar si se encontraron más coincidencias que antes
                    if len(good_matches) > max_matches:
                        max_matches = len(good_matches)
                        name = registered_names[i]

                # Dibujar el rectángulo y el nombre de la persona en la imagen
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar la imagen resultante
        cv2.imshow("Reconocimiento facial", frame)

    # Esperar a que se presione la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

cap.release()
cv2.destroyAllWindows()