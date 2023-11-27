import cv2
import numpy as np
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
# Cargar el clasificador preentrenado para detección de coches
# Load the cascade

car_cascade = cv2.CascadeClassifier(os.path.join(dir_path, "..", "xml", "haarcascade_car.xml"))

# Iniciar la cámara (0 indica la cámara predeterminada)
# cap = cv2.VideoCapture(0)
# To use a video file as input 
cap = cv2.VideoCapture(os.path.join(dir_path, "..", "video", "video_2.mp4"))


while True:
    # Capturar el fotograma de la cámara
    ret, frame = cap.read()

    # Convertir a escala de grises para el clasificador
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar coches en la imagen
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

    # Dibujar rectángulos alrededor de los coches detectados
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Mostrar el resultado
    cv2.imshow('Car Detection', frame)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar la captura de la cámara y cerrar las ventanas
cap.release()