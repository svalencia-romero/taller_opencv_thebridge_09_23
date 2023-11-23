import cv2
import numpy as np

# Cargar el clasificador preentrenado para detección de coches
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')  # Asegúrate de tener el archivo XML en tu directorio

# Iniciar la cámara (0 indica la cámara predeterminada)
cap = cv2.VideoCapture(0)

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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de la cámara y cerrar las ventanas
cap.release()