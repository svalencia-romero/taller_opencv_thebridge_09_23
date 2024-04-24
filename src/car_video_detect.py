import cv2
import numpy as np
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

# Cargar el clasificador preentrenado para detección de coches

car_cascade = cv2.CascadeClassifier(os.path.join(dir_path, "..", "xml", "haarcascade_car.xml"))

# Iniciar la cámara (0 indica la cámara predeterminada)
# cap = cv2.VideoCapture(0)
# To use a video file as input 
# https://www.pexels.com/video/a-biker-traversing-a-road-built-on-mountain-sides-3055765/
cap = cv2.VideoCapture(os.path.join(dir_path, "..", "video", "Video_3_1_res.mp4"))

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray,1.3,7, minSize=(50,50))
    
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y), (x+w,y+h),(0,0,255),2)
    
    cv2.imshow('Car Detector', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Cerrar las ventanas
cap.release()