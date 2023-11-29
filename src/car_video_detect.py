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
cap = cv2.VideoCapture(os.path.join(dir_path, "..", "video", "Video_3_1.mp4"))

