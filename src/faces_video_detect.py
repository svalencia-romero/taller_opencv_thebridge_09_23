import os
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))

# Load the cascade
face_cascade = cv2.CascadeClassifier(os.path.join(dir_path, "..", "xml", "haarcascade_frontalface_default.xml"))

# Para utilizar la webcam. 
cap = cv2.VideoCapture(0)

# Utilizar video como input 
# cap = cv2.VideoCapture(os.path.join(dir_path, "..", "video", "video_2.mp4"))





    