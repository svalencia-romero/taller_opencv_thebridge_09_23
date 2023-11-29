import os
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))

face_cascade = cv2.CascadeClassifier(os.path.join(dir_path, "..", "xml", "haarcascade_frontalface_default.xml"))

# Read the input image
img = cv2.imread(os.path.join(dir_path, "..", "img", "faces", "180006.jpg"))



