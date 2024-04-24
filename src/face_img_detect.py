import os
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))

face_cascade = cv2.CascadeClassifier(os.path.join(dir_path, "..", "xml", "haarcascade_frontalface_default.xml"))


img = cv2.imread(os.path.join(dir_path, "..", "img", "faces", "180006.jpg"))

# Convertimos a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Con esto conseguimos difuminar la imagen y asi detectar mejor las imagenes
img_filtered = cv2.blur(img, (3, 3))

# Detectamos las caras
faces = face_cascade.detectMultiScale(img_filtered, 1.1, 4)

# Dibujamos rectangulo en cada cara
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output

cv2.imshow('img', img)
cv2.waitKey()

