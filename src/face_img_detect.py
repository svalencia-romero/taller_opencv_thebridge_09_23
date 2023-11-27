### Comprobado ok 1
import os
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))

face_cascade = cv2.CascadeClassifier(os.path.join(dir_path, "..", "xml", "haarcascade_frontalface_default.xml"))

# Read the input image
img = cv2.imread(os.path.join(dir_path, "..", "img", "faces", "22.jpeg"))

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Con esto conseguimos difuminar la imagen y asi detectar mejor las imagenes
img_filtered = cv2.blur(img, (3, 3))

# Detect faces
faces = face_cascade.detectMultiScale(img_filtered, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output

cv2.imshow('img', img)
cv2.waitKey()

