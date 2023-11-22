# import os
# import numpy as np 
# import matplotlib.pyplot as plt
# import cv2 as cv 

# # Nos posicionamos en el mismo directorio en el que queremos ejecutar el script
# dir_path = os.path.dirname(os.path.realpath(__file__))

# img = cv.imread(os.path.join(dir_path, "..", "img", "test", "23.jpeg"))
# cv.imshow('Grupo de 4 personas', img)

# #Lo convertimos a gris 
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# # La aplicamos una reduccion de tamaño
# img_small = cv.resize(gray, (800, 400))

# #Con esto conseguimos difuminar la imagen y asi detectar mejor las imagenes
# img_filtered = cv.blur(img_small, (3, 3))

# cv.imshow('Personas en gris', img_filtered)

# # Aqui llamamos a nuestro modelo entrenado para reconocer caras
# haar_cascade = cv.CascadeClassifier(os.path.join(dir_path, "..", "xml", 'haar_face.xml'))

# faces_rect = haar_cascade.detectMultiScale(img_small, scaleFactor=1.01, minNeighbors=5, minSize=(50,50))

# print(f'Numero de caras = {len(faces_rect)}')

# for (x,y,w,h) in faces_rect:
#     cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0), thickness=2)

# plt.imshow('Caras detectadas', img)
# plt.waitKey(0)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import warnings 

warnings.filterwarnings("ignore")

# Nos posicionamos en el mismo directorio en el que queremos ejecutar el script
dir_path = os.path.dirname(os.path.realpath(__file__))

img = cv.imread(os.path.join(dir_path, "..", "img", "test", "22.jpeg"))
cv.imshow('Grupo de 6 personas', img)

# Lo convertimos a gris
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Aplicamos una reducción de tamaño
img_small = cv.resize(gray, (800, 400))

# Con esto conseguimos difuminar la imagen y así detectar mejor las imágenes
img_filtered = cv.blur(img_small, (3, 3))

cv.imshow('Personas en gris', img_filtered)

# Aquí llamamos a nuestro modelo entrenado para reconocer caras
haar_cascade = cv.CascadeClassifier(os.path.join(dir_path, "..", "xml", 'haarcascade_frontalface_default.xml'))

faces_rect = haar_cascade.detectMultiScale(img_small, scaleFactor=1.01, minNeighbors=5, minSize=(50, 50))

print(f'Numero de caras = {len(faces_rect)}')

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)

# Mostrar la imagen con OpenCV
cv.imshow('Caras detectadas', img)
cv.waitKey(0)
# cv.destroyAllWindows()
"""
import os
import cv2
import PIL.Image

dir_path = os.path.dirname(os.path.realpath(__file__))
# Load the cascade
face_cascade=cv2.CascadeClassifier('os.path.join(dir_path, "..", "xml", "haarcascade_frontalface_default.xml"')

# Read the input image
img = cv2.imread(os.path.join(dir_path, "..", "img", "test", "22.jpeg"))

# Detect faces
faces = face_cascade.detectMultiScale(img, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Display the output
cv2.imwrite(os.path.join(dir_path, "..", "img", "imagenes_detectadas", "imagen_nueva.jpeg"), img)

Real=PIL.Image.open(os.path.join(dir_path, "..", "img", "imagenes_detectadas", "imagen_nueva.jpeg"))