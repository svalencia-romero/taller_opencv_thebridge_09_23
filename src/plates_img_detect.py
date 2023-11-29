import cv2
import os
import imutils
import matplotlib.pyplot as plt
import numpy as np
import easyocr
import pandas as pd
# Obtener la ruta del directorio actual
dir_path = os.path.dirname(os.path.realpath(__file__))

# Obtener la lista de archivos de imágenes en la carpeta 'plates'
# fotos = os.listdir(dir_path + '/img/plates')

# Lista para almacenar las matrículas detectadas
matriculas = []

carpeta_img = (os.path.join(dir_path, "..", "img", "plates"))
image_files = [f for f in os.listdir(carpeta_img) if f.endswith(('.jpeg', '.jpg', '.png'))]
print(image_files)

