import cv2 as cv
import os
import imutils
import matplotlib.pyplot as plt
import numpy as np
import easyocr
import pandas as pd

# Obtener la ruta del directorio actual
dir_path = os.path.dirname(os.path.realpath(__file__))

# Obtener la lista de archivos de imágenes en la carpeta 'plates'
fotos = os.listdir(dir_path + '/static/plates')

# Lista para almacenar las matrículas detectadas
matriculas = []

