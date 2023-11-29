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
for img_file in image_files:

    image_path = os.path.join(carpeta_img, img_file)
    
    img = cv2.imread(image_path)  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Recortamos la imagen
    alto, ancho, _ = img.shape

    margen = 18
    # Definir las coordenadas para el recorte
    x1 = margen
    y1 = margen
    x2 = ancho - margen
    y2 = alto - margen

    # Recortar la imagen
    img = img[y1:y2, x1:x2]

    # Cambiamos el tamaño 
    img = cv2.resize(img, (620,480) )
    

    # Convertimos a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # Escala de grises
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
    

    # Contamos los contornos cerrados que hay en la imagen
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:8]
    screenCnt = None

    # Aquí se muestra cómo dibujar los contornos en la imagen original.
    image_with_contours = img.copy()
    cv2.drawContours(image_with_contours, cnts, -1, (0, 255, 0), 2)  # Dibuja todos los contornos en verde

    # Hacemos un bucle por todos los contornos que tenemos 
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        # si nuestro contorno mas similar tiene 4 lados, entonces
        # asumimos que lo hemos encontrado, por eso paramos
        if len(approx) == 4:
            screenCnt = approx
            break

    x,y,w,h = cv2.boundingRect(c)
    epsilon = 0.9*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,epsilon,True)               

    mask = np.zeros_like(img)
    
    if screenCnt is not None and len(screenCnt) > 0:
    # Dibuja el contorno en la máscara
        cv2.drawContours(mask, [screenCnt], -1, (255, 255, 255), -1)
    else:
        print("No se detectó un contorno válido.")
        continue

    # Aplica la máscara para obtener solo la región de la matrícula
    license_plate = cv2.bitwise_and(img, mask)                

    indices = np.where(mask == 255)
    x = indices[0]
    y = indices[1]
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    # Topx +1 recortamos desde arriba, Bottomx +1 alargamos hacia abajo (Vertical)
    # Topy +1 acorta hacia la derecha, Bottomy +1 alarga hacia la derecha (Horizontal)
    Cropped = gray[topx:bottomx, topy:bottomy]
    reader = easyocr.Reader(['es'])
    result = reader.readtext(Cropped)
    if result and len(result) > 0:
        matriculas.append(result[0][1])
    else:
        print("No se detectó texto en la región de la matrícula.")
print(matriculas)
