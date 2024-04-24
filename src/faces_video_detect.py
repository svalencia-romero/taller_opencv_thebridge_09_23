import os
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))

# Load the cascade
face_cascade = cv2.CascadeClassifier(os.path.join(dir_path, "..", "xml", "haarcascade_frontalface_default.xml"))

# Para utilizar la webcam. 
cap = cv2.VideoCapture(0)

# Utilizar video como input 
# cap = cv2.VideoCapture(os.path.join(dir_path, "..", "video", "video_2.mp4"))

while True:
    _,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 2)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('imagen',img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

        
# Release the VideoCapture object
cap.release()



    