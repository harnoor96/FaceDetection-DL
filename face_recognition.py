from cv2 import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

cascadePath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_TRIPLEX

id = 0

# names = ['Harnoor', 'Bhavna']

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3) # Setting minimum window width
minH = 0.1 * cam.get(4) # Setting minimum window height

while(True):
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 2,
        minSize = (int(minW), int(minH))
    )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if(confidence < 100):
            id = id #names[id]
            confidence = "{0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "{0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+5, y), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 2)
    
    cv2.imshow('camera', img)

    timeout = cv2.waitKey(10) & 0xff
    print(timeout)
    if timeout == 30:
        break

print('\n [INFO] Exiting program...')
cam.release()
cv2.destroyAllWindows()