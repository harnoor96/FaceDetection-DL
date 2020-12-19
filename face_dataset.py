
from cv2 import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n Enter User ID: ')
face_name = input('\n Enter First Name: ')

print("\n [INFO] Initializing face capture...")

# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert images to grayscale
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces: #x,y - coordinates, w - width, h - height

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(face_name) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # wait for sometime for user to adjust and capture more images
    if k == 30: # timeout for capturing all images
        break
    elif count >= 30: 
         break

# Do a bit of cleanup
print("\n[INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()


#cv2.rectangle(image, start_point, end_point, color, thickness)
#image: It is the image on which rectangle is to be drawn.
#start_point: It is the starting coordinates of rectangle. The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
#end_point: It is the ending coordinates of rectangle. The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
#color: It is the color of border line of rectangle to be drawn. For BGR, we pass a tuple. eg: (255, 0, 0) for blue color.
#thickness: It is the thickness of the rectangle border line in px. Thickness of -1 px will fill the rectangle shape by the specified color.