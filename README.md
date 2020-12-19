# FaceDetection-DL
Libraries required:
OpenCV, NumPy, Pillow, Face
To install OpenCV: pip install opencv-python, pip install opencv-contrib-python
To install NumPy: pip install numpy
To install Pillow: pip install pillow
To install Face: pip install face

HAARCASCADE
Haar Cascade is a machine learning object detection algorithm used to identify objects in an image or video
In this project, we will use Haarcascade 'Frontal Face' algorithm to detect faces in the image

LBPH Face Recognizer algorithm
The local binary pattern histogram(LBPH) algorithm is a simple solution on face recognition problem, which can recognize both front face and side face.
In this project, we will use the pre-trained model stored in the form of .yml file.

Folder structure
-
    Face_dataset.py
    training.py
    Face_recognition.py
    haarcascade_frontalface_default.xml
    - Dataset
    - Trainer
        trainer.xml