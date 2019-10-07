# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import os 
import numpy as np
import cv2 as cv
import pickle as pkl
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()

y_labels = []
x_train = []
current_id = 0
label_id = {}

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            print(label, path)
            
            if not label in label_id:
                label_id[label] = current_id
                current_id += 1
            id_ = label_id[label]
            print(label_id)
            
            #pil_image = Image.open(path)
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
            print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            
            for(x, w, y, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

#print(y_labels)
#print(x_train)

with open("labels.pickle", 'wb') as f:
    pkl.dump(label_id, f)
            
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")

