# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import numpy as np
import cv2 as cv
import pickle as pkl

cascade_face = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
cascade_eye = cv.CascadeClassifier('haarcascade_eye.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pkl.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv.VideoCapture(1)

while(True):
    
    ret, frm = cap.read()
    gray = cv.cvtColor(frm, cv.COLOR_BGR2GRAY)
    faces = cascade_face.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    #Get Value From Detection Faces
    for(x, y, w, h) in faces:
        #print(x, y, w, h)
        f_roi_gray = gray[y:y+h, x:x+w]
        f_roi_color = frm[y:y+h, x:x+w]
        
        #Get Trainning File
        id_, conf = recognizer.predict(f_roi_gray)
        if conf>=0 and conf<=10000:
            print(id_)
            print(labels[id_])
            font = cv.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 0)
            stroke = 2
            cv.putText(frm, name, (x,y), font, 1.2, color, stroke, cv.LINE_AA)
        
        #f_img_item_gray = "myImageGrayFace.jpg"
        f_img_item_color = "myImageColorFace.jpg"
        #cv.imwrite(f_img_item_gray, f_roi_gray)
        cv.imwrite(f_img_item_color, f_roi_color)
        
        #draw box
        color = (0, 255, 0)
        stroke = 2
        coorX = x + w
        coorY = y + h
        cv.rectangle(frm, (x, y), (coorX, coorY), color, stroke)
        
    cv.imshow('Frame', frm)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

