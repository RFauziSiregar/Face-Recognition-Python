# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import numpy as np
import cv2 as cv
import pickle as pkl

cascade_face = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
cascade_eye = cv.CascadeClassifier('haarcascade_eye.xml')

cap = cv.VideoCapture(1)

while(True):
    
    ret, frm = cap.read()
    gray = cv.cvtColor(frm, cv.COLOR_BGR2GRAY)
    faces = cascade_face.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    #Get Value From Detection Faces
    for(x, y, w, h) in faces:
        print(x, y, w, h)
        f_roi_gray = gray[y:y+h, x:x+w]
        f_roi_color = frm[y:y+h, x:x+w]
        f_img_item_gray = "myImageGrayFace.jpg"
        f_img_item_color = "myImageColorFace.jpg"
        cv.imwrite(f_img_item_gray, f_roi_gray)
        cv.imwrite(f_img_item_color, f_roi_color)
        
        #draw box
        color = (255, 0, 0)
        stroke = 2
        coorX = x + w
        coorY = y + h
        cv.rectangle(frm, (x, y), (coorX, coorY), color, stroke)
        
        #Get Value From Eyes
        eyes = cascade_eye.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for(x, y, w, h) in eyes:
            print(x, y, w, h)
            e_roi_gray = gray[y:y+h, x:x+w]
            e_roi_color = frm[y:y+h, x:x+w]
            e_img_item_gray = "myImageGrayEye.jpg"
            e_img_item_color = "myImageColorEye.jpg"
            cv.imwrite(e_img_item_gray, e_roi_gray)
            cv.imwrite(e_img_item_color, e_roi_color)
            #draw box
            color = (255, 0, 255)
            stroke = 2
            coorX = x + w
            coorY = y + h
            cv.rectangle(frm, (x, y), (coorX, coorY), color, stroke)
    
    cv.imshow('Frame', frm)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

    
    


