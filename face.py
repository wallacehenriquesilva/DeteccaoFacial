import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')

img = cv2.imread('img/face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    cv2.rectangle(img,(x,y),(x+w,y-16),(255,100,0),-1)
    #cv2.putText(img, 'Bruna', (x+10, y-2), cv2.FONT_HERSHEY_PLAIN,1.2,(255, 255, 255))

    i = 0
    listEyesEx = []
    listEyesEy = []
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,100,0),1)
        #scv2.line(roi_color, (ex, ey), (ew, eh), (255,100,0), 1)
        #cv2.rectangle(roi_color,(ex,ey),(ew, eh),(255,100,0),-1)
        listEyesEx.append(ex)
        listEyesEy.append(ey)
    val = listEyesEx[1] - listEyesEx[0]
    if(val == 92):
        cv2.putText(img, 'Bruno', (x+10, y-2), cv2.FONT_HERSHEY_PLAIN,1.2,(255, 255, 255))
    elif(val == 50):
    	cv2.putText(img, 'Bruna', (x+10, y-2), cv2.FONT_HERSHEY_PLAIN,1.2,(255, 255, 255))

print(val)

cv2.imshow('Face com partes detectadas',img)
cv2.waitKey(0)
cv2.destroyAllWindows()