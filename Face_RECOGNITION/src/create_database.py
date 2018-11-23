import cv2
import os
import time


count = 0
person_name = input("Enter your name")
database_dir = 'C:/Users/Dell/Desktop/AI Workshop/Face_RECOGNITION/database'
path = os.path.join(database_dir, person_name)
if not os.path.isdir(path):
    os.mkdir(path)


face_cascade=cv2.CascadeClassifier("../trained_models/haarcascade_frontalface_default.xml")
cap=cv2.VideoCapture(0)
while (count < 100):
    ret_val,frame=cap.read()
    if ret_val == True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#        gray = cv2.resize(gray,(100,100))
        faces = face_cascade.detectMultiScale(gray,1.1,5)
        if list(faces):
            face = max(faces, key = lambda x : x[2]*x[3])
            roi = gray[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
            roi = cv2.resize(roi,(100,100))
            cv2.imwrite(path+"/"+str(count)+".jpg",roi)
            
        time.sleep(2)
        count+=1
        
        cv2.imshow("IMAGE", gray)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()
            
