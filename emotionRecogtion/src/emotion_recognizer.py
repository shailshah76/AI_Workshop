import cv2
from keras.models import load_model
import numpy

face_models_path='../trained_models/face_detection_model/haarcascade_frontalface_default.xml'
emotion_models_path='../trained_models/emotion_models_path/emotion_recog_2_0.379773691676.model'

emotion_labels=['angry','fear','happy','sad','surprise','neutral']

face_detection=cv2.CascadeClassifier(face_models_path)

emotion_model=load_model(emotion_models_path)


emotion_model_input_size=emotion_model.input_shape[1:3]

cap=cv2.VideoCapture(0)
while True:
    ret_val, frame=cap.read()
    if ret_val==True:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_detection.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
                gray_face=gray[y:y+h,x:x+w]
                gray_face=cv2.resize(gray_face,emotion_model_input_size)
                pre_processed_img=gray_face.astype('float32')
                pre_processed_img/=255
                expanded_dimen_img=numpy.expand_dims(pre_processed_img,0)
                expanded_dimen_img=numpy.expand_dims(expanded_dimen_img,-1)
                emotion_probabilities=emotion_model.predict(expanded_dimen_img)
                emotion_max_prob=numpy.max(emotion_probabilities)
                emotion_label=numpy.argmax(emotion_probabilities)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
                cv2.putText(frame,emotion_labels[emotion_label],(x,y),cv2.FONT_HERSHEY_COMPLEX,4,(0,255,0),10)
                
        cv2.imshow("emotion_recognition",frame)
        key_pressed=cv2.waitKey(1) & 0xFF
        if key_pressed==27:
            break
cv2.destroyAllWindows()
cap.release()

