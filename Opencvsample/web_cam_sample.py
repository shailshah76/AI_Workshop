import cv2
import numpy as np
from keras.models import load_model

face_models_path = '../trained_models/face_detection_models/haarcascade_frontalface_default.xml'
emotion_models_path = '../trained_models/emotion_models_path/emotion_modelsemotion_recog_2_0.334512022652.model'

emotion_labels = ['angry','fear','happy','sad','surprise','neutral']
face_detection = cv2.CascadeClassifier(face_models_path)
emotion_model = load_model(emotion_models_path)
emotion_model_input_size = emotion_models_path.input_shape[1:3]

cap=cv2.VideoCapture(0)

while True:
    ret_val,frame=cap.read()#to read the dynamic input from the camera
    if ret_val == True:
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMulitiScale(gray,1.3,5)
        for x,y,w,h in faces:
            gray_face = gray[y:y+h,x:x+w]
            gray_face = cv2.resize(gray_face, emotion_model_input_size)
            pre_processed_img = gray_face.astype('float32')
            pre_processed_img /= 255
            expanded_dimen_img = np.expand_dims(pre_processed_img,0)
            expanded_dimen_img = np.expand_dims(expanded_dimen_img,-1)
            emotion_probabilities = emotion_model.predict(expanded_dimen_img)
            emotion_max_prob = np.max(emotion_probabilities)
            emotion_label = np.argmax(emotion_probabilities)
            
            cv2.rectangle(gray,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(gray,emotion_labels[emotion_label],(x,y),
                       cv2.FONT_HERSHEY_COMPLEX,4,(0,255,0),10)
        cv2.imshow("emotion_recognition",frame)
        if cv2.waitkey(1) == 27:
            break
        
