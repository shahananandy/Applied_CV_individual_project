import cv2
import keras
# from keras.models import load_model
import numpy as np
from pygame import mixer


mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_alt.xml')
left = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_lefteye_2splits.xml')
right = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_righteye_2splits.xml')
rpred = [5]
lpred = [5]

model = keras.models.load_model('model.h5')
cap = cv2.VideoCapture(0)
drowsy_list = []
while(True):
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    left_eye = left.detectMultiScale(gray)
    right_eye =  right.detectMultiScale(gray)

    print(left_eye, "left_eye")
    if len(left_eye) >= 1:
        l_eye=image[left_eye[0][1]:left_eye[0][1]+left_eye[0][3],left_eye[0][0]:left_eye[0][0]+left_eye[0][2]]
        l_eye = cv2.resize(cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY),(24,24))/255
        l_eye = np.expand_dims(l_eye.reshape(24,24,-1),axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=-1)

    if len(right_eye) >= 1:
        r_eye=image[right_eye[0][1]:right_eye[0][1]+right_eye[0][3],right_eye[0][0]:right_eye[0][0]+right_eye[0][2]]
        r_eye = cv2.resize(cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY),(24,24))/255
        r_eye = np.expand_dims(r_eye.reshape(24,24,-1),axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=-1)


    drowsy_list.append((rpred[0]==0 and lpred[0]==0))
    if len(drowsy_list) > 5:
            drowsy_list.pop(0)
    if sum(drowsy_list) >= 3:
        sound.play()
    else:
        sound.stop()
    cv2.imshow('output',image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
