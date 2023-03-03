import cv2
import dlib
from imutils import face_utils
from playsound import playsound
from pygame import mixer
import multiprocessing

def dist(x,y):
    return ((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5

mixer.init()
alarm = mixer.Sound('alarm_2.wav')



cap = cv2.VideoCapture(0)
drowsy_list = []
find = dlib.get_frontal_face_detector()
landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    _, image = cap.read()
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frames = find(grey, 0)

    for frame in frames:
        face = face_utils.shape_to_np(landmarks(grey, frame))

        for (a, b) in face:
            cv2.circle(image, (a, b), 2, (0, 0, 255), -1)
        drowsy_list.append((dist(face[37],face[41]) + dist(face[43],face[47]) + dist(face[44],face[46]) + dist(face[38],face[40]) ) / 4 < 8)
        if len(drowsy_list) > 8:
            drowsy_list.pop(0)
        if sum(drowsy_list) >= 3:
            alarm.play()
        else:
            alarm.stop()
            

    cv2.imshow("Image", image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
