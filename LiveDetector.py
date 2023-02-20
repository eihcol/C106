import cv2

face = cv2.CascadeClassifier("c:/users/miche/appdata/local/programs/python/python311/lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
eye = cv2.CascadeClassifier("c:/users/miche/appdata/local/programs/python/python311/lib/site-packages/cv2/data/haarcascade_eye.xml")

video = cv2.VideoCapture(0)
while(True):
    ret,frame = video.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(grey, 1.1, 5)
    eyes = eye.detectMultiScale(grey, 1.1, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x + w, y + h), (255,0,0),2)

    for (x,y,w,h) in eyes:
        cv2.rectangle(frame,(x,y), (x +w, y + h), (0,0,255),2)

    cv2.imshow("frame", frame)
    cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()