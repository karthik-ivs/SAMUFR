import cv2
import numpy as np
capture = cv2.VideoCapture(0) 
while True:
    ret, frame = capture.read() 
    if not ret:
        break
    face_cascade = cv2.CascadeClassifier("/Users/ivskarthik/Documents/vscode/pythonprojects/IDP/haarcascade_frontalface_alt2.xml")
    face_reacts = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    print(face_reacts)
    for (x, y, w, h) in face_reacts:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.waitKey(0)
cv2.destroyAllWindows()