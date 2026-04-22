import cv2
img=cv2.imread("/Users/ivskarthik/Documents/vscode/pythonprojects/IDP/idp_1_img.jpg")
img = cv2.resize(img, (400, 400))
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
face_cascade=cv2.CascadeClassifier("/Users/ivskarthik/Documents/vscode/pythonprojects/IDP/haarcascade_frontalface_alt2.xml")
face_reacts=face_cascade.detectMultiScale(img1, scaleFactor=1.1, minNeighbors=2)
print(face_reacts)
for (x, y, w, h) in face_reacts:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()