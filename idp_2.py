import cv2
import numpy as np
import matplotlib.pyplot as plt
img1=cv2.imread("/Users/ivskarthik/Documents/vscode/pythonprojects/IDP/idp_1_img.jpg")
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

img1=cv2.resize(img1,(400,400))
plt.imshow(img1)
plt.show()
img1=cv2.rectangle(img1,(18,53),(92,179),(0,255,0),2)
img1=cv2.rectangle(img1,(181,20),(243,111),(0,255,0),2)
img1=cv2.rectangle(img1,(288,62),(342,119),(0,255,0),2)
cv2.imshow("Image", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()