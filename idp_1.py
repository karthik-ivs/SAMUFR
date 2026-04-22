import numpy as np
import cv2
Image = cv2.imread('/Users/ivskarthik/Documents/vscode/pythonprojects/IDP/idp_1_img.jpg')
Image = cv2.resize(Image, (400, 400)) 
print(Image)
cv2.imshow('Modified Image', Image)
cv2.waitKey(0)
cv2.destroyAllWindows()