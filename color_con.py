#color conversion GREY SCALE

import cv2 
import numpy as np
'''
bgr_img= cv2.imread('D:\Data Science\Computer vision\parrot.jpg')
img_grey= cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

cv2.imshow('grey scale', img_grey)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#HSV

bgr_img= cv2.imread('D:\Data Science\Computer vision\parrot.jpg')
img_grey= cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

cv2.imshow('grey scale', img_grey)
cv2.waitKey(0)
cv2.destroyAllWindows()