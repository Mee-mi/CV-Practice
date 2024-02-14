import cv2 as cv
import numpy as np

#image read
img= cv.imread('D:\Data Science\Computer vision\parrot.jpg')

#scaling

# scale_factor= 0.3
# resize_img= cv.resize(img, None, fx= scale_factor, fy= scale_factor, interpolation=cv.INTER_LINEAR)

#rotation
'''
# Rotation angle in degrees
angle = 75
# Get the image center
center = (img.shape[1] // 2, img.shape[0] // 2)
# Perform rotation
rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
rotated_img = cv.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
'''
#translation
translation_matrix= np.float32([[1, 0, 50], [0, 1, 30]])
 #Perform translation
translated_img = cv.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))

# Display the original and resized images
cv.imshow('Original Image', img)
cv.imshow('Resized Image', translated_img)
cv.waitKey(0)
cv.destroyAllWindows()