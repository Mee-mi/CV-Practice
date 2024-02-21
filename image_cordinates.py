import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img= cv.imread('Computer vision/car.jpeg')
gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.imshow(gray)
plt.show()