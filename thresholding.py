import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read the image in grayscale
img = cv.imread('D:\Data Science\Computer vision\parrot.jpg', cv.IMREAD_GRAYSCALE)
'''
# Apply simple thresholding
_, simple_threshold = cv.threshold(img, 128, 255, cv.THRESH_BINARY)

# Display the results
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(simple_threshold, cmap='gray'), plt.title('Simple Thresholding')
plt.show()
'''

'''
# Apply adaptive thresholding
adaptive_threshold = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

# Display the results
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(adaptive_threshold, cmap='gray'), plt.title('Adaptive Thresholding')
plt.show()
'''

# Apply Otsu's thresholding
_, otsu_threshold = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Display the results
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(otsu_threshold, cmap='gray'), plt.title("Otsu's Thresholding")
plt.show()