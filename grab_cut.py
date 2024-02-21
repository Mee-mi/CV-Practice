import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread('Computer vision/Kheops-Pyramid.jpg')
original = image.copy()

# Create a mask and initialize it with zeros
mask = np.zeros(image.shape[:2], np.uint8)

# Set the rectangle region for initial segmentation
rect = (129, 112, 450, 550)

# Initialize the foreground and background models
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# Apply GrabCut algorithm
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Modify the mask to create a binary mask for foreground and background
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Apply the binary mask to the original image
result = original * mask2[:, :, np.newaxis]

# Display the results
plt.figure(figsize=(10, 5))

plt.subplot(131), plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(132), plt.imshow(mask, cmap='gray'), plt.title('Initial Mask')
plt.subplot(133), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Segmented Image')

plt.show()

