# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # Read the image
# image = cv2.imread('Computer vision\circle.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply thresholding to create a binary image
# _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# # Perform morphological operations to remove noise and improve segmentation
# kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# # Sure background area
# sure_bg = cv2.dilate(opening, kernel, iterations=3)

# # Distance Transform
# dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg, sure_fg)

# # Marker labeling
# _, markers = cv2.connectedComponents(sure_fg)

# # Add 1 to all labels so that sure background is not 0, but 1
# markers = markers + 1

# # Mark the region of unknown with 0
# markers[unknown == 255] = 0

# # Apply Watershed Algorithm
# cv2.watershed(image, markers)

# # Overlay the segmentation result on the original image
# image[markers == -1] = [0, 0, 255]

# # Display the results
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title('Watershed Algorithm for Image Segmentation')
# plt.show()




import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread('Computer vision\circle.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the original image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Apply thresholding to create a binary image
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Display the binary image after thresholding
plt.imshow(thresh, cmap='gray')
plt.title('Binary Image after Thresholding')
plt.show()

# Perform morphological operations to remove noise and improve segmentation
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Display the image after morphological operations
plt.imshow(opening, cmap='gray')
plt.title('Image after Morphological Operations')
plt.show()

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Display the image after dilating for sure background
plt.imshow(sure_bg, cmap='gray')
plt.title('Image after Dilation for Sure Background')
plt.show()

# Distance Transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Display the image after distance transform
plt.imshow(sure_fg, cmap='gray')
plt.title('Image after Distance Transform')
plt.show()

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Display the image of the unknown region
plt.imshow(unknown, cmap='gray')
plt.title('Unknown Region')
plt.show()

# Marker labeling
_, markers = cv2.connectedComponents(sure_fg)

# Add 1 to all labels so that sure background is not 0, but 1
markers = markers + 1

# Mark the region of unknown with 0
markers[unknown == 255] = 0

# Apply Watershed Algorithm
cv2.watershed(image, markers)

# Overlay the segmentation result on the original image
image[markers == -1] = [0, 0, 255]

# Display the final segmented image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Final Segmented Image')
plt.show()
