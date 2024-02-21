# import cv2
# import numpy as np

# # Read the image
# image = cv2.imread('Computer vision\parrot.jpg')

# # Define the Region of Interest (ROI) coordinates (example coordinates, adjust as needed)
# x, y, width, height = 100, 50, 150, 120
# roi = image[y:y+height, x:x+width]

# # Calculate the average color of the ROI
# average_color = np.mean(roi, axis=(0, 1))

# # Display the average color
# print("Average Color (BGR):", average_color)

# # Convert BGR to RGB for displaying with matplotlib
# average_color_rgb = average_color[::-1]  # Reverse the order (BGR to RGB)
# print("Average Color (RGB):", average_color_rgb)

# # Display the ROI and the entire image
# cv2.imshow('ROI', roi)
# cv2.imshow('Original Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##extreme points

import cv2
import numpy as np

# Read the parrot image
parrot_img = cv2.imread('Computer vision\parrot.jpg')

# Assuming 'cnt' is the contour of the parrot (modify as needed)
# Example contour points for demonstration purposes
cnt = np.array([[[37, 183]], [[366, 40]], [[563, 248]], [[354, 484]]])

# Find extreme points
leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

# Draw circles at the extreme points on the image
cv2.circle(parrot_img, leftmost, 5, (0, 255, 0), -1)       # Draw circle at leftmost
cv2.circle(parrot_img, rightmost, 5, (0, 255, 0), -1)      # Draw circle at rightmost
cv2.circle(parrot_img, topmost, 5, (0, 255, 0), -1)        # Draw circle at topmost
cv2.circle(parrot_img, bottommost, 5, (0, 255, 0), -1)     # Draw circle at bottommost   # Draw line from topmost to rightmost

# Display the image with the lines connecting the extreme points
cv2.imshow('Parrot Image with Extreme Points', parrot_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

