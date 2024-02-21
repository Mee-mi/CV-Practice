import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load images
img1 = cv2.imread('image1.jpg', 0)  # query image
img2 = cv2.imread('image2.jpg', 0)  # train image

# Initiate ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Create a Brute-Force Matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort them in ascending order of their distances
matches = sorted(matches, key=lambda x: x.distance)

# Draw the first 10 matches
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

# Display the result
plt.imshow(img3)
plt.show()