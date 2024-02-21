# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # Read the image
# image = cv2.imread('Computer vision\images.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply edge detection (you can use Canny or other edge detection methods)
# edges = cv2.Canny(gray, 50, 150)

# # Perform Hough Line Transform
# lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

# # Draw detected lines on the original image
# for line in lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * (a))
#     x2 = int(x0 - 1000 * (-b))
#     y2 = int(y0 - 1000 * (a))
#     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

# # Display the results
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title('Hough Line Transform')
# plt.show()


import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread('Computer vision\circle.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection (you can use Canny or other edge detection methods)
edges = cv2.Canny(gray, 50, 150)

# Perform Hough Circle Transform
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=50)

# Draw detected circles on the original image
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 1)

# Display the results
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Hough Circle Transform')
plt.show()
