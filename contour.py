
# import cv2 as cv
# import matplotlib.pyplot as plt

# # Read the parrot image
# parrot_img = cv.imread('D:\Data Science\Computer vision\parrot.jpg')

# # Convert the image to grayscale
# gray_img = cv.cvtColor(parrot_img, cv.COLOR_BGR2GRAY)

# # Find contours in the grayscale image
# contours, _ = cv.findContours(gray_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# # Draw the contours on a copy of the original image
# contour_img = parrot_img.copy()
# cv.drawContours(contour_img, contours, -1, (0, 255, 0), 12)



# # Display the original image and the image with contours
# plt.imshow(cv.cvtColor(contour_img, cv.COLOR_BGR2RGB))
# plt.title('Parrot Image with Contours')
# plt.show()



import cv2 as cv
import matplotlib.pyplot as plt

# Read the parrot image
parrot_img = cv.imread('D:\Data Science\Computer vision\parrot.jpg')

# Convert the image to grayscale
gray_img = cv.cvtColor(parrot_img, cv.COLOR_BGR2GRAY)

# Find contours in the grayscale image
contours, _ = cv.findContours(gray_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw the contours on a copy of the original image
contour_img = parrot_img.copy()
cv.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# Draw the approximated contours on a separate image
contour_img_approx = parrot_img.copy()
for contour in contours:
    epsilon = 0.1 * cv.arcLength(contour, True)  # Adjust epsilon value for noticeable simplification
    approx = cv.approxPolyDP(contour, epsilon, True)
    cv.drawContours(contour_img_approx, [approx], -1, (0, 255, 0), 2)

# Display the original image and the image with contours
plt.imshow(cv.cvtColor(contour_img, cv.COLOR_BGR2RGB))
plt.title('Parrot Image with Contours')
plt.show()

# Display the image with approximated contours
plt.imshow(cv.cvtColor(contour_img_approx, cv.COLOR_BGR2RGB))
plt.title('Parrot Image with Approximated Contours')
plt.show()
