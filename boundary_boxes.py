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

# # Draw bounding boxes around the contours
# contour_img_with_boxes = parrot_img.copy()
# for contour in contours:
#     x, y, w, h = cv.boundingRect(contour)
#     cv.rectangle(contour_img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 12)

# # Display the original image with contours
# plt.imshow(cv.cvtColor(contour_img, cv.COLOR_BGR2RGB))
# plt.title('Parrot Image with Contours')
# plt.show()

# # Display the original image with bounding boxes
# plt.imshow(cv.cvtColor(contour_img_with_boxes, cv.COLOR_BGR2RGB))
# plt.title('Parrot Image with Bounding Boxes')
# plt.show()


#Rectagnle around the eye


# import cv2 as cv
# import matplotlib.pyplot as plt

# # Read the parrot image
# parrot_img = cv.imread('D:\Data Science\Computer vision\parrot.jpg')

# # Specify the approximate x, y coordinates of the eye
# eye_x, eye_y = 307, 138
# eye_width, eye_height = 60, 50

# # Draw a bounding box around the specified eye region
# cv.rectangle(parrot_img, (eye_x, eye_y), (eye_x + eye_width, eye_y + eye_height), (0, 255, 0), 2)

# # Display the image with the bounding box around the eye
# plt.imshow(cv.cvtColor(parrot_img, cv.COLOR_BGR2RGB))
# plt.title('Parrot Image with Bounding Box Around Eye')
# plt.show()



#circle around the eye

import cv2 as cv
import matplotlib.pyplot as plt

# Read the parrot image
parrot_img = cv.imread('D:\Data Science\Computer vision\parrot.jpg')

# Specify the approximate x, y coordinates of the eye
eye_x, eye_y = 307, 138

# Experiment with adjusting center coordinates
circle_center = (eye_x + 30, eye_y + 30)

# Experiment with adjusting   
eye_radius = 30

# Draw a circle around the specified eye region
cv.circle(parrot_img, circle_center, eye_radius, (0, 255, 0), 2)

# Display the image with the adjusted circle around the eye
plt.imshow(cv.cvtColor(parrot_img, cv.COLOR_BGR2RGB))
plt.title('Parrot Image with Adjusted Circle Around Eye')
plt.show()
