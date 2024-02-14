import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv.imread('D:\Data Science\Computer vision\parrot.jpg', cv.IMREAD_GRAYSCALE)

'''
# Create a rectangular kernel
kernel = np.ones((5, 5), np.uint8)

# Apply erosion
erosion = cv.erode(img, kernel, iterations=1)

# Display the results
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(erosion, cmap='gray'), plt.title('Erosion')
plt.show()
'''

'''
#thresholding
def on_trackbar_low(value):
    global low_threshold
    low_threshold = value
    update_canny()

def on_trackbar_high(value):
    global high_threshold
    high_threshold = value
    update_canny()

def update_canny():
    edges = cv.Canny(image, low_threshold, high_threshold)
    cv.imshow('Canny Edge Detection', edges)

# Read the image
image = cv.imread('D:\Data Science\Computer vision\parrot.jpg', cv.IMREAD_GRAYSCALE)

# Initialize threshold values
low_threshold = 50
high_threshold = 150

# Create a window for the trackbars
cv.namedWindow('Canny Edge Detection')

# Create trackbars for low and high thresholds
cv.createTrackbar('Low Threshold', 'Canny Edge Detection', low_threshold, 255, on_trackbar_low)
cv.createTrackbar('High Threshold', 'Canny Edge Detection', high_threshold, 255, on_trackbar_high)

# Initialize the Canny edge detection
update_canny()

# Wait for the user to close the window
cv.waitKey(0)
cv.destroyAllWindows()

'''

#image blending
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read two images
img1 = cv.imread('D:\Data Science\Computer vision\bottlecap.jpg')
img2 = cv.imread('D:\Data Science\Computer vision\parrot.jpg')

# Resize images to the same dimensions (optional)
print(img1.shape)
print(img2.shape)
#img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))
# Blend images using addWeighted function
alpha = 0.5  # Weight for the first image
beta = 0.5   # Weight for the second image
gamma = 0    # Scalar added to each sum
blended = cv.addWeighted(img1, alpha, img2, beta, gamma)

# Display the results
plt.subplot(131), plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB)), plt.title('Image 1')
plt.subplot(132), plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB)), plt.title('Image 2')
plt.subplot(133), plt.imshow(cv.cvtColor(blended.astype(np.uint8), cv.COLOR_BGR2RGB)), plt.title('Blended Image')
plt.show()
