import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image in grayscale
image = cv2.imread('Computer vision\parrot.jpg', cv2.IMREAD_GRAYSCALE)
# Apply the 2D Fourier Transform
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)

# Calculate magnitude spectrum for visualization
magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)

# Display the original image and its magnitude spectrum
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Magnitude Spectrum')
plt.show()
