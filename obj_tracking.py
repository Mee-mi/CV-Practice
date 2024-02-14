import cv2 as cv
import numpy as np

parrot_img= cv.imread('D:/Data Science/Computer vision/parrot.jpg')

while True:
    #convert image to hsv
    hsv= cv.cvtColor(parrot_img, cv.COLOR_BGR2HSV)

    # Define range of yellow-green color in HSV
    lower_yellow_green = np.array([40, 40, 40])
    upper_yellow_green = np.array([80, 255, 255])
    mask_green= cv.inRange(hsv, lower_yellow_green, upper_yellow_green)
    res_green= cv.bitwise_and(parrot_img, parrot_img, mask=mask_green)

    ## Define range of red color in HSV
    lower_red= np.array([0, 100, 100])
    upper_red= np.array([10, 255, 255])
    mask_red= cv.inRange(hsv, lower_red, upper_red)
    res_red= cv.bitwise_and(parrot_img, parrot_img, mask=mask_red)

    # Combine results for red and green
    result = cv.addWeighted(res_red, 1, res_green, 1, 0)  #alpha, beta and gamma parameter rn we have only two so turn two 1's

    # Display the original image and results
    cv.imshow('Original Image', parrot_img)
    cv.imshow('Red Mask', mask_red)
    cv.imshow('Green Mask', mask_green)
    cv.imshow('Red Result', res_red)
    cv.imshow('Green Result', res_green)
    cv.imshow('Combined Result', result)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()


#object tracking with more colors