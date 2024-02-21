# #Bruite force matcher

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # Load images
# img1 = cv2.imread('image1.jpg', 0)  # query image
# img2 = cv2.imread('image2.jpg', 0)  # train image

# # Initiate ORB detector
# orb = cv2.ORB_create()

# # Find the keypoints and descriptors with ORB
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)

# # Create a Brute-Force Matcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# # Match descriptors
# matches = bf.match(des1, des2)

# # Sort them in ascending order of their distances
# matches = sorted(matches, key=lambda x: x.distance)

# # Draw the first 10 matches
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

# # Display the result
# plt.imshow(img3)
# plt.show()


#####################################################################################
#FLANN  based Matcher:

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # Load images
# img1 = cv2.imread('Computer vision/query.jpeg', 0)  # query image
# img2 = cv2.imread('Computer vision/img (2).jpeg', 0)  # train image

# # Initiate SIFT detector
# sift = cv2.SIFT_create()

# # Find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)

# # FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)  # or pass an empty dictionary

# flann = cv2.FlannBasedMatcher(index_params, search_params)

# # Match descriptors
# matches = flann.knnMatch(des1, des2, k=2)

# # Need to draw only good matches, so create a mask
# matchesMask = [[0, 0] for i in range(len(matches))]

# # Ratio test as per Lowe's paper
# for i, (m, n) in enumerate(matches):
#     if m.distance < 0.7 * n.distance:
#         matchesMask[i] = [1, 0]

# draw_params = dict(matchColor=(0, 255, 0),
#                    singlePointColor=(255, 0, 0),
#                    matchesMask=matchesMask,
#                    flags=0)

# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

# # Display the result
# plt.imshow(img3)
# plt.show()

#####################################################################################
#Meanshift


# import cv2 as cv
# import numpy as np

# # Capture video
# cap = cv.VideoCapture('Computer vision/pexels_videos_1572547 (1440p).mp4')  # Replace 'your_video.mp4' with the path to your video file

# # Create background subtractor
# fgbg = cv.createBackgroundSubtractorMOG2()

# # Set up initial values
# ret, frame = cap.read()
# r, h, c, w = 1100, 500, 1300, 600 # Initial window location
# track_window = (c, r, w, h)
# roi = frame[r:r+h, c:c+w]
# hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
# roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
# cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
# term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

# # Create 'output' window
# cv.namedWindow('output', cv.WINDOW_NORMAL)

# # Retrieve video dimensions
# width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# # Set dimensions for 'output' window
# cv.resizeWindow('output', width, height)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Apply background subtraction
#     fgmask = fgbg.apply(frame)

#     # Apply meanshift to get the new location
#     dst = cv.calcBackProject([frame], [0], roi_hist, [0, 180], 1)
#     ret, track_window = cv.meanShift(dst, track_window, term_crit)

#     # Draw the tracking window on the image
#     x, y, w, h = track_window
#     img2 = cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)

#     # Display the result
#     # cv.imshow('Frame', frame)
#     # cv.imshow('Foreground', fgmask)
#     # cv.imshow('Tracking', img2)

#     # Display in the 'output' window
#     cv.imshow('output', frame)

#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break

# cap.release()
# cv.destroyAllWindows()


#####################################################################################
#craft shift
import numpy as np
import cv2 as cv

# Load your video
cap = cv.VideoCapture('Computer vision/pexels_videos_1572547 (1440p).mp4')

# Create a window named 'output'
cv.namedWindow('output', cv.WINDOW_NORMAL)

# Retrieve the video's width and height
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Set the dimensions of the output window
cv.resizeWindow('output', width, height)

# Set up the initial window location for mean-shift and cam-shift
r, h, c, w = r, h, c, w = 1000, 500, 1100, 600 # Initial window location
track_window = (c, r, w, h)

# Set up the ROI for tracking
roi = None

# Create flag for mean-shift and cam-shift
mean_shift = True

# Set up termination criteria
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Display the current frame in the 'output' window
    cv.imshow('output', frame)

    if roi is None:
        # Initialize the ROI in the first frame
        roi = frame[r:r+h, c:c+w]
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    # Apply mean-shift or cam-shift to get the new location and size
    if mean_shift:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
    else:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv.CamShift(dst, track_window, term_crit)

    # Draw the tracking window on the image
    if mean_shift:
        x, y, w, h = track_window
        img2 = cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
    else:
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame, [pts], True, 255, 2)

    # Display the frame with tracking
    cv.imshow('output', img2)

    # Break the loop if 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv.destroyAllWindows()
