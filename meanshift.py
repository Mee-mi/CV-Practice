


import cv2 as cv
import numpy as np

# Capture video
cap = cv.VideoCapture('Computer vision/pexels_videos_1572547 (1440p).mp4')  # Replace 'your_video.mp4' with the path to your video file

# Create background subtractor
fgbg = cv.createBackgroundSubtractorMOG2()

# Set up initial values
ret, frame = cap.read()
r, h, c, w = 1100, 500, 1300, 600 # Initial window location
track_window = (c, r, w, h)
roi = frame[r:r+h, c:c+w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

# Create 'output' window
cv.namedWindow('output', cv.WINDOW_NORMAL)

# Retrieve video dimensions
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Set dimensions for 'output' window
cv.resizeWindow('output', width, height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Apply meanshift to get the new location
    dst = cv.calcBackProject([frame], [0], roi_hist, [0, 180], 1)
    ret, track_window = cv.meanShift(dst, track_window, term_crit)

    # Draw the tracking window on the image
    x, y, w, h = track_window
    img2 = cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)

    # Display the result
    # cv.imshow('Frame', frame)
    # cv.imshow('Foreground', fgmask)
    # cv.imshow('Tracking', img2)

    # Display in the 'output' window
    cv.imshow('output', frame)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()