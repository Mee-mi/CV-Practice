# # from __future__ import print_function
# # import sys
# # import cv2
# # from random import randint

# # trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

# # def createTrackerByName(trackerType):
# #     if trackerType == trackerTypes[0]:
# #         tracker = cv2.TrackerBoosting_create()
# #     elif trackerType == trackerTypes[1]:
# #         tracker = cv2.TrackerMIL_create()
# #     elif trackerType == trackerTypes[2]:
# #         tracker = cv2.TrackerKCF_create()
# #     elif trackerType == trackerTypes[3]:
# #         tracker = cv2.TrackerTLD_create()
# #     elif trackerType == trackerTypes[4]:
# #         tracker = cv2.TrackerMedianFlow_create()
# #     elif trackerType == trackerTypes[5]:
# #         tracker = cv2.TrackerGOTURN_create()
# #     elif trackerType == trackerTypes[6]:
# #         tracker = cv2.TrackerMOSSE_create()
# #     elif trackerType == trackerTypes[7]:
# #         tracker = cv2.TrackerCSRT_create()
# #     else:
# #         tracker = None
# #         print('Incorrect tracker name')
# #         print('Available trackers are:')
# #         for t in trackerTypes:
# #             print(t)
# #     return tracker

# # # Set video to load
# # videoPath = "Computer vision/CV-Practice/pexels-kelly-lacy-5473765 (2160p).mp4"

# # # Create a video capture object to read videos
# # cap = cv2.VideoCapture(videoPath)

# # # Get the dimensions of the video
# # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # # Create a window and resize it
# # cv2.namedWindow("MultiTracker", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("MultiTracker", width, height)

# # # Read first frame
# # success, frame = cap.read()
# # if not success:
# #     print('Failed to read video')
# #     sys.exit(1)

# # # Select boxes
# # bboxes = []
# # colors = []

# # while True:
# #     bbox = cv2.selectROI('MultiTracker', frame)
# #     bboxes.append(bbox)
# #     colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
# #     print("Press q to quit selecting boxes and start tracking")
# #     print("Press any other key to select the next object")
# #     k = cv2.waitKey(0) & 0xFF
# #     if k == 113:  # q is pressed
# #         break

# # print('Selected bounding boxes {}'.format(bboxes))

# # # Specify the tracker type
# # trackerType = "CSRT"

# # # Create MultiTracker object
# # if cv2.__version__.startswith('3'):
# #     multiTracker = cv2.MultiTracker()
# # else:
# #     multiTracker = cv2.MultiTracker_create()

# # # Initialize MultiTracker
# # for bbox in bboxes:
# #     multiTracker.add(createTrackerByName(trackerType), frame, bbox)

# # # Process video and track objects
# # while cap.isOpened():
# #     success, frame = cap.read()
# #     if not success:
# #         break

# #     # get updated location of objects in subsequent frames
# #     success, boxes = multiTracker.update(frame)

# #     # draw tracked objects
# #     for i, newbox in enumerate(boxes):
# #         p1 = (int(newbox[0]), int(newbox[1]))
# #         p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
# #         cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

# #     # show frame
# #     cv2.imshow('MultiTracker', frame)

# #     # quit on ESC button
# #     if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
# #         break

# # # Release the video capture object and close all windows
# # cap.release()
# # cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Dictionary mapping tracker names to OpenCV tracker creation functions
# TrDict = {'csrt': cv2.TrackerCSRT_create,
#           'kcf' : cv2.TrackerKCF_create,
#           'boosting' : cv2.TrackerBoosting_create,
#           'mil': cv2.TrackerMIL_create,
#           'tld': cv2.TrackerTLD_create,
#           'medianflow': cv2.TrackerMedianFlow_create,
#           'mosse':cv2.TrackerMOSSE_create}

# # Create a MultiTracker to handle multiple object tracking
# trackers = cv2.legacy.MultiTracker_create()

# # Open the video file for reading
# v = cv2.VideoCapture('Computer vision/CV-Practice/pexels-kelly-lacy-5473765 (2160p).mp4')
# ret, frame = v.read()

# # Number of objects to track (you can change this based on your requirement)
# k = 4

# # Loop to manually select regions of interest (ROIs) for tracking in the first frame
# for i in range(k):
#     cv2.imshow('Frame', frame)
#     bbi = cv2.selectROI('Frame', frame)
    
#     # Create a CSRT tracker for each ROI and add it to the MultiTracker
#     tracker_i = TrDict['csrt']()
#     trackers.add(tracker_i, frame, bbi)

# # Initialize variables for saving results
# frameNumber = 2
# baseDir = r'D:/Data Science/Computer vision/CV-Practices'

# # Main loop for processing video frames
# while True:
#     # Read the next frame from the video
#     ret, frame = v.read()
#     if not ret:
#         break

#     # Update the positions of the tracked objects in the current frame
#     (success, boxes) = trackers.update(frame)

#     # Save the bounding box coordinates of each tracked object for each frame in a text file
#     np.savetxt(baseDir + '/frame_' + str(frameNumber) + '.txt', boxes, fmt='%f')
#     frameNumber += 1

#     # Draw rectangles around the tracked objects on the current frame
#     for box in boxes:
#         (x, y, w, h) = [int(a) for a in box]
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

#     # Display the current frame
#     cv2.imshow('Frame', frame)

#     # Exit the loop if the 'q' key is pressed
#     key = cv2.waitKey(5) & 0xFF
#     if key == ord('q'):
#         break

# # Release the video capture object and close all OpenCV windows
# v.release()
# cv2.destroyAllWindows()

