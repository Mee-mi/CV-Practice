import cv2
import sys

# Set up tracker.
# Instead of MIL, you can also use
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[2]

if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
elif tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
elif tracker_type == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
elif tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

# Read video
video = cv2.VideoCapture('Computer vision/CV-Practice/pexels_videos_1572547 (1440p).mp4')

# Exit if the video is not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read the first frame.
ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

# Define an initial bounding box
bbox = (1000, 500, 1100, 600)

# Get the dimensions of the video
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a window and resize it
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking", width, height)

# Uncomment the line below to select a different bounding box
bbox = cv2.selectROI("Tracking",frame, False)

# Initialize the tracker with the first frame and bounding box
ok = tracker.init(frame, bbox)



# Move the window to a specific location on the screen
cv2.moveWindow("Tracking", 100, 100)  # Adjust the position as needed

while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()

    # Update the tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw the bounding box
    # p1 represents the top-left corner of the bounding box as a tuple (x, y).
    # p2 represents the bottom-right corner of the bounding box as a tuple (x, y).
    # thats how roi is made
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the tracker type on the frame
    cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display the FPS on the frame
    cv2.putText(frame, "FPS: " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display the result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()