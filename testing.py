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

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Display the current frame in the 'output' window
    cv.imshow('output', frame)

    # Break the loop if 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv.destroyAllWindows()
