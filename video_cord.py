

import cv2

def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at coordinates (x={x}, y={y}")

cap = cv2.VideoCapture('Computer vision/CV-Practice/pexels_videos_1572547 (1440p).mp4')

# Get video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)  # Create a resizable window
cv2.resizeWindow('Video', width, height)  # Resize the window to match video dimensions
cv2.setMouseCallback('Video', get_coordinates)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
