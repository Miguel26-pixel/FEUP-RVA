'''
Adapted from https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html

This script detects ArUco markers in a video feed from a webcam or DroidCam.
'''

import cv2
import cv2.aruco as aruco

# Load aruco dict
dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

parameters = aruco.DetectorParameters()

# Specify the camera: 0 for webcam, 1 for DroidCam
# To use DroidCam, install the app on your phone and connect to the same WiFi network as your computer
camera = 0

# replace with your DroidCam IP and port
ip = '192.168.1.68'
port = '4747'

# Start the video capture
if camera == 0:
    cap = cv2.VideoCapture(0)
elif camera == 1:
    cap = cv2.VideoCapture(f'http://{ip}:{port}/video')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = aruco.detectMarkers(frame, dict, parameters=parameters)

    # If markers are detected, highlight them in the image
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), markerCorners, markerIds)

    cv2.imshow('frame', frame_markers)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
