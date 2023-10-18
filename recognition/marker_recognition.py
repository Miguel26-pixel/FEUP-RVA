import cv2
import numpy as np

# Load image
img = cv2.imread('./markers/test_images/img4.jpeg')
img_area = img.shape[0] * img.shape[1]
area_threshold = 0.0015 * img_area


# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area = cv2.contourArea(contour)
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    if len(approx) == 4 and cv2.isContourConvex(approx) and area > area_threshold:
        # Get the corners of the marker
        corners = approx.reshape((4, 2))
        corners = corners.astype(np.float32)
        # Draw contours and corners
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)

'''
        # Camera matrix (assuming an identity matrix for simplicity)
        camera_matrix = np.eye(3)

        # Distortion coefficients (assuming  distortion for simplicity)
        dist_coeffs = np.zeros((4, 1))

        # Marker 3D model points (assuming a square marker of side length 1)
        marker_size = 1.0
        obj_points = np.array([[0, 0, 0],
                               [marker_size, 0, 0],
                               [marker_size, marker_size, 0],
                               [0, marker_size, 0]], dtype=np.float32)

        

        # Solve PnP
        _, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs)

        # Project 3D points to image plane
        img_points, _ = cv2.projectPoints(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
                                           rvec, tvec, camera_matrix, dist_coeffs)

        # Draw the axis
        img = cv2.line(img, tuple(corners[0].astype(int)), tuple(img_points[1].ravel().astype(int)), (0, 0, 255), 5)
        img = cv2.line(img, tuple(corners[0].astype(int)), tuple(img_points[2].ravel().astype(int)), (0, 255, 0), 5)
        img = cv2.line(img, tuple(corners[0].astype(int)), tuple(img_points[3].ravel().astype(int)), (255, 0, 0), 5)
'''


# Display image
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# TODO: identify the marker and draw the axis