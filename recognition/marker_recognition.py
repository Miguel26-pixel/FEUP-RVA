import cv2
import numpy as np

# Load image
img = cv2.imread('./markers/test_images/img5.jpg',)

# Aquired image is usually mirrored, flip it if necessary
img = cv2.flip(img, 1)

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_area = img.shape[0] * img.shape[1]
area_threshold = 0.0015 * img_area

# Load markers
markers = ['./markers/aruco/marker_0.png', './markers/aruco/marker_1.png', './markers/aruco/marker_2.png', './markers/aruco/marker_3.png','./markers/aruco/marker_4.png','./markers/aruco/marker_5.png']
markers = [cv2.imread(marker,cv2.IMREAD_GRAYSCALE) for marker in markers]
marker_size = markers[0].shape[0]

# Thresholding
_, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


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

        # Apply perspective transform to get marker isolated
        marker = np.array([[0, 0], [marker_size, 0], [marker_size, marker_size], [0, marker_size]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(corners, marker)
        marker = cv2.warpPerspective(img_gray, transform, (marker_size, marker_size))

        # Threshold marker
        _, marker = cv2.threshold(marker, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Resize marker
        marker = cv2.resize(marker, (marker_size, marker_size))


        # Match template 4 times by rotating the marker by 90 degrees each time
        best_of_each_marker = []
        for x in markers:
            vals = []
            for i in range(1, 5):
                res = cv2.matchTemplate(marker, x, cv2.TM_CCORR_NORMED)
                vals.append(res.max())
                marker = cv2.rotate(marker, cv2.ROTATE_90_CLOCKWISE)

            best_of_each_marker.append(max(vals))

        # print index of marker with highest matchTemplate value
        print(np.argmax(best_of_each_marker))
        
        # display id of marker with highest matchTemplate value next to the marker
        cv2.putText(img, str(np.argmax(best_of_each_marker)), (int(corners[0][0]), int(corners[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2, cv2.LINE_AA)
        


# Display image
img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

