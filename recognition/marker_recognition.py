import cv2 as cv
import numpy as np

# Load images
img = cv.imread('./markers/test_images/img1.jpeg',)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

markers = ['./markers/artoolkit/marker1.png', './markers/artoolkit/marker2.png', './markers/artoolkit/marker3.png', './markers/artoolkit/marker4.png', './markers/artoolkit/marker5.png']
markers = [cv.imread(marker,cv.IMREAD_GRAYSCALE) for marker in markers]

# Define corners of the marker
marker_corners = np.array([[0, 0], [0, markers[0].shape[0]], [markers[0].shape[1], markers[0].shape[0]], [markers[0].shape[1], 0]], dtype=np.float32)

# Create SIFT object
sift = cv.SIFT_create()

# List to store keypoints and descriptors for each marker
markers_kp_des = []

for marker in markers:
    kp, des = sift.detectAndCompute(marker, None)
    markers_kp_des.append((kp, des))

# Apply thresholding gaussian blur
img_thresh = cv.adaptiveThreshold(img_gray, 200, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 55, 2)

# Connected components (Blob) detection
num_labels, labels = cv.connectedComponents(img_thresh)
labels = ((labels / num_labels) * 255).astype(np.uint8) # Normalize the label matrix

# Find contours in the threshold image
contours, _ = cv.findContours(img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # Approximate the contour
    epsilon = 0.01 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)

    # If quad
    if len(approx) == 4:
        # Compute contour area
        area = cv.contourArea(contour)
        x,y,w,h = cv.boundingRect(contour)
        roi=img_gray[y:y+h,x:x+w]

        if area>10000:
            # Find the keypoints and descriptors with SIFT
            kp2, des2 = sift.detectAndCompute(roi, None)

            # create BFMatcher object
            bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

            for kp1, des1 in markers_kp_des:
                # Match descriptors.
                matches = bf.match(des1,des2)

                # Sort them in the order of their distance.
                matches = sorted(matches, key = lambda x:x.distance)

                if len(matches)>4:
                    # Apply ransac 
                    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
                    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)

                    cv.drawContours(img, [approx], 0, (0, 255, 0),4)

# Show the image
img = cv.resize(img, (0, 0), fx=0.4, fy=0.4)
cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()