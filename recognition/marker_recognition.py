import cv2
import numpy as np
from imageChange import *


# Define constants
AREA_THRESHOLD = 0.01
OBJ_POINTS = np.array([[0.5, 0.5, 0], [-0.5, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 0]], dtype=np.float32)
DIST_COEFFS = np.zeros(4)

# Camera Specs
# Focal length of camera lens in mm (check manufacturer specs)
FOCAL_LENGTH_MM= 24
# Usually between 4-7 in smartphones
SENSOR_WIDTH_MM= 7
# width of your image sensor in pixels (check manufacturer specs)
SENSOR_WIDTH_PIXELS=1080


def get_image(img_num,flip=False):
    img = cv2.imread('./markers/test_images/img'+str(img_num)+'.jpg')
    
    # Flip image if needed
    if flip:
        img = cv2.flip(img, 1)
        
    return img

def get_image_area(img):
    return img.shape[0] * img.shape[1]

def calculate_area_threshold(img):
    return AREA_THRESHOLD * get_image_area(img)


def get_markers():
    markers = ['./markers/aruco/marker_0.png', './markers/aruco/marker_1.png', './markers/aruco/marker_2.png', './markers/aruco/marker_3.png','./markers/aruco/marker_4.png','./markers/aruco/marker_5.png']
    markers = [cv2.imread(marker,cv2.IMREAD_GRAYSCALE) for marker in markers]
    return markers

def get_marker_size(marker):
    return marker.shape[0]

def apply_threshold(img):
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def get_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_focal_length_pixels():
    return (FOCAL_LENGTH_MM * SENSOR_WIDTH_PIXELS) / SENSOR_WIDTH_MM

def get_camera_matrix(img):
    focal_length = get_focal_length_pixels()
    return np.array([[focal_length, 0, img.shape[1]/2], [0, focal_length, img.shape[0]/2], [0, 0, 1]], dtype=np.float32)

def get_corners(approx):
    return approx.reshape((4, 2)).astype(np.float32)

def apply_perspective_transform(img,corners, marker_size):
    marker = np.array([[0, 0], [marker_size, 0], [marker_size, marker_size], [0, marker_size]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners, marker)
    marker = cv2.warpPerspective(img, transform, (marker_size, marker_size))
    return marker

def resize_img(img,img_size):
    return cv2.resize(img, (img_size, img_size))



def match_template(marker, markers):
    best_of_each_marker = []
    for x in markers:
        vals = []
        for i in range(1, 5):
            res = cv2.matchTemplate(marker, x, cv2.TM_CCORR_NORMED)
            vals.append(res.max())
            marker = cv2.rotate(marker,cv2.ROTATE_90_CLOCKWISE)

        best_of_each_marker.append(max(vals))

    return np.argmax(best_of_each_marker)

def get_rvec_tvec(img,corners,camera_matrix):
    # Solve PnP
    _, rvec,tvec=cv2.solvePnP(OBJ_POINTS,corners,camera_matrix,DIST_COEFFS)
    return rvec,tvec


def draw_3d_object(img, rvec, tvec, camera_matrix):
    # draw cube
    axis=np.float32([[0.5,0.5,0],[0.5,-0.5,0],[-0.5,-0.5,0],[-0.5,0.5,0],[0.5,0.5,1],[0.5,-0.5,1],[-0.5,-0.5,1],[-0.5,0.5,1]])
    imgpts,jac=cv2.projectPoints(axis,rvec,tvec,camera_matrix,DIST_COEFFS   )
    imgpts=np.int32(imgpts).reshape(-1,2)
    img=cv2.drawContours(img,[imgpts[:4]],-1,(0,255,0),3)
    for i,j in zip(range(4),range(4,8)):
        img=cv2.line(img,tuple(imgpts[i]),tuple(imgpts[j]),(0,0,255),3)

    img=cv2.drawContours(img,[imgpts[4:]],-1,(0,255,0),3)
    return img

def process_countours(contours, img,img_gray,marker_size,markers):
    area_list = {}
    contour_list = {}
    for contour in contours:
        area = cv2.contourArea(contour)
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and area > calculate_area_threshold(img):
            corners = get_corners(approx)
            marker = apply_perspective_transform(img_gray,corners,marker_size)
            marker = apply_threshold(marker)
            marker = resize_img(marker,marker_size)

            corresponding_marker = match_template(marker, markers)
            area_list[area] = corresponding_marker
            contour_list[area] = contour

            # if want to display the id of the marker next to the marker
            cv2.putText(img,str(corresponding_marker),(int(corners[0][0]),int(corners[0][1])),cv2.FONT_HERSHEY_SIMPLEX ,5,(255 ,255 ,255),3,cv2.LINE_AA)

            # get camera matrix
            camera_matrix = get_camera_matrix(img)

            rvec,tvec = get_rvec_tvec(img,corners,camera_matrix)

            # draw cube
            img = draw_3d_object(img,rvec,tvec,camera_matrix)

    return img,area_list,contour_list

def display_image(img,resized=False):
    if len(img.shape)==2:
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        
    if resized:
        img=cv2.resize(img,(0 ,0),fx=0.25,fy=0.25)
    cv2.imshow('Image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():

    # 5,6,7
    # set flip to True if you want to flip the image
    image = get_image(5,flip=True)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # load markers dictionary
    markers = get_markers()

    # get marker size
    marker_size = get_marker_size(markers[0])

    # apply threshold to image
    image_threshold = apply_threshold(image_gray)

    # get contours
    image_contours = get_contours(image_threshold)

    image,area_list,contour_list = process_countours(image_contours,image,image_gray,marker_size,markers)

    max_area = max(area_list.keys())

    max_marker = area_list[max_area]

    print(max_marker)

    max_contour = contour_list[max_area]

    sword_image = cv2.imread('sword_2.png', cv2.IMREAD_UNCHANGED)

    #sword_image = take_background(sword_image)

    rotated_image = rotate_image(sword_image, max_contour)

    rotated_image = rotate_to_vertical(rotated_image)

    final_image = merge_images(rotated_image, image, max_contour)

    display_image(final_image,resized=True)


if __name__ == '__main__':
    main()