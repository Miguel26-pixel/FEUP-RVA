import cv2
import numpy as np
from PIL import Image


def rotate_image(image, large_contour):

    ellipse = cv2.fitEllipse(large_contour)
    angle = ellipse[2]


    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image


def take_background(image):

    lower_color = np.array([0, 0, 0])
    upper_color = np.array([100, 100, 100]) 


    mask = cv2.inRange(image, lower_color, upper_color)


    mask = cv2.bitwise_not(mask)


    result = cv2.bitwise_and(image, image, mask=mask)

    return result



def merge_images(image1, image2, large_contour):
    # Get the dimensions of image1
    image1_height, image1_width = image1.shape[:2]

    x, y, w, h = cv2.boundingRect(large_contour)

    # Define the position where you want to place image1 based on the largest contour
    x_position = x
    y_position = y

    # Ensure that the placement coordinates are within the bounds of image2
    x_position = max(0, x_position)
    y_position = max(0, y_position)

    # Define the scaling factor to make image1 larger
    scale_factor = 2  # You can adjust this value as needed

    # Calculate the dimensions for placement
    roi_height = int(h * scale_factor)
    roi_width = int(w * scale_factor)

    # Resize image1 to match the placement dimensions
    image1_resized = cv2.resize(image1, (roi_width, roi_height))

    # Create a region of interest (ROI) for the placement
    roi = image2[y_position:y_position + roi_height, x_position:x_position + roi_width]

    # Use the alpha channel (if available) to blend image1 with the ROI
    if image1.shape[2] == 4:  # Check if there's an alpha channel
        for c in range(0, 3):
            image2[y_position:y_position + image1_height, x_position:x_position + image1_width, c] = (
                image1[:, :, c] + (1 - image1[:, :, 3] / 255.0)
            )
    else:
        # If there's no alpha channel, simply copy the RGB values
        for c in range(0, 3):
            image2[y_position:y_position + image1_height, x_position:x_position + image1_width, c] = (
                image1[:, :, c]
            )

    return image2




def rotate_image_angle(image):
    rotated_image = cv2.flip(image, flipCode=0)
    return rotated_image

def rotate_to_vertical(image):

    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated_image




