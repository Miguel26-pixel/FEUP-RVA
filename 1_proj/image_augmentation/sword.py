import cv2
import numpy as np
import mediapipe as mp

# Load YOLO model for person detection
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load the sword image with transparency
sword_img = cv2.imread('sword.png', cv2.IMREAD_UNCHANGED)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

while True:
    ret, frame = cap.read()
    

    if not ret:
        break

    # # Detect people in the frame
    # blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # net.setInput(blob)
    # outs = net.forward(output_layers)

    # class_ids = []
    # confidences = []
    # boxes = []

    # for out in outs:
    #     for detection in out:
    #         scores = detection[5:]
    #         class_id = np.argmax(scores)
    #         confidence = scores[class_id]
    #         if confidence > 0.5 and class_id == 0:  # Class ID 0 corresponds to a person in COCO dataset
    #             center_x = int(detection[0] * frame.shape[1])
    #             center_y = int(detection[1] * frame.shape[0])
    #             w = int(detection[2] * frame.shape[1])
    #             h = int(detection[3] * frame.shape[0])

    #             x = int(center_x - w / 2)
    #             y = int(center_y - h / 2)

    #             boxes.append([x, y, w, h])
    #             confidences.append(float(confidence))
    #             class_ids.append(class_id)

    # indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # if len(indexes) > 0:
    #     for i in range(len(boxes)):
    #         if i in indexes:
    #             x, y, w, h = boxes[i]

    #             # Calculate the position and orientation of the sword relative to the person
    #             person_frame = frame[y:y+h, x:x+w]
                
    #             # Convert the person_frame to RGB for pose estimation
    #             person_frame_rgb = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
    #             results = pose.process(person_frame_rgb)

    #             if results.pose_landmarks:
    #                 # Extract landmarks of interest, e.g., shoulder and hand keypoints
    #                 shoulder_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    #                 hand_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

    #                 # Calculate the position and orientation of the sword relative to the person based on the landmarks
    #                 sword_x = int(w * hand_landmark.x * person_frame.shape[1])
    #                 sword_y = int(h * hand_landmark.y * person_frame.shape[0])
    #                 # Adjust sword_x and sword_y based on your specific requirements

    #                 # Overlay the sword image onto the person
    #                 sword_resized = cv2.resize(sword_img, (w, h))
    #                 frame[y + sword_y:y + sword_y + sword_resized.shape[0], x + sword_x:x + sword_x + sword_resized.shape[1]] = sword_resized

    cv2.imshow('Live Augmentation', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
