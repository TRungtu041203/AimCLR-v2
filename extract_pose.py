# Hand and body keypoint detector by using mediapipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.tasks.python import vision
import cv2
import matplotlib.pyplot as plt 
import mediapipe as mp
import numpy as np
import time
from mediapipe.framework.formats import landmark_pb2
import zmq
import time
from ddnet import data_generator, loaded_model, C
from statistics import mode
context = zmq.Context()

model = loaded_model #model load từ bên ddnet.py
########## Define the video path and model path, initialize mediapipe ##########
video_path = r"D:\Ha Anh\mediapipe\test\s6_LeVietDuc_1.mp4"
model_path = r"models/pose_landmarker_full.task" #mediapipe landmarker
hand_model_path = r"models/hand_landmarker.task"
# Import mediapipe and its dependencies
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)
hand_option = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.VIDEO, num_hands=2, min_hand_detection_confidence=0.2)
landmarker = PoseLandmarker.create_from_options(options)
hand_landmarker = HandLandmarker.create_from_options(hand_option)
video_reader = cv2.VideoCapture(video_path)  


bottom_line_y = 0
trigger_line = 0
state_dict = {
    0: 'Idle',
    1: 'Moving',
    2: 'Trigger'
}
state_color = {
    0: (0, 255, 0),
    1: (0, 0, 255),
    2: (255, 0, 0)
}
previous_moving_hand_index = 0
moving_hand_index = 0
number_diff_move = 0

current_state = 0 # 0: idle, 1: register, 2: trigger



########## Extract hand and arm pose ##########
def extract_hand(result):
    rh = lh = np.zeros((21, 3))
    
    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        assert len(handedness) == 1, "Unexpected Error in handedness"
        
        if handedness[0].category_name == 'Left':
            lh = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
        if handedness[0].category_name == 'Right':
            rh = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
        
    hand_pose = np.concatenate((rh, lh), axis=0)
    return hand_pose


def extract_arm(result):
    extracted_keypoints = [
        solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
        solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
        solutions.pose.PoseLandmark.LEFT_ELBOW.value,
        solutions.pose.PoseLandmark.RIGHT_ELBOW.value,
        solutions.pose.PoseLandmark.LEFT_WRIST.value,
        solutions.pose.PoseLandmark.RIGHT_WRIST.value
    ]
    n_keypoints = len(extracted_keypoints)
    arm_pose = np.zeros((n_keypoints, 3))
    
    for landmarks in result.pose_landmarks:
        extracted_lms = [landmarks[i] for i in extracted_keypoints]
        for i in range(n_keypoints):
            lm = extracted_lms[i]
            arm_pose[i] = (lm.x, lm.y, lm.z)
            
    return arm_pose   
 
def fill_missing_pose(pose_concat, previous_2_hands):
    if previous_2_hands[0][0][0] != 0 and pose_concat[0][0] == 0:
        pose_concat[0:21] = previous_2_hands[0]
    if previous_2_hands[1][0][0] != 0 and pose_concat[21][0] == 0:
        pose_concat[21:42] = previous_2_hands[1]
    return pose_concat

def draw_concat_pose(image, concat_pose, handedness=None, thickness=2):
    annotated_image = np.copy(image)

    right_hand = concat_pose[0:21]
    left_hand = concat_pose[21:42]
    arms = concat_pose[42:]

    # [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
    arm_connections = [
        (0, 2), (2, 4),  # Left arm
        (1, 3), (3, 5),  # Right arm
        (0, 1)           # Shoulders
    ]

    height, width, _ = annotated_image.shape

    def to_landmark_list(points):
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        landmark_list.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=p[0], y=p[1], z=p[2]) for p in points
        ])
        return landmark_list

    # Draw right hand
    if np.any(right_hand[0] != 0):
        hand_landmarks_proto = to_landmark_list(right_hand)
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

    # Draw left hand
    if np.any(left_hand[0] != 0):
        hand_landmarks_proto = to_landmark_list(left_hand)
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

    # Draw arms
    if np.any(arms[0] != 0):
        pose_landmarks_proto = to_landmark_list(arms)
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            arm_connections,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )

    return annotated_image


previous_2_hands = np.zeros((2, 21, 3))

import os

video_folder_path = r"D:\HaAnh\mediapipe\test"
predict_folder_path = r"D:\HaAnh\mediapipe\extract_pose"

for i in os.listdir(video_folder_path):
    frame_count = 0 

    start_time = time.time()

    video_path = os.path.join(video_folder_path, i)
    video_name, _ = os.path.splitext(i)
    file_name = os.path.join(predict_folder_path, f"{video_name}.npy")

    landmarker = PoseLandmarker.create_from_options(options)
    hand_landmarker = HandLandmarker.create_from_options(hand_option)

    video_reader = cv2.VideoCapture(video_path)

    all_poses = []
    
    while True:
        ret, image = video_reader.read()
        if not ret:
            break
        frame_count += 1
        # Process the image
        #image = cv2.resize(image, (1280, 960))
        image = cv2.resize(image, (640, 480))

        # with PoseLandmarker.create_from_options(options) as landmarker:
        numpy_frame_from_opencv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        frame_timestamp_ms = int(video_reader.get(cv2.CAP_PROP_POS_MSEC))
        # print(frame_timestamp_ms)
        extract_time = time.time()  
        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        hand_landmarker_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        print('extract time:', time.time() - extract_time)
            # print(pose_landmarker_result)
        # get width and height of the image
        height, width, _ = image.shape
        pose = extract_arm(pose_landmarker_result)
        hand_pose = extract_hand(hand_landmarker_result)
        
        hand_pose = fill_missing_pose(hand_pose, previous_2_hands)
        

        if hand_pose[0][0] != 0:
            previous_2_hands[0] = hand_pose[0:21]
        if hand_pose[21][0] != 0:
            previous_2_hands[1] = hand_pose[21:42]
        # concatenate pose and hand_pose
        concat_pose = np.concatenate((hand_pose, pose), axis=0)

        all_poses.append(concat_pose)


        # if pose_list.full():
        #         pose_list.get_nowait()  # Remove the oldest item from the queue
        image = draw_concat_pose(image,concat_pose)
        cv2.imshow('Result',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    all_poses = np.stack(all_poses, axis=0)  # Convert list to numpy array with shape (num_frames, num_keypoints, 3)
    np.save(file_name, all_poses)  # Save the poses to a file

    landmarker.close()
    hand_landmarker.close()
    video_reader.release()
    # plot


cv2.destroyAllWindows()

