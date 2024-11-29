import torch
import cv2
import time
import logging
import threading
import numpy as np
from datetime import datetime
from MetalTheft.motion_detection import detect_motion
from ultralytics import YOLO
from MetalTheft.utils.utils import save_snapshot
from MetalTheft.vid_stabilisation import VideoStabilizer
from MetalTheft.send_email import EmailSender
from MetalTheft.roi_selector import ROISelector
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.aws import AWSConfig
from Video_classification.testing import VideoActionTester

logging.getLogger('ultralytics').setLevel(logging.WARNING) 

# Initialize objects
model = YOLO('yolov8n.pt', verbose=True)
email = EmailSender()
roi_selector = ROISelector()
mongo_handler = MongoDBHandler()
stabiliser = VideoStabilizer()
aws = AWSConfig()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
alpha = 0.8


# Initialize the action recognition model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_path = '/home/alluvium/Desktop/Namdeo/CMR_Project/Video_classification/action_recognition_model.pth'
model_path = 'action_recognition_model.pth'
label_map = {'CricketBowling': 0, 'JavelineThrow': 1, 'ThrowDiscus': 2}  # Replace with your label map
action_tester = VideoActionTester(model_path=model_path, device=device, label_map=label_map, frames_per_clip=32)


def verify_motion_in_video(video_path, roi_1_pts_np, roi_2_pts_np, snapshot_path, motion_threshold=350):
    cap_verify = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)
    
    if not cap_verify.isOpened():
        print(f"Error: Could not open video for verification: {video_path}")
        return "No"
    
    person_detected = False
    action_detected = False
    action_checked = False  # To ensure action recognition is only called once
    frame_count = 0  # To keep track of the number of frames

    while cap_verify.isOpened():
        ret, frame = cap_verify.read()
        if not ret:
            break

        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)  # Blur for noise reduction
        frame_count += 1  # Increment frame count

        # Motion detection in ROI 1
        _, thresh_ROI1, _, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)
        motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > motion_threshold
        
        # Motion detection in ROI 2
        _, thresh_ROI2, _, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np)
        motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > motion_threshold

        # Person detection after 20 frames
        if frame_count > 20:
            _, _, person_detected, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)

        # Trigger action recognition only once when motion and person detection are satisfied
        if motion_in_roi1 and person_detected and not action_checked:
            action_result = action_tester.predict(video_path, threshold=0.5)
            action_checked = True  # Mark action recognition as checked

            if action_result == 'Yes':
                if action_result=='Throw': # Check if the recognized action matches
                    print(f"Action detected: {action_result}")
                    roi_color = (0, 0, 255)  # Red for positive detection
                    cv2.fillPoly(frame, [roi_1_pts_np], roi_color)

                    # Save snapshot and video
                    current_time = datetime.now()
                # snapshot_url = aws.upload_snapshot_to_s3bucket(snapshot_path)
                # mongo_handler.save_snapshot_to_mongodb(snapshot_url, current_time)
                
                # video_url = aws.upload_video_to_s3bucket(video_path)
                # mongo_handler.save_video_to_mongodb(video_url, current_time)

                # # Send alert email in a separate thread
                # threading.Thread(target=email.send_alert_email, args=(snapshot_path, video_url)).start()

                return "Yes"

    cap_verify.release()
    return "No"







# import cv2
# import time
# import logging
# import threading
# import torch
# import numpy as np
# from datetime import datetime
# from MetalTheft.motion_detection import detect_motion
# from ultralytics import YOLO
# from MetalTheft.utils.utils import save_snapshot
# from MetalTheft.vid_stabilisation import VideoStabilizer
# from MetalTheft.send_email import EmailSender
# from MetalTheft.roi_selector import ROISelector
# from MetalTheft.mongodb import MongoDBHandler
# from MetalTheft.aws import AWSConfig
# from Video_classification.testing import VideoActionTester
# logging.getLogger('ultralytics').setLevel(logging.WARNING) 

# # Initialize objects
# model = YOLO('yolov8n.pt', verbose=True)
# email = EmailSender()
# roi_selector = ROISelector()
# mongo_handler = MongoDBHandler()
# stabiliser = VideoStabilizer()
# aws = AWSConfig()
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# alpha = 0.8


# # Initialize the action recognition model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_path = '/home/alluvium/Desktop/Namdeo/CMR_Project/Video_classification/action_recognition_model.pth'
# label_map = {'CricketBowling': 0, 'JavelineThrow': 1, 'ThrowDiscus': 2}  # Replace with your label map
# action_tester = VideoActionTester(model_path=model_path, device=device, label_map=label_map, frames_per_clip=32)

# def verify_motion_in_video(video_path, roi_1_pts_np, roi_2_pts_np, snapshot_path, motion_threshold=350):
    
#     cap_verify = cv2.VideoCapture(video_path)
#     fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)
    
#     if not cap_verify.isOpened():
#         print(f"Error: Could not open video for verification: {video_path}")
#         return "No"
    
#     person_detected = False
#     frame_count = 0  # To keep track of the number of frames

#     while cap_verify.isOpened():
#         ret, frame = cap_verify.read()
#         if not ret:
#             break

#         blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)  # Blur for noise reduction
#         frame_count += 1  # Increment frame count

#         # Motion in ROI 1
#         _, thresh_ROI1, _, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)
#         motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > motion_threshold
        
        
#         # Motion in ROI 2
#         _, thresh_ROI2, _, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np)
#         motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > motion_threshold
#         # cv2.polylines(frame, [roi_1_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)

#         # After 20 frames, start person detection
#         if frame_count > 20:
#             _, _, person_detected, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)

#         action_result = action_tester.predict(video_path, threshold=0.5)
#         action_detected = (action_result == 'CricketBowling')  # Check if the recognized action matches

#         # If motion is detected in either ROI and a person is detected after 20 frames
#         if (motion_in_roi1) and person_detected and action_detected:
#             motion_in_roi2_to_roi1 = True   
#             roi_color = (0, 0, 255) if motion_in_roi2_to_roi1 else (0, 255, 0)   
#             cv2.fillPoly(frame, [roi_1_pts_np], roi_color) 
            

#             # snapshot_path = save_snapshot(frame)
#             # if snapshot_path:
#             current_time = datetime.now()
#             snapshot_url = aws.upload_snapshot_to_s3bucket(snapshot_path)
#             mongo_handler.save_snapshot_to_mongodb(snapshot_url, current_time)
            
#             start_time = current_time
#             video_url = aws.upload_video_to_s3bucket(video_path)
#             mongo_handler.save_video_to_mongodb(video_url, start_time)
#             threading.Thread(target=email.send_alert_email, args=(snapshot_path, video_url)).start()

#             return "Yes"

#     cap_verify.release()
#     return "No"


