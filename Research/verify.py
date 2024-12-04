import torch
import cv2
import time
import logging
import threading
import numpy as np
from datetime import datetime
from MetalTheft.motion_detection import detect_motion
from ultralytics import YOLO
from MetalTheft.utils.utils import save_snapshot, draw_motion_contours, save_video
from MetalTheft.vid_stabilisation import VideoStabilizer
from MetalTheft.send_email import EmailSender
from MetalTheft.roi_selector import ROISelector
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.aws import AWSConfig
from collections import deque
from vidgear.gears import CamGear
import os
motion_detected_flag = False
motion_buffer_duration = 5  # Duration before and after motion
moton_buffer_fps = 30  
motion_frame_buffer = deque(maxlen=moton_buffer_fps * motion_buffer_duration)  
recording_after_motion = False
recording_end_time = None
logging.getLogger('ultralytics').setLevel(logging.WARNING) 
motion_frame_buffer = deque(maxlen=moton_buffer_fps * motion_buffer_duration)  
# Initialize objects
model = YOLO('yolov8n.pt', verbose=True)
email = EmailSender()
roi_selector = ROISelector()
mongo_handler = MongoDBHandler()
out_motion =  None
stabiliser = VideoStabilizer()
aws = AWSConfig()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
alpha = 0.8
last_motion_time = None

# def verify_motion_in_video(video_path, roi_1_pts_np, roi_2_pts_np, snapshot_path, camera_id, result_queue):
#     global last_motion_time
    
#     video_path = os.path.abspath(video_path)
#     stream = CamGear(source=video_path, logging=True).start()
#     fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)
#     frame_count = 0  # To keep track of the number of frames
#     frame_count += 1  
#     motion_direction_window = 1
#     while True:
#         frame = stream.read()
#         if frame is None or frame.size == 0:
#             break

#         blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)  # Blur for noise reduction

#         _, thresh_ROI1, _, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)
#         motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > 300
#         if motion_in_roi1:
#             roi1_motion_time = time.time()

#         _, thresh_ROI2, person_in_roi2, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np)
#         motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > 300
#         if motion_in_roi2:
#             roi2_motion_time = time.time()

#         if frame_count > 15:
#             _, _, person_in_roi2, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np)

#         current_time11 = datetime.now()
#         if (motion_in_roi1 and person_in_roi2):
#             if last_motion_time is None or (current_time11 - last_motion_time).total_seconds() > 3:
#                 if (roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window):
#                     last_motion_time = current_time11

#         if (motion_in_roi1 and person_in_roi2 ):
#             if (roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window):                  
#                 start_time = datetime.now()
#                 video_url = aws.upload_video_to_s3bucket(video_path, camera_id)                  
#                 snapshot_url = aws.upload_snapshot_to_s3bucket(snapshot_path, camera_id)
#                 threading.Thread(target=mongo_handler.save_snapshot_to_mongodb, args=(snapshot_url, start_time, camera_id)).start()
#                 threading.Thread(target=mongo_handler.save_video_to_mongodb, args=(video_url, start_time, camera_id)).start()
#                 threading.Thread(target=email.send_alert_email, args=(snapshot_path, video_url, camera_id)).start()

#                 result_queue.put("Yes")
#                 stream.stop()
#                 return
            
#         frame_count += 1


#     result_queue.put("No")
#     stream.stop()








def verify_motion_in_video(video_path, roi_1_pts_np, roi_2_pts_np, snapshot_path, camera_id):
    global last_motion_time
    
    video_path = os.path.abspath(video_path)
    stream = CamGear(source=video_path, logging=True).start()
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)
    frame_count = 0  # To keep track of the number of frames
    frame_count += 1  
    motion_direction_window = 1
    while True:
        frame = stream.read()
        if frame is None or frame.size == 0:
            break

        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)  # Blur for noise reduction

        _, thresh_ROI1, _, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)
        motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > 300
        if motion_in_roi1:
            roi1_motion_time = time.time()

        _, thresh_ROI2, person_in_roi2, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np)
        motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > 300
        if motion_in_roi2:
            roi2_motion_time = time.time()

        if frame_count > 15:
            _, _, person_in_roi2, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np)

        current_time11 = datetime.now()
        if (motion_in_roi1 and person_in_roi2):
            if last_motion_time is None or (current_time11 - last_motion_time).total_seconds() > 3:
                if (roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window):
                    last_motion_time = current_time11

        if (motion_in_roi1 and person_in_roi2 ):
            if (roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window):                  
                start_time = datetime.now()
                video_url = aws.upload_video_to_s3bucket(video_path, camera_id)                  
                snapshot_url = aws.upload_snapshot_to_s3bucket(snapshot_path, camera_id)
                threading.Thread(target=mongo_handler.save_snapshot_to_mongodb, args=(snapshot_url, start_time, camera_id)).start()
                threading.Thread(target=mongo_handler.save_video_to_mongodb, args=(video_url, start_time, camera_id)).start()
                threading.Thread(target=email.send_alert_email, args=(snapshot_path, video_url, camera_id)).start()

                stream.stop()
                return
            
        frame_count += 1
    stream.stop()



























# def verify_motion_in_video(video_path, roi_1_pts_np, roi_2_pts_np, snapshot_path, camera_id, motion_threshold=300):
#     global last_motion_time

#     # Ensure the video path is valid
#     video_path = os.path.abspath(video_path)
#     if not os.path.exists(video_path):
#         raise FileNotFoundError(f"Error: The video file '{video_path}' does not exist.")

#     # Initialize the video stream
#     stream = CamGear(source=video_path, logging=True).start()
#     fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)

#     object_counter = 0  # Initialize object counter
#     frame_count = 0  # Frame counter
#     last_motion_time = None
#     roi1_motion_time = None
#     roi2_motion_time = None
#     motion_direction_window = 1  # Allowed time difference between ROIs for direction

#     try:
#         while True:
#             frame = stream.read()
#             if frame is None or frame.size == 0:
#                 break

#             blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)  # Blur for noise reduction
#             frame_count += 1

#             # Detect motion in ROI 1
#             _, thresh_ROI1, _, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)
#             motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > motion_threshold
#             if motion_in_roi1:
#                 roi1_motion_time = time.time()

#             # Detect motion in ROI 2
#             _, thresh_ROI2, person_in_roi2, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np)
#             motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > motion_threshold
#             if motion_in_roi2:
#                 roi2_motion_time = time.time()

#             # Check direction and person presence after sufficient frames
#             if frame_count > 20 and motion_in_roi1 and person_in_roi2:
#                 if last_motion_time is None or (datetime.now() - last_motion_time).total_seconds() > 3:
#                     if (roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window):
#                         object_counter += 1  # Increment object count
#                         last_motion_time = datetime.now()

#                         # Handle uploads and alerts
#                         video_url = aws.upload_video_to_s3bucket(video_path, camera_id)
#                         snapshot_url = aws.upload_snapshot_to_s3bucket(snapshot_path, camera_id)
#                         threading.Thread(target=mongo_handler.save_snapshot_to_mongodb, args=(snapshot_url, datetime.now(), camera_id)).start()
#                         threading.Thread(target=mongo_handler.save_video_to_mongodb, args=(video_url, datetime.now(), camera_id)).start()
#                         threading.Thread(target=email.send_alert_email, args=(snapshot_path, video_url, camera_id)).start()

#     except Exception as e:
#         logging.error(f"An error occurred: {e}")
#     finally:
#         # Release resources
#         stream.stop()

#     return object_counter


# object_counter = None

# def verify_motion_in_video(video_path, roi_1_pts_np, roi_2_pts_np, snapshot_path, camera_id, motion_threshold=300):
#     global last_motion_time

#     video_path = os.path.abspath(video_path)
#     if not os.path.exists(video_path):
#         raise FileNotFoundError(f"Error: The video file '{video_path}' does not exist.")

#     stream = CamGear(source=video_path, logging=True).start()
#     fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)

#     object_counter = 0  # Initialize object counter
#     frame_count = 0  # Frame counter
#     last_motion_time = None
#     roi1_motion_time = None
#     roi2_motion_time = None
#     motion_direction_window = 1  

#     try:
#         while True:
#             frame = stream.read()
#             if frame is None or frame.size == 0:
#                 break

#             blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)  # Blur for noise reduction
#             frame_count += 1

#             # Detect motion in ROI 1
#             _, thresh_ROI1, _, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)
#             motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > motion_threshold
#             if motion_in_roi1:
#                 roi1_motion_time = time.time()

#             # Detect motion in ROI 2
#             _, thresh_ROI2, person_in_roi2, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np)
#             motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > motion_threshold
#             if motion_in_roi2:
#                 roi2_motion_time = time.time()

#             # Check direction and person presence after sufficient frames
#             if frame_count > 20 and motion_in_roi1 and person_in_roi2:
#                 if last_motion_time is None or (datetime.now() - last_motion_time).total_seconds() > 3:
#                     if (roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window):
#                         object_counter += 1  # Increment object count
#                         last_motion_time = datetime.now()

#                         # Handle uploads and alerts
#                         video_url = aws.upload_video_to_s3bucket(video_path, camera_id)
#                         snapshot_url = aws.upload_snapshot_to_s3bucket(snapshot_path, camera_id)
#                         threading.Thread(target=mongo_handler.save_snapshot_to_mongodb, args=(snapshot_url, datetime.now(), camera_id)).start()
#                         threading.Thread(target=mongo_handler.save_video_to_mongodb, args=(video_url, datetime.now(), camera_id)).start()
#                         threading.Thread(target=email.send_alert_email, args=(snapshot_path, video_url, camera_id)).start()

#     except Exception as e:
#         logging.error(f"An error occurred: {e}")
#     finally:
#         stream.stop()

#     logging.info(f"Total objects detected: {object_counter}")
#     return object_counter


# def verify_motion_in_video(video_path, roi_1_pts_np, roi_2_pts_np, snapshot_path, camera_id, motion_threshold=300):
#     global last_motion_time

#     video_path = os.path.abspath(video_path)
#     if not os.path.exists(video_path):
#         raise FileNotFoundError(f"Error: The video file '{video_path}' does not exist.")

#     stream = CamGear(source=video_path, logging=True).start()
#     fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)

#     object_counter = 0  # Initialize object counter
#     frame_count = 0  # Frame counter
#     last_motion_time = None
#     roi1_motion_time = None
#     roi2_motion_time = None
#     motion_direction_window = 1  

#     try:
#         while True:
#             frame = stream.read()
#             if frame is None or frame.size == 0:
#                 break

#             blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)  # Blur for noise reduction
#             frame_count += 1

#             # Detect motion in ROI 1
#             _, thresh_ROI1, _, detections = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)
#             motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > motion_threshold
#             if motion_in_roi1:
#                 roi1_motion_time = time.time()

#             # Detect motion in ROI 2
#             _, thresh_ROI2, person_in_roi2, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np)
#             motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > motion_threshold
#             if motion_in_roi2:
#                 roi2_motion_time = time.time()

#             # Check direction and person presence after sufficient frames
#             if frame_count > 15 and motion_in_roi1 and person_in_roi2:
#                 if last_motion_time is None or (datetime.now() - last_motion_time).total_seconds() > 3:
#                     if (roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window):
#                         # Use the number of detected objects instead of a simple increment
#                         object_counter += len(detections) if detections else 1
#                         last_motion_time = datetime.now()

#                         # Handle uploads and alerts
#                         video_url = aws.upload_video_to_s3bucket(video_path, camera_id)
#                         snapshot_url = aws.upload_snapshot_to_s3bucket(snapshot_path, camera_id)
#                         threading.Thread(target=mongo_handler.save_snapshot_to_mongodb, args=(snapshot_url, datetime.now(), camera_id)).start()
#                         threading.Thread(target=mongo_handler.save_video_to_mongodb, args=(video_url, datetime.now(), camera_id)).start()
#                         threading.Thread(target=email.send_alert_email, args=(snapshot_path, video_url, camera_id)).start()

#     except Exception as e:
#         logging.error(f"An error occurred: {e}")
#         return 0  # Return 0 if an error occurs
#     finally:
#         stream.stop()

#     logging.info(f"Total objects detected: {object_counter}")
#     return object_counter






