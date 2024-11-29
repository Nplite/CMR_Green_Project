import cv2
import time
import numpy as np
import threading
from datetime import datetime
from collections import deque
from MetalTheft.motion_detection import detect_motion
from ultralytics import YOLO
from MetalTheft.utils.utils import save_snapshot, save_video
from MetalTheft.constant import *
from MetalTheft.vid_stabilisation import VideoStabilizer
from MetalTheft.exception import MetalTheptException
from MetalTheft.send_email import EmailSender
from MetalTheft.utils.utils import save_snapshot, normalize_illumination, save_video, draw_boxes
from MetalTheft.motion_detection import detect_motion
from MetalTheft.roi_selector import ROISelectorq
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.aws import AWSConfig


model = YOLO('yolov8n.pt', verbose=True)  
motion_buffer_duration = 5  # Duration before and after motion
moton_buffer_fps = 30  
motion_frame_buffer = deque(maxlen=moton_buffer_fps * motion_buffer_duration)  # Main camera feed buffer



email = EmailSender()
roi_selector = ROISelector()
mongo_handler = MongoDBHandler()
stabiliser = VideoStabilizer()
aws = AWSConfig()
model = YOLO('yolov8n.pt')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
alpha = 0.8


# # Function to verify motion in saved video (from the additional code provided)
# def verify_motion_in_video(video_path_data, roi_1_pts_np, roi_2_pts_np, motion_threshold=350, motion_direction_window = 1):
#     motion_detected_flag  = False
#     counter = 1 
#     recording_after_motion =  False
#     recording_end_time = None
#     out_motion = None 
#     cap_verify = cv2.VideoCapture(video_path_data)
#     fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)

#     if not cap_verify.isOpened():
#         print(f"Error: Could not open video for verification: {video_path}")
#         return False
    

#     while cap_verify.isOpened():
#         ret, frame = cap_verify.read()
#         if not ret:
#             break
        
#         combined_frame1 = frame.copy()
#         combined_frame2 = frame.copy()
    
#         blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)  # Blur for noise reduction

#         # Motion in ROI 1
#         combined_frame1, thresh_ROI1, person_detected, detections = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)
#         motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > motion_threshold
#         cv2.polylines(combined_frame1, [roi_1_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
#         roi_color_1 = (0, 0, 255) if motion_in_roi1 and person_detected else (0, 255, 0)
#         motion_mask_1 = np.zeros_like(combined_frame1)
#         cv2.fillPoly(motion_mask_1, [roi_1_pts_np], (0, 255, 0))
#         combined_frame1 = cv2.addWeighted(combined_frame1, alpha, motion_mask_1, 1-alpha, 0)

#         # Motion in ROI 2
#         combined_frame2, thresh_ROI2, _, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np )
#         motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > motion_threshold
#         cv2.polylines(combined_frame2, [roi_2_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
#         motion_mask_2 = np.zeros_like(frame)
#         cv2.fillPoly(motion_mask_2, [roi_2_pts_np], (0,0,0))  
#         combined_frame2 = cv2.addWeighted(combined_frame2, alpha, motion_mask_2, 1-alpha, 0)

#         # Combine ROI1 & ROI2 in one frame
#         combined_frame = cv2.add(combined_frame1, combined_frame2)
#         motion_frame_buffer.append(combined_frame.copy())
        

#         if motion_in_roi2:
#             roi2_motion_time = time.time()
#             contours, _ = cv2.findContours(thresh_ROI2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             for contour in contours:
#                 if cv2.contourArea(contour) < 200:    # Ignore small contours
#                     continue
#                 x, y, w, h = cv2.boundingRect(contour)
#                 if cv2.pointPolygonTest(roi_2_pts_np, (x, y), False) >= 0:
#                     cv2.drawContours(combined_frame, contours, -1, (0, 255, 0), 2)


#         if motion_in_roi1:
#             roi1_motion_time = time.time()
#             contours, _ = cv2.findContours(thresh_ROI1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             for contour in contours:
#                 if cv2.contourArea(contour) < 200:    # Ignore small contours
#                     continue
#                 x, y, w, h = cv2.boundingRect(contour)
#                 if cv2.pointPolygonTest(roi_1_pts_np, (x, y), False) >= 0:
#                     cv2.drawContours(combined_frame, contours, -1, (0, 255, 0), 2)

#         if motion_in_roi1 and person_detected:
#             if roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window:
#                 motion_in_roi2_to_roi1 = True  # Direction is confirmed from ROI2 to ROI1 
#                 roi_color = (0, 0, 255) if motion_in_roi2_to_roi1 else (0, 255, 0)
#                 cv2.fillPoly(combined_frame, [roi_1_pts_np], roi_color)    

#             else:
#                 motion_in_roi2_to_roi1 = False 
        
#             # Motion detected: start or continue recording
#             if roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window:
                       
#                 if not motion_detected_flag:
#                     out_motion = None 
#                     video_path = save_video()
#                     out_motion = cv2.VideoWriter(video_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))

#                     while motion_frame_buffer:
#                         out_motion.write(motion_frame_buffer.popleft())
                
#                     snapshot_path = save_snapshot(combined_frame)
#                     if snapshot_path:
#                         current_time=  datetime.now()
#                         # snapshot_url = aws.upload_snapshot_to_s3bucket(snapshot_path)
#                         # mongo_handler.save_snapshot_to_mongodb(snapshot_url, current_time)
                    
#                     start_time = current_time
#                     motion_detected_flag = True
#                     motion_frame_buffer.clear() 
                    
#                     recording_after_motion = False


#                 # Write current frame
#                 out_motion.write(combined_frame)

#         elif motion_detected_flag:
#             # No motion: start countdown for post-motion recording
#             if not recording_after_motion:
#                 recording_end_time = time.time() + 8  # Record for 5 more seconds
#                 recording_after_motion = True

#             # Write post-motion frames
#             if time.time() <= recording_end_time:
#                 out_motion.write(combined_frame)
#             else:
#                 # Stop recording after 5 seconds of no motion
#                 motion_detected_flag = False
#                 recording_after_motion = False
                
#                 # Release the video writer
#                 if out_motion is not None:
#                     out_motion.release()
                
#                 # Clear the buffer and reset all relevant flags and variables
#                 motion_frame_buffer.clear()  # Clear buffer for the next motion event
#                 roi1_motion_time = None
#                 roi2_motion_time = None
#                 recording_end_time = None
#                 out_motion = None
#                 counter += 1
                
                
#                 # video_url = aws.upload_video_to_s3bucket(video_path)
#                 # mongo_handler.save_video_to_mongodb(video_url, start_time)
#                 # threading.Thread(target=email.send_alert_email, args=(snapshot_path, video_url)).start()



#     cap_verify.release()




def verify_motion_in_video(video_path_data, roi_1_pts_np, roi_2_pts_np, motion_threshold=350, motion_direction_window=1):
    motion_detected_flag = False
    counter = 1  # Add counter to track videos
    recording_after_motion = False
    recording_end_time = None
    out_motion = None
    cap_verify = cv2.VideoCapture(video_path_data)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)

    if not cap_verify.isOpened():
        print(f"Error: Could not open video for verification: {video_path_data}")
        return False

    while cap_verify.isOpened():
        ret, frame = cap_verify.read()
        if not ret:
            break
        
        combined_frame1 = frame.copy()
        combined_frame2 = frame.copy()
        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)  # Blur for noise reduction

        # Motion in ROI 1
        combined_frame1, thresh_ROI1, person_detected, detections = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)
        motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > motion_threshold
        cv2.polylines(combined_frame1, [roi_1_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
        motion_mask_1 = np.zeros_like(combined_frame1)
        cv2.fillPoly(motion_mask_1, [roi_1_pts_np], (0, 255, 0))
        combined_frame1 = cv2.addWeighted(combined_frame1, alpha, motion_mask_1, 1-alpha, 0)

        # Motion in ROI 2
        combined_frame2, thresh_ROI2, _, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np)
        motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > motion_threshold
        cv2.polylines(combined_frame2, [roi_2_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
        motion_mask_2 = np.zeros_like(frame)
        cv2.fillPoly(motion_mask_2, [roi_2_pts_np], (0,0,0))  
        combined_frame2 = cv2.addWeighted(combined_frame2, alpha, motion_mask_2, 1-alpha, 0)

        # Combine ROI1 & ROI2 in one frame
        combined_frame = cv2.add(combined_frame1, combined_frame2)
        motion_frame_buffer.append(combined_frame.copy())

        # Process motion detection logic (similar to before)
        if motion_in_roi2:
            roi2_motion_time = time.time()

        if motion_in_roi1:
            roi1_motion_time = time.time()

        if motion_in_roi1 and person_detected:
            if roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window:
                motion_in_roi2_to_roi1 = True  # Direction is confirmed from ROI2 to ROI1 
                roi_color = (0, 0, 255) if motion_in_roi2_to_roi1 else (0, 255, 0)   
                cv2.fillPoly(combined_frame, [roi_1_pts_np], roi_color) 
                
            if not motion_detected_flag:
                # Create the video writer only for even-numbered videos
                if counter % 2 == 1:  # Save only even-numbered videos
                    out_motion = None 
                    video_path = save_video()
                    out_motion = cv2.VideoWriter(video_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))

                    while motion_frame_buffer:
                        out_motion.write(motion_frame_buffer.popleft())

                    snapshot_path = save_snapshot(combined_frame)
                    if snapshot_path:
                        current_time = datetime.now()
                        # snapshot_url = aws.upload_snapshot_to_s3bucket(snapshot_path)
                        # mongo_handler.save_snapshot_to_mongodb(snapshot_url, current_time)
                    
                    start_time = current_time
                    motion_detected_flag = True
                    motion_frame_buffer.clear()

                recording_after_motion = False

            if out_motion is not None:  # Only write to file if it's an even-numbered video
                out_motion.write(combined_frame)

        elif motion_detected_flag:
            # No motion: start countdown for post-motion recording
            if not recording_after_motion:
                recording_end_time = time.time() + 8  # Record for 5 more seconds
                recording_after_motion = True

            if out_motion is not None and time.time() <= recording_end_time:
                out_motion.write(combined_frame)
            else:
                if out_motion is not None:
                    out_motion.release()
                    motion_detected_flag = False
                    recording_after_motion = False

                    # Increment the counter after a video has been saved
                    counter += 1

                    if counter % 2 == 1:  # Upload only even-numbered videos to MongoDB
                        video_url = aws.upload_video_to_s3bucket(video_path)
                        mongo_handler.save_video_to_mongodb(video_url, start_time)
                        threading.Thread(target=email.send_alert_email, args=(snapshot_path, video_url)).start()

                # Clear the buffer and reset all relevant flags and variables
                    motion_frame_buffer.clear()
                    roi1_motion_time = None
                    roi2_motion_time = None
                    recording_end_time = None
                    out_motion = None


    cap_verify.release()



