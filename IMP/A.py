import cv2
import numpy as np
import sys
import time
import queue
import threading
import logging
from collections import deque
from datetime import datetime
from MetalTheft.constant import *
from MetalTheft.exception import MetalTheptException
from MetalTheft.send_email import EmailSender
from MetalTheft.utils.utils import save_snapshot, normalize_illumination, save_video, draw_boxes
from MetalTheft.motion_detection import detect_motion
from MetalTheft.roi_selector import ROISelector
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.aws import AWSConfig
# from M import verify_motion_in_video11
from Research.verify import verify_motion_in_video
from ultralytics import YOLO
logging.getLogger('ultralytics').setLevel(logging.WARNING) 



# Flags and variables to track ROI selections
roi_1_set, roi_2_set = False, False
roi_1_pts_np, roi_2_pts_np = None, None
alpha, counter = 0.6, 1 
last_motion_time, start_time = None, None
motion_direction_window = 1
motion_detected_flag = False
out_motion = None
email = EmailSender()
roi_selector = ROISelector()
mongo_handler = MongoDBHandler()
aws = AWSConfig()
model = YOLO('yolov8n.pt', verbose=True)  
rtsp_url = RTSP_URL 
cap = cv2.VideoCapture('DATA/09.10.2024.mp4')


# Get the actual FPS from the camera
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:  # Fallback if the camera does not provide FPS
    fps = 30

motion_buffer_duration = 5  # Duration before and after motion
moton_buffer_fps = 30
motion_frame_buffer = deque(maxlen=moton_buffer_fps * motion_buffer_duration)  # Main camera feed buffer

# Variables to track recording after motion stops
recording_after_motion = False
recording_end_time = None


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
height ,width =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

cv2.namedWindow('IP Camera Feed')
cv2.setMouseCallback('IP Camera Feed', roi_selector.select_point)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    sys.exit()

while True:
    try:
        # Read a frame from the video feed
        ret, frame = cap.read()
        # frame = normalize_illumination(frame)
        if not ret:
            print("Error: Failed to receive frame from video. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            continue

        # If ROI 1 is not set, prompt user to select it
        if not roi_1_set:
            cv2.putText(frame, "Select first ROI for motion detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('IP Camera Feed', frame)
            if roi_selector.is_roi_selected():
                roi_1_pts_np = roi_selector.get_roi_points()
                roi_1_set = True
                roi_selector.reset_roi()

        # If ROI 1 is set and ROI 2 is not, prompt user to select ROI 2
        elif not roi_2_set:
            cv2.putText(frame, "Select second ROI for highlighting", (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('IP Camera Feed', frame)
            if roi_selector.is_roi_selected():
                roi_2_pts_np = roi_selector.get_roi_points()
                roi_2_set = True
                roi_selector.reset_roi()

        elif roi_1_set and roi_2_set:
            combined_frame1 = frame.copy()
            combined_frame2 = frame.copy()
            motion_frame_buffer.append(frame.copy())
            
            blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)  # Blur for noise reduction

            # Motion in ROI 1
            combined_frame1, thresh_ROI1, person_detected, detections = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)
            motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > 350
            cv2.polylines(combined_frame1, [roi_1_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
            roi_color_1 = (0, 0, 255) if motion_in_roi1 and person_detected else (0, 255, 0)
            motion_mask_1 = np.zeros_like(combined_frame1)
            cv2.fillPoly(motion_mask_1, [roi_1_pts_np], (0, 255, 0))
            combined_frame1 = cv2.addWeighted(combined_frame1, alpha, motion_mask_1, 1-alpha, 0)

            # Motion in ROI 2
            combined_frame2, thresh_ROI2, _, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np )
            motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > 350
            cv2.polylines(combined_frame2, [roi_2_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
            motion_mask_2 = np.zeros_like(frame)
            cv2.fillPoly(motion_mask_2, [roi_2_pts_np], (0,0,0))  
            combined_frame2 = cv2.addWeighted(combined_frame2, alpha, motion_mask_2, 1-alpha, 0)

            # Combine ROI1 & ROI2 in one frame
            combined_frame = cv2.add(combined_frame1, combined_frame2)


            if motion_in_roi2:
                roi2_motion_time = time.time()
                contours, _ = cv2.findContours(thresh_ROI2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) < 200:  # Ignore small contours
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    if cv2.pointPolygonTest(roi_2_pts_np, (x, y), False) >= 0:
                        cv2.drawContours(combined_frame, contours, -1, (0, 255, 0), 2)

            if motion_in_roi1:
                roi1_motion_time = time.time()
                contours, _ = cv2.findContours(thresh_ROI1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) < 200:  # Ignore small contours
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    if cv2.pointPolygonTest(roi_1_pts_np, (x, y), False) >= 0:
                        cv2.drawContours(combined_frame, contours, -1, (0, 255, 0), 2)


            if motion_in_roi1 and person_detected:
                if roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window:
                    motion_in_roi2_to_roi1 = True  # Direction is confirmed from ROI2 to ROI1 
                    roi_color = (0, 0, 255) if motion_in_roi2_to_roi1 else (0, 255, 0)   
                    cv2.fillPoly(combined_frame, [roi_1_pts_np], roi_color)    

                else:
                    motion_in_roi2_to_roi1 = False 

                # Motion detected: start or continue recording
                if roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window:
                    if not motion_detected_flag:
                        video_path = save_video()
                        out_motion = cv2.VideoWriter(video_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
                        snapshot_path = save_snapshot(combined_frame)
                        while motion_frame_buffer:
                            out_motion.write(motion_frame_buffer.popleft())
                            
                        # Set motion flags
                        motion_detected_flag = True
                        recording_after_motion = False
                        motion_frame_buffer.clear()  # Clear the buffer
                    
                    # Write current frame
                    out_motion.write(frame)  
                    recording_end_time = time.time() + 8 

            elif motion_detected_flag:
                if time.time() <= recording_end_time:
                    out_motion.write(frame)
                else:
                    # Stop recording after 5 seconds of no motion
                    motion_detected_flag = False
                    recording_after_motion = False
                    out_motion.release()
                    out_motion = None
                    roi1_motion_time = None
                    roi2_motion_time = None
                    recording_end_time = None
                    out_motion = None
                    counter += 1
                    
                    

                    # Technic 1:
                    result = threading.Thread(target= verify_motion_in_video, args = (video_path, roi_1_pts_np, roi_2_pts_np, snapshot_path)).start()
                    # print(f"Motion and person detected: {result}")



                    # Technic 2:
                    # Example usage with threading and queue
                    # result_queue = queue.Queue()
                    # motion_thread = threading.Thread(target=verify_motion_in_video, args=(video_path, roi_1_pts_np, roi_2_pts_np, 350, 1, result_queue))
                    # motion_thread.start()
                    # motion_thread.join()
                    # # Get the result from the queue
                    # result = result_queue.get()
                    # start_time = datetime.now()
                    # print(f"Motion and person detected: {result}")
                    # if result == "Yes":
                    #     snapshot_url = aws.upload_snapshot_to_s3bucket(snapshot_path)
                    #     mongo_handler.save_snapshot_to_mongodb(snapshot_url, start_time)
                    #     video_url = aws.upload_video_to_s3bucket(video_path)
                    #     mongo_handler.save_video_to_mongodb(video_url, start_time)
                    #     threading.Thread(target=email.send_alert_email, args=(snapshot_path, video_url)).start()
            


            draw_boxes(combined_frame, detections)

            # Display motion detection results
            cv2.imshow('Motion', thresh_ROI1)
            cv2.imshow("imgRegion", combined_frame)

        else:
            # Default display if no ROIs are set
            cv2.imshow('IP Camera Feed', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):                
            roi_1_set = False
            roi_2_set = False
            roi_selector.reset_roi()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise MetalTheptException(e, sys) from e

# Release resources
cap.release()
cv2.destroyAllWindows()

