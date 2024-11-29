import cv2
import numpy as np
import sys
import logging
from collections import deque
from datetime import datetime
from MetalTheft.constant import *
from MetalTheft.vid_stabilisation import VideoStabilizer
from MetalTheft.exception import MetalTheptException
from MetalTheft.send_email import EmailSender
from MetalTheft.utils.utils import save_snapshot, normalize_illumination, save_video, draw_boxes
from MetalTheft.motion_detection import detect_motion, detect_motion_Roi2
from MetalTheft.roi_selector import ROISelector
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.aws import AWSConfig
from ultralytics import YOLO
from ultralytics import YOLO
logging.getLogger('ultralytics').setLevel(logging.WARNING) 

# Flags and variables to track ROI selections
roi_1_set = False
roi_2_set = False
roi_1_pts_np = None  # First ROI (for motion detection)
roi_2_pts_np = None  # Second ROI (for contour highlighting)

# Initialization of required components and variables
motion_detected_flag = False
start_time = None
counter = 1
buffer_duration_seconds = 2
fps = 30  
frame_buffer = deque(maxlen=fps * buffer_duration_seconds)

email = EmailSender()
roi_selector = ROISelector()
mongo_handler = MongoDBHandler()
stabiliser = VideoStabilizer()
aws = AWSConfig()
model = YOLO('yolov8n.pt')  
rtsp_url = RTSP_URL 
# cap = cv2.VideoCapture('DATA/Project Theft 03-10-2024.mp4')
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    sys.exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(3))  # Width of the frames
frame_height = int(cap.get(4))  # Height of the frames
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter('Project2.mp4', fourcc, 20.0, (frame_width, frame_height))               # Full video recorder



cv2.namedWindow('IP Camera Feed')
cv2.setMouseCallback('IP Camera Feed', roi_selector.select_point)
# fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)
fgbg = cv2.createBackgroundSubtractorMOG2()


while True:
    try:
        # Read a frame from the video feed
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to receive frame from video. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            continue

        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)  # Blur for noise reduction

        # If ROI 1 is not set, prompt user to select it
        if not roi_1_set:
            cv2.putText(frame, "Select first ROI for motion detection", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('IP Camera Feed', frame)

            if roi_selector.is_roi_selected():
                roi_1_pts_np = roi_selector.get_roi_points()
                roi_1_set = True
                roi_selector.reset_roi()

        # If ROI 1 is set and ROI 2 is not, prompt user to select ROI 2
        elif not roi_2_set:
            cv2.putText(frame, "Select second ROI for highlighting", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('IP Camera Feed', frame)

            if roi_selector.is_roi_selected():
                roi_2_pts_np = roi_selector.get_roi_points()
                roi_2_set = True
                roi_selector.reset_roi()

        # If both ROIs are set, start processing motion and detection
        elif roi_1_set and roi_2_set:
            combined_frame, thresh_ROI1, person_detected, detections = detect_motion(
                frame, blurred_frame, model, fgbg, roi_1_pts_np
            )

            # Check for motion in the first ROI
            motion_in_roi = cv2.countNonZero(thresh_ROI1) > 250
            roi_color_1 = (0, 0, 255) if motion_in_roi and person_detected else (0, 255, 0)

            # Highlight ROI 1 based on motion detection
            motion_mask = np.zeros_like(combined_frame)
            cv2.fillPoly(motion_mask, [roi_1_pts_np], roi_color_1)
            combined_frame = cv2.addWeighted(combined_frame, 1.0, motion_mask, 0.3, 0)

            # Overlay the second ROI mask on top
            # motion_mask_2 = np.zeros_like(frame)
            # cv2.fillPoly(motion_mask_2, [roi_2_pts_np], (25, 155, 255))
            # combined_frame = cv2.addWeighted(combined_frame, 1.0, motion_mask_2, 0.5, 0)
                        # Process second ROI motion detection and draw boxes
                        
            combined_frame, thresh_ROI2 = detect_motion_Roi2(combined_frame, fgbg, roi_2_pts_np)
            out.write(combined_frame)
            if motion_in_roi:
                # Find and draw contours for motion in ROI
                contours, _ = cv2.findContours(thresh_ROI1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) < 200:  # Ignore small contours
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    if cv2.pointPolygonTest(roi_1_pts_np, (x, y), False) >= 0:
                        cv2.drawContours(combined_frame, contours, -1, (0, 255, 0), 2)

                # Handle detection events, saving video and sending snapshots
                if not motion_detected_flag:
                    video_path = save_video(counter)
                    out_motion = cv2.VideoWriter(video_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
                    snapshot_path = save_snapshot(combined_frame)
                    
                    if snapshot_path:
                        current_time = datetime.now()
                        # aws.upload_snapshot_to_s3bucket(snapshot_path)
                        # mongo_handler.save_snapshot_to_mongodb(snapshot_url, current_time)

                    while frame_buffer:
                        out_motion.write(frame_buffer.popleft())

                    start_time = current_time
                    motion_detected_flag = True
                    out_motion.release()
                    # aws.upload_video_to_s3bucket(video_path)
                    # mongo_handler.save_video_to_mongodb(video_url, start_time)
                    # email.send_alert_email(snapshot_path, video_url)

            else:
                if motion_detected_flag:
                    end_time = datetime.now()
                    if (end_time - start_time).total_seconds() > 1:
                        counter += 1
                    motion_detected_flag = False


            draw_boxes(combined_frame, detections)

            # Display motion detection results
            cv2.imshow('Motion', thresh_ROI1)
            cv2.imshow('Motion2_det', thresh_ROI2)
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
out.release()
cap.release()
cv2.destroyAllWindows()
