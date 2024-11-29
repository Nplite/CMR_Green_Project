import cv2
import numpy as np
import cv2
import sys
import numpy as np
import threading
import logging
from collections import deque
from ultralytics import YOLO
from datetime import datetime
from MetalTheft.constant import *
from MetalTheft.vid_stabilisation import VideoStabilizer
from MetalTheft.exception import MetalTheptException
from MetalTheft.send_email import EmailSender
from MetalTheft.utils.utils import save_snapshot, normalize_illumination, save_video, draw_boxes
from MetalTheft.motion_detection import detect_motion
from MetalTheft.roi_selector import ROISelector
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.aws import AWSConfig
logging.getLogger('ultralytics').setLevel(logging.WARNING) 

motion_detected_flag = False
start_time = None
counter = 1
buffer_duration_seconds = 2
fps = 30  
frame_buffer = deque(maxlen=fps * buffer_duration_seconds)
roi_1_set = None
roi_2_set = None
roi_3_set = None

email = EmailSender()
roi_selector = ROISelector()
mongo_handler = MongoDBHandler()
stabiliser = VideoStabilizer()
aws = AWSConfig()
model = YOLO('yolov8n.pt')
rtsp_url = RTSP_URL
# cap = cv2.VideoCapture(rtsp_url)
cap = cv2.VideoCapture('rtsp://admin:secure@123@201.202.202.38/cam/realmonitor?channel=1&subtype=0')

frame_width = int(cap.get(3))  # Width of the frames
frame_height = int(cap.get(4))  # Height of the frames
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter('Project1.mp4', fourcc, 20.0, (frame_width, frame_height))            



fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow('IP Camera Feed')
cv2.setMouseCallback('IP Camera Feed', roi_selector.select_point, param=None)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)


if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    sys.exit()

while True:
    try:
        ret, frame = cap.read()
        
        
        if not ret:
            print("Error: Failed to receive frame from RTSP stream. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            continue

        # frame = normalize_illumination(frame)
        # frame = stabiliser.stabilised_frame(frame)
        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)


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

        # If ROI 1 is not set, prompt user to select it
        elif not roi_3_set:
            cv2.putText(frame, "Select Third ROI for motion detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('IP Camera Feed', frame)
            if roi_selector.is_roi_selected():
                roi_3_pts_np = roi_selector.get_roi_points()
                roi_3_set = True
                roi_selector.reset_roi()
        
        elif roi_1_set and roi_2_set and roi_3_set:

            combined_frame, thresh, person_detected, detections = detect_motion(
                frame, blurred_frame, model, fgbg, roi_1_set)

            motion_in_roi = cv2.countNonZero(thresh) > 350

            roi_color = (0, 0, 255) if motion_in_roi and person_detected else (0, 255, 0)
            motion_mask = np.zeros(combined_frame.shape, dtype=np.uint8)
            cv2.fillPoly(motion_mask, [roi_1_set], roi_color)
            alpha = 1.0
            beta = 0.8
            combined_frame = cv2.addWeighted(combined_frame, alpha, motion_mask, 1 - beta, 0)
            frame_buffer.append(combined_frame.copy())
            out.write(combined_frame)

            

            if motion_in_roi:
                # Find contours in the motion mask (thresh)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if cv2.contourArea(contour) < 200:  # Filter out small motions
                        continue
  
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if the bounding box is inside the ROI
                    if cv2.pointPolygonTest(roi_1_set, (x, y), False) >= 0:
                        # Draw the bounding box around the detected motion
                        # cv2.rectangle(combined_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.drawContours(combined_frame, contours, -1, (0, 255, 0), 2)

            if motion_in_roi and person_detected:
                if not motion_detected_flag:
                    video_path = save_video()
                    out_motion = cv2.VideoWriter(video_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
                    snapshot_path = save_snapshot(combined_frame)
                    if snapshot_path:
                        
                        current_time = datetime.now()
                        # snapshot_url = aws.upload_snapshot_to_s3bucket(snapshot_path)
                        # mongo_handler.save_snapshot_to_mongodb(snapshot_url, current_time)

                    while frame_buffer:
                        out_motion.write(frame_buffer.popleft())

                    start_time = current_time
                    motion_detected_flag = True
                    out_motion.release()
                    # video_url = aws.upload_video_to_s3bucket(video_path)
                    # mongo_handler.save_video_to_mongodb(video_url, start_time)
                    # threading.Thread(target=email.send_alert_email, args=(snapshot_path, video_url)).start()

            else:
                if motion_detected_flag:
                    end_time = datetime.now()
                    if (end_time - start_time).total_seconds() > 1:
                        counter += 1
                    motion_detected_flag = False

            draw_boxes(combined_frame, detections)
            cv2.imshow('Motion', thresh)
            cv2.imshow("imgRegion", combined_frame)

        else:
            cv2.imshow('IP Camera Feed', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):                
            roi_selector.reset_roi()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise MetalTheptException(e, sys) from e
    
out.release()
cap.release()
cv2.destroyAllWindows()


