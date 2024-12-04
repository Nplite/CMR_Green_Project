import cv2
import numpy as np
import sys
import time
import logging
from collections import deque
from datetime import datetime
from MetalTheft.constant import *
from MetalTheft.exception import MetalTheptException
from MetalTheft.send_email import EmailSender
from MetalTheft.utils.utils import save_snapshot, save_video, draw_boxes, draw_motion_contours
from MetalTheft.motion_detection import detect_motion, person_detection_ROI3
from MetalTheft.roi_selector import ROISelector
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.aws import AWSConfig
from ultralytics import YOLO
from Research.verify import verify_motion_in_video
logging.getLogger('ultralytics').setLevel(logging.WARNING) 
roi_1_set, roi_2_set, roi_3_set = False, False, False
roi_1_pts_np, roi_2_pts_np = None, None
alpha, counter = 0.6, 1 
last_motion_time, start_time = None, None
motion_direction_window = 1
motion_detected_flag,person_detected_flag = False, False
roi2_motion_time = None
email = EmailSender()
roi_selector = ROISelector()
mongo_handler = MongoDBHandler()
aws = AWSConfig()
# model = YOLO('/home/alluvium/Desktop/Namdeo/CMR_Project/yolov8n.engine',task ='detect', verbose=True)  
model = YOLO('yolov8n.pt',verbose=True)  
RTSP = 'rtsp://admin:Control@12@201.202.202.20/cam/realmonitor?channel=1&subtype=0'
cap = cv2.VideoCapture(RTSP)
Manual_Thresh = 300
object_counter = 0

# Get the actual FPS from the camera
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:  # Fallback if the camera does not provide FPS
    fps = 30

motion_buffer_duration = 5  # Duration before and after motion
moton_buffer_fps = 30  
motion_frame_buffer = deque(maxlen=moton_buffer_fps * motion_buffer_duration)  
recording_after_motion = False
recording_end_time = None


# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# height ,width =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

cv2.namedWindow('IP Camera Feed')
cv2.setMouseCallback('IP Camera Feed', roi_selector.select_point)
# fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)
fgbg = cv2.createBackgroundSubtractorMOG2()

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    sys.exit()

# Full Video Recorder
frame_width = int(cap.get(3))  
frame_height = int(cap.get(4)) 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
# Full_Video = cv2.VideoWriter('MetalTheft', fourcc, 20.0, (frame_width, frame_height))               


while True:
    try:
        # Read a frame from the video feed
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (1480,800))


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
            combined_frame1 = frame.copy()
            combined_frame2 = frame.copy()
            combined_frame3 = frame.copy()
            motion_frame_buffer.append(frame.copy())
            blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)  # Blur for noise reduction

            # Motion in ROI 1
            combined_frame1, thresh_ROI1,_, detections = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)
            motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > Manual_Thresh
            cv2.polylines(combined_frame1, [roi_1_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
            # roi_color_1 = (0, 0, 255) if motion_in_roi1 and person_detected else (0, 255, 0)
            motion_mask_1 = np.zeros_like(combined_frame1)
            cv2.fillPoly(motion_mask_1, [roi_1_pts_np], (0, 255, 0))
            combined_frame1 = cv2.addWeighted(combined_frame1, alpha, motion_mask_1, 1-alpha, 0)

            # Motion in ROI 2
            combined_frame2, thresh_ROI2, person_in_roi2, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np )
            motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > 300
            cv2.polylines(combined_frame2, [roi_2_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
            motion_mask_2 = np.zeros_like(frame)
            cv2.fillPoly(motion_mask_2, [roi_2_pts_np], (0,0,0))  
            combined_frame2 = cv2.addWeighted(combined_frame2, alpha, motion_mask_2, 1-alpha, 0)

            # combined_frame3, person_in_roi3 = person_detection_ROI3(frame=frame, roi_points=roi_3_pts_np, model=model)
            # cv2.polylines(combined_frame3, [roi_3_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
            combined_frame3, person_in_roi3, person_count = person_detection_ROI3(frame=frame, roi_points=roi_3_pts_np, model=model)
            cv2.polylines(combined_frame3, [roi_3_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
            

            if motion_in_roi2:
                roi2_motion_time = time.time()
                # draw_motion_contours(frame=combined_frame2, thresh=thresh_ROI2, roi_pts = roi_2_pts_np)

            if motion_in_roi1:
                roi1_motion_time = time.time()
                draw_motion_contours(frame=combined_frame1, thresh=thresh_ROI1, roi_pts = roi_1_pts_np)

            # Combine ROI1 & ROI2 in one frame
            combined_frame = cv2.add(cv2.add(combined_frame1, combined_frame2), combined_frame3)

            current_time11 = datetime.now()
            if motion_in_roi1:# and person_in_roi2:
                if last_motion_time is None or (current_time11 - last_motion_time).total_seconds() > 3:
                    if (roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window):
                        motion_in_roi2_to_roi1 = True  # Direction is confirmed from ROI2 to ROI1 
                        roi_color = (0, 0, 255) if motion_in_roi2_to_roi1 else (0, 255, 0)
                        cv2.fillPoly(combined_frame, [roi_1_pts_np], roi_color)  
                        object_counter += 1  
                        last_motion_time = current_time11
                    else:
                        motion_in_roi2_to_roi1 = False


            if (motion_in_roi1 ):#and person_in_roi2) or person_in_roi3:
                if (roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window) or person_in_roi3:
                    if not motion_detected_flag:
                        
                        video_path = save_video()
                        out_motion = cv2.VideoWriter(video_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
                        snapshot_path = save_snapshot(combined_frame)
                        
                        # Write buffered frames before motion
                        while motion_frame_buffer:
                            out_motion.write(motion_frame_buffer.popleft())
                        # Set motion flags
                        motion_detected_flag = True
                        recording_after_motion = False
                        motion_frame_buffer.clear()  # Clear the buffer

                    # Write current frame
                    out_motion.write(frame)

            elif motion_detected_flag:
                # No motion: start countdown for post-motion recording
                if not recording_after_motion:
                    recording_end_time = time.time() + 5  # Record for 5 more seconds
                    recording_after_motion = True

                # Write post-motion frames
                if time.time() <= recording_end_time:
                    
                    out_motion.write(frame)
                else:
                    # Stop recording after 5 seconds of no motion
                    motion_detected_flag = False
                    recording_after_motion = False
                    out_motion.release()
                    

                    # video_url = aws.upload_video_to_s3bucket(video_path)
                    # mongo_handler.save_video_to_mongodb(video_url, start_time)
                    email.send_alert_email(snapshot_path, video_url)

                    # verify_motion_in_video(video_path, roi_1_pts_np, roi_2_pts_np, snapshot_path)
                    counter += 1

            
            elif person_in_roi3:
                if not person_detected_flag:
                    video_path = save_video()
                    out_motion = cv2.VideoWriter(video_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
                    snapshot_path = save_snapshot(combined_frame)
                    
                    # Write buffered frames before motion
                    while motion_frame_buffer:
                        out_motion.write(motion_frame_buffer.popleft())

                    # Set motion flags
                    person_detected_flag = True
                    recording_after_motion = False
                    motion_frame_buffer.clear()  # Clear the buffer

                # Write current frame
                out_motion.write(combined_frame)

            elif person_detected_flag:
                if not recording_after_motion:
                    recording_end_time = time.time() + 5  # Record for 5 more seconds
                    recording_after_motion = True

                # Write post-motion frames
                if time.time() <= recording_end_time:
                    
                    out_motion.write(combined_frame)
                else:
                    person_detected_flag = False
                    recording_after_motion = False
                    out_motion.release()
                    
                    start_time = datetime.now()
                    video_url = aws.upload_video_to_s3bucket(video_path)
                    mongo_handler.save_video_to_mongodb(video_url, start_time)
                    email.send_alert_email(snapshot_path, video_url)
                    counter += 1

            
            cv2.putText(combined_frame, f'Object Count: {object_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(combined_frame, f'Person Count: {person_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # Full_Video.write(combined_frame)
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
            roi_3_set = False
            roi_selector.reset_roi()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise MetalTheptException(e, sys) from e

# Release resources
# Full_Video.release()
cap.release()
cv2.destroyAllWindows()
