import numpy as np
import sys
import time
import logging
from collections import deque
from datetime import datetime
from vidgear.gears import CamGear, WriteGear
from MetalTheft.constant import *
from MetalTheft.exception import MetalTheptException
from MetalTheft.send_email import EmailSender
from MetalTheft.utils.utils import save_snapshot, save_video, draw_boxes, draw_motion_contours
from MetalTheft.motion_detection import detect_motion, person_detection_ROI3
from MetalTheft.roi_selector import ROISelector
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.aws import AWSConfig
from ultralytics import YOLO
import cv2

logging.getLogger('ultralytics').setLevel(logging.WARNING) 

# Flags and variables to track ROI selections
roi_1_set, roi_2_set, roi_3_set = False, False, False
roi_1_pts_np, roi_2_pts_np = None, None
alpha, counter = 0.6, 1 
last_motion_time, start_time = None, None
motion_direction_window = 1
motion_detected_flag, person_detected_flag = False, False
roi2_motion_time = None

# Initialize components
email = EmailSender()
roi_selector = ROISelector()
mongo_handler = MongoDBHandler()
aws = AWSConfig()
model = YOLO('/home/alluvium/Desktop/Namdeo/CMR_Project/yolov8n.engine', verbose=True)

# VidGear stream setup
stream_options = {
    "THREADED_QUEUE_MODE": True,  # Enable thread-safe mode
}

# Initialize video stream
# stream = CamGear(source=RTSP_URL, stream_mode=True, logging=True, **stream_options).start()
stream = CamGear(source='DATA/23.10.2024 Theft project.mp4', logging=True).start()

# Get frame dimensions for the first frame
first_frame = stream.read()
height, width = first_frame.shape[:2] if first_frame is not None else (720, 1280)

# Motion buffer setup
motion_buffer_duration = 5
moton_buffer_fps = 30
motion_frame_buffer = deque(maxlen=moton_buffer_fps * motion_buffer_duration)

# Recording setup
object_counter = -1

# Define output parameters for WriteGear
output_params = {
    "-input_framerate": moton_buffer_fps,
    "-vcodec": "libx264",
    "-preset": "ultrafast",
    "-crf": 22
}

# Full video recorder setup
full_video_writer = WriteGear(
    output='MetalTheft_Trial4.mp4',  # Changed from output_filename to output
    compression_mode=True,
    logging=True,
    **output_params
)

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Window setup
cv2.namedWindow('IP Camera Feed')
cv2.setMouseCallback('IP Camera Feed', roi_selector.select_point)

while True:
    try:
        # Read frame from the video stream
        frame = stream.read()
        if frame is None:
            print("Error: Failed to receive frame from video. Reconnecting...")
            stream.stop()
            stream = CamGear(source=RTSP_URL, stream_mode=True, logging=True).start()
            continue

        # ROI Selection Logic
        if not roi_1_set:
            cv2.putText(frame, "Select first ROI for motion detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('IP Camera Feed', frame)
            if roi_selector.is_roi_selected():
                roi_1_pts_np = roi_selector.get_roi_points()
                roi_1_set = True
                roi_selector.reset_roi()

        elif not roi_2_set:
            cv2.putText(frame, "Select second ROI for highlighting", (10, 30),  
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('IP Camera Feed', frame)
            if roi_selector.is_roi_selected():
                roi_2_pts_np = roi_selector.get_roi_points()
                roi_2_set = True
                roi_selector.reset_roi()

        elif not roi_3_set:
            cv2.putText(frame, "Select Third ROI for motion detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
            blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)

            # Motion detection in ROI 1
            combined_frame1, thresh_ROI1, _, detections = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)
            motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > 100
            cv2.polylines(combined_frame1, [roi_1_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
            motion_mask_1 = np.zeros_like(combined_frame1)
            cv2.fillPoly(motion_mask_1, [roi_1_pts_np], (0, 255, 0))
            combined_frame1 = cv2.addWeighted(combined_frame1, alpha, motion_mask_1, 1-alpha, 0)

            # Motion detection in ROI 2
            combined_frame2, thresh_ROI2, person_in_roi2, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np)
            motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > 350
            cv2.polylines(combined_frame2, [roi_2_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
            motion_mask_2 = np.zeros_like(frame)
            cv2.fillPoly(motion_mask_2, [roi_2_pts_np], (0,0,0))
            combined_frame2 = cv2.addWeighted(combined_frame2, alpha, motion_mask_2, 1-alpha, 0)

            # Person detection in ROI 3
            combined_frame3, person_in_roi3, person_count = person_detection_ROI3(frame=frame, roi_points=roi_3_pts_np, model=model)
            cv2.polylines(combined_frame3, [roi_3_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)

            if motion_in_roi2:
                roi2_motion_time = time.time()

            if motion_in_roi1:
                roi1_motion_time = time.time()
                draw_motion_contours(frame=combined_frame1, thresh=thresh_ROI1, roi_pts=roi_1_pts_np)

            # Combine all ROIs
            combined_frame = cv2.add(cv2.add(combined_frame1, combined_frame2), combined_frame3)

            current_time11 = datetime.now()
            if (motion_in_roi1 and person_in_roi2):
                if last_motion_time is None or (current_time11 - last_motion_time).total_seconds() > 3:
                    if (roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window):
                        motion_in_roi2_to_roi1 = True
                        roi_color = (0, 0, 255) if motion_in_roi2_to_roi1 else (0, 255, 0)
                        cv2.fillPoly(combined_frame, [roi_1_pts_np], roi_color)
                        object_counter += 1
                        last_motion_time = current_time11
                    else:
                        motion_in_roi2_to_roi1 = False

            # Motion detection and recording logic
            if (motion_in_roi1 and person_in_roi2) or person_in_roi3:
                if (roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window) or person_in_roi3:
                    if not motion_detected_flag:
                        video_path = save_video()
                        writer = WriteGear(
                            output=video_path,  # Changed from output_filename to output
                            compression_mode=True,
                            logging=True,
                            **output_params
                        )
                        snapshot_path = save_snapshot(combined_frame)
                        
                        # Write buffered frames
                        while motion_frame_buffer:
                            writer.write(motion_frame_buffer.popleft())
                        
                        motion_detected_flag = True
                        recording_after_motion = False
                        motion_frame_buffer.clear()

                    writer.write(combined_frame)

            elif motion_detected_flag:
                if not recording_after_motion:
                    recording_end_time = time.time() + 5
                    recording_after_motion = True

                if time.time() <= recording_end_time:
                    writer.write(combined_frame)
                else:
                    motion_detected_flag = False
                    recording_after_motion = False
                    writer.close()
                    counter += 1

            # Person detection recording logic
            elif person_in_roi3:
                if not person_detected_flag:
                    video_path = save_video()
                    writer = WriteGear(
                        output=video_path,  # Changed from output_filename to output
                        compression_mode=True,
                        logging=True,
                        **output_params
                    )
                    snapshot_path = save_snapshot(combined_frame)
                    
                    while motion_frame_buffer:
                        writer.write(motion_frame_buffer.popleft())
                    
                    person_detected_flag = True
                    recording_after_motion = False
                    motion_frame_buffer.clear()

                writer.write(combined_frame)

            elif person_detected_flag:
                if not recording_after_motion:
                    recording_end_time = time.time() + 5
                    recording_after_motion = True

                if time.time() <= recording_end_time:
                    writer.write(combined_frame)
                else:
                    person_detected_flag = False
                    recording_after_motion = False
                    writer.close()
                    
                    start_time = datetime.now()
                    video_url = aws.upload_video_to_s3bucket(video_path)
                    mongo_handler.save_video_to_mongodb(video_url, start_time)
                    email.send_alert_email(snapshot_path, video_url)
                    counter += 1

            # Add text overlays
            cv2.putText(combined_frame, f'Object Count: {object_counter}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(combined_frame, f'Person Count: {person_count}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Write to full video and draw detection boxes
            full_video_writer.write(combined_frame)
            draw_boxes(combined_frame, detections)

            # Display results
            cv2.imshow('Motion', thresh_ROI1)
            cv2.imshow("imgRegion", combined_frame)

        else:
            cv2.imshow('IP Camera Feed', frame)

        # Key handling
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

# Cleanup
full_video_writer.close()
stream.stop()
cv2.destroyAllWindows()