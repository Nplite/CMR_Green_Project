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
from vidgear.gears import CamGear, WriteGear

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



# Define output parameters for WriteGear
output_params = {
    "-input_framerate": moton_buffer_fps,
    "-vcodec": "libx264",
    "-preset": "ultrafast",
    "-crf": 22
}

def verify_motion_in_video(video_path, roi_1_pts_np, roi_2_pts_np, snapshot_path, camera_id, motion_threshold=300):

    global motion_detected_flag  
    global recording_after_motion
    global recording_end_time
    global last_motion_time
    global out_motion
    
    # Initialize video stream
    # stream = CamGear(source=RTSP_URL, stream_mode=True, logging=True, **stream_options).start()
    stream = CamGear(source=video_path, logging=True).start()


    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)

    frame_count = 0  # To keep track of the number of frames
    while True:
        frame = stream.read()
        if frame is None or frame.size == 0:
            break

        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)  # Blur for noise reduction
        frame_count += 1  
        motion_direction_window = 1
        combined_frame1 = frame.copy()
        combined_frame2 = frame.copy()
        motion_frame_buffer.append(frame.copy())


    
        # Motion in ROI 1
        combined_frame1, thresh_ROI1,_, detections = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)
        motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > motion_threshold
        cv2.polylines(combined_frame1, [roi_1_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
      
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


        # Person detection after 20 frames
        if frame_count > 20:
            _, _, person_in_roi2, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np)


        if motion_in_roi2:
            roi2_motion_time = time.time()
            draw_motion_contours(frame=combined_frame2, thresh=thresh_ROI2, roi_pts = roi_2_pts_np)

        if motion_in_roi1:
            roi1_motion_time = time.time()
            draw_motion_contours(frame=combined_frame1, thresh=thresh_ROI1, roi_pts = roi_1_pts_np)

        # Combine ROI1 & ROI2 in one frame
        combined_frame = cv2.add(combined_frame1, combined_frame2)

        current_time11 = datetime.now()
        if (motion_in_roi1 and person_in_roi2):
            if last_motion_time is None or (current_time11 - last_motion_time).total_seconds() > 3:
                if (roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window):
                    motion_in_roi2_to_roi1 = True  # Direction is confirmed from ROI2 to ROI1 
                    roi_color = (0, 0, 255) if motion_in_roi2_to_roi1 else (0, 255, 0)
                    cv2.fillPoly(combined_frame, [roi_1_pts_np], roi_color)  
                    
                    last_motion_time = current_time11
                else:
                    motion_in_roi2_to_roi1 = False


        if (motion_in_roi1 and person_in_roi2 ):
            if (roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window):
                if not motion_detected_flag:
                    
                    video_path = save_video()
                    out_motion = WriteGear(output=video_path, compression_mode=True, logging=True, **output_params)
                    # snapshot_path = save_snapshot(combined_frame)
                    while motion_frame_buffer:
                        out_motion.write(motion_frame_buffer.popleft())
                    # Set motion flags
                    motion_detected_flag = True
                    recording_after_motion = False
                    motion_frame_buffer.clear()  # Clear the buffer

                # Write current frame
                out_motion.write(combined_frame)

        elif motion_detected_flag:
            # No motion: start countdown for post-motion recording
            if not recording_after_motion:
                recording_end_time = time.time() + 5  # Record for 5 more seconds
                recording_after_motion = True

            # Write post-motion frames
            if time.time() <= recording_end_time:
                
                out_motion.write(combined_frame)
            else:
                # Stop recording after 5 seconds of no motion
                motion_detected_flag = False
                recording_after_motion = False
                out_motion.close()


                
                start_time = datetime.now()
                video_url = aws.upload_video_to_s3bucket(video_path, camera_id)                  
                snapshot_url = aws.upload_snapshot_to_s3bucket(snapshot_path, camera_id)
                threading.Thread(target=mongo_handler.save_snapshot_to_mongodb, args=(snapshot_url, start_time, camera_id)).start()
                threading.Thread(target=mongo_handler.save_video_to_mongodb, args=(video_url, start_time, camera_id)).start()
                threading.Thread(target=email.send_alert_email, args=(snapshot_path, video_url, camera_id)).start()
              
                return "Yes"

    stream.stop()
    return "No"




