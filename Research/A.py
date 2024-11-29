import torch
import cv2
import time
import logging
import threading
import numpy as np
from datetime import datetime
from collections import deque
from vidgear.gears import CamGear
from MetalTheft.motion_detection import detect_motion
from ultralytics import YOLO
from MetalTheft.utils.utils import save_snapshot, draw_motion_contours, save_video
from MetalTheft.vid_stabilisation import VideoStabilizer
from MetalTheft.send_email import EmailSender
from MetalTheft.roi_selector import ROISelector
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.aws import AWSConfig
import os
import queue

# Logging configuration
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Global variables
motion_detected_flag = False
motion_buffer_duration = 5  # Duration before and after motion in seconds
motion_buffer_fps = 30
motion_frame_buffer = deque(maxlen=motion_buffer_fps * motion_buffer_duration)
recording_after_motion = False
recording_end_time = None
last_motion_time = None
alpha = 0.8

# Initialize objects
model = YOLO('yolov8n.pt', verbose=True)
email = EmailSender()
roi_selector = ROISelector()
mongo_handler = MongoDBHandler()
stabiliser = VideoStabilizer()
aws = AWSConfig()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Function to verify motion in video
def verify_motion_in_video(video_path, roi_1_pts_np, roi_2_pts_np, snapshot_path, camera_id, result_queue, motion_threshold=300):
    global last_motion_time

    video_path = os.path.abspath(video_path)
    stream = CamGear(source=video_path, logging=True).start()
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)
    motion_direction_window = 1  # Window for motion direction validation
    frame_count = 0

    while True:
        frame = stream.read()
        if frame is None or frame.size == 0:
            break

        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)

        # Detect motion in ROI 1
        _, thresh_ROI1, _, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)
        motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > motion_threshold
        if motion_in_roi1:
            roi1_motion_time = time.time()

        # Detect motion and person in ROI 2
        _, thresh_ROI2, person_in_roi2, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np)
        motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > motion_threshold
        if motion_in_roi2:
            roi2_motion_time = time.time()

        # Validate motion direction and presence
        if frame_count > 15:
            _, _, person_in_roi2, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np)

        current_time = datetime.now()
        if motion_in_roi1 and person_in_roi2:
            if last_motion_time is None or (current_time - last_motion_time).total_seconds() > 3:
                if roi2_motion_time and (roi1_motion_time - roi2_motion_time) <= motion_direction_window:
                    last_motion_time = current_time

                    # Process motion event
                    start_time = datetime.now()
                    video_url = aws.upload_video_to_s3bucket(video_path, camera_id)
                    snapshot_url = aws.upload_snapshot_to_s3bucket(snapshot_path, camera_id)

                    threading.Thread(target=mongo_handler.save_snapshot_to_mongodb, args=(snapshot_url, start_time, camera_id)).start()
                    threading.Thread(target=mongo_handler.save_video_to_mongodb, args=(video_url, start_time, camera_id)).start()
                    threading.Thread(target=email.send_alert_email, args=(snapshot_path, video_url, camera_id)).start()

                    result_queue.put("Yes")
                    stream.stop()
                    return

        frame_count += 1

    result_queue.put("No")
    stream.stop()
object_counter = 0


# Example of queue operation and threading
result_queue = queue.Queue()
thread = threading.Thread(
    target=verify_motion_in_video,
    args=("video_path.mp4", [[100, 100], [200, 100], [200, 200], [100, 200]],
          [[150, 150], [250, 150], [250, 250], [150, 250]],
          "snapshot.jpg", "camera_id", result_queue)
)
thread.start()
thread.join()  # Wait for the thread to finish
result = result_queue.get()

print("******************************************************************************")
print("Motion detection result:", result)

if result == "Yes":
    object_counter += 1
