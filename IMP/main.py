import cv2
import sys
import numpy as np
import threading
import os
from ultralytics import YOLO
from datetime import datetime
from MetalTheft.constant import *
from MetalTheft.exception import MetalTheptException
from MetalTheft.send_email import EmailSender
from MetalTheft.utils.utils import  save_snapshot
from MetalTheft.motion_detection import detect_motion



motion_detected_flag = False
start_time = None
counter = 1


# Initialize video capture with RTSP stream
rtsp_url = "rtsp://ProjectTheft2024:Theft@2024@103.106.195.202:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(rtsp_url)
# cap = cv2.VideoCapture('./Videos/npl.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()
roi_pts = [517, 116, 529, 124, 552, 713, 384, 719]
email = EmailSender()



while True:
    try:
        ret, frame = cap.read()
        if not ret:
            break

        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)

        roi_pts_np = np.array(roi_pts, np.int32).reshape((-1, 1, 2))

        # Call the motion detection function
        combined_frame, thresh, person_detected, detections = detect_motion(
            frame, blurred_frame, model, fgbg, roi_pts_np
        )

        # Check if there is motion in the ROI
        motion_in_roi = cv2.countNonZero(thresh) > 300

        # If motion is detected in the ROI and a person is detected, trigger email notification
        if motion_in_roi and person_detected:
            if not motion_detected_flag:
                snapshot_path = save_snapshot(combined_frame)
                # if snapshot_path:
                    # threading.Thread(target=email.send_email, args=(snapshot_path,)).start()   
                start_time = datetime.now()
                motion_detected_flag = True
        else:
            if motion_detected_flag:
                end_time = datetime.now()
                if (end_time - start_time).total_seconds() > 1:
                    counter += 1
                motion_detected_flag = False

        if motion_detected_flag:
            roi_color = (0, 0, 255)  # Red
        else:
            roi_color = (0, 255, 0)  # Green

        # Create a mask image filled with the appropriate color
        motion_mask = np.zeros(combined_frame.shape, dtype=np.uint8)
        cv2.fillPoly(motion_mask, [roi_pts_np], roi_color)

        # Apply the mask to the frame using cv2.addWeighted()
        alpha = 0.8
        combined_frame = cv2.addWeighted(combined_frame, alpha, motion_mask, 1 - alpha, 0)


        # Show the threshold frame if it is defined
        cv2.imshow('Motion', thresh)

        # Show the original frame with bounding boxes
        cv2.imshow("imgRegion", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        raise MetalTheptException(e, sys) from e

cap.release()
cv2.destroyAllWindows()




