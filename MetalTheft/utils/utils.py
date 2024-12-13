
import sys
import os
import cv2
import time
import cv2
import asyncio
import smtplib
from datetime import datetime
from MetalTheft.constant import *
# from MetalTheft.logger import logging
from MetalTheft.exception import MetalTheptException
import traceback
import logging

module_directory = os.path.abspath("MetalTheft")
sys.path.insert(0, module_directory)
from aws import AWSConfig
from send_email import EmailSender
from mongodb import MongoDBHandlerSaving
from dotenv import load_dotenv
load_dotenv()
CC_EMAIL = os.getenv('CC_EMAIL')
mongo_handler_saving = MongoDBHandlerSaving()
aws = AWSConfig()
email = EmailSender(cc_email = CC_EMAIL)




def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"


def send_data_to_dashboard(video_path, snapshot_path, start_time, camera_id):
    try:
        video_url = aws.upload_video_to_s3bucket(video_path, camera_id)
        snapshot_url = aws.upload_snapshot_to_s3bucket(snapshot_path, camera_id)
        mongo_handler_saving.save_snapshot_to_mongodb(snapshot_url, start_time, camera_id)
        mongo_handler_saving.save_video_to_mongodb(video_url, start_time, camera_id)
        email.send_alert_email(snapshot_path, video_url, camera_id)

        logging.info(f"Data sending completed for camera_Id: {camera_id}.")
    except Exception as e:
        logging.info(f"Error sending data for camera {camera_id}: {str(e)}")

def save_snapshot(frame, camera_id):
    try:
        # Validate frame
        if frame is None:
            raise ValueError("Frame is None. Ensure the input frame is valid and not empty.")

        # Generate date and time strings
        date_str = datetime.now().strftime('%Y-%m-%d')
        time_str = datetime.now().strftime('%H-%M-%S')  # Use '-' for compatibility

        # Create directory structure
        directory = os.path.join('snapshots', date_str)
        os.makedirs(directory, exist_ok=True)

        # Generate filename with camera_id
        filename = os.path.join(directory, f'camera_{camera_id}_{time_str}.jpg')

        # Save the snapshot
        success = cv2.imwrite(filename, frame)
        if not success:
            raise IOError(f"Failed to write frame to file: {filename}")

        print(f"Snapshot saved: {filename}")
        logging.info(f"Snapshot saving is done and it save in this path: {filename}")
        return filename

    except Exception as e:
        raise MetalTheptException(e,sys) from e

def save_video():
    try:
    
        date_str = datetime.now().strftime('%Y-%m-%d')
        time_str = datetime.now().strftime('%H:%M:%S')
        directory = os.path.join('video', date_str)
        os.makedirs(directory, exist_ok=True)
        video_path = os.path.join(directory, f'{time_str}.mp4')
        print(video_path)
        logging.info(f"Video saving is done and it save in this path: {video_path}")

        return video_path
    
    except Exception as e:
        raise MetalTheptException(e,sys) from e



def draw_boxes(frame, detections):
    try:
        if detections is not None:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if cls == 0:
                    conf = float(conf)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        return frame
    
    except Exception as e:
        raise MetalTheptException(e,sys) from e



def normalize_illumination(frame):
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel to manage illumination
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))
        frame_normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return frame_normalized
    except Exception as e:
        raise MetalTheptException(e, sys) from e



def draw_motion_contours(frame, thresh, roi_pts):
    try:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 50:  # Ignore small contours
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.pointPolygonTest(roi_pts, (x, y), False) >= 0:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
    except Exception as e:
        raise MetalTheptException(e, sys) from e




