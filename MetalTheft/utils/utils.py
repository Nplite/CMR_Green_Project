
import sys
import os
import cv2
import time
import cv2
import smtplib
from datetime import datetime
from MetalTheft.constant import *
# from MetalTheft.logger import logging
from MetalTheft.exception import MetalTheptException





def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    
def save_snapshot(frame):
    try:
        date_str = datetime.now().strftime('%Y-%m-%d')
        time_str = datetime.now().strftime('%H:%M:%S')
        directory = os.path.join('snapshots', date_str)
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f'{time_str}.jpg')
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved: {filename}")
        return filename  # Return the file path after saving the snapshot
    
    except Exception as e:
        raise MetalTheptException(e, sys) from e


def save_video():
    date_str = datetime.now().strftime('%Y-%m-%d')
    time_str = datetime.now().strftime('%H:%M:%S')
    directory = os.path.join('video', date_str)
    os.makedirs(directory, exist_ok=True)
    video_path = os.path.join(directory, f'{time_str}.mp4')
    print(video_path)

    return video_path


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
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel to manage illumination
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    frame_normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return frame_normalized

def draw_motion_contours(frame, thresh, roi_pts):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 50:  # Ignore small contours
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.pointPolygonTest(roi_pts, (x, y), False) >= 0:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

