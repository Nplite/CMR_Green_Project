import cv2
import time
import numpy as np
from MetalTheft.utils.utils import draw_boxes, normalize_illumination
from MetalTheft.motion_detection import detect_motion
from MetalTheft.vid_stabilisation import VideoStabilizer

stabiliser = VideoStabilizer()

class MotionVerification:
    def __init__(self, saved_video_path, roi_1_pts_np, roi_2_pts_np, model, motion_direction_window=1, alpha=0.6, fps=30):
        self.cap = cv2.VideoCapture(saved_video_path)
        self.roi_1_pts_np = roi_1_pts_np
        self.roi_2_pts_np = roi_2_pts_np
        self.model = model
        self.motion_direction_window = motion_direction_window
        self.alpha = alpha
        self.fps = fps
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)
        self.motion_in_roi1 = False
        self.motion_in_roi2 = False
        self.motion_direction_verified = False
        self.last_motion_time_roi2 = None
        self.last_motion_time_roi1 = None

    def verify_motion_direction(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video file.")
                break
            
            blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
            frame = normalize_illumination(frame)
            # frame = stabiliser.stabilised_frame(frame)

            # Detect motion in ROI 1
            frame_roi1, thresh_ROI1, person_detected, _ = detect_motion(frame, blurred_frame, self.model, self.fgbg, self.roi_1_pts_np)
            self.motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > 350
            
            # Detect motion in ROI 2
            frame_roi2, thresh_ROI2, _, _ = detect_motion(frame, blurred_frame, self.model, self.fgbg, self.roi_2_pts_np)
            self.motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > 350

            if self.motion_in_roi2:
                self.last_motion_time_roi2 = cv2.getTickCount() / cv2.getTickFrequency()

            if self.motion_in_roi1:
                self.last_motion_time_roi1 = cv2.getTickCount() / cv2.getTickFrequency()

            combined_frame = cv2.add(frame_roi1, frame_roi2)

            # Draw Motion counters ROI2
            if self.motion_in_roi2:
                
                contours, _ = cv2.findContours(thresh_ROI2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) < 200:  # Ignore small contours
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    if cv2.pointPolygonTest(self.roi_2_pts_np, (x, y), False) >= 0:
                        cv2.drawContours(combined_frame, contours, -1, (0, 255, 0), 2)
           
            # Draw Motion counters ROI1
            if self.motion_in_roi1:
            
                contours, _ = cv2.findContours(thresh_ROI1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) < 200:  # Ignore small contours
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    if cv2.pointPolygonTest(self.roi_1_pts_np, (x, y), False) >= 0:
                        cv2.drawContours(combined_frame, contours, -1, (0, 255, 0), 2)

                
            # Verify the direction: motion must happen in ROI 2 before ROI 1 within the direction window
            if self.motion_in_roi1 and person_detected:
                if (self.last_motion_time_roi1 - self.last_motion_time_roi2) <= self.motion_direction_window:
                    self.motion_direction_verified = True
                    
                else:
                    self.motion_direction_verified = False

                if self.motion_direction_verified:
                    roi_color = (0, 0, 255)  
                else:
                    roi_color = (0, 255, 0)  

        
           
                cv2.fillPoly(combined_frame, [self.roi_1_pts_np], roi_color)
                 # combined_frame = cv2.addWeighted(frame_roi1, self.alpha, frame_roi2, 1 - self.alpha, 0)
            # cv2.imshow('Verified Motion Detection', combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()














# import cv2
# import time
# import numpy as np
# import threading
# from datetime import datetime
# from collections import deque
# from MetalTheft.motion_detection import detect_motion
# from ultralytics import YOLO
# from MetalTheft.utils.utils import save_snapshot, save_video
# from MetalTheft.constant import *
# from MetalTheft.vid_stabilisation import VideoStabilizer
# from MetalTheft.send_email import EmailSender
# from MetalTheft.utils.utils import save_snapshot, normalize_illumination, save_video, draw_boxes
# from MetalTheft.motion_detection import detect_motion
# from MetalTheft.roi_selector import ROISelector
# from MetalTheft.mongodb import MongoDBHandler
# from MetalTheft.aws import AWSConfig

# # Initialize YOLO model
# model = YOLO('yolov8n.pt', verbose=True)  

# # Settings for motion detection
# motion_threshold = 350
# motion_direction_window = 1  # Time window to check motion direction
# recording_fps = 30  

# # Initialize helper classes
# email = EmailSender()
# roi_selector = ROISelector()
# mongo_handler = MongoDBHandler()
# stabilizer = VideoStabilizer()
# aws = AWSConfig()

# # Video writer codec
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# # Alpha for blending
# alpha = 0.8

# # Function to record the entire video and save when motion is detected
# def verify_motion_in_video(video_path, roi_1_pts_np, roi_2_pts_np):
#     cap_verify = cv2.VideoCapture(video_path)
    
#     if not cap_verify.isOpened():
#         print("Error: Could not open video source")
#         return
    
#     width = int(cap_verify.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap_verify.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     out = cv2.VideoWriter(video_path, fourcc, recording_fps, (width, height))
    
#     fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)

#     motion_detected_flag = False
#     roi1_motion_time = None
#     roi2_motion_time = None
#     snapshot_path = None
#     video_url = None

#     while cap_verify.isOpened():
#         ret, frame = cap_verify.read()
#         if not ret:
#             break

#         blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)  # Blur for noise reduction

#         # Motion in ROI 1
#         combined_frame1, thresh_ROI1, person_detected, detections = detect_motion(frame, blurred_frame, model, fgbg, roi_1_pts_np)
#         motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > motion_threshold
#         cv2.polylines(combined_frame1, [roi_1_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)

#         # Motion in ROI 2
#         combined_frame2, thresh_ROI2, _, _ = detect_motion(frame, blurred_frame, model, fgbg, roi_2_pts_np)
#         motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > motion_threshold
#         cv2.polylines(combined_frame2, [roi_2_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)

#         # Combine ROI1 & ROI2 in one frame
#         combined_frame = cv2.add(combined_frame1, combined_frame2)

#         # Check motion in ROI 1 and ROI 2
#         if motion_in_roi1:
#             roi1_motion_time = time.time()
#         if motion_in_roi2:
#             roi2_motion_time = time.time()

#         out.write(frame)

#         # Check if motion direction from ROI 2 to ROI 1
#         if motion_in_roi1 and person_detected:
#             if roi2_motion_time is not None and (roi1_motion_time - roi2_motion_time) <= motion_direction_window:
#                 motion_in_roi2_to_roi1 = True
#                 roi_color = (0, 0, 255) if motion_in_roi2_to_roi1 else (0, 255, 0)   
#                 cv2.fillPoly(combined_frame, [roi_1_pts_np], roi_color)  
#                 if not motion_detected_flag:
#                     snapshot_path = save_snapshot(frame)
#                     print(f"Snapshot saved at: {snapshot_path}")
#                     motion_detected_flag = True
#             else:
#                 motion_in_roi2_to_roi1 = False 

#         # Reset flags and release video writer after motion stops
#         if motion_detected_flag:
#             out.release()
#             print(f"Video saved at: {video_path}")
#             start_time = datetime.now()

#             # Upload video to S3 and save in MongoDB
#             video_url = aws.upload_video_to_s3bucket(video_path)
#             mongo_handler.save_video_to_mongodb(video_url, start_time)

#             # Send email alert with snapshot and video URL
#             threading.Thread(target=email.send_alert_email, args=(snapshot_path, video_url)).start()

#             motion_detected_flag = False

#         # Display the video feed
#         cv2.imshow('Motion Detection', combined_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release resources
#     cap_verify.release()
#     cv2.destroyAllWindows()



