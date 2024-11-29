import cv2
import numpy as np
import sys
import threading
import time
import logging
from collections import deque
from datetime import datetime
from MetalTheft.constant import *
from MetalTheft.exception import MetalTheptException
from MetalTheft.send_email import EmailSender
from MetalTheft.utils.utils import save_snapshot, normalize_illumination, save_video, draw_boxes, draw_motion_contours
from MetalTheft.motion_detection import detect_motion
from MetalTheft.roi_selector import ROISelector
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.aws import AWSConfig
from ultralytics import YOLO
from threading import Thread, Lock
import queue
logging.getLogger('ultralytics').setLevel(logging.WARNING) 

class CameraStream:
    def __init__(self, rtsp_url, camera_id):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(rtsp_url)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS) or 30)  # Default to 30 FPS if unknown
        self.frame_queue = queue.Queue(maxsize=30)
        self.stopped = False
        self.lock = Lock()
        
    def start(self):
        thread = Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        return self
        
    def update(self):
        while True:
            if self.stopped:
                return

            with self.lock:
                if not self.frame_queue.full():
                    ret, frame = self.cap.read()
                    if not ret:
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.rtsp_url)
                        continue

                    # Clear stale frames to keep the queue fresh
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break

                    self.frame_queue.put(frame)

            time.sleep(1 / self.fps)  #
                
    def read(self):
        try:
            return True, self.frame_queue.get_nowait()
        except queue.Empty:
            return False, None
    
    def stop(self):
        self.stopped = True
        with self.lock:
            self.cap.release()

class CameraProcessor:
    def __init__(self, camera_id, rtsp_url):
        self.camera_id = camera_id
        self.stream = CameraStream(rtsp_url, camera_id)
        self.roi_1_set = False
        self.roi_2_set = False
        self.roi_1_pts_np = None
        self.roi_2_pts_np = None
        self.alpha = 0.6
        self.counter = 1
        self.last_motion_time = None
        self.start_time = None
        self.motion_direction_window = 1
        self.motion_detected_flag = False
        self.roi1_motion_time = None
        self.roi2_motion_time = None
        self.recording_after_motion = False
        self.recording_end_time = None
        self.video_path  = None
        self.snapshot_path = None
        self.is_processing = False
        self.setup_complete = False
        self.email = EmailSender()
        self.mongo_handler = MongoDBHandler()
        self.aws = AWSConfig()
        self.model = YOLO('yolov8n.pt', verbose=True)
        
        # Initialize components
        self.roi_selector = ROISelector()
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)
        
        # Initialize motion buffer
        self.motion_buffer_duration = 5
        self.motion_buffer_fps = 30
        self.motion_frame_buffer = deque(maxlen=self.motion_buffer_fps * self.motion_buffer_duration)
        
        # Video writer setup
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_motion = None
        
        # Window name
        self.window_name = f'Camera {self.camera_id}'

    def setup_roi(self):
        """Handle ROI setup for this camera"""
        print(f"\nSetting up ROIs for Camera {self.camera_id}")
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.roi_selector.select_point)

        while not (self.roi_1_set and self.roi_2_set):
            ret, frame = self.stream.read()
            if not ret:
                continue

            if not self.roi_1_set:
                cv2.putText(frame, f"Select first ROI for camera {self.camera_id} (Click 4 points)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if self.roi_selector.is_roi_selected():
                    self.roi_1_pts_np = self.roi_selector.get_roi_points()
                    self.roi_1_set = True
                    self.roi_selector.reset_roi()
                    print(f"Camera {self.camera_id}: First ROI set")
                    time.sleep(0.5)  # Small delay for visual feedback
                    
            elif not self.roi_2_set:
                cv2.putText(frame, f"Select second ROI for camera {self.camera_id} (Click 4 points)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if self.roi_selector.is_roi_selected():
                    self.roi_2_pts_np = self.roi_selector.get_roi_points()
                    self.roi_2_set = True
                    self.roi_selector.reset_roi()
                    print(f"Camera {self.camera_id}: Second ROI set")
                    time.sleep(0.5)  # Small delay for visual feedback

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            elif key == ord('r'):
                self.reset_roi()
                
        self.setup_complete = True
        print(f"Camera {self.camera_id} setup complete!")
        return True

    def reset_roi(self):
            """Reset the ROI for this specific camera feed."""
            self.roi_1_set = False
            self.roi_2_set = False
            self.roi_selector.reset_roi()
            self.setup_complete = False
            print(f"Camera {self.camera_id}: ROI reset. Please set ROIs again.")
            self.setup_roi()  # Trigger ROI setup only for this camera

    def process_frame(self, frame, model):
        if not self.setup_complete:
            return frame, None

        try:
            combined_frame1 = frame.copy()
            combined_frame2 = frame.copy()
            self.motion_frame_buffer.append(frame.copy())
            blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)

            # Process ROI 1
            combined_frame1, thresh_ROI1, person_detected, detections = detect_motion(frame, blurred_frame, model, self.fgbg, self.roi_1_pts_np)
            motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > 100
            motion_mask_1 = np.zeros_like(combined_frame1)
            cv2.fillPoly(motion_mask_1, [self.roi_1_pts_np], (0, 255, 0))
            combined_frame1 = cv2.addWeighted(combined_frame1, 0.6, motion_mask_1, 0.4, 0)
            
            # Process ROI 2
            combined_frame2, thresh_ROI2, _, _ = detect_motion(frame, blurred_frame, model, self.fgbg, self.roi_2_pts_np)
            motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > 350
            motion_mask_2 = np.zeros_like(frame)
            cv2.fillPoly(motion_mask_2, [self.roi_2_pts_np], (0,0,0))  
            combined_frame2 = cv2.addWeighted(combined_frame2, 0.6, motion_mask_2, 0.4, 0)

            # Combine frames and process motion
            combined_frame = self.process_motion(frame,
                combined_frame1, combined_frame2, motion_in_roi1, 
                motion_in_roi2, person_detected, thresh_ROI1, thresh_ROI2, detections)
            
            return combined_frame, thresh_ROI1
            
        except Exception as e:
            print(f"Error processing camera {self.camera_id}: {str(e)}")
            return frame, None

    def process_motion(self, frame, combined_frame1, combined_frame2, motion_in_roi1, 
                      motion_in_roi2, person_detected, thresh_ROI1, thresh_ROI2, detections):
        # Draw ROIs
        cv2.polylines(combined_frame1, [self.roi_1_pts_np], True, (0, 255, 0), 1)
        cv2.polylines(combined_frame2, [self.roi_2_pts_np], True, (0, 255, 0), 1)
        
        # Process motion in ROI 1 & ROI2
        if motion_in_roi1:
            self.roi1_motion_time = time.time()
            draw_motion_contours(combined_frame1, thresh_ROI1, self.roi_1_pts_np)
       
        if motion_in_roi2:
            self.roi2_motion_time = time.time()
            draw_motion_contours(combined_frame2, thresh_ROI2, self.roi_2_pts_np)
        
        # Combine frames
        combined_frame = cv2.add(combined_frame1, combined_frame2)

        
        # Handle recording logic
        if motion_in_roi1 and person_detected:
            if self.roi2_motion_time is not None and (self.roi1_motion_time - self.roi2_motion_time) <= self.motion_direction_window:
                motion_in_roi2_to_roi1 = True  # Direction is confirmed from ROI2 to ROI1 
                roi_color = (0, 0, 255) if motion_in_roi2_to_roi1 else (0, 255, 0)
                cv2.fillPoly(combined_frame, [self.roi_1_pts_np], roi_color)    

            else:
                motion_in_roi2_to_roi1 = False 

            # Motion detected: start or continue recording
            if self.roi2_motion_time is not None and (self.roi1_motion_time - self.roi2_motion_time) <= self.motion_direction_window:
                if not self.motion_detected_flag:
                    self.video_path = save_video()
                    self.out_motion = cv2.VideoWriter(self.video_path, self.fourcc, 30.0, (frame.shape[1], frame.shape[0]))
                    self.snapshot_path = save_snapshot(combined_frame)
                    while self.motion_frame_buffer:
                        self.out_motion.write(self.motion_frame_buffer.popleft())
                    self.motion_detected_flag = True
                    self.recording_after_motion = False
                    self.motion_frame_buffer.clear()  # Clear the buffer

                # Write current frame
                self.out_motion.write(combined_frame)

        elif self.motion_detected_flag:
            if not self.recording_after_motion:
                self.recording_end_time = time.time() + 5 
                self.recording_after_motion = True

            if time.time() <= self.recording_end_time:
                self.out_motion.write(combined_frame)
            else:
                self.motion_detected_flag = False
                self.recording_after_motion = False
                self.out_motion.release()
                start_time = datetime.now()
                

                # video_url = self.aws.upload_video_to_s3bucket(self.video_path)
                # self.mongo_handler.save_video_to_mongodb(video_url, start_time)
                # self.email.send_alert_email(self.snapshot_path, video_url)


                video_url = self.aws.upload_video_to_s3bucket(self.video_path)
                threading.Thread(target=self.mongo_handler.save_video_to_mongodb, args=(video_url, start_time))
                threading.Thread(target=self.email.send_alert_email, args=(self.snapshot_path, video_url, self.camera_id)).start()
                self.counter += 1
        
            
        # Draw detection boxes
        draw_boxes(combined_frame, detections)
        
        return combined_frame


class MultiCameraSystem:
    def __init__(self, camera_urls):
        self.camera_processors = {}
        self.model = YOLO('yolov8n.pt', verbose=True)
        
        # Initialize camera processors
        for camera_id, url in camera_urls.items():
            self.camera_processors[camera_id] = CameraProcessor(camera_id, url)
            self.camera_processors[camera_id].stream.start()
            
    def setup_cameras(self):
        """Set up ROIs for each camera sequentially"""
        for camera_id, processor in self.camera_processors.items():
            if not processor.setup_complete:
                if not processor.setup_roi():
                    return False
        return True

    def run(self):
        try:
            if not self.setup_cameras():
                print("Camera setup incomplete. Exiting...")
                return

            print("\nAll cameras configured! Starting monitoring...")
            
            while True:
                for camera_id, processor in self.camera_processors.items():
                    ret, frame = processor.stream.read()
                    if not ret:
                        continue
                    
                    processed_frame, motion = processor.process_frame(frame, self.model)

                    # Display results
                    cv2.imshow(processor.window_name, processed_frame)
                    # if motion is not None:
                    #     cv2.imshow(f'Motion {camera_id}', motion)

                    time.sleep(1 / 30)  # Throttle the display loop to match 30 FPS

                key = cv2.waitKey(int(1000 / 30)) & 0xFF 
                if key == ord('q'):
                    break

                elif key == ord('r'):
                    # Ask for camera ID to reset ROI for that specific camera
                    camera_id = int(input("Enter Camera ID to reset ROI: "))
                    self.reset_camera_roi(camera_id)
                    
                # elif key == ord('r'):
                #     print("\nResetting all ROIs...")
                #     self.reset_all_rois()
                #     if not self.setup_cameras():
                #         break     

        except Exception as e:
            print(f"An error occurred in main loop: {str(e)}")
            raise MetalTheptException(e, sys) from e
        finally:
            self.cleanup()

            
    def reset_all_rois(self):
        for processor in self.camera_processors.values():
            processor.reset_roi()

    def reset_camera_roi(self, camera_id):
        """Reset ROI for a specific camera based on its ID."""
        if camera_id in self.camera_processors:
            print(f"\nResetting ROI for Camera {camera_id}...")
            self.camera_processors[camera_id].reset_roi()
        else:
            print(f"Camera {camera_id} not found.")
            
    def cleanup(self):
        for processor in self.camera_processors.values():
            processor.stream.stop()
        cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    # Define camera URLs
    camera_urls = {
        # 1: RTSP_URL,
        2: 'DATA/17.10.2024.mp4',
        3: 'DATA/18-10-2024.mp4'

    }
    
    # Initialize and run the system
    multi_camera_system = MultiCameraSystem(camera_urls)
    multi_camera_system.run()