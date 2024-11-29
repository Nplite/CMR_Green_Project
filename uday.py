import threading
from typing import Dict
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from MetalTheft.mongodb import MongoDBHandler
import logging
logging.getLogger('ultralytics').setLevel(logging.WARNING) 
from MetalTheft.constant import *   
from MetalTheft.mongodb import MongoDBHandler
from Research.constant import CameraProcessor, MultiCameraSystem, CameraStream
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import Dict, Optional
import cv2
import numpy as np
from typing import List
import threading
import time
import logging
from collections import deque
from datetime import datetime
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import sys
import yaml
import torch
import threading
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
from threading import Thread, Lock
import queue
logging.getLogger('ultralytics').setLevel(logging.WARNING) 
from MetalTheft.constant import *
from MetalTheft.exception import MetalTheptException
from MetalTheft.send_email import EmailSender
from MetalTheft.utils.utils import save_snapshot, save_video, draw_boxes, draw_motion_contours
from MetalTheft.motion_detection import detect_motion, person_detection_ROI3
from MetalTheft.roi_selector import ROISelector
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.aws import AWSConfig
from ultralytics import YOLO

mongo_handler = MongoDBHandler()
aws = AWSConfig()





class CameraStream:
    def __init__(self, rtsp_url, camera_id):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(rtsp_url)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS) or 30)  # Default to 30 FPS if unknown
        self.frame_queue = queue.Queue(maxsize=30)
        self.stopped = False
        self.lock = Lock()
        print("***********************************************", self.fps )
        
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

class MotionStatusTracker:
    def __init__(self, timeout=10):
        """
        Initialize Motion Status Tracker
        
        :param timeout: Duration (in seconds) to consider motion as active
        """
        self._lock = threading.Lock()
        self._motion_status: Dict[int, Dict[str, any]] = {}
        self._timeout = timeout

    def update_motion_status(self, camera_id: int, is_motion_detected: bool):
        """
        Update motion status for a specific camera
        
        :param camera_id: ID of the camera
        :param is_motion_detected: Boolean indicating motion detection status
        """
        with self._lock:
            self._motion_status[camera_id] = {
                'is_motion_detected': is_motion_detected,
                'timestamp': datetime.now() }

    def get_motion_status(self, camera_id: int) -> Dict[str, any]:
        """
        Get current motion status for a camera
        
        :param camera_id: ID of the camera
        :return: Dictionary with motion status and timestamp
        """
        with self._lock:
            status = self._motion_status.get(camera_id, {
                'is_motion_detected': False,
                'timestamp': datetime.now()
            })
            
            # Check if status is still valid based on timeout
            if status['is_motion_detected']:
                time_since_detection = datetime.now() - status['timestamp']
                if time_since_detection > timedelta(seconds=self._timeout):
                    status['is_motion_detected'] = False
            
            return status

# Global motion status tracker
motion_status_tracker = MotionStatusTracker()

# Modify CameraProcessor to update motion status
class CameraProcessor:
    def __init__(self, camera_id, config):
        try:
            self.camera_id = camera_id
            self.stream = CameraStream(config['RTSP_URL'], camera_id)
            
            # Load ROI points from config with validation
            self.validate_and_load_roi_points(config)
            
            # Load other parameters from config with defaults
            self.thresh_value = config.get('thresh_roi', 25)
            self.motion_buffer_duration = config.get('motion_buffer_duration', 5)
            self.motion_buffer_fps = config.get('motion_buffer_fps', 30)
            self.motion_direction_window = config.get('motion_direction_window', 2.0)
            
            # Initialize processing attributes
            self.aws = AWSConfig()
            self.mongo_handler = MongoDBHandler()
            self.email = EmailSender()
            self.setup_complete = True
            self.alpha = 0.6
            self.counter = 1
            self.last_motion_time = None
            self.current_time = None
            self.start_time = None
            self.motion_detected_flag = False
            self.roi1_motion_time = None
            self.roi2_motion_time = None
            self.recording_after_motion = False
            self.recording_end_time = None
            self.video_path = None
            self.snapshot_path = None
            self.video_url = None
            self.snapshot_url = None
            self.is_processing = False
            self.object_counter = 0
            
            # Initialize components with error handling
            self.initialize_components()
            
            # Initialize motion buffer
            self.motion_frame_buffer = deque(maxlen=self.motion_buffer_fps * self.motion_buffer_duration)
            self.current_people_in_roi3 = 0
            
            # Video writer setup
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out_motion = None
            
            # Window name
            self.window_name = f'Camera {self.camera_id}'
            
        except Exception as e:
            logging.error(f"Error initializing CameraProcessor for camera {camera_id}: {str(e)}")
            raise

    def validate_and_load_roi_points(self, config):
        """Validate and load ROI points from config"""
        try:
            required_rois = ['roi_1_pts_np', 'roi_2_pts_np', 'roi_3_pts_np']
            for roi in required_rois:
                if roi not in config:
                    raise ValueError(f"Missing {roi} in configuration")
                points = np.array(config[roi], np.int32)
                if points.size == 0:
                    raise ValueError(f"Empty points array for {roi}")
                setattr(self, roi, points)
            
            self.roi_1_set = True
            self.roi_2_set = True
            self.roi_3_set = True
            
        except Exception as e:
            logging.error(f"Error loading ROI points: {str(e)}")
            raise

    def initialize_components(self):
        """Initialize all necessary components with error handling"""
        try:
            # Initialize YOLO model
            self.model = YOLO('/home/alluvium/Desktop/Namdeo/CMR_Project/yolov8n.engine', task = 'detect', verbose=True)
            
            # Initialize background subtractor
            self.fgbg = cv2.createBackgroundSubtractorMOG2(
                history=500, 
                varThreshold=20, 
                detectShadows=True
            )
            
        except Exception as e:
            logging.error(f"Error initializing components: {str(e)}")
            raise


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
            motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > self.thresh_value
            motion_mask_1 = np.zeros_like(combined_frame1)
            cv2.fillPoly(motion_mask_1, [self.roi_1_pts_np], (0, 255, 0))
            combined_frame1 = cv2.addWeighted(combined_frame1, 0.6, motion_mask_1, 0.4, 0)
            
            # Process ROI 2
            combined_frame2, thresh_ROI2, _, _ = detect_motion(frame, blurred_frame, model, self.fgbg, self.roi_2_pts_np)
            motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > self.thresh_value
            motion_mask_2 = np.zeros_like(frame)
            cv2.fillPoly(motion_mask_2, [self.roi_2_pts_np], (0,0,0))  
            combined_frame2 = cv2.addWeighted(combined_frame2, 0.6, motion_mask_2, 0.4, 0)

            # Process ROI 2
            combined_frame3, person_in_roi3, person_counter = person_detection_ROI3(frame=frame, roi_points= self.roi_3_pts_np, model= self.model)
            cv2.polylines(combined_frame3, [self.roi_3_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
            self.current_people_in_roi3 = len(person_in_roi3)
            


            combined_frame = self.process_motion(frame,
                combined_frame1, combined_frame2, combined_frame3, motion_in_roi1, person_counter,
                motion_in_roi2, person_detected, person_in_roi3, thresh_ROI1, thresh_ROI2, detections)
            
            return combined_frame, thresh_ROI1
            
        except Exception as e:
            print(f"Error processing camera {self.camera_id}: {str(e)}")
            return frame, None

    def process_motion(self, frame, combined_frame1, combined_frame2, combined_frame3, motion_in_roi1, 
                      person_counter, motion_in_roi2, person_detected, person_in_roi3, thresh_ROI1, thresh_ROI2, detections):

        cv2.polylines(combined_frame1, [self.roi_1_pts_np], True, (0, 255, 0), 1)
        cv2.polylines(combined_frame2, [self.roi_2_pts_np], True, (0, 255, 0), 1)
        
        # Process motion in ROI 1 & ROI2
        if motion_in_roi1:
            self.roi1_motion_time = time.time()
            draw_motion_contours(combined_frame1, thresh_ROI1, self.roi_1_pts_np)
       
        if motion_in_roi2:
            self.roi2_motion_time = time.time()
            # draw_motion_contours(combined_frame2, thresh_ROI2, self.roi_2_pts_np)

        # combined_frame = cv2.add(combined_frame1, combined_frame2)
        combined_frame = cv2.add(cv2.add(combined_frame1, combined_frame2), combined_frame3)

        # Handle recording logic
        self.current_time = datetime.now()
        if (motion_in_roi1 and person_detected):
            if self.last_motion_time is None or (self.current_time - self.last_motion_time).total_seconds() > 3:
                if self.roi2_motion_time is not None and (self.roi1_motion_time - self.roi2_motion_time) <= self.motion_direction_window:
                    motion_in_roi2_to_roi1 = True  
                    # print("**********************************************\ncamera_id:", self.camera_id, '\nThreshold:', self.thresh_value)
                    roi_color = (0, 0, 255) if motion_in_roi2_to_roi1 else (0, 255, 0)
                    cv2.fillPoly(combined_frame, [self.roi_1_pts_np], roi_color)  
                    self.last_motion_time = self.current_time  
                    self.object_counter += 1

                else:
                    motion_in_roi2_to_roi1 = False 

        if (motion_in_roi1 and person_detected) or person_in_roi3:
            if (self.roi2_motion_time is not None and (self.roi1_motion_time - self.roi2_motion_time) <= self.motion_direction_window) or person_in_roi3:
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

                self.video_url = self.aws.upload_video_to_s3bucket(self.video_path, self.camera_id)                  
                self.snapshot_url = self.aws.upload_snapshot_to_s3bucket(self.snapshot_path, self.camera_id)
                
                threading.Thread(target=self.mongo_handler.save_snapshot_to_mongodb, args=(self.snapshot_url, start_time, self.camera_id)).start()
                threading.Thread(target=self.mongo_handler.save_video_to_mongodb, args=(self.video_url, start_time, self.camera_id)).start()
                threading.Thread(target=self.email.send_alert_email, args=(self.snapshot_path, self.video_url, self.camera_id)).start()
                self.counter += 1

        # Draw detection boxes
        cv2.putText(combined_frame, f'Object Count: {self.object_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(combined_frame, f'Person Count: {person_counter}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        draw_boxes(combined_frame, detections)

        is_motion_detected = (motion_in_roi1 and person_detected) or person_in_roi3

        # Update motion status tracker
        self.motion_status_tracker.update_motion_status(
            camera_id=self.camera_id, 
            is_motion_detected=is_motion_detected)

        # Rest of the existing process_motion method remains the same
        # ... (existing motion processing logic)




        return combined_frame

# Modify MultiCameraSystem
class MultiCameraSystem:

    def __init__(self, config_path):
        self.camera_processors = {}
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = YOLO('yolov8n.pt').to(device)
            
            # self.model = YOLO('/home/alluvium/Desktop/Namdeo/CMR_Project/yolov8n.engine', task = 'detect', verbose=True)
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {str(e)}")
            raise
            
        self.is_running = False
        self.processing_thread = None
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' )
        # Load configuration
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Failed to load configuration: {str(e)}")
            raise
        
        # Initialize cameras from config
        self.initialize_cameras()

    def get_camera_count(self) -> int:
        """Get the total number of cameras in the system"""
        return len(self.camera_processors)
    
    def get_object_counts(self, camera_id: Optional[int] = None) -> dict:
        """Get object counts for all cameras or a specific camera"""
        if camera_id is not None:
            if camera_id in self.camera_processors:
                processor = self.camera_processors[camera_id]
                return {camera_id: {"object_count": processor.object_counter}}
            else:
                raise HTTPException(status_code=404, detail=f"Camera ID {camera_id} not found")
        
        # If no specific camera_id is provided, return counts for all cameras
        object_counts = {
            cam_id: {"object_count": processor.object_counter}
            for cam_id, processor in self.camera_processors.items()
        }
        return object_counts

    def get_people_counts(self, camera_id: Optional[int] = None) -> dict:
        """Get the current number of people in ROI3 for a specific camera"""
        if camera_id is not None:
            if camera_id in self.camera_processors:
                processor = self.camera_processors[camera_id]
                return {camera_id: {"current_people_in_roi3": processor.current_people_in_roi3}}
            else:
                raise HTTPException(status_code=404, detail=f"Camera ID {camera_id} not found")

        # If no specific camera_id is provided, return counts for all cameras
        people_counts = {
            cam_id: {"current_people_in_roi3": processor.current_people_in_roi3}
            for cam_id, processor in self.camera_processors.items()
        }
        return people_counts
    
    def initialize_cameras(self):
        """Initialize all cameras with error handling"""
        for camera_name, camera_config in self.config['cameras'].items():
            try:
                camera_id = int(camera_name.split('_')[1])
                processor = CameraProcessor(camera_id, camera_config)
                processor.stream.start()
                self.camera_processors[camera_id] = processor
                logging.info(f"Initialized camera {camera_id} with configuration")
            except Exception as e:
                logging.error(f"Failed to initialize camera {camera_name}: {str(e)}")
                # Continue with other cameras if one fails
                continue

    def _process_cameras(self):
        """Main processing loop with enhanced error handling"""
        logging.info("\nAll cameras configured! Starting monitoring...")
        
        while self.is_running:
            try:
                for camera_id, processor in self.camera_processors.items():
                    if processor.stream.stopped:
                        continue
                        
                    ret, frame = processor.stream.read()
                    if not ret:
                        continue
                    
                    try:
                        processed_frame, motion = processor.process_frame(frame, self.model)
                        cv2.imshow(processor.window_name, processed_frame)
                    except Exception as e:
                        logging.error(f"Error processing camera {camera_id}: {str(e)}")
                        continue

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop()
                    break
                elif key == ord('a'):
                    self.reset_all_rois()
                    
            except Exception as e:
                logging.error(f"Error in main processing loop: {str(e)}")
                time.sleep(1)  # Prevent tight loop on error
                continue

    def start(self):
        """Start processing with error handling"""
        try:
            if not self.camera_processors:
                raise Exception("No cameras were successfully initialized")
                
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._process_cameras)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logging.info("Camera system started successfully")
        except Exception as e:
            logging.error(f"Failed to start camera system: {str(e)}")
            raise

    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        for processor in self.camera_processors.values():
            processor.stream.stop()
        cv2.destroyAllWindows()

    def stop(self):
        """Safely stop all cameras and processing"""
        self.is_running = False
        for processor in self.camera_processors.values():
            processor.stream.stop()
        cv2.destroyAllWindows()
        logging.info("Camera system stopped")



    def get_camera_motion_status(self, camera_id: Optional[int] = None) -> dict:
        """Get motion detection status for all cameras or a specific camera"""
        if camera_id is not None:
            if camera_id in self.camera_processors:
                status = motion_status_tracker.get_motion_status(camera_id)
                return {camera_id: status}
            else:
                raise HTTPException(status_code=404, detail=f"Camera ID {camera_id} not found")
        
        # If no specific camera_id is provided, return status for all cameras
        motion_statuses = {
            cam_id: motion_status_tracker.get_motion_status(cam_id)
            for cam_id in self.camera_processors.keys()
        }
        return motion_statuses

# Add new API endpoint in FastAPI app
@app.get("/cameras/motion-status/{camera_id}")
async def get_motion_status(camera_id: int):
    """Get motion detection status for a specific camera"""
    try:
        status = camera_system.get_camera_motion_status(camera_id=camera_id)
        return {"motion_status": status}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})