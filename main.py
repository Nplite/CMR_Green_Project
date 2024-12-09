import asyncio
import logging
import threading
import time
import cv2
import yaml
import numpy as np
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import concurrent.futures
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import logging
import cv2
import numpy as np
import threading
import time
from datetime import datetime
from collections import deque
import sys
import yaml
import torch
from threading import Thread, Lock
import queue
from ultralytics import YOLO
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.constant import *
from MetalTheft.exception import MetalTheptException
from MetalTheft.send_email import EmailSender
from MetalTheft.utils.utils import save_snapshot, save_video, draw_boxes, draw_motion_contours
from MetalTheft.motion_detection import detect_motion, person_detection_ROI
from MetalTheft.roi_selector import ROISelector
from MetalTheft.aws import AWSConfig
logging.getLogger('ultralytics').setLevel(logging.WARNING)
mongo_handler = MongoDBHandler()
aws = AWSConfig()


class AsyncCameraStream:
    def __init__(self, rtsp_url, camera_id):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(rtsp_url)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS) or 30)  # Default to 30 FPS if unknown
        self.frame_queue = asyncio.Queue(maxsize=30)
        self.stopped = False
        self.lock = asyncio.Lock()
        
        
    async def start(self):
        """Start the async camera stream"""
        self.stream_task = asyncio.create_task(self.update())
        return self
        
    async def update(self):
        """Continuously update frames"""
        while not self.stopped:
            async with self.lock:
                if not self.frame_queue.full():
                    ret, frame = self.cap.read()
                    if not ret:
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.rtsp_url)
                        await asyncio.sleep(1)
                        continue

                    # Clear any existing frames in the queue
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                    await self.frame_queue.put(frame)

            await asyncio.sleep(1 / self.fps)

    async def read(self):
        """Read a frame from the queue"""
        try:
            frame = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)
            return True, frame
        except asyncio.TimeoutError:
            return False, None
    
    async def stop(self):
        """Stop the camera stream"""
        self.stopped = True
        async with self.lock:
            self.cap.release()

class AsyncCameraProcessor:
    def __init__(self, camera_id, config):
        try:
            self.camera_id = camera_id
            # self.stream = CameraStream(config['RTSP_URL'], camera_id)
            self.model = YOLO('yolov8n.engine', task='detect', verbose=True)
            
            # Load ROI points from config with validation
            self.validate_and_load_roi_points(config)
            
            # Load other parameters from config with defaults
            self.thresh_value = config.get('thresh_roi', 25)
            self.motion_buffer_duration = config.get('motion_buffer_duration', 5)
            self.motion_buffer_fps = config.get('motion_buffer_fps', 30)
            self.motion_direction_window = config.get('motion_direction_window', 2.0)
            
            # Initialize processing attributes
            self.config = config 
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
            self.frame_count  = 0
            self.frame_count += 1
            
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
            logging.error(f"Error initializing components for camera {self.camera_id}: {str(e)}")
            raise



    def initialize_components(self):
        """Initialize camera components"""
        try:
            self.stream = AsyncCameraStream(self.config['RTSP_URL'], self.camera_id)
            self.model = YOLO('yolov8n.engine', task='detect', verbose=True)
            self.fgbg = cv2.createBackgroundSubtractorMOG2(
                history=500, 
                varThreshold=20, 
                detectShadows=True
            )
            
            # Validate and load ROI points
            required_rois = ['roi_1_pts_np', 'roi_2_pts_np', 'roi_3_pts_np']
            for roi in required_rois:
                if roi not in self.config:
                    raise ValueError(f"Missing {roi} in configuration")
                points = np.array(self.config[roi], np.int32)
                if points.size == 0:
                    raise ValueError(f"Empty points array for {roi}")
                setattr(self, roi, points)

            
            self.setup_complete = True
            logging.info(f"Camera {self.camera_id} initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing components for camera {self.camera_id}: {str(e)}")
            raise

    async def process_frame(self, frame, model):
        """Process a single frame"""
        if not self.setup_complete:
            return frame

        try:
            combined_frame1 = frame.copy()
            combined_frame2 = frame.copy()
            person_detected = None

            blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)

            combined_frame1, thresh_ROI1, _, detections = detect_motion(frame, blurred_frame, model, self.fgbg, self.roi_1_pts_np)
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
            combined_frame3, person_in_roi3, person_counter = person_detection_ROI(frame=frame, roi_points= self.roi_3_pts_np, model= self.model)
            cv2.polylines(combined_frame3, [self.roi_3_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
            self.current_people_in_roi3 = len(person_in_roi3)
            
            # if self.frame_count > 20:
            #     _, _, person_detected, _ = detect_motion(frame, blurred_frame, model, self.fgbg, self.roi_2_pts_np)

            
            if self.frame_count > 20:  # ADD COMBINEFRAME2 IN THIS PLACE SO WE CAN TRACK PERSON HERE.
                combined_frame2, person_detected, _ = person_detection_ROI(frame=combined_frame2, roi_points= self.roi_2_pts_np, model= self.model)

            combined_frame = self.process_motion(frame,
                combined_frame1, combined_frame2, combined_frame3, motion_in_roi1, person_counter,
                motion_in_roi2, person_detected, person_in_roi3, thresh_ROI1, thresh_ROI2, detections)
            
            self.frame_count += 1
            
            return combined_frame, thresh_ROI1
            
        except Exception as e:
            print(f"Error processing camera {self.camera_id}: {str(e)}")
            return frame, None

    def process_motion(self, frame, combined_frame1, combined_frame2, combined_frame3, motion_in_roi1, 
                      person_counter, motion_in_roi2, person_detected, person_in_roi3, thresh_ROI1, thresh_ROI2, detections):
        # Draw ROIs
        cv2.polylines(combined_frame1, [self.roi_1_pts_np], True, (0, 255, 0), 1)
        cv2.polylines(combined_frame2, [self.roi_2_pts_np], True, (0, 255, 0), 1)
        
        # Process motion in ROI 1 & ROI2
        if motion_in_roi1:
            self.roi1_motion_time = time.time()
            draw_motion_contours(combined_frame1, thresh_ROI1, self.roi_1_pts_np)
       
        if motion_in_roi2:
            self.roi2_motion_time = time.time()
            # draw_motion_contours(combined_frame2, thresh_ROI2, self.roi_2_pts_np)

      
        combined_frame = cv2.add(cv2.add(combined_frame1, combined_frame2), combined_frame3)
        self.motion_frame_buffer.append(combined_frame.copy())

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

        if (motion_in_roi1 and person_detected) or (person_in_roi3 and person_detected):
            if (self.roi2_motion_time is not None and (self.roi1_motion_time - self.roi2_motion_time) <= self.motion_direction_window) or person_in_roi3:
                if not self.motion_detected_flag:
                    self.video_path = save_video()
                    self.out_motion = cv2.VideoWriter(self.video_path, self.fourcc, 30.0, (frame.shape[1], frame.shape[0]))
                    self.snapshot_path = save_snapshot(combined_frame, camera_id=self.camera_id)

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

        return combined_frame


class AsyncMultiCameraSystem:
    def __init__(self, config_path):
        self.camera_processors = {}
        self.is_running = False
        self.processing_task = None
        self.model = YOLO('yolov8n.engine', task = 'detect', verbose=True)

        # Setup logging
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
    
    async def initialize_cameras(self):
        """Asynchronously initialize all cameras"""
        for camera_name, camera_config in self.config['cameras'].items():
            try:
                camera_id = int(camera_name.split('_')[1])
                processor = AsyncCameraProcessor(camera_id, camera_config)
                await processor.stream.start()
                self.camera_processors[camera_id] = processor
                logging.info(f"Initialized camera {camera_id}")
            except Exception as e:
                logging.error(f"Failed to initialize camera {camera_name}: {str(e)}")

    async def process_cameras(self):
        """Asynchronous camera processing loop"""
        logging.info("Starting camera processing...")
        
        while self.is_running:
            tasks = []
            for camera_id, processor in self.camera_processors.items():
                # Create a task to read and process frame for each camera
                tasks.append(self.process_single_camera(camera_id, processor))
            
            # Run all camera processing tasks concurrently
            await asyncio.gather(*tasks)
            
            # Small delay to prevent tight loop
            await asyncio.sleep(0.01)

    async def process_single_camera(self, camera_id, processor):
        """Process a single camera's frame"""
        try:
            ret, frame = await processor.stream.read()
            if not ret:
                return
            
            # Process the frame
            processed_frame = await processor.process_frame(frame, self.model)
            
            # Optional: Display or further process the frame
            # In a real-world scenario, you might want to stream or save the processed frame
        except Exception as e:
            logging.error(f"Error processing camera {camera_id}: {str(e)}")

    async def start(self):
        """Start the camera system"""
        try:
            # Initialize cameras
            await self.initialize_cameras()
            
            # Check if any cameras were initialized
            if not self.camera_processors:
                raise Exception("No cameras were successfully initialized")
            
            # Set running flag and start processing
            self.is_running = True
            self.processing_task = asyncio.create_task(self.process_cameras())
            logging.info("Camera system started successfully")
        except Exception as e:
            logging.error(f"Failed to start camera system: {str(e)}")
            raise

    async def stop(self):
        """Gracefully stop the camera system"""
        self.is_running = False
        
        # Stop all camera streams
        stop_tasks = [processor.stream.stop() for processor in self.camera_processors.values()]
        await asyncio.gather(*stop_tasks)
        
        # Cancel processing task if it exists
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logging.info("Camera system stopped")

# FastAPI Application with Async Endpoints
app = FastAPI(title="Async Camera Surveillance System API")

# Global async camera system
async_camera_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize camera system on startup"""
    global async_camera_system
    async_camera_system = AsyncMultiCameraSystem('MetalTheft/camera.yaml')
    await async_camera_system.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup when shutting down"""
    global async_camera_system
    if async_camera_system:
        await async_camera_system.stop()

@app.get("/")
async def read_root():
    """Root endpoint"""
    return {"message": "Welcome to the Async Camera Surveillance System"}

@app.get("/cameras/count")
async def get_camera_count():
    """Get the total number of cameras in the system"""
    global async_camera_system
    return {"camera_count": len(async_camera_system.camera_processors)}

@app.get("/cameras/object-count/{camera_id}")
async def get_object_count(camera_id: int):
    """Get object count for a specific camera"""
    global async_camera_system
    try:
        # This is a placeholder. You'll need to implement actual object counting logic
        return {"camera_id": camera_id, "object_count": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    import uvicorn
    
    try:
        # Run the FastAPI application with uvicorn
        uvicorn.run(
            "main:app",  # Replace 'your_script_name' with the actual name of your script file without the `.py` extension
            host="0.0.0.0", 
            port=8080, 
            reload=True
        )
    except Exception as e:
        logging.error(f"Failed to start server: {str(e)}")
