import asyncio
import logging
import threading
import time
from typing import Optional, List, Dict
from collections import deque
import cv2
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
from MetalTheft.aws import AWSConfig
from MetalTheft.constant import *
from MetalTheft.exception import MetalTheptException
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.motion_detection import detect_motion, person_detection_ROI3
from MetalTheft.send_email import EmailSender
from MetalTheft.utils.utils import save_snapshot, save_video, draw_boxes, draw_motion_contours

class AsyncCameraStream:
    def __init__(self, rtsp_url: str, camera_id: int):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(rtsp_url)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS) or 30)
        self.frame_queue = asyncio.Queue(maxsize=30)
        self.stopped = False
        self._lock = asyncio.Lock()

    async def start(self):
        asyncio.create_task(self._update())

    async def _update(self):
        while not self.stopped:
            async with self._lock:
                if not self.frame_queue.full():
                    ret, frame = self.cap.read()
                    if not ret:
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.rtsp_url)
                        continue

                    # Clear stale frames
                    while not self.frame_queue.empty():
                        self.frame_queue.get_nowait()

                    await self.frame_queue.put(frame)

                await asyncio.sleep(1 / self.fps)

    async def read(self):
        try:
            return True, await self.frame_queue.get()
        except asyncio.QueueEmpty:
            return False, None

    def stop(self):
        self.stopped = True
        self.cap.release()

class AsyncCameraProcessor:
    def __init__(self, camera_id: int, config: Dict):
        self.camera_id = camera_id
        self.stream = AsyncCameraStream(config['RTSP_URL'], camera_id)
        self.model = YOLO('yolov8n.engine', task='detect', verbose=True)
        self.object_counter = 0
        
        self.current_people_in_roi3 = 0
        self.validate_and_load_roi_points(config)
        
        # Load other parameters from config with defaults
        self.thresh_value = config.get('thresh_roi', 25)
        self.motion_buffer_duration = config.get('motion_buffer_duration', 5)
        self.motion_buffer_fps = config.get('motion_buffer_fps', 30)
        self.motion_direction_window = config.get('motion_direction_window', 2.0)
        self.motion_frame_buffer = deque(maxlen=self.motion_buffer_fps * self.motion_buffer_duration)
        self.initialize_components()
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

    async def process_frame(self, frame, model):
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
            

            
            combined_frame, person_counter = await self._detect_and_process_motion(frame,
                combined_frame1, combined_frame2, combined_frame3, motion_in_roi1, person_counter,
                motion_in_roi2, person_detected, person_in_roi3, thresh_ROI1, thresh_ROI2, detections)
            
            # return combined_frame, person_counter
            return combined_frame, thresh_ROI1
            

        
        except Exception as e:
            print(f"Error processing camera {self.camera_id}: {str(e)}")
            return frame, None

    async def _detect_and_process_motion(self, frame, combined_frame1, combined_frame2, combined_frame3, motion_in_roi1, 
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

                def upload_and_process(video_path, snapshot_path, camera_id, start_time ):
                    video_url = self.aws.upload_video_to_s3bucket(video_path, camera_id)
                    snapshot_url = self.aws.upload_snapshot_to_s3bucket(snapshot_path, camera_id)
                    self.mongo_handler.save_snapshot_to_mongodb(snapshot_url, start_time, camera_id)
                    self.mongo_handler.save_video_to_mongodb(video_url, start_time, camera_id)
                    self.email.send_alert_email(snapshot_path, video_url, camera_id)

                threading.Thread(target=upload_and_process, args=(self.video_path, self.snapshot_path,self.camera_id, start_time)).start()
                self.counter += 1
                    
        cv2.putText(combined_frame, f'Object Count: {self.object_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(combined_frame, f'Person Count: {person_counter}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        draw_boxes(combined_frame, detections)
        
        return combined_frame

class AsyncMultiCameraSystem:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.camera_processors = {}
        self.is_running = False
        self.websocket_clients = set()
        self.model = YOLO('/home/alluvium/Desktop/Namdeo/CMR_Project/yolov8n.engine', task = 'detect', verbose=True)

    async def initialize_cameras(self):
        for camera_name, camera_config in self.config['cameras'].items():
            camera_id = int(camera_name.split('_')[1])
            processor = AsyncCameraProcessor(camera_id, camera_config)
            await processor.stream.start()
            self.camera_processors[camera_id] = processor

    async def start(self):
        await self.initialize_cameras()
        self.is_running = True
        asyncio.create_task(self._process_cameras())

    async def _process_cameras(self):
        while self.is_running:
            for camera_id, processor in self.camera_processors.items():
                ret, frame = await processor.stream.read()
                if ret:
                    processed_frame, person_counter = await processor.process_frame(frame, self.model)
                    await self._broadcast_frame(camera_id, processed_frame)
            await asyncio.sleep(0.01)

    async def _broadcast_frame(self, camera_id: int, frame):
        for client in self.websocket_clients:
            try:
                await client.send_json({
                    "camera_id": camera_id,
                    "frame": frame.tolist()  # Convert to list for JSON serialization
                })
            except Exception as e:
                self.websocket_clients.remove(client)

    def stop(self):
        self.is_running = False
        for processor in self.camera_processors.values():
            processor.stream.stop()

# FastAPI Application
app = FastAPI(title="Async Camera Surveillance System")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global camera system instance
camera_system = AsyncMultiCameraSystem('config.yaml')

@app.on_event("startup")
async def startup_event():
    await camera_system.start()

@app.websocket("/ws/camera-feed")
async def websocket_camera_feed(websocket: WebSocket):
    await websocket.accept()
    camera_system.websocket_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        camera_system.websocket_clients.remove(websocket)

@app.get("/cameras/count")
async def get_camera_count():
    return {"camera_count": len(camera_system.camera_processors)}

@app.get("/cameras/object-counts")
async def get_object_counts():
    counts = {
        camera_id: processor.object_counter 
        for camera_id, processor in camera_system.camera_processors.items()
    }
    return {"counts": counts}

@app.get("/cameras/people-counts")
async def get_people_counts():
    counts = {
        camera_id: processor.current_people_in_roi3
        for camera_id, processor in camera_system.camera_processors.items()
    }
    return {"counts": counts}

@app.on_event("shutdown")
async def shutdown_event():
    camera_system.stop()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


